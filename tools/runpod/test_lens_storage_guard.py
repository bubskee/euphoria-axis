"""Integrity regression tests for stale/short object-store views and recovery."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lens_storage_guard import RowSet, CheckedReplay, ImmutableJournal, encode, publish, read_exact, read_sealed


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.identity = {'spec_sha256': 'frozen', 'model_revision': 'pinned'}
        self.fields = ['condition', 'prompt_idx', 'layer', 'lens']

    def tearDown(self):
        self.temp.cleanup()

    def row(self, n):
        return {'condition': 'baseline', 'prompt_idx': 0, 'layer': n,
                'lens': 'J', 'ranks': [n + 1, 7], **self.identity}

    def test_short_read_cannot_pass_by_hashing_itself(self):
        rows = RowSet(self.identity, self.fields)
        rows.add(encode(self.row(0)), 'short but valid JSON/hash')
        with self.assertRaisesRegex(ValueError, '1 missing'):
            rows.materialize([rows.key(self.row(0)), rows.key(self.row(1))])

    def test_mirrors_merge_exact_original_bytes_and_reject_changed_rank(self):
        rows = RowSet(self.identity, self.fields)
        a, b = encode(self.row(0)), encode(self.row(1))
        rows.add(a, 'legacy'); rows.add(a + b, 'complete source')
        self.assertEqual(rows.materialize([rows.key(self.row(i)) for i in range(2)]), a + b)
        bad = self.row(0); bad['ranks'][0] += 1
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            rows.add(encode(bad), 'conflicting observation')

    def test_wrong_identity_duplicate_and_torn_observations_fail(self):
        rows = RowSet(self.identity, self.fields)
        bad = self.row(0); bad['model_revision'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'Identity'):
            rows.add(encode(bad), 'wrong model')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            rows.add(encode(self.row(0)) * 2, 'duplicate')
        with self.assertRaisesRegex(ValueError, 'Unterminated'):
            rows.add(encode(self.row(0))[:-1], 'torn')

    def test_read_back_mismatch_is_a_failure_not_a_new_expected_hash(self):
        path = self.root / 'file'; path.write_bytes(b'prefix')
        with self.assertRaises(IOError): read_exact(path, b'prefix and tail', attempts=1)
        with patch.object(Path, 'read_bytes', return_value=b'stale'):
            with patch('lens_storage_guard.time.sleep'):
                with self.assertRaises(IOError): publish(path, b'complete')

    def test_wal_recovers_a_truncated_mutable_alias_without_recomputing(self):
        path = self.root / 'journal.jsonl'
        with patch.object(ImmutableJournal, 'root_mirror', self.root / 'mirror'):
            first = ImmutableJournal(path, self.identity, self.fields)
            for i in range(3):
                row = self.row(i)
                for k in self.identity: row.pop(k)
                first.append(row)
            expected = b''.join(encode(self.row(i)) for i in range(3))
            self.assertEqual(path.read_bytes(), expected)
            path.write_bytes(encode(self.row(0)))
            resumed = ImmutableJournal(path, self.identity, self.fields)
            self.assertEqual(len(resumed.records), 3)
            self.assertEqual(resumed.require([resumed.key(self.row(i)) for i in range(3)]),
                             __import__('hashlib').sha256(expected).hexdigest())
            self.assertEqual(path.read_bytes(), expected)
            self.assertTrue(list((self.root / 'mirror').rglob('*.jsonl')))
            self.assertTrue(list(path.parent.glob('journal.immutable-wal/*/*.jsonl')))

    def test_unacknowledged_network_wal_failure_does_not_complete_row(self):
        path = self.root / 'j.jsonl'
        with patch.object(ImmutableJournal, 'root_mirror', self.root / 'mirror'):
            journal = ImmutableJournal(path, self.identity, self.fields)
            with patch('lens_storage_guard.seal', side_effect=[self.root / 'mirror-object', IOError('volume down')]):
                with self.assertRaisesRegex(IOError, 'volume down'):
                    journal.append({k:v for k,v in self.row(0).items() if k not in self.identity})
            self.assertFalse(journal.records)

    def test_replay_requires_all_known_rows_to_match_before_completing(self):
        rows=RowSet(self.identity,self.fields)
        rows.add(encode(self.row(0)),'original')
        expected=[rows.key(self.row(i)) for i in range(2)]
        persisted=[]
        replay=CheckedReplay(rows,expected,{('baseline',0)},persisted.append)
        replay.append({k:v for k,v in self.row(1).items() if k not in self.identity})
        with self.assertRaisesRegex(ValueError,'Not every surviving'):
            replay.finish()
        replay.append({k:v for k,v in self.row(0).items() if k not in self.identity})
        self.assertEqual(replay.finish(),encode(self.row(0))+encode(self.row(1)))
        self.assertEqual(persisted,[encode(self.row(1))])
        self.assertEqual(len(replay.added),1)

    def test_replay_mismatch_does_not_replace_original_or_persist_rank(self):
        rows=RowSet(self.identity,self.fields);original=encode(self.row(0))
        rows.add(original,'original');persisted=[]
        replay=CheckedReplay(rows,[rows.key(self.row(0))],{('baseline',0)},persisted.append)
        bad=self.row(0);bad['ranks'][0]+=1
        with self.assertRaisesRegex(ValueError,'disagrees'):
            replay.append({k:v for k,v in bad.items() if k not in self.identity})
        self.assertEqual(rows.raw[rows.key(self.row(0))],original)
        self.assertFalse(persisted)

    def test_resumed_journal_retains_frozen_iteration_order(self):
        path=self.root/'ordered.jsonl'
        order=[('baseline',0,i,'J') for i in range(3)]
        with patch.object(ImmutableJournal,'root_mirror',self.root/'mirror'), \
             patch.object(ImmutableJournal,'expected_keys',{'ordered.jsonl':order}):
            j=ImmutableJournal(path,self.identity,self.fields)
            for i in (2,0,1):j.append({k:v for k,v in self.row(i).items() if k not in self.identity})
            self.assertEqual(list(j.records),order)
            self.assertEqual(path.read_bytes(),b''.join(encode(self.row(i)) for i in range(3)))

    def test_sealed_hash_rejects_valid_json_from_a_short_generation(self):
        data=encode(self.row(0))+encode(self.row(1))
        name=__import__('hashlib').sha256(data).hexdigest()+'-unique.jsonl'
        p=self.root/name;p.write_bytes(encode(self.row(0)))
        with self.assertRaisesRegex(IOError,'hash mismatch'):read_sealed(p,attempts=1)

    def test_uncommitted_pending_objects_are_not_resume_inputs(self):
        path=self.root/'pending.jsonl'
        wal=self.root/'pending.immutable-wal'/'aa';wal.mkdir(parents=True)
        (wal/'interrupted.jsonl.pending').write_bytes(b'{"unfinished')
        with patch.object(ImmutableJournal,'root_mirror',self.root/'mirror'):
            j=ImmutableJournal(path,self.identity,self.fields)
            self.assertFalse(j.records)

    def approved_scope(self):
        keys=[('baseline',i,0,'J') for i in range(23)]
        return {**self.identity,'status':'APPROVED BY USER',
                'human_authorization':'Approve the 23-prompt recovery','model':'qwen36-27b',
                'missing_rank_keys':keys,'exact_original_prompt_conditions':[key[:2] for key in keys],
                'requested_case_bound':23}

    def test_approved_scope_allows_only_exact_original_keys_or_recovered_subset(self):
        from lens_storage_guard import approved_case_bound
        scope=self.approved_scope()
        self.assertEqual(approved_case_bound(scope,'qwen36-27b',self.identity,scope['missing_rank_keys']),23)
        self.assertEqual(approved_case_bound(scope,'qwen36-27b',self.identity,scope['missing_rank_keys'][:2]),23)
        with self.assertRaises(AssertionError):
            approved_case_bound(scope,'qwen36-27b',self.identity,[('baseline',24,0,'J')])

    def test_scope_cannot_apply_to_other_models_or_changed_identities(self):
        from lens_storage_guard import approved_case_bound
        scope=self.approved_scope()
        with self.assertRaises(AssertionError):approved_case_bound(scope,'qwen38-27b',self.identity,[])
        with self.assertRaises(AssertionError):approved_case_bound(scope,'qwen36-27b',{'spec_sha256':'other'},[])

    def test_scope_requires_actual_authorization_and_exact_case_bound(self):
        from lens_storage_guard import approved_case_bound
        scope=self.approved_scope();scope['status']='AWAITING AUTHORIZATION'
        with self.assertRaises(AssertionError):approved_case_bound(scope,'qwen36-27b',self.identity,[])
        scope=self.approved_scope();scope['requested_case_bound']=24
        with self.assertRaises(AssertionError):approved_case_bound(scope,'qwen36-27b',self.identity,[])

    def test_interrupted_marker_publication_recovers_preserved_metadata(self):
        from lens_storage_guard import seal, digest
        from lens_repair import preserved_artifact
        original=self.root/'model';before=original/'storage-incidents/recovery-test'
        data=b'{"original":"completion manifest"}\n'
        saved=seal(before/'before',data,'.json')
        (before/'BEFORE.json').write_text(json.dumps({'before':{'battery.complete.json':
            {'immutable_backup':str(saved),'sha256':digest(data)}}}))
        self.assertEqual(preserved_artifact(original,'battery.complete.json'),data)


if __name__ == '__main__': unittest.main()
