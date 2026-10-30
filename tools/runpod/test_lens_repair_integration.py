"""Opt-in CPU integration against copies of an already accepted real appendix."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lens_storage_guard import RowSet, digest, lens_keys
import lens_repair


@unittest.skipUnless(os.environ.get('RUN_REAL_STORAGE_INTEGRATION') == '1', 'opt-in real-artifact CPU test')
class RealArtifactIntegration(unittest.TestCase):
    def test_exact_repair_and_interrupted_marker_resume_on_isolated_real_copy(self):
        slug='qwen35-27b'
        real=lens_repair.ORIGINAL_STORE/'runs'/slug
        spec,h=lens_repair.specification(slug)
        names=['run_header.json','m1.complete.json','m2.complete.json','m3.complete.json',
               'm4.complete.json','m5.complete.json','m6.complete.json','m7.complete.json',
               'core_scorecard.json','m7_scorecard.json','lenses.complete.json','battery.complete.json',
               'appendix_scorecard.json','lens_ranks.jsonl','lens_provenance.json',
               'lens_family_gains.json','lens_family_gains.csv','probe_terms.csv','m2_direct_probes.csv']
        source={name:(real/name).read_bytes() for name in names}
        hashes={name:digest(data) for name,data in source.items()}
        wanted=lens_keys(spec)
        corpus=RowSet({'spec_sha256':h,'model_revision':spec['model_revision']},lens_repair.FIELDS)
        corpus.add(source['lens_ranks.jsonl'],'accepted real 3.5 appendix')
        self.assertEqual(len(corpus.records),69300)
        with tempfile.TemporaryDirectory(prefix='lens-repair-real-') as temp:
            temp=Path(temp);store=temp/'store';original=store/'runs'/slug
            original.mkdir(parents=True)
            for name,data in source.items():(original/name).write_bytes(data)
            chunks=original/'lens_ranks.durable-chunks';chunks.mkdir()
            (chunks/'prefix.jsonl').write_bytes(b'')
            (chunks/'consolidated.json').write_bytes((real/'lens_ranks.durable-chunks/consolidated.json').read_bytes())
            lines=source['lens_ranks.jsonl'].splitlines(keepends=True)
            (original/'lens_ranks.jsonl').write_bytes(b''.join(line for i,line in enumerate(lines) if i not in {13000,65000}))
            plan={'frozen_files':json.loads(source['run_header.json'])['code_sha256'],
                  'replay_cases':[],'missing_keys':[wanted[13000],wanted[65000]]}
            def stage(name):
                root=temp/name;directory=root/'runs'/slug;directory.mkdir(parents=True)
                for file in ('probe_terms.csv','m2_direct_probes.csv','lens_provenance.json'):
                    (directory/file).write_bytes(source[file])
                return root
            with patch.object(lens_repair,'ORIGINAL_STORE',store):
                first=stage('first')
                lens_repair.install_repaired(first,spec,h,corpus,wanted,plan)
                self.assertEqual((original/'lens_ranks.jsonl').read_bytes(),source['lens_ranks.jsonl'])
                for name in ('lens_family_gains.json','lens_family_gains.csv','appendix_scorecard.json'):
                    self.assertEqual((original/name).read_bytes(),source[name])
                # A crash after an invalidated marker must be safely resumable.
                (original/'lenses.complete.json').unlink()
                second=stage('second')
                lens_repair.install_repaired(second,spec,h,corpus,wanted,plan)
                self.assertEqual((original/'lens_ranks.jsonl').read_bytes(),source['lens_ranks.jsonl'])
                self.assertTrue((second/'ACCEPTED.json').exists())
                self.assertTrue(list((original/'storage-incidents').glob('recovery-*/BEFORE.json')))
        for name,wanted_hash in hashes.items():
            self.assertEqual(digest((real/name).read_bytes()),wanted_hash,'Production artifact changed: '+name)
        print(json.dumps({'real_cpu_integration':'PASS','rows':69300,
                          'truncated_copy_repaired':True,'interrupted_marker_resume':True,
                          'frozen_gain_tables_and_scorecard_byte_identical':True,
                          'production_artifacts_unchanged':True}),flush=True)


if __name__=='__main__':unittest.main()
