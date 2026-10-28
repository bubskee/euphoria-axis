"""Durability/resume checks against the original Journal's actual bytes."""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from gen27b_common import Journal
from lens_io_resume import ChunkedJournal, pause_present


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.identity = {'spec_sha256':'frozen', 'model_revision':'pinned'}
        self.keys = ['condition','prompt_idx','layer','lens']

    def tearDown(self):
        self.temp.cleanup()

    def row(self, n):
        return {'condition':'baseline','prompt_idx':n // 63,'layer':n % 63,
                'lens':'J','ranks':[n+1,5,42]}

    def test_identical_bytes_across_restart_and_post_consolidation_restart(self):
        ref = Journal(self.root/'reference.jsonl',self.identity,self.keys)
        old = Journal(self.root/'new.jsonl',self.identity,self.keys)
        for i in range(21):
            ref.append(self.row(i)); old.append(self.row(i))
        prefix = old.path.read_bytes()
        new = ChunkedJournal(old.path,self.identity,self.keys)
        for i in range(21,200):
            ref.append(self.row(i)); new.append(self.row(i))
        # Legacy artifact and all pre-existing row bytes remain untouched mid-run.
        self.assertEqual(old.path.read_bytes(),prefix)
        new = ChunkedJournal(old.path,self.identity,self.keys)
        for i in range(200,333):
            ref.append(self.row(i)); new.append(self.row(i))
        self.assertEqual(new.path.read_bytes(),ref.path.read_bytes())
        again = ChunkedJournal(old.path,self.identity,self.keys)
        self.assertEqual(len(again.records),333)
        self.assertEqual(again.path.read_bytes(),ref.path.read_bytes())
        self.assertEqual(again.prefix.read_bytes(),prefix)

    def test_torn_final_chunk_repairs_without_losing_completed_rows(self):
        j = ChunkedJournal(self.root/'r.jsonl',self.identity,self.keys)
        for i in range(140): j.append(self.row(i))
        with j.chunks[-1].open('ab') as f: f.write(b'{"torn":')
        r = ChunkedJournal(self.root/'r.jsonl',self.identity,self.keys)
        self.assertEqual(len(r.records),140)
        r.append(self.row(140))
        self.assertEqual(len(r.path.read_bytes().splitlines()),141)

    def test_duplicate_and_identity_changes_are_rejected(self):
        path=self.root/'r.jsonl';j=ChunkedJournal(path,self.identity,self.keys)
        j.append(self.row(0))
        with self.assertRaises(AssertionError): j.append(self.row(0))
        with self.assertRaises(AssertionError): ChunkedJournal(path,{'spec_sha256':'changed'},self.keys)

    def test_interior_corruption_is_never_silently_repaired(self):
        path=self.root/'r.jsonl';j=ChunkedJournal(path,self.identity,self.keys)
        for i in range(129): j.append(self.row(i))
        with j.chunks[0].open('ab') as f: f.write(b'not-json\n')
        with self.assertRaises(json.JSONDecodeError): ChunkedJournal(path,self.identity,self.keys)

    def test_real_empty_pause_stops_and_only_missing_object_means_no_pause(self):
        path = self.root/'PAUSE'
        self.assertFalse(pause_present(path))
        path.write_bytes(b'')
        self.assertTrue(pause_present(path))
        path.write_text('human pause')
        self.assertTrue(pause_present(path))
        # Directory metadata can claim the deleted object exists while reads return ENOENT.
        with patch.object(Path,'open',side_effect=FileNotFoundError()):
            self.assertFalse(pause_present(path))
        with patch.object(Path,'open',side_effect=PermissionError()):
            with self.assertRaises(PermissionError): pause_present(path)

    def test_deleted_object_cache_is_refreshed_before_read(self):
        path = self.root/'PAUSE';path.write_text('stale cached entry')
        def refresh(p, attr, value):
            self.assertEqual(attr,'.invalidate')
            p.unlink()
        with patch('lens_io_resume.os.setxattr',side_effect=refresh):
            self.assertFalse(pause_present(path))


if __name__ == '__main__': unittest.main()
