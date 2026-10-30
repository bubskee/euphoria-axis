"""Storage-only launch wrapper for the unchanged frozen O-lens reader."""
import json
import runpy
import sys
from pathlib import Path

from lens_storage_guard import ImmutableJournal, digest, publish

FROZEN_READER = 'd56101743d6d6a90cce670f35dc8bb8d1ac7dd27e8cdb4721cf1aa55edb5e4ff'
FROZEN_SPEC = 'cda4add79a2232ae0909a3a69cea985b0f28f28ba5639a609fe6cd54f5044c5f'


def expected_panel(spec):
    cases = [case['id'] for case in spec['cases']]
    layers = spec['source_layers']
    return {
        'jr_readouts.jsonl': [(case, layer, lens) for lens in ('J', 'R')
                             for case in cases for layer in layers],
        'verbalizations.jsonl': [(case, layer, seed) for case in cases
                                for layer in layers for seed in spec['generation']['seeds']],
    }


def run(verify_only=False):
    import gen27b_common as common
    from lens_io_resume import install_pause_guard
    install_pause_guard()
    spec_path = common.REPO/'doc/protocols/olens36-appendix/spec.json'
    assert digest(spec_path.read_bytes()) == FROZEN_SPEC, 'Frozen O-lens spec changed'
    assert common.sha(common.REPO/'tools/runpod/olens36_appendix.py') == FROZEN_READER, 'Frozen O-lens reader changed'
    spec = json.loads(spec_path.read_text())
    assert len(spec['cases']) == 12 and len(spec['source_layers']) == 3
    wanted = expected_panel(spec)
    assert all(len(rows) == 72 and len(set(rows)) == 72 for rows in wanted.values())
    ImmutableJournal.expected_keys = wanted
    ImmutableJournal.registry = {}
    common.Journal = ImmutableJournal
    original_sha, original_atomic = common.sha, common.atomic_json
    out = common.STORE/'olens36-appendix'
    assert not (out/'FAILED.json').exists(), 'Preserve and review the prior O-lens failure before resuming'
    if verify_only:
        complete = json.loads((out/'COMPLETE.json').read_text())
        identity = json.loads((out/'run_header.json').read_text())
        assert complete['identity'] == identity
        assert identity['spec_sha256'] == FROZEN_SPEC and identity['runner_sha256'] == FROZEN_READER
        for name, keys in wanted.items():
            fields = ['case','layer','lens' if name == 'jr_readouts.jsonl' else 'seed']
            journal = ImmutableJournal(out/name, identity, fields)
            assert journal.require(keys) == complete['artifacts'][name]
        for name, expected_hash in complete['artifacts'].items():
            assert original_sha(out/name) == expected_hash, 'O-lens manifest mismatch: ' + name
        print('O-lens expected rows and all completion hashes verified', flush=True)
        return
    def guarded_sha(path):
        journal = ImmutableJournal.registry.get(str(Path(path)))
        if journal:
            journal.flush()
        return original_sha(path)
    def guarded_atomic(path, value):
        if Path(path) == out/'COMPLETE.json':
            for name, expected in wanted.items():
                journal = ImmutableJournal.registry[str(out/name)]
                checked_hash = journal.require(expected)
                assert value['artifacts'][name] == checked_hash, 'O-lens artifact hash changed at completion boundary'
            receipt = {'status':'EXACT JOURNAL COMPLETENESS VERIFIED',
                       'expected_rows':{name:len(rows) for name,rows in wanted.items()},
                       'guard_sha256':digest(Path(__file__).read_bytes()),
                       'journal_guard_sha256':digest(Path(__file__).with_name('lens_storage_guard.py').read_bytes()),
                       'numerical_reader_sha256':original_sha(common.REPO/'tools/runpod/olens36_appendix.py')}
            publish(out/'storage_verification.json', (json.dumps(receipt,indent=2)+'\n').encode())
        return original_atomic(path, value)
    common.sha, common.atomic_json = guarded_sha, guarded_atomic
    runpy.run_path(str(common.REPO/'tools/runpod/olens36_appendix.py'), run_name='__main__')
    assert (out/'COMPLETE.json').exists(), 'No accepted O-lens completion'
    for name, expected in wanted.items():
        journal = ImmutableJournal.registry[str(out/name)]
        assert original_sha(out/name) == journal.require(expected)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    run(parser.parse_args().verify_only)
