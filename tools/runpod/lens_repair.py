"""Auditable storage recovery around the unchanged frozen numerical runners.

Original artifacts are retained. A GPU replay, only when necessary, repeats only
affected original prompts and must exactly reproduce every surviving rank row.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from lens_storage_guard import RowSet, CheckedReplay, digest, encode, lens_keys, publish, seal, read_sealed, approved_case_bound

FIELDS = ['condition', 'prompt_idx', 'layer', 'lens']
REPO = Path(__file__).resolve().parents[2]
LOCAL = Path(os.environ.get('GEN27B_LOCAL', '/root/gen27b'))
ORIGINAL_STORE = Path(os.environ.get('GEN27B_STORE', '/workspace/gen27b'))


def metadata(path, value):
    publish(Path(path), (json.dumps(value, indent=2) + '\n').encode())


def specification(slug):
    path = REPO / 'doc/protocols/gen27b' / (slug + '.json')
    spec = json.loads(path.read_text())
    assert spec['status'] == 'frozen-before-27b-data'
    return spec, digest(path.read_bytes())


def stop_check():
    from lens_io_resume import install_pause_guard
    install_pause_guard()
    import gen27b_common
    gen27b_common.check_stop()


def verify_frozen(slug):
    import gen27b_common as common
    spec, h = common.read_spec(slug)
    original = ORIGINAL_STORE / 'runs' / slug
    header = json.loads((original / 'run_header.json').read_text())
    assert header['spec_sha256'] == h and header['model_revision'] == spec['model_revision']
    for name, wanted in header['code_sha256'].items():
        assert common.sha(REPO / 'tools/runpod' / name) == wanted, name
    assert common.sha(ORIGINAL_STORE / 'environment.lock.txt') == header['environment_sha256']
    return spec, h


def read_corpus(slug, active=False):
    spec, h = verify_frozen(slug)
    identity = {'spec_sha256': h, 'model_revision': spec['model_revision']}
    corpus = RowSet(identity, FIELDS)
    directory = ORIGINAL_STORE / 'runs' / slug
    sources = []
    # Immutable historical observations rescue rows from regressed later views.
    archive_root = LOCAL / 'monitor-backups'
    for archive in sorted(archive_root.glob(slug.replace('-27b', '') + '-lens*.tar.gz')):
        sidecar = Path(str(archive) + '.json')
        if not sidecar.exists():
            continue
        meta = json.loads(sidecar.read_text())
        assert digest(archive.read_bytes()) == meta['archive_sha256'], archive
        with tarfile.open(archive, 'r:gz') as t:
            for member in t.getmembers():
                name = Path(member.name).name
                if name == 'lens_ranks.jsonl' or name == 'prefix.jsonl' or (name.startswith('chunk-') and name.endswith('.jsonl')):
                    data = t.extractfile(member).read()
                    corpus.add(data, str(archive) + ':' + member.name)
                    sources.append({'source': str(archive) + ':' + member.name, 'sha256': digest(data)})
    for mirror in (LOCAL / 'storage-recovery/observations' / slug,
                   ORIGINAL_STORE / 'storage-recovery/observations' / slug):
        if mirror.exists():
            for path in sorted(mirror.glob('*.jsonl')):
                data = read_sealed(path)
                corpus.add(data, str(path)); sources.append({'source': str(path), 'sha256': digest(data)})
    # Reuse replay data only after a durable receipt proves both the exact
    # surviving-row comparisons and byte-identical frozen scientific outputs.
    receipts = ORIGINAL_STORE / 'storage-recovery/validated-replays' / slug
    if receipts.exists():
        for path in sorted(receipts.glob('*.json')):
            receipt = json.loads(read_sealed(path))
            assert receipt['spec_sha256'] == h
            assert receipt['frozen_science_outputs_byte_identical'] is True
            for name, wanted_hash in receipt['frozen_scientific_outputs'].items():
                assert digest((directory / name).read_bytes()) == wanted_hash
            candidate = read_sealed(receipt['candidate'])
            assert digest(candidate) == receipt['canonical_sha256']
            corpus.add(candidate, str(path))
            sources.append({'source': str(path), 'sha256': digest(candidate)})
    chunks = directory / 'lens_ranks.durable-chunks'
    paths = [directory / 'lens_ranks.jsonl', chunks / 'prefix.jsonl']
    selected = sorted(chunks.glob('chunk-*.jsonl'))
    if active:
        selected = selected[:-1]
    paths += selected
    for path in paths:
        if path.exists():
            data = path.read_bytes()
            corpus.add(data, str(path)); sources.append({'source': str(path), 'sha256': digest(data)})
    wanted = lens_keys(spec)
    assert not (set(corpus.records) - set(wanted)), 'Unexpected condition/prompt/layer/lens'
    probes = list(csv.DictReader((directory / 'probe_terms.csv').open()))
    for row in corpus.records.values():
        assert len(row['ranks']) == len(probes)
        assert all(isinstance(r, int) and r > 0 for r in row['ranks'])
    return spec, h, corpus, wanted, sources


def observe(slug, interval):
    """Passive closed-chunk snapshots; never invalidate a live writer's inodes."""
    spec, h = verify_frozen(slug)
    identity = {'spec_sha256': h, 'model_revision': spec['model_revision']}
    seen = {}
    destination = ORIGINAL_STORE / 'storage-recovery/observations' / slug
    local = LOCAL / 'storage-recovery/observations' / slug
    directory = ORIGINAL_STORE / 'runs' / slug / 'lens_ranks.durable-chunks'
    while True:
        stop_check()
        finished = (ORIGINAL_STORE / 'ALL_APPENDICES_ATTEMPTED').exists()
        paths = sorted(directory.glob('chunk-*.jsonl'))
        if not finished:
            paths = paths[:-1]
        for path in paths:
            if seen.get(path.name, b'').count(b'\n') == 128:
                continue
            stop_check()
            data = path.read_bytes()
            previous = seen.get(path.name, b'')
            shorter, longer = sorted((previous, data), key=len)
            assert longer.startswith(shorter), 'Conflicting chunk observation: ' + path.name
            if len(data) <= len(previous):
                continue
            rows = RowSet(identity, FIELDS); rows.add(data, str(path))
            name = path.stem + '-' + digest(data) + '.jsonl'
            local.mkdir(parents=True, exist_ok=True)
            # Keep independent POSIX evidence, then acknowledge a persistent copy.
            publish(local / name, data)
            seal(destination, data)
            seen[path.name] = data
        if finished:
            break
        time.sleep(interval)


def prepare_stage(slug, approved_scope=None):
    spec, h, corpus, wanted, sources = read_corpus(slug)
    scope = None
    if approved_scope:
        scope = json.loads(Path(approved_scope).read_text())
        reference = Path(scope['known_original_reference_path']).read_bytes()
        assert digest(reference) == scope['known_original_reference_sha256']
        corpus.add(reference, 'explicitly approved preserved original reference')
    missing = corpus.missing(wanted)
    cases = sorted({k[:2] for k in missing})
    bound = 16
    if scope is not None:
        bound = approved_case_bound(scope, slug, corpus.identity, missing)
        original = ORIGINAL_STORE/'runs'/slug
        assert scope['frozen_numerical_sources'] == json.loads((original/'run_header.json').read_text())['code_sha256']
        for name, expected_hash in scope['frozen_scientific_output_sha256'].items():
            assert digest((original/name).read_bytes()) == expected_hash, name
    assert len(cases) <= bound and len(missing) <= len(wanted) * .02, 'Recovery exceeds bounded missing-row scope'
    run_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
    stage_store = LOCAL / 'storage-recovery/stages' / (slug + '-' + run_id)
    stage = stage_store / 'runs' / slug
    stage.mkdir(parents=True, exist_ok=False)
    original = ORIGINAL_STORE / 'runs' / slug
    names = ['run_header.json', 'm1.complete.json', 'directions.pt', 'm1_summary.json',
             'm1_grouped_cv.csv', 'probe_terms.csv', 'm2_direct_probes.csv', 'lens_provenance.json']
    for name in names:
        (stage / name).write_bytes((original / name).read_bytes())
    (stage_store / 'anchor').mkdir()
    (stage_store / 'anchor/drift_check.json').write_bytes((ORIGINAL_STORE / 'anchor/drift_check.json').read_bytes())
    (stage_store / 'environment.lock.txt').write_bytes((ORIGINAL_STORE / 'environment.lock.txt').read_bytes())
    reference = b''.join(corpus.raw[k] for k in wanted if k in corpus.records)
    publish(stage_store / 'reference.jsonl', reference)
    immutable_reference = seal(ORIGINAL_STORE / 'storage-recovery/references' / slug, reference)
    plan = {'slug': slug, 'spec_sha256': h, 'model_revision': spec['model_revision'],
            'reference_sha256': digest(reference), 'missing_keys': missing, 'replay_cases': cases,
            'immutable_reference': str(immutable_reference),
            'expected_rows': len(wanted), 'source_observations': sources,
            'frozen_files': json.loads((original / 'run_header.json').read_text())['code_sha256'],
            'original_store': str(ORIGINAL_STORE), 'stage_store': str(stage_store),
            'authorization': 'User requested storage repair and missing-row recovery while 3.6 continues.'}
    if scope is not None:
        plan['explicit_scope_authorization'] = scope
        plan['explicit_scope_sha256'] = digest(Path(approved_scope).read_bytes())
    (stage_store / 'recovery_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    seal(ORIGINAL_STORE / 'storage-recovery/plans', (json.dumps(plan, indent=2) + '\n').encode(), '.json')
    return stage_store, spec, h, corpus, wanted, plan


def verify_scientific_outputs(stage, spec, h, corpus, wanted):
    """Execute the frozen postprocessing statements verbatim, only on the CPU."""
    import numpy as np
    import pandas as pd
    import gen27b_common as common
    from gen27b_score import appendix_score
    stage = Path(stage)
    publish(stage / 'lens_ranks.jsonl', corpus.materialize(wanted))
    source = REPO / 'tools/runpod/gen27b_lenses.py'
    tree = ast.parse(source.read_text(), filename=str(source))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'readouts')
    start = next(i for i,n in enumerate(function.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id == 'gains' for t in n.targets))
    tail = ast.Module(body=function.body[start:], type_ignores=[])
    namespace = {'np': np, 'pd': pd, 'journal': SimpleNamespace(records=corpus.records, path=stage/'lens_ranks.jsonl'),
                 'd': stage, 'spec_sha': h, 'lenses': dict.fromkeys(spec['lenses']),
                 'layers': list(range(spec['n_layers']-1)),
                 'arms': [('baseline',0,None)] + [(c,0,None) for c in dict.fromkeys(k[0] for k in wanted) if c!='baseline'],
                 'probes': pd.read_csv(stage/'probe_terms.csv'), 'atomic_json': common.atomic_json,
                 'complete': common.complete}
    exec(compile(tail, str(source), 'exec'), namespace)
    appendix_score(stage, spec, h)
    original = ORIGINAL_STORE / 'runs' / spec['slug']
    for name in ('lens_family_gains.json','lens_family_gains.csv','appendix_scorecard.json'):
        assert (stage/name).read_bytes() == (original/name).read_bytes(), 'Frozen science output changed: ' + name


def replay_worker(stage_store):
    """Run the original numerical function; only its journal object is substituted."""
    stage_store = Path(stage_store)
    plan = json.loads((stage_store/'recovery_plan.json').read_text())
    original_store = Path(plan['original_store'])
    # Must be set before importing any frozen module.
    os.environ['GEN27B_STORE'] = str(stage_store)
    import gen27b_common as common
    from lens_io_resume import install_pause_guard
    install_pause_guard()
    # Stop signals and the real user PAUSE always refer to the original store.
    from lens_io_resume import pause_present
    def check_stop():
        if common.STOP or pause_present(original_store/'PAUSE'):
            raise common.Interrupted('Recovery paused; original units preserved')
    common.check_stop = check_stop
    common.install_signals()
    spec,h = common.read_spec(plan['slug'])
    assert h == plan['spec_sha256'] and spec['model_revision'] == plan['model_revision']
    for name,wanted_sha in plan['frozen_files'].items():
        assert common.sha(REPO/'tools/runpod'/name) == wanted_sha
    reference = (stage_store/'reference.jsonl').read_bytes(); assert digest(reference)==plan['reference_sha256']
    identity={'spec_sha256':h,'model_revision':spec['model_revision']}
    rows=RowSet(identity,FIELDS);rows.add(reference,'preserved originals')
    expected=lens_keys(spec); replay_cases={tuple(x) for x in plan['replay_cases']}
    replay=CheckedReplay(rows,expected,replay_cases,
        lambda data:seal(original_store/'storage-recovery/replayed-rows'/plan['slug'],data))
    class ReplayJournal:
        def __init__(self,path,identity_arg,fields):
            assert identity_arg==identity and fields==FIELDS
            self.records=rows.records;self.canonical=Path(path)
        def contains(self,**row):
            return replay.contains(**row)
        def append(self,row):
            replay.append(row)
        @property
        def path(self):
            publish(self.canonical,replay.finish())
            return self.canonical
    import gen27b_lenses
    gen27b_lenses.Journal=ReplayJournal
    from gen27b_run import run
    run(plan['slug'],'lenses')
    required={k for k in expected if k[:2] in replay_cases}
    assert replay.verified==required and set(replay.added)=={tuple(x) for x in plan['missing_keys']}
    assert common.is_complete(stage_store/'runs'/plan['slug'],'lenses',h)
    result={'status':'FROZEN REPLAY VERIFIED','replayed_cases':plan['replay_cases'],
            'matched_original_rows':len(replay.verified)-len(replay.added),'new_missing_rows':replay.added,
            'numerical_source_hashes':plan['frozen_files'],'spec_sha256':h}
    (stage_store/'replay_verification.json').write_text(json.dumps(result,indent=2)+'\n')


def preserved_artifact(original, name):
    """Resume an interrupted metadata transaction from its immutable before-copy."""
    path = Path(original) / name
    try:
        return path.read_bytes()
    except FileNotFoundError:
        for receipt in sorted((Path(original)/'storage-incidents').glob('recovery-*/BEFORE.json'), reverse=True):
            record = json.loads(receipt.read_text()).get('before', {}).get(name)
            if record:
                data = read_sealed(record['immutable_backup'])
                assert digest(data) == record['sha256']
                return data
        raise


def install_repaired(stage_store, spec, h, corpus, wanted, plan):
    original=ORIGINAL_STORE/'runs'/spec['slug'];stage=Path(stage_store)/'runs'/spec['slug']
    body=corpus.materialize(wanted)
    prefix=(original/'lens_ranks.durable-chunks/prefix.jsonl').read_bytes()
    assert body.startswith(prefix), 'Legacy prefix must remain byte-identical'
    verify_scientific_outputs(stage,spec,h,corpus,wanted)
    stop_check()
    validated = ORIGINAL_STORE/'storage-recovery/validated-replays'/spec['slug']
    candidate = seal(validated/'objects',body)
    receipt = {'spec_sha256':h,'model_revision':spec['model_revision'],
               'canonical_sha256':digest(body),'candidate':str(candidate),
               'frozen_science_outputs_byte_identical':True,
               'frozen_scientific_outputs':{name:digest((original/name).read_bytes())
                    for name in ('lens_family_gains.json','lens_family_gains.csv','appendix_scorecard.json')},
               'replayed_cases':plan['replay_cases'],'original_missing_keys':plan['missing_keys'],
               'numerical_source_hashes':plan['frozen_files']}
    seal(validated,(json.dumps(receipt,indent=2)+'\n').encode(),'.json')
    old={name:preserved_artifact(original,name) for name in
         ('lens_ranks.jsonl','lenses.complete.json','battery.complete.json','lens_ranks.durable-chunks/consolidated.json')}
    import gen27b_common as common
    consolidated=json.loads(old['lens_ranks.durable-chunks/consolidated.json'])
    try:
        already_valid = (old['lens_ranks.jsonl']==body and
                         consolidated.get('records')==len(wanted) and
                         consolidated.get('canonical_sha256')==digest(body) and
                         common.is_complete(original,'lenses',h) and common.is_complete(original,'battery',h))
    except (AssertionError, FileNotFoundError):
        already_valid = False  # Metadata from an interrupted publication is repairable.
    if already_valid:
        metadata(Path(stage_store)/'ACCEPTED.json',{'status':'ALREADY COMPLETE AND FROZEN OUTPUTS VERIFIED','rows':len(wanted)})
        return
    incident=original/'storage-incidents'/('recovery-'+Path(stage_store).name)
    before={}
    for name,data in old.items():
        saved=seal(incident/'before',data,'.jsonl' if name.endswith('jsonl') else '.json')
        before[name]={'sha256':digest(data),'immutable_backup':str(saved)}
    repair={'status':'STORAGE-ONLY REPAIR','rows':len(wanted),'canonical_sha256':digest(body),
            'before':before,'spec_sha256':h,'model_revision':spec['model_revision'],
            'prefix_sha256':digest(prefix),'frozen_science_outputs_byte_identical':True,
            'frozen_numerical_source_hashes':plan['frozen_files'],
            'replayed_cases':plan['replay_cases'],'recovered_missing_keys':plan['missing_keys'],
            'tool_sha256':digest(Path(__file__).read_bytes()),
            'guard_sha256':digest(Path(__file__).with_name('lens_storage_guard.py').read_bytes())}
    metadata(incident/'BEFORE.json',repair)
    # Remove only already-preserved invalid markers before changing their artifact.
    for name in ('lenses.complete.json','battery.complete.json'):
        try:(original/name).unlink()
        except FileNotFoundError:pass  # An interrupted repair already archived it.
    publish(original/'lens_ranks.jsonl',body)
    metadata(original/'storage_recovery.json',repair)
    consolidated.update(canonical_sha256=digest(body),records=len(wanted),
                        actual_verified_rows=len(wanted),storage_recovery_sha256=digest((original/'storage_recovery.json').read_bytes()))
    metadata(original/'lens_ranks.durable-chunks/consolidated.json',consolidated)
    lenses=json.loads(old['lenses.complete.json']);lenses['artifacts']['lens_ranks.jsonl']=digest(body)
    lenses['artifacts']['storage_recovery.json']=digest((original/'storage_recovery.json').read_bytes())
    metadata(original/'lenses.complete.json',lenses)
    battery=json.loads(old['battery.complete.json']);battery['artifacts']['lenses.complete.json']=digest((original/'lenses.complete.json').read_bytes())
    battery['artifacts']['storage_recovery.json']=lenses['artifacts']['storage_recovery.json']
    metadata(original/'battery.complete.json',battery)
    assert (original/'lens_ranks.jsonl').read_bytes()==body
    assert common.is_complete(original,'lenses',h) and common.is_complete(original,'battery',h)
    metadata(Path(stage_store)/'ACCEPTED.json',{'status':'REPAIRED AND VERIFIED','rows':len(wanted),'repair':repair})


def repair(slug, allow_replay, approved_scope=None):
    stop_check()
    stage_store,spec,h,corpus,wanted,plan=prepare_stage(slug,approved_scope)
    print(json.dumps({'stage_store':str(stage_store),'missing_rows':len(plan['missing_keys']),
                      'replay_cases':plan['replay_cases']}),flush=True)
    if plan['missing_keys']:
        if not allow_replay:
            metadata(stage_store/'NEEDS_GPU.json',plan);return 3
        # A replay can evict disposable weights and use the GPU only under the
        # same inherited lock that protects the original research controller.
        import fcntl
        lock_fd = int(os.environ['GEN27B_RECOVERY_LOCK_FD'])
        assert os.fstat(lock_fd).st_ino == (LOCAL/'battery.lock').stat().st_ino
        assert os.fstat(lock_fd).st_dev == (LOCAL/'battery.lock').stat().st_dev
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import subprocess
        model_path=LOCAL/'models'/spec['model_id'].split('/')[-1]
        if not (model_path/'VERIFIED.json').exists():
            # Delete only disposable model weights, after acquiring the GPU lock.
            for other in (LOCAL/'models').iterdir():
                if other.is_dir() and other!=model_path:shutil.rmtree(other)
            subprocess.run([sys.executable,str(REPO/'tools/runpod/gen27b_fetch.py'),
                            spec['model_id'],spec['model_revision'],str(model_path)],check=True)
        env={**os.environ,'GEN27B_STORE':str(stage_store)}
        subprocess.run([sys.executable,__file__,'replay-worker','--stage-store',str(stage_store)],env=env,check=True)
        recovered=(stage_store/'runs'/slug/'lens_ranks.jsonl').read_bytes()
        corpus.add(recovered,'verified frozen replay stage')
    install_repaired(stage_store,spec,h,corpus,wanted,plan)
    return 0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['repair','observe','replay-worker'])
    parser.add_argument('--slug',choices=['qwen35-27b','qwen36-27b','qwen38-27b'])
    parser.add_argument('--allow-replay',action='store_true')
    parser.add_argument('--approved-scope',type=Path)
    parser.add_argument('--stage-store',type=Path)
    parser.add_argument('--interval',type=int,default=60)
    args=parser.parse_args()
    if args.mode=='replay-worker':replay_worker(args.stage_store);return 0
    import gen27b_common as common
    common.install_signals()
    if args.mode=='observe':observe(args.slug,args.interval);return 0
    return repair(args.slug,args.allow_replay,args.approved_scope)


if __name__=='__main__':raise SystemExit(main())
