#!/usr/bin/env python3
"""Run one model's frozen battery, or the independent 4B environment gate."""
import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from gen27b_common import (LOCAL, PAPER, PROTOCOL, SOURCE, STORE, Interrupted, atomic_json,
                          complete, execution_metadata, fit_direction, install_signals, is_complete,
                          progress, read_spec, sha, unit)
from gen27b_runtime import Runtime


def anchor():
    spec = json.loads((PROTOCOL / 'anchor.json').read_text())
    output = STORE / 'anchor'
    output.mkdir(parents=True, exist_ok=True)
    old = output / 'drift_check.json'
    if old.exists():
        record = json.loads(old.read_text())
        assert record['environment_sha256'] == sha(STORE / 'environment.lock.txt')
        assert record['spec_sha256'] == sha(PROTOCOL / 'anchor.json') and record['pass']
        return
    r = Runtime(spec)
    fast = r.check_fast_path()
    rows = json.loads((SOURCE / 'datasets/3.1_pain_and_control_datasets.json').read_text())['datasets']['S2_1P']['sentences']
    x = r.capture([a['prompt'] for a in rows], layers=[30])[:, 0]
    y = np.array([a['category'] in ['A1', 'A2', 'A3', 'A4', 'A5'] for a in rows])
    groups = np.array([a['set'] for a in rows])
    scores = []
    for train, test in GroupKFold(5).split(x, y, groups):
        _, den, _ = fit_direction(x[train][y[train]], x[train][~y[train]])
        scores.append(float(roc_auc_score(y[test], (x[test] @ unit(den)).numpy())))
    ref_cv = pd.read_csv(PAPER.parent / 'denoised_direction_cv.csv')
    ref_auc = float(ref_cv[ref_cv.layer == 30].auc.mean())
    vectors = torch.load(PAPER / 'steering_vectors.pt', map_location='cpu', weights_only=True)
    s1 = torch.load(PAPER / 's1_comparison/s1_direction_block30.pt', map_location='cpu', weights_only=True)['matched_vector']
    references = [json.loads(a) for a in (PAPER / 'plane_sweep/generations.jsonl').read_text().splitlines()]
    comparisons = []
    for name, coeff, vector in [('baseline', 0., None), ('S2_-0.5', -.5, vectors['S2']), ('S1_-0.5', -.5, s1)]:
        ref = next(a for a in references if a['condition'] == name and a['prompt_idx'] == 0)
        mine = r.generate(ref['prompt'], coeff, vector)
        a, b = mine['token_ids'], ref['token_ids']
        first = next((i for i, (aa, bb) in enumerate(zip(a, b)) if aa != bb), None)
        if first is None and len(a) != len(b):
            first = min(len(a), len(b))
        comparisons.append({'condition': name, 'exact_tokens': a == b, 'first_divergence': first,
                            'ours': mine['text'], 'reference': ref['text']})
    record = {'spec_sha256': sha(PROTOCOL / 'anchor.json'), 'environment_sha256': sha(STORE / 'environment.lock.txt'),
              'pass': abs(np.mean(scores) - ref_auc) <= .005, 'reference_auc': ref_auc,
              'mean_auc': float(np.mean(scores)), 'folds': scores, 'fast_path_calls': fast,
              'generations_report_only': comparisons,
              'lens_gate': 'Deferred to appendix; core gate is independent of lens availability.'}
    atomic_json(old, record)
    assert record['pass'], '4B environment AUC drift exceeds 0.005; stop before 27B observations'
    progress('4B environment gate passed', auc=record['mean_auc'], reference=ref_auc)


def run(slug, phase):
    spec, digest = read_spec(slug)
    directory = STORE / 'runs' / slug
    directory.mkdir(parents=True, exist_ok=True)
    gate = json.loads((STORE / 'anchor/drift_check.json').read_text())
    assert gate['pass'] and gate['environment_sha256'] == sha(STORE / 'environment.lock.txt')
    header = execution_metadata(spec, digest)
    header_path = directory / 'run_header.json'
    if header_path.exists():
        previous = json.loads(header_path.read_text())
        for key in ['model_id', 'model_revision', 'spec_sha256', 'torch', 'transformers', 'gpu', 'environment_sha256']:
            assert previous[key] == header[key], f'Resume identity changed: {key}'
        if previous['code_sha256'] != header['code_sha256']:
            # Do not silently mix different runner implementations in a resumable run.
            raise RuntimeError('Runner changed since this model started; record and review an instrumentation amendment before resuming.')
    else:
        atomic_json(header_path, header)
    if phase == 'score':
        from gen27b_score import core_score, buttons_score
        core_score(directory, spec, digest)
        if is_complete(directory, 'm7', digest):
            buttons_score(directory, spec, digest)
        return
    runtime = Runtime(spec)
    fast = runtime.check_fast_path()
    atomic_json(directory / 'fast_path.json', {'calls': fast})
    from gen27b_core import Core
    core = Core(runtime, spec, digest, directory)
    if phase in ['all', 'core']:
        core.m1()
        atomic_json(directory / 'steering_gate.json', runtime.check_steering(runtime.vectors['S2']))
        core.run()
        from gen27b_score import core_score
        core_score(directory, spec, digest)
    else:
        assert is_complete(directory, 'm1', digest)
        core.m1()
        atomic_json(directory / 'steering_gate.json', runtime.check_steering(runtime.vectors['S2']))
    if phase in ['all', 'buttons']:
        from gen27b_buttons import Buttons
        from gen27b_score import buttons_score
        Buttons(runtime, spec, digest, directory).run()
        buttons_score(directory, spec, digest)
    if phase in ['all', 'lenses'] and not is_complete(directory, 'lenses', digest):
        try:
            from gen27b_lenses import readouts
            readouts(runtime, spec, digest, directory)
            from gen27b_score import appendix_score
            appendix_score(directory, spec, digest)
        except Interrupted:
            raise
        except Exception:
            atomic_json(directory / 'appendix_deferred.json', {'reason': traceback.format_exc(), 'spec_sha256': digest})
            print('Lens appendix deferred; core results are retained.', flush=True)
    if phase == 'all':
        assert all(is_complete(directory, name, digest) for name in ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7'])
        paths = [directory / p for p in ['run_header.json', 'core_scorecard.json', 'm7_scorecard.json', 'directions.pt',
                 'm1_grouped_cv.csv', 'm3_source_projections.csv', 'm4_self_other.csv', 'm5_matched_pairs.csv',
                 'm6_generations.jsonl', 'm7_unsteered.jsonl', 'steering_gate.json']]
        for extra in ['m7_steered.jsonl', 'appendix_scorecard.json', 'appendix_deferred.json']:
            if (directory / extra).exists():
                paths.append(directory / extra)
        complete(directory, 'battery', paths, digest)
    progress('model battery finished', model=slug, phase=phase)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--anchor', action='store_true')
    p.add_argument('--slug', choices=['qwen35-27b', 'qwen36-27b', 'qwen38-27b'])
    p.add_argument('--phase', choices=['all', 'core', 'buttons', 'lenses', 'score'], default='all')
    a = p.parse_args()
    install_signals()
    try:
        if a.anchor:
            anchor()
        else:
            assert a.slug
            run(a.slug, a.phase)
    except Interrupted as e:
        progress('paused', reason=str(e))
        sys.exit(130)
