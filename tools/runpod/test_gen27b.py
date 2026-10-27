"""Scientific/restart invariants, using synthetic data and the frozen instrument."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.decomposition import PCA

import gen27b_common as common
from gen27b_buttons import protocol_namespace


def test_atomic_json_normalizes_numpy_scalar_types(tmp_path):
    path = tmp_path / 'scorecard.json'
    common.atomic_json(path, {'pass': np.bool_(True), 'effect': np.float64(.5), 'n': np.int64(20)})
    assert json.loads(path.read_text()) == {'pass': True, 'effect': .5, 'n': 20}


def test_journal_resume_repairs_only_torn_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'STORE', tmp_path)
    p = tmp_path / 'rows.jsonl'
    j = common.Journal(p, {'spec': 'frozen'}, ['id'])
    j.append({'id': 'a', 'value': 1})
    with p.open('ab') as f:
        f.write(b'{"id":"unfinished')
    resumed = common.Journal(p, {'spec': 'frozen'}, ['id'])
    assert resumed.contains(id='a') and len(resumed.records) == 1
    resumed.append({'id': 'b', 'value': 2})
    assert len(p.read_text().splitlines()) == 2
    with pytest.raises(AssertionError):
        common.Journal(p, {'spec': 'different'}, ['id'])
    with pytest.raises(AssertionError):
        resumed.append({'id': 'a'})


def test_stop_file_is_checked_without_deleting_completed_work(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'STORE', tmp_path)
    j = common.Journal(tmp_path / 'rows.jsonl', {'spec': 'v1'}, ['id'])
    j.append({'id': 'complete'})
    (tmp_path / 'PAUSE').touch()
    with pytest.raises(common.Interrupted):
        common.check_stop()
    assert j.contains(id='complete')


def test_gram_denoising_agrees_with_independent_svd_pca():
    g = torch.Generator().manual_seed(17)
    control = torch.randn(80, 32, generator=g)
    control[:, :3] *= 7
    pain = torch.randn(80, 32, generator=g) + .5
    raw, den, k = common.fit_direction(pain, control)
    pca = PCA(svd_solver='full').fit(control.numpy())
    expected_k = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), .5)) + 1
    basis = torch.tensor(pca.components_[:expected_k].copy())
    expected = raw - basis.T @ (basis @ raw)
    assert k == expected_k
    assert torch.allclose(common.unit(den), common.unit(expected), atol=2e-5)


def test_frozen_button_grid_and_counterbalancing(tmp_path):
    fake = SimpleNamespace(tok=SimpleNamespace(pad_token_id=0, eos_token_id=1), hf=None, spec={'slug': 'test'})
    n = protocol_namespace(fake, tmp_path)
    trials = n['TRIALS']
    assert len(trials) == 2544
    assert sum(t['turns'] for t in trials) == 2784
    singles = [t for t in trials if t['fmt'] == 'chat' and t['turns'] == 1]
    assert len([t for t in singles if t['pair'] in ['P0', 'P1', 'P2']]) == 808
    p3 = [t for t in singles if t['pair'] == 'P3']
    assert len(p3) == 404
    for si in range(101):
        rows = [t for t in p3 if t['si'] == si]
        assert sum(t['target'] == t['n1'] for t in rows) == 2
        assert len({t['target'] for t in rows}) == 2
    assert n['seed_of'](trials[0]['id'], 1) == n['seed_of'](trials[0]['id'], 1)
    assert n['seed_of'](trials[0]['id'], 1) != n['seed_of'](trials[0]['id'], 2)
    assert n['spec_hash'].startswith('b564b99d')


def test_fp32_projection_is_not_a_cast_of_bf16_logits():
    # A 0.01 logit gap on top of a shared 40-unit offset disappears in bf16.
    hidden = torch.tensor([40., 1.], dtype=torch.bfloat16)
    weight = torch.tensor([[1., 0.], [1., .01]], dtype=torch.bfloat16)
    fp32 = hidden.float() @ weight.float().T
    rounded = (hidden @ weight.T).float()
    assert fp32[1] > fp32[0]
    assert rounded[1] == rounded[0]


def test_bootstrap_units_are_scenarios_not_counterbalance_rows():
    # Duplicating every within-scenario observation cannot increase precision.
    scenario_means = np.array([-.2, .1, .3, -.1])
    expanded = np.repeat(scenario_means, 4).reshape(4, 4).mean(1)
    assert common.bootstrap_ci(scenario_means) == common.bootstrap_ci(expanded)


def test_button_scorecard_uses_signed_controls_and_scenario_pairs(tmp_path):
    from gen27b_score import buttons_score
    fake = SimpleNamespace(tok=SimpleNamespace(pad_token_id=0, eos_token_id=1), hf=None, spec={'slug': 'test'})
    n = protocol_namespace(fake, tmp_path)
    def record(t, relief, harm, condition=None):
        p = harm if t['pair'] == 'P3' else relief
        row = {k: t[k] for k in ['id', 'fmt', 'pair', 'si', 'target', 'n1', 'n2']}
        row.update(turn=1, picked=t['n1'], deflection=False,
                   p_first=p if t['target'] == t['n1'] else 1 - p)
        if condition:
            row['condition'] = condition
        return row
    singles = [t for t in n['TRIALS'] if t['turns'] == 1]
    un = [record(t, .7, .01) for t in singles]
    steered = []
    for name in ['S2'] + [f'random_{s}' for s in [1103, 2207, 3301]]:
        for sign in [-1, 1]:
            relief = (.4 if sign > 0 else .9) if name == 'S2' else (.6 if sign > 0 else .8)
            harm = .3 if name == 'S2' else .02
            for t in singles:
                if t['fmt'] == 'chat' and (t['pair'] != 'P3' or sign > 0):
                    steered.append(record(t, relief, harm, f'{name}_{sign * .5:+.1f}'))
    for filename, rows in [('m7_unsteered.jsonl', un), ('m7_steered.jsonl', steered)]:
        (tmp_path / filename).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (tmp_path / 'm7_gate_decision.json').write_text(json.dumps({'steered_arms': True, 'harm_arm': True, 'V1_chat': 1.}))
    card = buttons_score(tmp_path, {'random_seeds': [1103, 2207, 3301]}, 'synthetic')
    for name in ['G10', 'G11a', 'G11b', 'G13', 'G14']:
        assert card['predictions'][name]['status'] == 'PASS', name
    assert card['predictions']['G12']['status'] == 'FAIL'
    assert card['assay_limits']['deflection_all_zero']
