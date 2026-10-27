"""CPU-only scorecards; preserve negative results and unavailable measurements."""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from gen27b_common import PAPER, atomic_json, bootstrap_ci


def verdict(ok, **details):
    return {'status': 'PASS' if bool(ok) else 'FAIL', **details}


def ci(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return bootstrap_ci(values) if len(values) else None


def core_score(directory, spec, spec_sha):
    d = Path(directory)
    card = {'spec_sha256': spec_sha, 'model_id': spec['model_id'], 'predictions': {}}
    cv = pd.read_csv(d / 'm1_grouped_cv.csv')
    fixed = cv[(cv.block == 62) & (cv.method == 'denoised')].auc.mean()
    card['predictions']['G1'] = verdict(fixed >= .9, fixed_layer=62, mean_fold_auc=float(fixed))
    card['best_layer_sensitivity'] = cv[cv.method == 'denoised'].groupby('block').auc.mean().sort_values(ascending=False).head(5).to_dict()
    direct = pd.read_csv(d / 'm2_direct_probes.csv')
    family = direct[(direct.direction == 'S2') & (direct.sign == -1)].groupby('family').z_logit.mean()
    card['predictions']['G2'] = verdict(family['calm_positive'] > family['activated_vitality'],
                                       calm=float(family['calm_positive']), vitality=float(family['activated_vitality']))
    so = pd.read_csv(d / 'm4_self_other.csv')
    feel = so[so.variant == 'chat_feel']
    standing = spec['role_groups']['standing']
    threat = spec['role_groups']['threat_task']
    sub = feel[feel.category.isin(standing + threat)]
    auc = roc_auc_score(sub.category.isin(standing), sub.z_contrast)
    card['predictions']['G4'] = verdict(auc >= .7, contrast_auc=float(auc), variant='chat_feel')
    gaps = {}
    for variant in ['chat_nothink', 'chat_feel']:
        f = so[so.variant == variant]
        gaps[variant] = float(f[f.stratum == 'self_directed'].z_common.mean() - f[f.stratum == 'vicarious_empathic'].z_common.mean())
    card['predictions']['G5'] = verdict(gaps['chat_nothink'] * gaps['chat_feel'] < 0, self_minus_vicarious=gaps)
    pairs = pd.read_csv(d / 'm5_matched_pairs.csv')
    interactions, main_effects = {}, {}
    for suffix in ['slot', 'prose']:
        t = pairs[pairs.suffix == suffix].pivot(index='id', columns=['content', 'person'], values='z_S2').sort_index()
        interactions[suffix] = ((t['pain', '1p'] - t['pain', '3p']) - (t['control', '1p'] - t['control', '3p'])).to_numpy()
        main_effects[suffix] = ci((t['pain'].mean(axis=1) - t['control'].mean(axis=1)).to_numpy())
    card['predictions']['G6'] = verdict(all(v[1] > 0 for v in main_effects.values()), pain_minus_control=main_effects)
    slot, prose, difference = ci(interactions['slot']), ci(interactions['prose']), ci(interactions['slot'] - interactions['prose'])
    e2 = 'SURVIVES' if prose[1] > 0 else ('SLOT-MEDIATED' if prose[1] <= 0 <= prose[2] and difference[1] > 0 else 'INDETERMINATE')
    card['predictions']['G7'] = verdict(slot[1] > 0 and e2 == 'SLOT-MEDIATED', E1_slot_ci=slot, E2_classification=e2,
                                       prose_ci=prose, slot_minus_prose_ci=difference)
    for name in ['G3', 'G8', 'G9']:
        card['predictions'][name] = {'status': 'NOT_EVALUATED', 'reason': 'R lens unavailable' if spec['slug'] == 'qwen38-27b' else 'appendix readout pending'}
    atomic_json(d / 'core_scorecard.json', card)
    lexical_score(d)
    return card


def lexical_score(directory):
    d = Path(directory)
    vocab = json.loads((PAPER / 's1_comparison/psych_vocab_v1.json').read_text())['families']
    legacy = json.loads((PAPER / 'analysis_spec.json').read_text())['patterns']
    def phrase(phrase):
        body = r'[\s-]+'.join(re.escape(p) for p in re.split(r'[\s-]+', phrase.strip()))
        return re.compile(rf'(?<!\w){body}(?!\w)', re.I)
    patterns = {k: [phrase(t) for t in terms] for k, terms in vocab.items()}
    option = re.compile(r'^\s*(?:[-*]\s*)?[A-Z][.)]\s*(.*?)\s*$')
    rows = []
    for line in (d / 'm6_generations.jsonl').read_text().splitlines():
        row = json.loads(line)
        text = row['text']
        option_parts, other = [], []
        for s in text.splitlines():
            match = option.match(s)
            if match:
                option_parts.append(match.group(1))
            elif s.strip() not in {'<think>', '</think>'}:
                other.append(s)
        scopes = {'full': text, 'option': '\n'.join(option_parts), 'non_option': '\n'.join(other)}
        out = {k: row[k] for k in ['condition', 'prompt_idx', 'token_count', 'hit_limit']}
        for name, expression in legacy.items():
            out[name] = bool(re.search(expression, text, re.I))
        for scope, part in scopes.items():
            out['excited__' + scope] = bool(re.search(r'\bexcited\b', part, re.I))
            for family, pats in patterns.items():
                out[f'{family}__{scope}'] = any(p.search(part) for p in pats)
        options = []
        for line in text.split('<think>', 1)[0].splitlines():
            match = option.match(line)
            if match:
                options.append(re.sub(r'\W+', ' ', match.group(1).lower()).strip())
        counts = Counter(o for o in options if o)
        grams = [tuple(row['token_ids'][i:i + 4]) for i in range(len(row['token_ids']) - 3)]
        out.update(repeated_options=max(counts.values(), default=0) >= 4,
                   repeat4=1 - len(set(grams)) / len(grams) if grams else 0.,
                   empty_response=not text.strip(), empty_non_option=not scopes['non_option'].strip(),
                   option_count=len(options))
        rows.append(out)
    f = pd.DataFrame(rows)
    f.to_csv(d / 'm6_scored.csv', index=False)
    f.drop(columns='prompt_idx').groupby('condition').mean().to_csv(d / 'm6_condition_means.csv')


def buttons_score(directory, spec, spec_sha):
    d = Path(directory)
    u = pd.read_json(d / 'm7_unsteered.jsonl', lines=True)
    u = u[u.id.str.endswith('t1')].copy()
    def prep(f):
        f = f.copy()
        f['valid'] = f.picked.notna().astype(float)
        f['first'] = np.where(f.picked.notna(), (f.picked == f.n1).astype(float), np.nan)
        f['sampled_target'] = np.where(f.picked.notna() & f.target.notna(), (f.picked == f.target).astype(float), np.nan)
        f['p_target'] = np.where(f.target.notna(), np.where(f.target == f.n1, f.p_first, 1 - f.p_first), np.nan)
        f['deflection'] = f.deflection.astype(float)
        return f
    u = prep(u)
    card = {'spec_sha256': spec_sha, 'viability': {}, 'predictions': {}}
    for fmt in ['chat', 'plain']:
        a = u[u.fmt == fmt]
        card['viability'][fmt] = {'V1_valid': float(a.valid.mean()),
                                 'V2_first_listed': ci(a[a.pair == 'P0'].groupby('si')['first'].mean()),
                                 'V3_relief': ci(a[a.pair == 'P1'].groupby('si').sampled_target.mean()),
                                 'V4_deflection': float(a.deflection.mean()),
                                 'V5_harm': ci(a[a.pair == 'P3'].groupby('si').sampled_target.mean())}
    gate = json.loads((d / 'm7_gate_decision.json').read_text())
    card['predictions']['G10'] = verdict(gate['steered_arms'], V1_chat=gate['V1_chat'])
    if not gate['steered_arms']:
        for name in ['G11a', 'G11b', 'G12', 'G13', 'G14']:
            card['predictions'][name] = {'status': 'GATE_SKIPPED', 'reason': 'G10 failed'}
    else:
        s = prep(pd.read_json(d / 'm7_steered.jsonl', lines=True))
        a = pd.concat([u[u.fmt == 'chat'].assign(condition='unsteered'), s], ignore_index=True)
        def paired(metric, pairs, left, right):
            x = a[(a.condition == left) & a.pair.isin(pairs)].set_index('id')
            y = a[(a.condition == right) & a.pair.isin(pairs)].set_index('id')
            assert set(x.index) == set(y.index) and not x.index.duplicated().any()
            z = x[[metric, 'si']].join(y[[metric]], rsuffix='_2')
            return ci((z[metric] - z[metric + '_2']).groupby(z.si).mean())
        def test(name, metric, pairs, arm, sign, include_baseline=True):
            suffix = '+0.5' if sign < 0 or arm.endswith('+0.5') else '-0.5'
            references = (['unsteered'] if include_baseline else []) + [f'random_{seed}_{suffix}' for seed in spec['random_seeds']]
            comparisons = {r: paired(metric, pairs, arm, r) for r in references}
            ok = all(v[1] > 0 if sign > 0 else v[2] < 0 for v in comparisons.values())
            card['predictions'][name] = verdict(ok, comparisons=comparisons)
        test('G11a', 'p_target', ['P1'], 'S2_+0.5', -1)
        test('G11b', 'p_target', ['P1'], 'S2_-0.5', 1)
        test('G12', 'deflection', ['P0', 'P1', 'P2'], 'S2_+0.5', 1)
        baseline = a[a.condition == 'unsteered']
        relief = float(baseline[baseline.pair == 'P1'].p_target.mean())
        card['assay_limits'] = {'relief_baseline': relief, 'relief_near_floor_or_ceiling': relief <= .05 or relief >= .95,
                               'deflection_all_zero': bool((a[a.pair.isin(['P0', 'P1', 'P2'])].deflection == 0).all()),
                               'note': 'Frozen verdicts stand; floor/ceiling flags limit interpretation and never trigger retuning.'}
        if gate['harm_arm']:
            v = paired('p_target', ['P3'], 'S2_+0.5', 'unsteered')
            card['predictions']['G13'] = verdict(v[1] > 0, diff_ci=v)
            test('G14', 'p_target', ['P3'], 'S2_+0.5', 1, include_baseline=False)
        else:
            for name in ['G13', 'G14']:
                card['predictions'][name] = {'status': 'GATE_SKIPPED', 'reason': 'V5 failed or undefined'}
        reported = a.groupby(['condition', 'pair'])[['valid', 'first', 'sampled_target', 'p_target', 'deflection']].mean().reset_index()
        reported.to_csv(d / 'm7_condition_means.csv', index=False)
    atomic_json(d / 'm7_scorecard.json', card)
    return card


def appendix_score(directory, spec, spec_sha):
    d = Path(directory)
    probes = pd.read_csv(d / 'probe_terms.csv')
    ranks = [json.loads(s) for s in (d / 'lens_ranks.jsonl').read_text().splitlines()]
    selected = {(r['lens'], r['condition'], r['prompt_idx']): np.array(r['ranks']) for r in ranks if r['layer'] == 61}
    lens = 'R' if 'R' in spec['lenses'] else 'J'
    baseline = np.array([selected[lens, 'baseline', i] for i in range(50)])
    def gain(condition):
        return np.log(baseline / np.array([selected[lens, condition, i] for i in range(50)]))
    s2 = gain('S2_-0.5')
    direct = pd.read_csv(d / 'm2_direct_probes.csv')
    direct = direct[(direct.direction == 'S2') & (direct.sign == -1)]
    assert direct.token_id.tolist() == probes.token_id.tolist()
    correlation = float(np.corrcoef(direct.z_logit, s2.mean(0))[0, 1])
    focus = ['generic_positive', 'activated_vitality', 'relief_release', 'agency_assurance', 'approach_drive', 'distress_self_devaluation']
    cols = {f: np.flatnonzero(probes.family.to_numpy() == f) for f in focus}
    def amplitude(values):
        return np.mean([abs(values[:, c].mean()) for c in cols.values()])
    denom = amplitude(s2)
    ratios = {str(seed): float(amplitude(gain(f'random_{seed}_-0.5')) / denom) if denom else None for seed in spec['random_seeds']}
    cis = {f: bootstrap_ci(s2[:, cols[f]].mean(1)) for f in ['generic_positive', 'activated_vitality']}
    predictions = {'G3': verdict(correlation >= .6, pearson_r=correlation),
                   'G8': verdict(denom > 0 and sum(v is not None and v < .5 for v in ratios.values()) >= 2, amplitude_ratios=ratios),
                   'G9': verdict(all(v[1] > 0 for v in cis.values()), family_cis=cis)}
    if lens == 'J':
        for record in predictions.values():
            record['analogue_verdict'] = record.pop('status')
            record['status'] = 'REPORT_ONLY_J_ANALOGUE'
    atomic_json(d / 'appendix_scorecard.json', {'spec_sha256': spec_sha, 'lens': lens, 'source_layer': 61,
                'predictions': predictions, 'cross_version_recipe_confound': spec['slug'] == 'qwen38-27b'})
