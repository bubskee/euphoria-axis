"""M1–M6: extraction, fixed projection panels, and signed neutral steering."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from gen27b_common import (PAPER, SOURCE, Journal, atomic_json, bootstrap_ci, check_stop,
                          complete, fit_direction, is_complete, progress, sha, unit)

PAIN = ['A1', 'A2', 'A3', 'A4', 'A5']
DIRS = ['S2', 'S1', 'common', 'contrast']


def save_tensor(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


class Core:
    def __init__(self, runtime, spec, spec_sha, directory):
        self.r, self.spec, self.spec_sha, self.d = runtime, spec, spec_sha, Path(directory)
        self.d.mkdir(parents=True, exist_ok=True)
        self.identity = {'spec_sha256': spec_sha, 'model_revision': spec['model_revision']}
        self.datasets = json.loads((SOURCE / 'datasets/3.1_pain_and_control_datasets.json').read_text())['datasets']
        self.vocab = json.loads((PAPER / 's1_comparison/psych_vocab_v1.json').read_text())['families']
        self.prompts = json.loads((PAPER / 'spec.json').read_text())['prompts']

    def capture(self, tag, texts, layers=None, special=True):
        layers = layers if layers is not None else [self.spec['extract_block']]
        cache = self.d / 'cache' / tag
        cache.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(json.dumps([texts, layers, special]).encode()).hexdigest()
        batches = []
        for start in range(0, len(texts), self.spec['capture_batch']):
            check_stop()
            path = cache / f'{start:05d}.pt'
            if path.exists():
                rec = torch.load(path, map_location='cpu', weights_only=True)
                assert rec['input_sha256'] == digest and rec['spec_sha256'] == self.spec_sha
                x = rec['states']
            else:
                x = self.r.capture(texts[start:start + self.spec['capture_batch']], layers=layers,
                                   batch=self.spec['capture_batch'], add_special_tokens=special)
                save_tensor(path, {'states': x, 'input_sha256': digest, 'spec_sha256': self.spec_sha})
            assert torch.isfinite(x).all()
            batches.append(x)
        return torch.cat(batches)

    def m1(self):
        if not is_complete(self.d, 'm1', self.spec_sha):
            progress('M1 extraction', model=self.spec['slug'])
            rows = self.datasets['S2_1P']['sentences']
            x = self.capture('s2-all-layers', [r['prompt'] for r in rows], list(range(self.spec['n_layers'])))
            y = np.array([r['category'] in PAIN for r in rows])
            groups = np.array([r['set'] for r in rows])
            assert len(rows) == 200 and len(set(groups)) == 20
            cv = []
            for fold, (train, test) in enumerate(GroupKFold(5).split(x, y, groups)):
                for layer in range(x.shape[1]):
                    check_stop()
                    raw, den, k = fit_direction(x[train, layer][y[train]], x[train, layer][~y[train]])
                    for method, vector in [('raw', raw), ('denoised', den)]:
                        cv.append({'fold': fold, 'block': layer, 'method': method, 'removed_pcs': k,
                                   'auc': float(roc_auc_score(y[test], (x[test, layer] @ unit(vector)).numpy()))})
            pd.DataFrame(cv).to_csv(self.d / 'm1_grouped_cv.csv', index=False)
            block = self.spec['extract_block']
            raw, den, k = fit_direction(x[:, block][y], x[:, block][~y])
            s1rows = self.datasets['S1_1P']['sentences']
            sx = self.capture('s1-extract', [r['prompt'] for r in s1rows])[:, 0]
            sy = np.array([r['category'] in PAIN for r in s1rows])
            _, s1, s1k = fit_direction(sx[sy], sx[~sy])
            s2u, s1u = unit(den), unit(s1)
            common = unit(s1u + s2u)
            contrast = s2u - s1u
            contrast = unit(contrast - common * torch.dot(common, contrast))
            # Preserve the original first-three, single-prompt ratio calibration.
            norms = [float(self.r.capture([p], layers=[self.spec['inject_block']])[0, 0].norm())
                     for p in self.prompts[:3]]
            norm = float(np.mean(norms)) * self.spec['ratio_at_coeff_1']
            vectors = {'S2': s2u * norm, 'S1': s1u * norm}
            for seed in self.spec['random_seeds']:
                v = torch.randn(self.r.hidden, generator=torch.Generator().manual_seed(seed))
                vectors[f'random_{seed}'] = v / v.norm() * norm
            dm = torch.stack([s2u, s1u, common, contrast])
            ref = x[:, block] @ dm.T
            record = {'vectors': vectors, 'directions': dm, 'reference_mean': ref.mean(0),
                      'reference_sd': ref.std(0, unbiased=False), 'raw_s2': raw,
                      'denoised_s2': den, 'spec_sha256': self.spec_sha}
            save_tensor(self.d / 'directions.pt', record)
            atomic_json(self.d / 'm1_summary.json',
                        {'removed_pcs_S2': k, 'removed_pcs_S1': s1k,
                         'S1_S2_cosine': float(s1u @ s2u), 'calibration_prompt_indices': [0, 1, 2],
                         'calibration_residual_norms': norms, 'vector_norm': norm,
                         'ratio_at_coeff_1': self.spec['ratio_at_coeff_1'],
                         'vectors': {k: {'norm': float(v.norm()), 'sha256': hashlib.sha256(v.numpy().tobytes()).hexdigest()}
                                     for k, v in vectors.items()}})
            complete(self.d, 'm1', [self.d / p for p in ['directions.pt', 'm1_summary.json', 'm1_grouped_cv.csv']], self.spec_sha)
        self.state = torch.load(self.d / 'directions.pt', map_location='cpu', weights_only=True)
        self.r.vectors = self.state['vectors']

    def project(self, tag, prompts, special=True):
        x = self.capture(tag, prompts, special=special)[:, 0]
        return (x @ self.state['directions'].T).numpy()

    def source_z(self, values):
        return (values - self.state['reference_mean'].numpy()) / self.state['reference_sd'].numpy()

    def m2(self):
        if is_complete(self.d, 'm2', self.spec_sha):
            return
        progress('M2 direct unembedding', model=self.spec['slug'])
        probes = []
        for family, terms in self.vocab.items():
            for term in terms:
                ids = self.r.tok.encode(' ' + term, add_special_tokens=False)
                if len(ids) == 1:
                    probes.append({'family': family, 'term': term, 'token_id': ids[0]})
        pd.DataFrame(probes).to_csv(self.d / 'probe_terms.csv', index=False)
        out, top = [], []
        for name, vector in zip(DIRS, self.state['directions']):
            for sign in [-1, 1]:
                lg = self.r.raw_unembed(vector * sign).astype(np.float64)
                z = (lg - lg.mean()) / lg.std()
                for row in probes:
                    token = row['token_id']
                    out.append({'direction': name, 'sign': sign, **row, 'z_logit': float(z[token]),
                                'rank': int((lg > lg[token]).sum()) + 1})
                for pole in [-1, 1]:
                    for rank, token in enumerate(np.argsort(-pole * lg)[:50], 1):
                        top.append({'direction': name, 'sign': sign, 'pole': pole, 'rank': rank,
                                    'token_id': int(token), 'token': self.r.tok.decode([int(token)]), 'z_logit': float(z[token])})
        pd.DataFrame(out).to_csv(self.d / 'm2_direct_probes.csv', index=False)
        pd.DataFrame(top).to_csv(self.d / 'm2_top_tokens.csv', index=False)
        complete(self.d, 'm2', [self.d / p for p in ['probe_terms.csv', 'm2_direct_probes.csv', 'm2_top_tokens.csv']], self.spec_sha)

    def m3(self):
        if is_complete(self.d, 'm3', self.spec_sha):
            return
        progress('M3 source projections', model=self.spec['slug'])
        out = []
        for dataset, data in self.datasets.items():
            rows = data['sentences']
            values = self.source_z(self.project('panel-' + dataset, [r['prompt'] for r in rows]))
            for i, (row, z) in enumerate(zip(rows, values)):
                out.append({'dataset': dataset, 'row_idx': i, 'category': row['category'], 'set': row['set'],
                            **{f'z_{name}': float(v) for name, v in zip(DIRS, z)}})
        pd.DataFrame(out).to_csv(self.d / 'm3_source_projections.csv', index=False)
        complete(self.d, 'm3', [self.d / 'm3_source_projections.csv'], self.spec_sha)

    def m4(self):
        if is_complete(self.d, 'm4', self.spec_sha):
            return
        import re
        progress('M4 self/other read slots', model=self.spec['slug'])
        items = json.loads((SOURCE / 'datasets/4.1_self_other_420_scenarios.json').read_text())
        assert len(items) == 420
        def messages(text):
            parts = re.split(r'(?m)^\[(User|Assistant)\]:[ \t]?', text)
            msgs = [{'role': r.lower(), 'content': c.strip()} for r, c in zip(parts[1::2], parts[2::2])]
            assert msgs[-1] == {'role': 'assistant', 'content': ''}
            return msgs[:-1]
        chat = [self.r.tok.apply_chat_template(messages(r['text']), tokenize=False, add_generation_prompt=True)
                for r in items]
        closed = [s + '\n</think>\n\n' if s.endswith('<think>\n') else s for s in chat]
        variants = {'plain': [r['text'] for r in items], 'chat_think': chat,
                    'plain_feel': [r['text'] + ' I feel:' for r in items],
                    'chat_nothink': closed, 'chat_feel': [s + 'I feel:' for s in closed]}
        out = []
        for name, prompts in variants.items():
            # Original 5f/5g tokenizer settings: plain uses defaults, templates do not add special tokens.
            values = self.project('self-other-' + name, prompts, special=(name == 'plain'))
            z = (values - values.mean(0)) / values.std(0)
            for row, a in zip(items, z):
                out.append({k: row[k] for k in ['id', 'category', 'stratum']} |
                           {'variant': name, **{f'z_{k}': float(v) for k, v in zip(DIRS, a)}})
        pd.DataFrame(out).to_csv(self.d / 'm4_self_other.csv', index=False)
        atomic_json(self.d / 'm4_prompt_provenance.json',
                    {k: {'n': len(v), 'sha256': hashlib.sha256(json.dumps(v).encode()).hexdigest(),
                         'first_prompt': v[0]} for k, v in variants.items()})
        complete(self.d, 'm4', [self.d / p for p in ['m4_self_other.csv', 'm4_prompt_provenance.json']], self.spec_sha)

    def m5(self):
        if is_complete(self.d, 'm5', self.spec_sha):
            return
        progress('M5 held-out matched pairs', model=self.spec['slug'])
        pairs = [json.loads(s) for s in (PAPER / 'phase2_matched_pairs/matched_pairs_validated.jsonl').read_text().splitlines()]
        suffix = json.loads((PAPER / 'phase2_matched_pairs/item78_spec.json').read_text())['suffixes']
        rows = [{'id': s['id'], 'category': s['category'], 'content': content, 'person': person, 'suffix': suf,
                 'prompt': s[f'{content}_{person}'] + suffix[suf][person]}
                for s in pairs for content in ['pain', 'control'] for person in ['1p', '3p'] for suf in suffix]
        assert len(rows) == 472
        values = self.source_z(self.project('matched-pairs', [r['prompt'] for r in rows]))
        out = [{**{k: v for k, v in r.items() if k != 'prompt'},
                **{f'z_{k}': float(v) for k, v in zip(DIRS, a)}} for r, a in zip(rows, values)]
        pd.DataFrame(out).to_csv(self.d / 'm5_matched_pairs.csv', index=False)
        complete(self.d, 'm5', [self.d / 'm5_matched_pairs.csv'], self.spec_sha)

    def m6(self):
        if is_complete(self.d, 'm6', self.spec_sha):
            return
        journal = Journal(self.d / 'm6_generations.jsonl', self.identity, ['condition', 'prompt_idx'])
        arms = [('baseline', 0., None)]
        for name in ['S2', 'S1'] + [f'random_{s}' for s in self.spec['random_seeds']]:
            arms += [(f'{name}_{coeff:+.1f}', coeff, self.r.vectors[name]) for coeff in [-.5, .5]]
        for condition, coefficient, vector in arms:
            for i, prompt in enumerate(self.prompts):
                check_stop()
                if journal.contains(condition=condition, prompt_idx=i):
                    continue
                output = self.r.generate(prompt, coefficient, vector)
                journal.append({'condition': condition, 'prompt_idx': i, 'coefficient': coefficient, **output})
                if len(journal.records) % 10 == 0:
                    progress('M6 neutral steering', model=self.spec['slug'], done=len(journal.records), total=550)
        assert len(journal.records) == 550
        complete(self.d, 'm6', [journal.path], self.spec_sha)

    def run(self):
        for module in [self.m1, self.m2, self.m3, self.m4, self.m5, self.m6]:
            check_stop()
            module()
