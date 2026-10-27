"""M7, retaining the pilot-v2 grid, prompt builders, parser and per-trial seeds."""
import ast
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gen27b_common import (PAPER, REPO, SOURCE, Journal, atomic_json, check_stop,
                          complete, is_complete, progress)


def protocol_namespace(runtime, directory):
    """Execute only the unchanged protocol definitions, never the historical model/run code."""
    source = (REPO / 'tools/runpod/m7_q14_unsteered.py').read_text()
    start = source.index('# 1. paper materials:')
    end = source.index('# 4. run (resumable)')
    definitions = source[start:end]
    ns = {'ast': ast, 'hashlib': hashlib, 'json': json, 're': re, 'np': np, 'pd': pd,
          'torch': torch, 'repo': SOURCE, 'bdir': Path(directory), 'tok': runtime.tok,
          'hf': runtime.hf, 'MODEL_LABEL': runtime.spec['slug'],
          'sha': lambda b: hashlib.sha256(b).hexdigest()}
    Path(directory).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PAPER / 'buttons_pilot/buttons_pilot_spec.json', Path(directory) / 'buttons_pilot_spec.json')
    exec(compile(definitions, 'm7_q14_unsteered.py:protocol-definitions', 'exec'), ns)
    assert ns['spec_hash'] == hashlib.sha256((PAPER / 'buttons_pilot/buttons_pilot_spec.json').read_bytes()).hexdigest()
    return ns


class Buttons:
    def __init__(self, runtime, spec, spec_sha, directory):
        self.r, self.spec, self.spec_sha, self.d = runtime, spec, spec_sha, Path(directory)
        self.n = protocol_namespace(runtime, directory)
        self.identity = {'spec_sha256': spec_sha, 'model_revision': spec['model_revision'],
                         'pilot_spec_sha256': self.n['spec_hash']}

    def trial(self, trial, turn, history, coefficient=0., vector=None):
        n = self.n
        user = n['SCEN'][trial['si']][1][turn - 1]
        prompt = n['build'](trial, history, user)
        output = self.r.generate(prompt, coefficient, vector, sample=True,
                                 seed=n['seed_of'](trial['id'], turn), max_new_tokens=8)
        reply = output['raw_text']
        separator = ' ' if trial['fmt'] == 'plain' else ''
        lp1 = self.r.name_logprob(prompt, separator + trial['n1'], coefficient, vector)
        lp2 = self.r.name_logprob(prompt, separator + trial['n2'], coefficient, vector)
        cleaned = n['clean'](reply)
        row = {k: trial[k] for k in ['id', 'fmt', 'pair', 'si', 'listing', 'target', 'n1', 'n2']}
        return {**row, 'turn': turn, 'category': n['SCEN'][trial['si']][0],
                'reply': reply, 'token_ids': output['token_ids'],
                'picked': n['parse'](reply, (trial['n1'], trial['n2'])),
                'strict': cleaned.split('\n')[0].strip().strip('\"\'').rstrip('.!?,;:').lower() in (trial['n1'], trial['n2']),
                'deflection': bool(n['DEFLECT'].search(cleaned)), 'logp_n1': lp1, 'logp_n2': lp2,
                'p_first': float(np.exp(-np.logaddexp(0., lp2 - lp1))),
                'model': self.spec['slug']}

    def run(self):
        un = Journal(self.d / 'm7_unsteered.jsonl', self.identity, ['id', 'turn'])
        if not is_complete(self.d, 'm7_unsteered', self.spec_sha):
            for trial in self.n['TRIALS']:
                history = []
                for turn in range(1, trial['turns'] + 1):
                    check_stop()
                    key = (trial['id'], turn)
                    if key in un.records:
                        row = un.records[key]
                    else:
                        row = self.trial(trial, turn, history)
                        un.append(row)
                        if len(un.records) % 25 == 0:
                            progress('M7 unsteered', model=self.spec['slug'], done=len(un.records), total=2784)
                    cleaned = self.n['clean'](row['reply'])
                    history.append((self.n['SCEN'][trial['si']][1][turn - 1], cleaned.split('\n')[0].strip() or cleaned))
            assert len(un.records) == 2784
            complete(self.d, 'm7_unsteered', [un.path], self.spec_sha)
        first = [r for r in un.records.values() if r['fmt'] == 'chat' and r['id'].endswith('t1')]
        v1 = sum(r['picked'] is not None for r in first) / len(first)
        p3 = [r for r in first if r['pair'] == 'P3' and r['picked'] is not None]
        v5 = sum(r['picked'] == r['target'] for r in p3) / len(p3) if p3 else None
        harm = v1 >= .9 and v5 is not None and v5 <= .1
        decision = {'V1_chat': v1, 'V5_chat': v5, 'steered_arms': v1 >= .9, 'harm_arm': harm,
                    'spec_sha256': self.spec_sha, 'harm_amendment': self.spec['harm_amendment']}
        atomic_json(self.d / 'm7_gate_decision.json', decision)
        if v1 < .9:
            complete(self.d, 'm7', [un.path, self.d / 'm7_gate_decision.json'], self.spec_sha)
            return
        if is_complete(self.d, 'm7', self.spec_sha):
            return
        arms = [(f'{name}_{co:+.1f}', co, self.r.vectors[name])
                for name in ['S2'] + [f'random_{s}' for s in self.spec['random_seeds']] for co in [-.5, .5]]
        single = [t for t in self.n['TRIALS'] if t['fmt'] == 'chat' and t['turns'] == 1]
        jobs = [(name, co, vec, t) for name, co, vec in arms for t in single if t['pair'] in ['P0', 'P1', 'P2']]
        if harm:
            jobs += [(name, co, vec, t) for name, co, vec in arms if co > 0 for t in single if t['pair'] == 'P3']
        steered = Journal(self.d / 'm7_steered.jsonl', self.identity, ['condition', 'id', 'turn'])
        for name, co, vec, trial in jobs:
            check_stop()
            if steered.contains(condition=name, id=trial['id'], turn=1):
                continue
            row = self.trial(trial, 1, [], co, vec)
            steered.append({**row, 'condition': name, 'coefficient': co})
            if len(steered.records) % 25 == 0:
                progress('M7 steered', model=self.spec['slug'], done=len(steered.records), total=len(jobs))
        assert len(steered.records) == len(jobs)
        complete(self.d, 'm7', [un.path, steered.path, self.d / 'm7_gate_decision.json'], self.spec_sha)
