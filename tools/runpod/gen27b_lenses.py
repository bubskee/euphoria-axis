"""Optional, resumable appendix readouts using released lenses only."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from gen27b_common import LOCAL, STORE, Journal, atomic_json, check_stop, complete, progress, sha


def readouts(runtime, spec, spec_sha, directory):
    d = Path(directory)
    sys.path.insert(0, str(LOCAL / 'jacobian-lens'))
    import jlens
    native_model = jlens.from_hf(runtime.hf, runtime.tok, force_bos=False)
    lenses, provenance = {}, {}
    for label, artifact in spec['lenses'].items():
        path = hf_hub_download(artifact['repo'], artifact['file'], revision=artifact['revision'],
                               local_dir=str(LOCAL / 'lenses' / spec['slug']))
        digest = sha(path)
        assert digest == artifact['sha256']
        if artifact['format'] == 'miru-safetensors':
            sys.path.insert(0, str(LOCAL / 'miru-tracer/src'))
            from miru_tracer.core.lens_io import load_lens
            from miru_tracer.core._jlens.hf import from_hf
            lens = load_lens(path)
            native_model = from_hf(runtime.hf, runtime.tok, force_bos=False)
            meta = lens.fit_metadata
            assert meta['provenance']['model_commit_hash'] == spec['model_revision']
            assert meta['fit']['target_layer'] == 63
            provenance[label] = {'sha256': digest, 'fit_metadata': meta, 'reader_revision': spec['miru_code_revision']}
        else:
            lens = jlens.JacobianLens.load(path)
            payload = torch.load(path, map_location='cpu', weights_only=False)
            provenance[label] = {'sha256': digest, 'provenance': payload.get('provenance', {}),
                                 'reader_revision': '581d398613e5602a5af361e1c34d3a92ea82ba8e'}
        assert 61 in lens.source_layers
        lenses[label] = lens
    atomic_json(d / 'lens_provenance.json', provenance)
    probes = pd.read_csv(d / 'probe_terms.csv')
    ids = probes.token_id.tolist()
    from gen27b_common import PAPER
    prompts = json.loads((PAPER / 'spec.json').read_text())['prompts']
    arms = [('baseline', 0., None)] + [(f'{k}_{co:+.1f}', co, v) for k, v in runtime.vectors.items() for co in [-.5, .5]]
    layers = sorted(set.intersection(*(set(lens.source_layers) for lens in lenses.values())))
    journal = Journal(d / 'lens_ranks.jsonl', {'spec_sha256': spec_sha, 'model_revision': spec['model_revision']},
                      ['condition', 'prompt_idx', 'layer', 'lens'])
    for condition, co, vec in arms:
        for i, prompt in enumerate(prompts):
            check_stop()
            if all(journal.contains(condition=condition, prompt_idx=i, layer=l, lens=k) for l in layers for k in lenses):
                continue
            states = runtime.capture([prompt], layers=layers, coefficient=co, vector=vec)[0]
            for j, layer in enumerate(layers):
                for label, lens in lenses.items():
                    check_stop()
                    if journal.contains(condition=condition, prompt_idx=i, layer=layer, lens=label):
                        continue
                    with torch.inference_mode():
                        state = states[j].cuda()
                        transported = lens.transport(state, layer)
                        # Reproduce the release's native transport/unembed before accepting our wrapper.
                        matrix = lens.jacobians[layer].to(state.device)
                        assert torch.equal(transported, state @ matrix.T)
                        logits = native_model.unembed(transported).float().flatten()
                        target = logits[ids]
                        ranks = (logits[None, :] > target[:, None]).sum(1).add(1).cpu().tolist()
                    journal.append({'condition': condition, 'prompt_idx': i, 'layer': layer, 'lens': label, 'ranks': ranks})
            progress('appendix lens readout', model=spec['slug'], condition=condition, prompt_idx=i)
    gains = []
    for label in lenses:
        for layer in layers:
            for condition, co, vec in arms[1:]:
                base = np.array([journal.records[('baseline', i, layer, label)]['ranks'] for i in range(50)])
                other = np.array([journal.records[(condition, i, layer, label)]['ranks'] for i in range(50)])
                values = np.log(base / other)
                for family in probes.family.unique():
                    cols = np.flatnonzero(probes.family.to_numpy() == family)
                    per = values[:, cols].mean(1)
                    from gen27b_common import bootstrap_ci
                    gains.append({'lens': label, 'layer': layer, 'condition': condition, 'family': family,
                                  'gain': float(per.mean()), 'ci': bootstrap_ci(per) if layer == 61 else None})
    # JSON retains CIs; flat CSV is convenient for figures.
    atomic_json(d / 'lens_family_gains.json', gains)
    pd.DataFrame([{k: v for k, v in row.items() if k != 'ci'} for row in gains]).to_csv(d / 'lens_family_gains.csv', index=False)
    complete(d, 'lenses', [journal.path, d / 'lens_provenance.json', d / 'lens_family_gains.json', d / 'lens_family_gains.csv'], spec_sha)
