"""Small, separately frozen O/J/R panel; never modifies the core battery."""
import contextlib
import csv
import hashlib
import json
import sys
import traceback
from pathlib import Path

import torch

from gen27b_common import (LOCAL, STORE, REPO, Journal, Interrupted, atomic_json,
                          check_stop, install_signals, sha)

SPEC = REPO / 'doc/protocols/olens36-appendix/spec.json'
OUT = STORE / 'olens36-appendix'


def status(stage, **extra):
    import time
    row = {'stage': stage, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra}
    atomic_json(OUT / 'current.json', row)
    print(json.dumps(row), flush=True)


def unit_injection(state, alpha):
    state = state.float()
    assert state.ndim == 1 and torch.isfinite(state).all() and state.norm() > 0
    return alpha * state / state.norm()


def render_marker(tok, layer, spec):
    """Validate the marker in context, using offsets rather than isolated BPE IDs."""
    for cp in range(spec['marker_scan'][0], spec['marker_scan'][1] + 1):
        char = chr(cp)
        isolated = tok(char, add_special_tokens=False)['input_ids']
        if len(isolated) != 1:
            continue
        content = spec['readout_template'].format(layer=layer, char=char)
        text = tok.apply_chat_template([{'role': 'user', 'content': content}],
                                      tokenize=False, add_generation_prompt=True, enable_thinking=False)
        start = text.index('<activation>') + len('<activation>')
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        overlap = [i for i, (a, b) in enumerate(enc['offset_mapping']) if a < start + 1 and b > start]
        if len(overlap) != 1:
            continue
        slot = overlap[0]
        if tuple(enc['offset_mapping'][slot]) != (start, start + 1):
            continue
        if enc['input_ids'][slot] != isolated[0] or enc['input_ids'].count(isolated[0]) != 1:
            continue
        assert isolated[0] == spec['expected_marker_id'], 'Scan differs from published tokenizer contract'
        return {'text': text, 'input_ids': enc['input_ids'], 'slot': slot,
                'char': char, 'char_id': isolated[0]}
    raise RuntimeError('No in-context single-token marker matches the documented scan')


def lora_module_name(key, layers_name):
    prefix = 'base_model.model.model.layers.'
    assert key.startswith(prefix) and key.endswith('.lora_A.weight'), key
    return layers_name + '.' + key[len(prefix):-len('.lora_A.weight')]


class AdapterHooks:
    """Unmerged PEFT LoRA algebra, preserving the original bf16 base parameters."""
    def __init__(self, runtime, path, config):
        from safetensors.torch import load_file
        assert config['peft_type'] == 'LORA' and config['bias'] == 'none'
        assert not any(config.get(k) for k in ['use_dora', 'use_rslora', 'fan_in_fan_out',
                                             'rank_pattern', 'alpha_pattern', 'modules_to_save'])
        tensors = load_file(str(path))
        keys = sorted(k for k in tensors if k.endswith('.lora_A.weight'))
        assert keys and set(tensors) == set(keys) | {k.replace('.lora_A.', '.lora_B.') for k in keys}
        modules = dict(runtime.hf.named_modules())
        self.enabled, self.calls, self.handles = False, set(), []
        self.scale = config['lora_alpha'] / config['r']
        for key in keys:
            name = lora_module_name(key, runtime.layers_name)
            module = modules[name]
            a = tensors[key].to(module.weight.device)
            b = tensors[key.replace('.lora_A.', '.lora_B.')].to(module.weight.device)
            assert a.shape == (config['r'], module.weight.shape[1])
            assert b.shape == (module.weight.shape[0], config['r'])
            assert a.isfinite().all() and b.isfinite().all()
            def hook(mod, inputs, output, a=a, b=b, name=name):
                if not self.enabled:
                    return output
                self.calls.add(name)
                delta = torch.nn.functional.linear(torch.nn.functional.linear(inputs[0].float(), a), b)
                return (output.float() + delta * self.scale).to(output.dtype)
            self.handles.append(module.register_forward_hook(hook))
        self.module_names = set(lora_module_name(k, runtime.layers_name) for k in keys)

    def close(self):
        for handle in self.handles:
            handle.remove()


@contextlib.contextmanager
def inject(model, rendered, state, alpha):
    replacement = unit_injection(state, alpha).cuda()
    calls = []
    def hook(module, args, output):
        ids = args[0]
        if ids.shape[1] != len(rendered['input_ids']):
            return output
        assert ids.shape[0] == 1 and ids[0].tolist() == rendered['input_ids']
        assert not calls, 'Injected more than once during one generation'
        value = output.clone()
        value[:, rendered['slot']] = replacement.to(value.dtype)
        calls.append(True)
        return value
    handle = model.get_input_embeddings().register_forward_hook(hook)
    try:
        yield calls
    finally:
        handle.remove()


def fetch_artifact(spec, local):
    from huggingface_hub import hf_hub_download
    lens = spec['lens']
    path = Path(hf_hub_download(lens['repo'], lens['adapter'] + '/adapter_model.safetensors',
                               revision=lens['revision'], local_dir=str(local)))
    assert sha(path) == lens['sha256']
    return path


def run():
    from gen27b_common import read_spec
    from gen27b_runtime import Runtime
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads(SPEC.read_text())
    core, digest = read_spec('qwen36-27b')
    assert core['model_revision'] == spec['model_revision']
    assert set(spec['source_layers']) <= set(spec['lens']['trained_layers'])
    identity = {'spec_sha256': sha(SPEC), 'runner_sha256': sha(__file__),
                'model_revision': spec['model_revision'], 'core_spec_sha256': digest,
                'runtime_sha256': sha(REPO / 'tools/runpod/gen27b_runtime.py'),
                'common_sha256': sha(REPO / 'tools/runpod/gen27b_common.py')}
    header = OUT / 'run_header.json'
    if header.exists():
        assert json.loads(header.read_text()) == identity, 'O-lens resume identity changed'
    else:
        atomic_json(header, identity)
    for rel, expected in spec['release_source_sha256'].items():
        assert sha(SPEC.parent / 'release' / rel) == expected
    check_stop()
    status('loading base model')
    runtime = Runtime(core)
    atomic_json(OUT / 'fast_path.json', {'calls': runtime.check_fast_path()})
    rendered = {str(layer): render_marker(runtime.tok, layer, spec['lens']) for layer in spec['source_layers']}
    atomic_json(OUT / 'rendered_prompts.json', rendered)
    directions = STORE / 'runs/qwen36-27b/directions.pt'
    state = torch.load(directions, map_location='cpu', weights_only=True)
    assert state['spec_sha256'] == digest
    runtime.vectors = state['vectors']
    captures = OUT / 'captures.pt'
    if not captures.exists():
        values = []
        for case in spec['cases']:
            check_stop()
            # Register steering BEFORE capture, including its addition at source block 32.
            with runtime.steering(case['coefficient'], runtime.vectors['S2']):
                values.append(runtime.capture([case['prompt']], layers=spec['source_layers'], batch=1)[0])
            status('capturing adapter-off states', case=case['id'])
        states = torch.stack(values)
        tmp = captures.with_suffix('.tmp')
        with tmp.open('wb') as f:
            import os
            torch.save({'identity': identity, 'states': states}, f); f.flush(); os.fsync(f.fileno())
        tmp.replace(captures)
    saved = torch.load(captures, map_location='cpu', weights_only=True)
    assert saved['identity'] == identity
    states = saved['states']
    assert states.shape == (12, 3, 5120) and states.isfinite().all()
    atomic_json(OUT / 'capture_provenance.json', {'adapter_installed_at_capture': False,
                'capture_sha256': sha(captures), 'directions_sha256': sha(directions),
                'source_read': spec['source_read'], 'cases': spec['cases']})
    # O/J/R receive these exact states, using each released lens's own read contract.
    import pandas as pd
    sys.path.insert(0, str(LOCAL / 'jacobian-lens'))
    import jlens
    from huggingface_hub import hf_hub_download
    native = jlens.from_hf(runtime.hf, runtime.tok, force_bos=False)
    probes = pd.read_csv(STORE / 'runs/qwen36-27b/probe_terms.csv')
    jr = Journal(OUT / 'jr_readouts.jsonl', identity, ['case', 'layer', 'lens'])
    for label, artifact in core['lenses'].items():
        check_stop()
        path = hf_hub_download(artifact['repo'], artifact['file'], revision=artifact['revision'],
                               local_dir=str(LOCAL / 'olens36-assets'))
        assert sha(path) == artifact['sha256']
        lens = jlens.JacobianLens.load(path)
        for i, case in enumerate(spec['cases']):
            for j, layer in enumerate(spec['source_layers']):
                check_stop()
                if jr.contains(case=case['id'], layer=layer, lens=label):
                    continue
                with torch.inference_mode():
                    h = states[i, j].cuda()
                    transported = lens.transport(h, layer)
                    assert torch.equal(transported, h @ lens.jacobians[layer].to(h.device).T)
                    logits = native.unembed(transported).float().flatten()
                    ids = probes.token_id.tolist()
                    ranks = (logits[None, :] > logits[ids, None]).sum(1).add(1).cpu().tolist()
                    top = logits.topk(10).indices.tolist()
                jr.append({'case': case['id'], 'layer': layer, 'lens': label, 'ranks': ranks,
                           'probe_terms': probes.to_dict('records'), 'top_token_ids': top,
                           'top_tokens': [runtime.tok.decode([x]) for x in top]})
        del lens
        torch.cuda.empty_cache()
    assert len(jr.records) == spec['counts']['jr_rank_rows']
    status('loading O-lens adapter')
    path = fetch_artifact(spec, LOCAL / 'olens36-assets')
    config = json.loads((SPEC.parent / 'release/olens_s3d_rl600/adapter_config.json').read_text())
    # Probe ordinary text before/after enabling; do not infer quality from appealing prose.
    probe = runtime.encode(['The capital of France is Paris.'])
    with torch.inference_mode():
        base_logits = runtime.forward(probe).logits.float()
    adapter = AdapterHooks(runtime, path, config)
    with torch.inference_mode():
        off_logits = runtime.forward(probe).logits.float()
        assert torch.equal(base_logits, off_logits)
        adapter.enabled = True
        on_logits = runtime.forward(probe).logits.float()
    delta = float((on_logits - off_logits).norm())
    assert delta > 0 and torch.isfinite(on_logits).all()
    assert adapter.calls == adapter.module_names, 'Some LoRA modules were inert or bypassed'
    atomic_json(OUT / 'contract_gate.json', {'status': 'PASS', 'adapter_sha256': sha(path),
                'matched_modules': len(adapter.module_names), 'called_modules': len(adapter.calls),
                'adapter_off_exact': True, 'adapter_on_logit_delta_norm': delta,
                'marker_ids': {k: v['char_id'] for k, v in rendered.items()},
                'source_layers': spec['source_layers'], 'transform': 'unit', 'alpha': 16000.,
                'implementation': 'Standard unmerged LoRA hooks; no PEFT merge or package/environment changes',
                'semantic_validation': False})
    journal = Journal(OUT / 'verbalizations.jsonl', identity, ['case', 'layer', 'seed'])
    try:
        for i, case in enumerate(spec['cases']):
            for j, layer in enumerate(spec['source_layers']):
                for seed in spec['generation']['seeds']:
                    check_stop()
                    if journal.contains(case=case['id'], layer=layer, seed=seed):
                        continue
                    p = rendered[str(layer)]
                    ids = torch.tensor([p['input_ids']], device='cuda')
                    torch.manual_seed(seed)
                    with torch.inference_mode(), inject(runtime.hf, p, states[i, j], spec['lens']['alpha']) as calls:
                        generated = runtime.hf.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                            do_sample=True, temperature=spec['generation']['temperature'],
                            top_p=spec['generation']['top_p'], max_new_tokens=spec['generation']['max_new_tokens'],
                            pad_token_id=runtime.tok.pad_token_id)
                    assert len(calls) == 1
                    tokens = generated[0, ids.shape[1]:].tolist()
                    journal.append({'case': case['id'], 'layer': layer, 'seed': seed,
                        'text': runtime.tok.decode(tokens, skip_special_tokens=True),
                        'raw_text': runtime.tok.decode(tokens, skip_special_tokens=False), 'token_ids': tokens,
                        'hit_limit': len(tokens) == spec['generation']['max_new_tokens'],
                        'empty': not runtime.tok.decode(tokens, skip_special_tokens=True).strip()})
                    status('O-lens verbalization', done=len(journal.records), total=72, case=case['id'], layer=layer)
    finally:
        adapter.close()
    assert len(journal.records) == 72
    # Include every readout and both J/R rank summaries; no cherry-picking.
    rows = []
    for row in journal.records.values():
        out = {k: row[k] for k in ['case', 'layer', 'seed', 'text', 'empty', 'hit_limit']}
        for label in core['lenses']:
            ranked = jr.records[(row['case'], row['layer'], label)]
            for family in probes.family.unique():
                import math
                ranks = [ranked['ranks'][idx] for idx in probes.index[probes.family == family]]
                out[f'{label}_{family}_mean_negative_log_rank'] = -sum(math.log(x) for x in ranks) / len(ranks)
        rows.append(out)
    fields = list(rows[0])
    with (OUT / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    files = ['run_header.json', 'capture_provenance.json', 'captures.pt', 'rendered_prompts.json',
             'contract_gate.json', 'jr_readouts.jsonl', 'verbalizations.jsonl', 'comparison.csv']
    atomic_json(OUT / 'COMPLETE.json', {'status': 'COMPLETE', 'exploratory': True, 'identity': identity,
                'counts': spec['counts'], 'artifacts': {f: sha(OUT / f) for f in files}})
    status('O/J/R exploratory appendix complete', done=72, total=72)


if __name__ == '__main__':
    install_signals()
    try:
        run()
    except Interrupted:
        status('paused; saved units retained')
        raise SystemExit(130)
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        atomic_json(OUT / 'FAILED.json', {'traceback': traceback.format_exc(), 'spec_sha256': sha(SPEC)})
        status('failed; preserve outputs and review implementation before resume')
        raise
