"""Pinned bf16 model, raw block-output capture, and constant steering hooks."""
import contextlib
import json
import sys
from pathlib import Path

import torch
import transformers

from gen27b_common import LOCAL, check_stop


class Runtime:
    def __init__(self, spec):
        self.spec = spec
        self.path = LOCAL / 'models' / spec['model_id'].split('/')[-1]
        verified = json.loads((self.path / 'VERIFIED.json').read_text())
        assert verified['revision'] == spec['model_revision']
        self.hf = transformers.AutoModelForCausalLM.from_pretrained(
            str(self.path), dtype=torch.bfloat16, local_files_only=True).cuda().eval()
        self.tok = transformers.AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
        self.tok.padding_side = 'left'
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        candidates = [(n, m) for n, m in self.hf.named_modules()
                      if isinstance(m, torch.nn.ModuleList) and len(m) == spec['n_layers']
                      and hasattr(m[0], 'mlp') and hasattr(m[0], 'input_layernorm')]
        assert len(candidates) == 1, [n for n, _ in candidates]
        self.layers_name, self.layers = candidates[0]
        self.hidden = self.hf.get_output_embeddings().weight.shape[1]
        assert self.hidden == spec['hidden_size']
        self.head_fp32 = None
        self.vectors = {}

    def encode(self, texts, add_special_tokens=True):
        return self.tok(texts, padding=True, return_tensors='pt',
                        add_special_tokens=add_special_tokens).to('cuda')

    def forward(self, enc):
        return self.hf(**enc, use_cache=False, logits_to_keep=1)

    def check_fast_path(self):
        calls = set()
        def profile(frame, event, arg):
            if event == 'call' and any(k in frame.f_code.co_name for k in ('delta_rule', 'conv1d')):
                calls.add(f"{frame.f_globals.get('__name__')}.{frame.f_code.co_name}")
        sys.setprofile(profile)
        try:
            with torch.inference_mode():
                self.forward(self.encode(['hello world ' * 20]))
        finally:
            sys.setprofile(None)
        ok = any(c.startswith(('fla', 'causal_conv1d')) for c in calls) and not any('torch_' in c for c in calls)
        assert ok, f'Fast attention path not active: {sorted(calls)}'
        return sorted(calls)

    @contextlib.contextmanager
    def steering(self, coefficient=0., vector=None):
        if vector is None or coefficient == 0:
            yield
            return
        delta = vector.to(device='cuda', dtype=torch.bfloat16)
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                return (output[0] + coefficient * delta, *output[1:])
            return output + coefficient * delta
        handle = self.layers[self.spec['inject_block']].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def capture(self, texts, layers=None, batch=4, add_special_tokens=True, coefficient=0., vector=None):
        layers = layers if layers is not None else [self.spec['extract_block']]
        parts = {layer: [] for layer in layers}
        def hook_for(layer):
            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                parts[layer].append(hidden[:, -1].detach().float().cpu())
            return hook
        handles = [self.layers[layer].register_forward_hook(hook_for(layer)) for layer in layers]
        try:
            with self.steering(coefficient, vector), torch.inference_mode():
                for start in range(0, len(texts), batch):
                    check_stop()
                    enc = self.encode(texts[start:start + batch], add_special_tokens=add_special_tokens)
                    assert enc['attention_mask'][:, -1].all()
                    self.forward(enc)
        finally:
            for handle in handles:
                handle.remove()
        return torch.stack([torch.cat(parts[layer]) for layer in layers], dim=1)

    def generate(self, prompt, coefficient=0., vector=None, sample=False, seed=0, max_new_tokens=120):
        if sample:
            torch.manual_seed(seed)
        enc = self.encode([prompt], add_special_tokens=not sample)
        args = {'do_sample': sample, 'max_new_tokens': max_new_tokens,
                'pad_token_id': self.tok.pad_token_id}
        if sample:
            args.update(temperature=.7, top_p=.95)
        with self.steering(coefficient, vector), torch.inference_mode():
            out = self.hf.generate(**enc, **args)
        ids = out[0, enc['input_ids'].shape[1]:].tolist()
        return {'text': self.tok.decode(ids, skip_special_tokens=True),
                'raw_text': self.tok.decode(ids, skip_special_tokens=False), 'token_ids': ids,
                'token_count': len(ids), 'hit_limit': len(ids) == max_new_tokens}

    def name_logprob(self, prompt, name, coefficient=0., vector=None):
        """Compute the output projection itself in fp32, before log_softmax."""
        p = self.tok(prompt, add_special_tokens=False)['input_ids']
        full = self.tok(prompt + name, add_special_tokens=False)['input_ids']
        assert full[:len(p)] == p and len(full) > len(p), 'Continuation retokenizes the prompt'
        ids = torch.tensor([full], device='cuda')
        with self.steering(coefficient, vector), torch.inference_mode():
            out = self.hf(ids, output_hidden_states=True, use_cache=False, logits_to_keep=1)
            hidden = out.hidden_states[-1][0, len(p) - 1:-1].float()
            if self.head_fp32 is None:
                self.head_fp32 = self.hf.get_output_embeddings().weight.detach().float()
            logits = hidden @ self.head_fp32.T
            bias = getattr(self.hf.get_output_embeddings(), 'bias', None)
            if bias is not None:
                logits += bias.float()
            lp = torch.log_softmax(logits, dim=-1)
            value = lp.gather(1, ids[0, len(p):, None]).sum()
        return float(value)

    def raw_unembed(self, vector):
        weight = self.hf.get_output_embeddings().weight
        # Chunking avoids materializing another full 5 GB copy for this diagnostic.
        with torch.inference_mode():
            return torch.cat([(weight[i:i + 8192].float() @ vector.cuda().float()).cpu()
                              for i in range(0, weight.shape[0], 8192)]).numpy()

    def check_steering(self, vector):
        enc = self.encode(['The receipt is in the drawer. I feel:'])
        saved = []
        def capture(module, inputs, output):
            saved.append((output[0] if isinstance(output, tuple) else output).detach().clone())
        with torch.inference_mode():
            h = self.layers[self.spec['inject_block']].register_forward_hook(capture)
            try:
                self.forward(enc)
            finally:
                h.remove()
            with self.steering(.5, vector):
                h = self.layers[self.spec['inject_block']].register_forward_hook(capture)
                try:
                    self.forward(enc)
                finally:
                    h.remove()
        expected = saved[0] + .5 * vector.cuda().to(torch.bfloat16)
        assert torch.equal(expected, saved[1]), 'Steering must add coefficient * bf16 vector at every prefill position'
        return {'exact_prefill_addition': True, 'positions_checked': saved[0].shape[1], 'injection_block': self.spec['inject_block']}
