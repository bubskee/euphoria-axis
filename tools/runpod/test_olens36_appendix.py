"""CPU contract invariants for the separately frozen exploratory panel."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from olens36_appendix import AdapterHooks, SPEC, render_marker, unit_injection


class ContractTests(unittest.TestCase):
    def test_injection_preserves_direction_and_fixed_norm(self):
        x = torch.tensor([3., 4.])
        y = unit_injection(x, 16000.)
        self.assertTrue(torch.equal(y, unit_injection(x * 7, 16000.)))
        self.assertAlmostEqual(float(y.norm()), 16000., places=2)
        with self.assertRaises(AssertionError):
            unit_injection(torch.zeros(2), 16000.)

    def test_lora_hook_matches_independent_weight_update_and_disables_exactly(self):
        # Check actual loading, key matching and hooks, rather than only repeating a helper.
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        model.model.layers[0].q_proj = torch.nn.Linear(3, 2, bias=False)
        linear = model.model.layers[0].q_proj
        x = torch.tensor([[1., 2., -1.]])
        a = torch.tensor([[1., 0., 2.]])
        b = torch.tensor([[2.], [-1.]])
        before = linear(x).detach().clone()
        original = linear.weight.detach().clone()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'adapter.safetensors'
            prefix = 'base_model.model.model.layers.0.q_proj.'
            save_file({prefix+'lora_A.weight': a, prefix+'lora_B.weight': b}, str(path))
            hooks = AdapterHooks(SimpleNamespace(hf=model, layers_name='model.layers'), path,
                                 {'peft_type':'LORA','bias':'none','r':1,'lora_alpha':2})
            self.assertTrue(torch.equal(linear(x), before))
            hooks.enabled = True
            expected = x @ (original + 2 * b @ a).T
            self.assertTrue(torch.allclose(linear(x), expected, atol=1e-6))
            self.assertEqual(hooks.calls, hooks.module_names)
            hooks.enabled = False
            self.assertTrue(torch.equal(linear(x), before))
            self.assertTrue(torch.equal(linear.weight, original))
            hooks.close()

    def test_panel_count_and_contract_are_frozen(self):
        s = json.loads(SPEC.read_text())
        self.assertEqual(len({x['id'] for x in s['cases']}), 12)
        self.assertEqual(len(s['source_layers']) * len(s['cases']) * len(s['generation']['seeds']), 72)
        self.assertTrue(set(s['source_layers']) <= set(s['lens']['trained_layers']))
        self.assertNotIn(61, s['source_layers'])
        self.assertNotIn(62, s['source_layers'])

    def test_marker_is_checked_in_full_prompt(self):
        # First isolated candidate merges with a neighbour; second survives in context.
        class Tokenizer:
            def apply_chat_template(self, messages, **kw):
                assert kw['enable_thinking'] is False and kw['add_generation_prompt']
                return messages[0]['content']
            def __call__(self, text, **kw):
                if len(text) == 1:
                    return {'input_ids':[158983]}
                start = text.index('<activation>') + len('<activation>')
                end = start + 1
                if text[start] == chr(0x3200): end += 1
                return {'input_ids':[1,158983,2], 'offset_mapping':[(0,start),(start,end),(end,len(text))]}
        spec = dict(json.loads(SPEC.read_text())['lens'])
        spec['marker_scan'] = [0x3200, 0x3201]
        p = render_marker(Tokenizer(), 44, spec)
        self.assertEqual(p['char'], chr(0x3201))
        self.assertEqual(p['slot'], 1)


if __name__ == '__main__':
    unittest.main()
