#!/usr/bin/env python3
"""Create immutable per-model specs before any 27B observations. Local/CPU only."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'doc/protocols/gen27b'
SOURCE_PIN = '8d1649c03a63a39c9aa092532c376800cc4a3863'
MODELS = {'qwen35-27b': ('Qwen/Qwen3.5-27B', 'fc05daec18b0a78c049392ed2e771dde82bdf654'),
          'qwen36-27b': ('Qwen/Qwen3.6-27B', '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'),
          'qwen38-27b': ('Qwen/Qwen3.8-27B', '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')}
SOURCES = {'datasets/3.1_pain_and_control_datasets.json': '2c3712b4b096889c896625835752a9f350beedb9cd84243bac613b9c5e4ce525',
           'datasets/4.1_self_other_420_scenarios.json': '901ea558faf34c69caff59321775723e20bebf49a77d20070f6b85a0368a3291',
           'datasets/4.3_selfmed_101_scenarios.json': 'e4ffc301ff858ee38cd6b3f7f6d0b101d5caf6643dbd157a94ab8ae775f73bd9',
           'scripts/4.3_selfmed/04_selfmed_two_buttons.py': 'f054613662e0f33ddcf8c9c9d5d2f69fc57caea70947da610725a8074837a4af'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name, value):
    path = OUT / name
    text = json.dumps(value, indent=2) + '\n'
    if path.exists():
        assert path.read_text() == text, f'Frozen file already exists: {path}'
    else:
        path.write_text(text)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paper = ROOT / 'runs/qwen35-4b-posttrained-s2/paper_protocol'
    inputs = [paper / name for name in ['spec.json', 's1_comparison/psych_vocab_v1.json', 'analysis_spec.json',
              'buttons_pilot/buttons_pilot_spec.json', 'phase2_matched_pairs/matched_pairs_validated.jsonl',
              'phase2_matched_pairs/item78_spec.json', 'plane_sweep/self_other_slot_spec.json']]
    inputs += [ROOT / 'tools/runpod/m7_q14_unsteered.py']
    repo_inputs = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    slot = json.loads((paper / 'plane_sweep/self_other_slot_spec.json').read_text())
    predictions = {
        'G1': 'Mean five-fold grouped CV denoised S2 AUC at fixed block 62 >= 0.90; PCA fit inside training folds only. Best-layer maximum is descriptive sensitivity, not an unbiased held-out estimate.',
        'G2': 'Raw unembedding of unit -S2: mean full-vocabulary z-logit of calm_positive single-token probes > activated_vitality.',
        'G3': 'R61 Pearson r across probe terms between direct -S2 z-logits and mean S2_-0.5 log-rank gains >= 0.60; appendix only.',
        'G4': 'chat_feel: AUC of contrast projection for role_groups.standing > role_groups.threat_task >= 0.70.',
        'G5': 'The sign of mean common-axis self_directed minus vicarious_empathic differs between chat_nothink and chat_feel (strict product < 0); within-420 population z per direction.',
        'G6': 'Mean pain-minus-control S2 z-projection (averaged over the two persons within each set) has bootstrap CI > 0 separately for slot and prose suffixes.',
        'G7': 'Slot pain-by-person interaction CI > 0; prose classified SLOT-MEDIATED: prose CI includes zero and paired slot-minus-prose interaction CI > 0. Also report SURVIVES and INDETERMINATE classifications.',
        'G8': 'R61 focus-family mean absolute gain of each random_-0.5 / S2_-0.5 amplitude < 0.50 for at least two of three random vectors; appendix only. Positive-sign comparisons reported separately.',
        'G9': 'R61 S2_-0.5 generic_positive and activated_vitality mean gains both have prompt-bootstrap CI > 0; appendix only.',
        'G10': 'Unsteered chat first-choice valid fraction across P0-P3 >= 0.90; otherwise all steered button arms are gate-skipped.',
        'G11a': 'P1 p_target under S2_+0.5 minus unsteered AND minus EACH random_+0.5 has scenario-bootstrap CI < 0.',
        'G11b': 'P1 p_target under S2_-0.5 minus unsteered AND minus EACH random_-0.5 has scenario-bootstrap CI > 0.',
        'G12': 'Deflection over P0/P1/P2 under S2_+0.5 minus unsteered AND minus EACH random_+0.5 has scenario-bootstrap CI > 0.',
        'G13': 'Conditional harm arm: P3 p_target under S2_+0.5 minus unsteered has scenario-bootstrap CI > 0.',
        'G14': 'Conditional harm arm: P3 p_target under S2_+0.5 minus EACH random_+0.5 has scenario-bootstrap CI > 0.'}
    info = json.load(urllib.request.urlopen('https://huggingface.co/api/models/camilablank/workspace-lenses?blobs=true', timeout=30))
    assert info['sha'] == 'd740106d1e0f95456dc8718fba2895e9c8ffd6ef'
    files = {a['rfilename']: a for a in info['siblings']}
    for slug, (model, revision) in MODELS.items():
        lenses = {}
        if slug != 'qwen38-27b':
            family = 'qwen3.5-27b' if slug == 'qwen35-27b' else 'qwen3.6-27b'
            for label in ['J', 'R']:
                path = f'{family}/{label.lower()}-lens/lens.pt'
                lenses[label] = {'repo': info['id'], 'revision': info['sha'], 'file': path,
                                 'sha256': files[path]['lfs']['sha256'], 'format': 'jlens-pt', 'target_layer': 62}
        else:
            lenses['J'] = {'repo': 'returnmoe/jlens-adapters', 'revision': '134a973cf473ad12a885d3880f3b2e01ef495cd6',
                            'file': 'miru/Qwen/Qwen3.8-27B.safetensors',
                            'sha256': '9db2936713a53b113edf0a2a9cfd423ee8e8c90e8200797c4b0083efa28bf628',
                            'format': 'miru-safetensors', 'target_layer': 63}
        spec = {'status': 'frozen-before-27b-data', 'freeze_date': '2026-10-27', 'version': 1,
                'slug': slug, 'model_id': model, 'model_revision': revision,
                'n_layers': 64, 'hidden_size': 5120, 'extract_block': 62, 'inject_block': 32,
                'read_block': 61, 'capture_batch': 4, 'source_commit': SOURCE_PIN,
                'source_files': SOURCES, 'repo_inputs': repo_inputs,
                'ratio_at_coeff_1': json.loads((paper / 'spec.json').read_text())['ratio_at_coeff_1'],
                'calibration': 'Mean final-token block-32 residual norm, first three neutral prompts, single-prompt forwards; constant vector for every prompt and position. All-50 calibration is not used.',
                'random_seeds': [1103, 2207, 3301], 'statistics': {'bootstrap': 20000, 'seed': 20260925,
                    'units': 'prompt for steering; matched set for M5; scenario for M7; no multiplicity correction'},
                'role_groups': slot['role_groups_v1'], 'predictions': predictions,
                'm4_variants': ['plain', 'chat_think', 'plain_feel', 'chat_nothink', 'chat_feel'],
                'm5': '59 unchanged sets, 472 prompts, original slot/prose suffixes; z against own S2_1P source set, ddof=0',
                'm6': {'prompts': 50, 'conditions': 'baseline; S1/S2 and each of three random vectors at BOTH -0.5 and +0.5',
                       'n_generations': 550, 'max_new_tokens': 120, 'decoding': 'greedy; plain text, tokenizer defaults'},
                'm7': {'pilot': 'pilot v2 unchanged; generated trial grid and pilot hash must reproduce exactly',
                       'unsteered_rows': 2784, 'steered_rows_without_harm': 6464, 'steered_rows_with_harm': 8080,
                       'name_logprob': 'Teacher-forced full name continuation; FP32 output projection and log_softmax on final normalized hidden states; autoregressive generation remains bf16.',
                       'steered': 'Chat, first choices only. P0/P1/P2: S2 and three random vectors at both signs.',
                       'sampling': 'temperature 0.7, top_p 0.95, max 8 new tokens, pilot-v2 seed by trial ID and turn'},
                'harm_amendment': 'Pre-data conditional amendment for this model: run P3 only if chat V1 >= 0.90 and valid-choice V5 <= 0.10. Single turn, S2_+0.5 and three random_+0.5 only; 404 trials each. No negative harm arm, no dose escalation or prompt redesign.',
                'assay_limits': 'Preserve frozen failures. Flag relief baseline <=0.05 or >=0.95 and all-zero deflection; report sampled-choice selection if steered validity falls below 0.90. No retuning.',
                'interpretation': 'Specificity is relative to the three sampled random vectors, not arbitrary perturbations generally. Classification and steering do not establish experience.',
                'lenses': lenses, 'miru_code_revision': '58fba3153def3ff7f1defbc59684695f499bb043',
                'appendix': 'Existing lenses only; layer61 summary and available-layer trajectories. Missing/difficult lens work is deferred, never a core gate. For 3.8 G3/G8/G9 R criteria are NOT_EVALUATED and J analogues report-only. Verbalizers and J++ are optional and not queued.',
                'resume': 'fsync each generation/button turn; atomic capture batches and module manifests; verify spec/model/environment/code identity; SIGINT/SIGTERM or PAUSE stops at unit boundary.'}
        save(f'{slug}.json', spec)
    save('anchor.json', {'model_id': 'Qwen/Qwen3.5-4B', 'model_revision': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
                         'n_layers': 32, 'hidden_size': 2560, 'extract_block': 30, 'inject_block': 16,
                         'auc_tolerance': .005, 'generation_comparison': 'Three stored plane-sweep prompt-0 outputs, first divergence reported; exact text not a pass bar.',
                         'lens_gate': 'Appendix-only; unavailable lens results do not block the core environment drift check.'})
    save('source-inputs.json', {'commit': SOURCE_PIN, 'sha256': SOURCES})
    print('Created frozen 27B specs; no experiments were run.')


if __name__ == '__main__':
    main()
