## Part 3 — MixLoRA (bonus)

Notebook: `maforllms-ass1-pt3.ipynb` · Config: `config_part3.json` · 1× T4, fp16, seed 42

**Setup.** Qwen3-1.7B on Ukrainian GEC (`lang-uk/UberText-GEC`). Train/val 20,000/500 pairs, split by text hash.
Test: 1,000 human-accepted pairs from the annotation file, removed from training.
Every run: 1,000 steps, batch 16, lr 2e-4 (LoRA) / 1e-3 (router).

**Method.** Own `MixLoRAFFN`: 8 rank-8 LoRA experts on gate/up/down, sharing the frozen FFN, with a top-2 fp32 router,
plus rank-8 LoRA on attention. Balancing: none / aux loss 1e-2 / bias update (u=1e-4, selection only).
Baselines: LoRA r=16, r=44 (matches MixLoRA's 47.7M total params), r=13 (matches its 14.7M active params).
At init (B=0) the patched model reproduces the base model exactly (logit diff 0.000).

| Run | GLEU | Test loss | MaxVio | Train (min) |
|---|---|---|---|---|
| copy source (floor) | 69.98 | – | – | – |
| zero-shot base | 48.36 | – | – | – |
| mixlora_no_balance | 76.98 | 0.128 | 2.32 | 166 |
| mixlora_aux_1e-2 | 77.13 | 0.126 | 0.23 | 162 |
| mixlora_loss_free | 77.07 | 0.127 | 1.65 | 164 |
| lora_r16 | 76.95 | 0.127 | – | 87 |
| lora_r44 (total match) | **77.32** | **0.122** | – | 95 |
| lora_r13 (active match) | 76.85 | 0.128 | – | 90 |

MaxVio ranges from 0 (perfectly even) to 3 (the same 2 experts take every token); a random router scores 0.005.

**Findings.**
- Fine-tuning adds about +7 GLEU over copying the source. The untrained model scores below the copy floor.
- MixLoRA does not beat parameter-matched LoRA. LoRA r=44 has the lowest val loss at every checkpoint from step 400 and trains 1.7× faster. MixLoRA ties the active-matched LoRA.
- Without balancing, the router collapses within ~100 steps (MaxVio 2.32, entropy 0.65), with no quality loss. The aux loss balances at no cost. The bias update at u=1e-4 is too slow: the bias grows linearly to 0.10 and never catches up.

**Limitations.** One seed; GLEU gaps ≤ 0.47 are not significant. No predictions were registered before the runs.
