### Team: Bohdan Kolisnyk, Mykhailo Tkachenko, Denys Pushkaruk

## Part 1 - Context extension (RoPE)

Notebook: `Part-1/maforllms-ass1-pt1.ipynb` · Results: `Part-1/` · 1× T4, fp16, seed 42 · transformers 5.0.0, torch 2.10.0+cu128

**Setup.** EuroMoE-2.6B-A0.6B (2.61B total / 0.50B active params, 64 experts, top-8, GQA 8 query / 2 KV heads, head dim 128).
Native context L_train = 4,096, RoPE base 500,000, target s = 4 (16,384 tokens). Chosen for its 4,096 native context, Ukrainian in pretraining, and T4 feasibility.
Tokenizer fertility on our eval text: 2.38 tokens/word (3.0 chars/token).
Eval: 110 Kobza documents (hplt-2.0 58, fineweb-2 26, cultura-x 23, ubertext2.0 3), each ≥16,384 tokens (median 23,192), 0 exact duplicates. The first 16,384 tokens of each document are used.

**Method.** Three modes, implemented by editing `rotary_emb.inv_freq` in place: no scaling, position interpolation (inv_freq / s), NTK-aware base scaling (b' = b·s^(d/(d−2)) = 2.04M, c_i = s^(−2i/(d−2))).
Equivalence on the frequency table (limit 1e-5): none vs loaded model 3.0e-8, PI vs HF `linear` 7.5e-9, NTK vs derived c_i formula 6.0e-8 (HF has no static NTK type).
Loss: teacher-forced on held-out human text, 16 bins of 1,024 tokens, 110 docs × 3 modes (5.2 min per mode).
Passkey: 5-digit key at depth 10/50/90%, lengths 4,096/8,192/16,384, 5 trials per cell, greedy decoding (hit = key appears in the 10 generated tokens), 135 trials in 4.6 min.
A custom GQA-safe SDPA kernel is needed on T4 (no FlashAttention-2). Passkey uses a manual prefill/decode loop.

| Method | Loss 0–4k | 4–8k | 8–12k | 12–16k | Passkey L | 2L | 4L |
|---|---|---|---|---|---|---|---|
| no scaling | 1.82 | 5.81 | 6.74 | 7.86 | 10/15 | 0/15 | 0/15 |
| position interpolation | 4.87 | 4.88 | 4.92 | 4.92 | 0/15 | 0/15 | 0/15 |
| NTK-aware | 1.84 | 1.81 | 4.22 | 6.27 | 11/15 | **14/15** | 0/15 |

Loss is the mean over the position range; passkey is hits over 3 depths × 5 trials. Raw data: `position_binned_loss_results.json`, `passkey_results.json`; plot: `position_binned_loss.png`.

**Findings.**
- No scaling has a cliff right after L_train (1.81 → 4.08 in the first bin past 4,096, then 6–8), and passkey falls to 0 at 2L.
- Training-free PI is flat at ~4.9 everywhere, including inside L_train (+3.0 nats over baseline at the first bin). It never recovers short-range resolution, and passkey is 0/45 at every length.
- NTK-aware matches baseline in range (max gap 0.018 over bins 0–3) and stays ≤1.93 up to ~8k tokens (~1.9×L_train). It then rises (2.32 → 3.10 → 5.50 at 8.7k / 9.7k / 10.8k). Passkey is 14/15 at 2L against 0/15 for baseline.
- Perplexity and passkey agree: retrieval works exactly where loss is near baseline and is 0 once loss goes above ~4.
- We would ship NTK-aware. The nominal s = 4 is not reached without fine-tuning: usable context is ≈2×L_train, and 4×L_train fails.

**Limitations.** No fine-tuned comparison, no dynamic NTK or YaRN, no English/PG-19 contrast (Ukrainian only). Passkey has 5 trials per cell (resolution 0.2).
NTK at L_train, depth 0.5 scored 0.2, below baseline's 0.6. This is probably noise but was not re-run. One seed. No predictions were registered before the runs.
Saved output of notebook cell 7 shows slowest freq 6.14e-07 because the cell was re-run after the sweeps left the model in NTK mode (unscaled value: 2.455e-06).

## Part 3 - MixLoRA (bonus)

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
