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

## Part 2 - Load balancing (MoE)

Notebook: `Part-2/maforllms-ass1-pt2.ipynb` · Results: `Part-2/` · 1× T4, fp16, seeds 42/43 · transformers 5.16.1, torch 2.11.0+cu128

**Setup.** EuroMoE-2.6B-A0.6B: 24 layers, 64 routed + 0 shared experts, top-8, softmax router with renormalized top-k weights, 2.61B total / 0.50B active. Pretraining balancing is not published (config has `router_aux_loss_coef = 0.01`, unconfirmed).
Data: Kobza split by doc-id hash, train 2.46M / val 41k / test 123k tokens (1,024-token chunks), Part 1 eval docs excluded.
Training: 300 steps × 8,192 tokens, only attention and routers are trained (experts frozen to fit a T4), same data order per seed, 30.7 min per run.

**Method.** Own `BalancedRouter` (fp32 logits, same interface as HF) records f_i, MaxVio, load entropy and logsumexp per layer. Drift = Jaccard overlap of base and trained top-8 sets on the same test tokens.
Aux loss is N·Σf_i·P_i (1.0 at balance; HF's version equals k). Loss-free bias: fp32 buffer outside the optimizer, selection only, sign update with u = 1e-3 after every step.
Checks: top-k matches the HF router on 99.4% of tokens, bias never changes the gates, aux loss is non-zero (1.97) and changes with the coefficient.
Noise floor (random top-8 on 123k tokens): MaxVio **0.018 ± 0.003**; MaxVio ceiling is N/K − 1 = 7.

| Strategy | Test ppl | Δppl | MaxVio mean | MaxVio max | Entropy | Drift (Jaccard) | Seeds |
|---|---|---|---|---|---|---|---|
| base (no training) | 6.947 | – | 5.695 | 6.963 | 0.884 | 1.000 | – |
| no balancing | 6.756 | 0 | 5.719 | 6.976 | 0.881 | 0.902 | 2 |
| aux 1e-3 | 6.752 | −0.01% | 5.709 | 6.974 | 0.883 | 0.902 | 1 |
| aux 1e-3 + z-loss | 6.753 | 0.00% | 5.708 | 6.974 | 0.883 | 0.901 | 1 |
| aux 1e-2 | 6.761 | +0.08% | 5.647 | 6.961 | 0.894 | 0.884 | 2 |
| aux 1e-1 | 6.861 | +1.60% | 5.169 | 6.934 | 0.924 | 0.777 | 1 |
| loss-free bias | 22.18 | **+225%** | **0.559** | **1.40** | **0.997** | 0.437 | 2 |
| frozen router | **6.745** | −0.12% | 5.703 | 6.971 | 0.883 | 0.937 | 1 |
| random router (floor) | – | – | 0.018 | – | 1.000 | – | – |

Two-seed rows are means over seeds 42/43 (seed spread: ppl ±0.004, loss-free ppl 21.98 / 22.37). Raw data: `part2_runs/*.json`, `part2_results.csv`; plots: `part2_heatmaps_all.png`, `part2_curves.png`, `part2_base_layer_curves.png`.

**Findings.**
- The base model is badly imbalanced (MaxVio 5.7 of 7) and it gets worse with depth: MaxVio 4.8 / 5.7 / 6.6 for early / middle / late layers, layer 23 has 41 of 64 experts under 0.25× load. 24 experts in 15 layers are in the top-8 of >90% of tokens (implicit shared experts).
- Without balancing, 300 steps do not change balance (+0.02 MaxVio, about the noise floor). aux 1e-3 and the z-loss are the same as no balancing (router logsumexp is already ≈0.01).
- aux 1e-2 gives a small repeatable gain (MaxVio −0.07, same on both seeds) at no measurable cost (+0.08% ppl, within seed spread). aux 1e-1 balances more (entropy 0.924) for +1.6% ppl.
- Loss-free at u = 1e-3 balances best (MaxVio 0.56, entropy 0.997) but ppl triples (6.76 → 22.2) and drift is 0.44, on both seeds. Probable cause: the step is large next to softmax probabilities of ~1/64 (hypothesis, u = 1e-4 not run).
- Frozen-router ppl equals the trained routers within noise. Its drift is 0.937, so a large part of the drift comes from attention changing the router input.
- Predictions were registered before the runs (`predictions.json`); no_balance and aux 1e-3 rows used an earlier partial run. Biggest miss: loss-free ppl predicted +3%, measured +225%.

**Limitations.** Experts are frozen and the run is short, so this is not pretraining-style collapse. Most strategies have one seed, loss-free was run only at u = 1e-3, MaxVio saturates near its ceiling (entropy is more informative), Ukrainian only, Part 1 used transformers 5.0.0.

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
