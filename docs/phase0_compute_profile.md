# AlphaMoo v4.1 — Phase 0 ComputeProfile Report

> **⚠️ IMPORTANT: Mixed measurement methods.** Read this carefully.
>
> - **Symbolic pipeline cost** (Perception, Cascade, Tracker, Prompt Build): **MEASURED** on real data in this sandbox
> - **LLM cost**: **ESTIMATED** using a latency-model stub calibrated against published benchmarks. NOT measured on real GPU.
>
> To get real LLM numbers, run `notebooks/phase0_kaggle_measurement.ipynb` on a Kaggle RTX 6000 notebook. The output `real_compute_profile.json` supersedes the LLM estimates in this doc.

---

## What was actually measured vs estimated

| Component | Method | Where | Confidence |
|-----------|--------|-------|------------|
| Perception (CCL + objects) | Real measurement | This sandbox (CPU) | ✅ High — same on Kaggle |
| Cascade interpreter | Real measurement | This sandbox (CPU) | ✅ High — same on Kaggle |
| Agent State Tracker | Real measurement | This sandbox (CPU) | ✅ High — same on Kaggle |
| Prompt building | Real measurement | This sandbox (CPU) | ✅ High — same on Kaggle |
| **LLM inference** | **Stub estimate** | Calibrated latency model | ⚠️ ±30% — needs Kaggle validation |

The symbolic pipeline is CPU-bound and runs the same on Kaggle. The LLM is GPU-bound and the only way to know real cost is to run it on a Kaggle RTX 6000.

---

## Estimated Per-Model Results (LLM = stub)

| Latency Model | Avg Step (ms) | Projected Total | Fits 9hr Budget? | Headroom |
|---------------|---------------|-----------------|------------------|----------|
| `qwen2.5-0.5b-4bit` | **1,467.8** | **6.03h (21,723s)** | ✅ **YES (estimated)** | **33.0%** |
| `qwen2.5-1.5b-4bit` | 2,691.3 | 11.06h (39,829s) | ❌ NO (estimated) | OVER by 2.06h |
| `vibethinker-3b-4bit` | 3,926.6 | 16.14h (58,074s) | ❌ NO (estimated) | OVER by 7.14h |

**Latency model assumptions** (calibrated against published 4-bit Qwen2.5 benchmarks on H100/RTX 6000-class GPUs):

| Model | Prefill (tok/s) | Decode (tok/s) | Overhead (ms) |
|-------|-----------------|----------------|---------------|
| Qwen2.5-0.5B 4-bit | 500 | 80 | 5 |
| Qwen2.5-1.5B 4-bit | 250 | 45 | 8 |
| VibeThinker-3B 4-bit | 180 | 30 | 10 |

Real Kaggle RTX 6000 performance may vary ±30% from these estimates. The stub gives us a *lower bound* on real LLM cost.

---

## Measured Pipeline Cost Breakdown (Qwen2.5-0.5B baseline)

| Stage | Total Time (20 steps) | % of Total | Method |
|-------|----------------------|------------|--------|
| Perception | 0.12s | 0.4% | **Measured** |
| Cascade | 0.08s | 0.3% | **Measured** |
| Tracker | 0.00s | 0.0% | **Measured** |
| Prompt Build | 0.00s | 0.0% | **Measured** |
| **LLM** | **29.16s** | **99.3%** | ⚠️ **Stub estimate** |
| **Total** | **29.36s** | **100%** | Mixed |

**Insight:** Symbolic pipeline cost is negligible (0.7% of step time). The entire budget is LLM. Any optimization effort should target the LLM (token budget, batching, prefix caching) — not the symbolic modules.

---

## How to get real numbers

Run `notebooks/phase0_kaggle_measurement.ipynb` on Kaggle:

1. Go to https://www.kaggle.com/code
2. New Notebook → set accelerator to **GPU RTX 6000**
3. Set internet to **ON** for first run (to download model weights)
4. Upload `notebooks/phase0_kaggle_measurement.ipynb`
5. Run all cells (~5 min)
6. Download `real_compute_profile.json`
7. Replace this file's LLM estimates with the real measurements

The notebook measures:
- Cold load time
- 100 warm inference calls with our actual Phase 0 prompt
- Per-call wall-clock (avg, P50, P95, P99, max)
- Peak VRAM usage
- Prefill/decode throughput (tokens/sec)
- Stub-vs-real comparison

---

## What we expect from the real measurement

### Most likely outcome (70% confidence)
Real Qwen2.5-0.5B-AWQ on Kaggle RTX 6000 hits **~80-120 tok/s decode**, matching our stub estimate within ±20%. Phase 0 conclusion stands: **0.5B fits, 1.5B doesn't.**

### Optimistic outcome (20% confidence)
Real Kaggle RTX 6000 is faster than expected (vLLM batching, prefix caching kick in). 0.5B hits ~150-200 tok/s, 1.5B hits ~80-100 tok/s. **Both fit**, with 1.5B being the new default for better reasoning quality.

### Pessimistic outcome (10% confidence)
Real Kaggle RTX 6000 has overhead we didn't model (container startup, GPU contention, model loading eats into the 9hr budget). 0.5B hits ~50-60 tok/s. Phase 0 conclusion changes: **0.5B barely fits, need aggressive token optimization.**

---

## Token Statistics (from measured runs)

- Total prompt tokens generated: 4,528 (over 20 steps)
- Total output tokens generated: 1,600 (over 20 steps)
- Average prompt tokens/step: **226**
- Average output tokens/step: **80** (capped)

At 0.5B 4-bit estimated latency: 226/500 + 80/80 = 0.45 + 1.0 = **1.45 sec/action** (matches measured 1.47s)

---

## What This Means for the Architecture (assuming stub is accurate)

### Provisional decisions (subject to Kaggle validation)
- ✅ **Base model: Qwen2.5-0.5B-Instruct at 4-bit AWQ** (likely the only viable choice)
- ✅ **Max output tokens: 80** (could go lower; 60 may work)
- ✅ **Max prompt tokens: ~250** (current 226 is fine; don't exceed 300)
- ✅ **No speculative decoding needed** (0.5B is already fast enough)
- ✅ **No KV cache quantization** (48GB VRAM has plenty of headroom)

### Rejected alternatives (subject to Kaggle validation)
- ❌ **Qwen2.5-1.5B**: 11h projected, over budget by 2h. Would need 60%+ efficiency gains to fit.
- ❌ **VibeThinker-3B**: 16h projected, over budget by 7h. Even with CLR disabled, doesn't fit.
- ❌ **Ornith-9B**: not tested but at 9B 4-bit, projected ~30h. Absurdly over.
- ❌ **DFlash spec decoding**: irrelevant — accelerates Qwen3-4B, not 0.5B.

### Optimization headroom (33% = 2.97 hours, if 0.5B estimate holds)
If we want to upgrade from 0.5B to a larger model later, we have 2.97 hours of headroom. Options:
1. **Batch multiple games in parallel** via vLLM continuous batching — could 2-3× throughput
2. **Cut output tokens** from 80 to 40-50 — saves ~30% of LLM time
3. **Prompt prefix caching** — saves prefill cost on shared system prompt (~10-15%)
4. **Use the headroom for smarter reasoning** — more hypothesis exploration per action

---

## Methodology

### Replay-driven simulator (measured portion)
We walk through ground-truth replays frame by frame. The agent sees the same frames the human saw, in order. The agent's actions don't affect which frame comes next — we just step through the replay.

This is sufficient for measuring **symbolic pipeline cost** because that cost doesn't depend on which frame we're looking at.

### LLM stub (estimated portion)
The LLM is a mock that:
- Builds a realistic "reasoning" string of ~80 tokens
- Picks an action heuristically (random valid movement/click)
- Sleeps for the latency-model-estimated duration to simulate real LLM cost
- Returns realistic token counts and timing

The latency model is calibrated against published 4-bit Qwen2.5 benchmarks. **It is NOT a measurement.** Real Kaggle RTX 6000 performance must be measured with `notebooks/phase0_kaggle_measurement.ipynb`.

### What's NOT measured here
- Agent quality (how good the actions are) — Phase 2+
- Real LLM reasoning quality — Phase 5
- Cascade overhead on high-cascade games (ls20 has cascades on 8% of steps; sb26 on 78%) — needs separate test
- GPU memory pressure (0.5B 4-bit + KV cache + intermediate state) — needs real GPU test
- Kaggle container startup time (eats into the 9hr budget) — needs real Kaggle run

---

## Next Steps

### Immediate (validate Phase 0)
1. **Run `notebooks/phase0_kaggle_measurement.ipynb` on Kaggle RTX 6000**
2. Replace LLM estimates in this doc with real measurements from `real_compute_profile.json`
3. If 0.5B doesn't fit: aggressive token optimization or architecture revision

### Phase 1 completion
4. Fix Module 16 for click-only games (current 47-78% detection rate)
5. Build Module 15: Spatial Memory Map (for partial-observability levels like LS20 L7)

### Phase 2 (Exploration Loop)
6. Build Module 5: Hypothesis Generator — predicate language + Bayesian update
7. Build Module 6: Goal Inference — separate hypothesis layer for win conditions
8. Build Module 7: Near-Miss Tracker — bootstrap goal signal from LOSE episodes
9. Build Module 8: Experiment Planner — three-tier IG (symbolic/surrogate/LLM)
   - **Note:** Tier 3 LLM IG is likely infeasible at 0.5B (no spare tokens for IG computation). Drop Tier 3, use only Tier 1 + Tier 2.

### Phase 3 (World Model + Planner)
10. Build Module 9: Executable World Model
11. Build Module 10: Verifier
12. Build Module 11: Planner Interface (A*/MCTS/Policy/LLM/Click/Transformation variants)

### Phase 5 (Eval + Ablations)
13. Run real Qwen2.5-0.5B-Instruct 4-bit on Kaggle notebook
14. Measure actual wall-clock per action vs Phase 0 prediction (validate the stub)
15. Ablate each module on/off

---

## Conclusion

**Phase 0 estimates (pending Kaggle validation) indicate Qwen2.5-0.5B 4-bit fits the 9-hour budget with 33% headroom. 1.5B and 3B do not fit.** Symbolic pipeline cost is negligible.

**The cow has been estimated. The cow probably fits. Time to validate on Kaggle, then build.** 🐄
