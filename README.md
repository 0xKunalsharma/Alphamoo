# AlphaMoo v4.1 — ARC-AGI-3 Solver

> A Bayesian scientific discovery agent for ARC-AGI-3.
> Built to land in the upper leaderboard band by cracking games that pure-LLM RL approaches can't.

## Status (as of June 29, 2026)

- ✅ Phase 0.5 (Data Mining): complete
- ✅ Phase 1 (Grounding): partial — Perception, Cascade Interpreter, Vortex reader, frame renderer all working
- ⏳ Phase 2 (Exploration Loop): not started
- ⏳ Phase 3 (World Model + Planner): not started
- ⏳ Phase 4 (Context Compression): not started
- ⏳ Phase 5 (Eval + Ablations): not started
- ⏳ Phase -1 (Synthetic Pretraining): not started (needs GPU)

## What's Built

### Core modules (`src/alphamoo/`)

| Module | Purpose | Status |
|--------|---------|--------|
| `schemas.py` | All dataclasses (FrameRecord, ActionInput, SceneGraph, GameObject, Hypothesis, ...) | ✅ |
| `vtx_reader.py` | Parse Vortex `.vtx` files into clean Python `Replay` objects | ✅ |
| `frame_renderer.py` | Render 64×64 grids to PNG with ARC palette | ✅ |
| `data_mining.py` | Mine 14,798 actions for patterns (object events, action-outcomes) | ✅ |
| `perception.py` | CCL + object extraction + relation extraction | ✅ |
| `cascade_interpreter.py` | Diff multi-subframe cascades into discrete events | ✅ |
| `replay_viewer.py` | CLI to step through any replay with rich diagnostics | ✅ |

### Data assets (`data/`)

- 25 ground-truth replays in Vortex format (33MB compressed, 66MB extracted)
- 14,798 human actions across all 25 public demo games
- All replays reach WIN state
- 5 distinct action-set signatures across games (variety confirmed)

### Generated outputs (`download/`)

- `arc_palette_legend.png` — the 16-color ARC palette
- `ls20_frame0_initial.png` — initial LS20 frame
- `ls20_cascade_example.png` — 6-subframe cascade example
- `ls20_level{1-7}_complete.png` — level completion frames
- `mining/mining_report.md` — full data mining report
- `mining/mining_dump.json` — machine-readable mining output
- `perception_test/{game}_perception.png` — annotated frames showing detected objects
- `cascade_test/{game}_cascade_*.png` — visualizations of the biggest cascades

## Quick Start

### Install dependencies

```bash
pip install vortex-data pyarrow numpy scipy pillow
```

### Get the replay data

The 25 ground-truth replays (33MB compressed) live in the
[`Data` release](https://github.com/0xKunalsharma/Alphamoo/releases/tag/Data),
not in git. Download with:

```bash
python scripts/download_data.py
```

Or manually: download the `Data` release asset, unzip into `data/`.

### Run the test scripts

```bash
cd /home/z/my-project/alphamoo

# Test the VTX reader on all 25 replays
PYTHONPATH=src python scripts/test_reader.py

# Test the perception module on real frames
PYTHONPATH=src python scripts/test_perception.py

# Test the cascade interpreter on all cascades
PYTHONPATH=src python scripts/test_cascade.py

# Run the full data mining pipeline
PYTHONPATH=src python -m alphamoo.data_mining

# Inspect a specific replay
PYTHONPATH=src python -m alphamoo.replay_viewer ls20 --level-transitions-only
PYTHONPATH=src python -m alphamoo.replay_viewer sb26 --cascades-only --verbose
```

### Use the library

```python
import sys
sys.path.insert(0, '/home/z/my-project/alphamoo/src')

from alphamoo.vtx_reader import load_replay
from alphamoo.perception import perceive_with_diagnostics
from alphamoo.cascade_interpreter import interpret_cascade
from alphamoo.frame_renderer import save_frame_png

# Load a replay
replay = load_replay('/path/to/ls20-....vtx')

# Perceive the first frame
diag = perceive_with_diagnostics(replay.records[0].final_grid)
print(f"Background: {diag['background_color']}")
print(f"Objects: {diag['n_objects']}")
print(f"Perception time: {diag['full_perception_ms']:.2f}ms")

# Save a frame as PNG
save_frame_png(replay.records[0].final_grid, 'frame.png', scale=10)

# Interpret a cascade
final_scene, events = interpret_cascade(replay.records[7].frame)
print(f"Cascade produced {len(events)} events")
for e in events:
    print(f"  {e.type}")
```

## Key Findings from Data Mining

### Action distribution (across 14,798 actions)

| Action | Count | % |
|--------|-------|---|
| CLICK (6) | 4,359 | 29.4% |
| RIGHT (4) | 2,769 | 18.7% |
| LEFT (3) | 2,645 | 17.8% |
| UP (1) | 2,421 | 16.3% |
| DOWN (2) | 2,075 | 14.0% |
| INTERACT (5) | 392 | 2.6% |
| RESET (0) | 150 | 1.0% |
| UNDO (7) | 12 | 0.1% |

### Game variety (5 distinct action-set signatures)

- `[1,2,3,4,5,6]` — 5 games (full movement + interact + click)
- `[6]` — 5 games (click-only — no movement!)
- `[1,2,3,4,6]` — 3 games (movement + click, no interact)
- `[1,2,3,4,5]` — 3 games (movement + interact, no click)
- `[1,2,3,4]` — 3 games (movement only)
- Various 1-2 game variants (UNDO-supporting, etc.)

### Cascade statistics

- 4,586 cascade records out of 14,798 actions (31% of all actions)
- Max cascade size: 372 subframes (sb26 — match-3-style animations)
- Avg cascade size when present: ~10 subframes
- Event types detected: 89,836 color_changes, 9,701 appearances, 9,746 disappearances, 1,026 moves, 580 level_transitions

### Game difficulty (by total human actions)

- Hardest: `wa30` (1,564 actions), `lf52` (1,211), `dc22` (1,192), `re86` (1,071), `m0r0` (970)
- Easiest: `cd82` (136), `sb26` (153), `ft09` (163), `r11l` (167), `sc25` (216)

## Performance Benchmarks (Phase 0 — partial measurement)

| Operation | Method | Avg time | Budget % (1.85s/action) |
|-----------|--------|----------|--------------------------|
| Vortex load (per file) | **Measured** | ~50ms | 2.7% |
| Perception (per frame) | **Measured** | 3.71ms | 0.2% |
| Cascade interpretation (per cascade) | **Measured** | ~5-15ms | <1% |
| Frame rendering (PNG) | **Measured** | ~2ms | 0.1% |
| LLM inference (0.5B 4-bit) | ⚠️ **Stub estimate** | ~1,470ms | 79% |

**What's measured vs estimated:**
- ✅ Symbolic pipeline (Perception, Cascade, Tracker, Prompt Build): **measured on real data**
- ⚠️ LLM inference: **estimated via latency-model stub** calibrated against published Qwen2.5 4-bit benchmarks
- 🔴 Real Kaggle RTX 6000 LLM measurement: **pending** — run `notebooks/phase0_kaggle_measurement.ipynb` on Kaggle to get real numbers

**Estimate (pending validation):** Qwen2.5-0.5B 4-bit fits the 9-hour Kaggle budget with ~33% headroom. 1.5B and 3B overflow. See `docs/phase0_compute_profile.md` for full report.

## Architecture Reference

Full architecture spec: `/home/z/my-project/download/AlphaMoo_v4_Design.md` + `AlphaMoo_v4.1_Delta.md`

Module map:
```
src/alphamoo/
├── schemas.py            — All dataclasses (the data contract)
├── vtx_reader.py         — Vortex .vtx file parser
├── frame_renderer.py     — Grid → PNG rendering
├── perception.py         — CCL + object + relation extraction (Module 1)
├── cascade_interpreter.py — Multi-subframe diff (Module 14)
├── data_mining.py        — Pattern mining pipeline (Phase 0.5)
├── replay_viewer.py      — CLI inspection tool
└── __init__.py
```

## What's Next

### Immediate (next session)
1. **Agent State Tracker** (Module 16) — track the mutable agent (color, orientation, shape) across timesteps. LS20 confirmed the agent transforms; we need to follow it.
2. **Spatial Memory Map** (Module 15) — for partial-observability levels (LS20 L7)
3. **Phase 0 stub loop** — wire perception + cascade into a real agent loop, measure wall-clock per action

### Phase 2 (Exploration Loop)
4. Hypothesis Generator (Module 5)
5. Goal Inference Module (Module 6)
6. Near-Miss Tracker (Module 7)
7. Experiment Planner with three-tier IG (Module 8)

### Phase 3 (World Model + Planner)
8. Executable World Model (Module 9)
9. Verifier (Module 10)
10. Planner Interface with all 6 variants (Module 11)

### Phase 4 (Context Compression)
11. Context Compressor (Module 12)
12. ICL retrieval at level start
13. Synthetic LoRA pre-training pipeline

### Phase 5 (Eval)
14. Full Kaggle integration
15. Ablations: each module on/off
16. Tune confidence thresholds

## Strategic Context

- **Tufa Labs is at 1.21** on the leaderboard with 132 B200s, RL+LLM, 0.8B reasoning model
- **Our target: 0.5-0.8 band** — cracking games Tufa's flat LLM can't reason about
- **Secondary play: ARC Prize 2026 Paper Track** ($75K main + $375K bonus pool for >4.5/5 papers)
- **Compute budget: 9hr Kaggle runs, RTX 6000 48GB, no internet at inference**
- **Model: Qwen2.5-0.5B (default), 1.5B (stretch), 7B+ (off the table)**

## License & Acknowledgments

- ARC-AGI-3 by François Chollet / ARC Prize
- Vortex format by Spiral / Linux Foundation
- Tufa Labs' published research on synthetic pretraining for small reasoning models informed Phase -1 design
