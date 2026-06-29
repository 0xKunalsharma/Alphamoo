# Changelog

All notable changes to AlphaMoo will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 0 stub loop (planned) — wires perception + cascade + tracker into a real agent

### Changed
- Nothing yet

## [0.4.1] — 2026-06-29

### Added
- **Phase 0.5: Data Mining pipeline** — mines 14,798 actions across 25 games for object events, action-outcome pairs, level-completion patterns. Output: `mining_report.md` + `mining_dump.json`.
- **Phase 1: Perception module** (`perception.py`) — CCL + object extraction + relation extraction. Avg 3.71ms per frame, 0.2% of action budget.
- **Module 14: Cascade Interpreter** (`cascade_interpreter.py`) — diffs multi-subframe cascades into discrete events. Tested on 4,586 cascades up to 372 subframes.
- **Module 16: Agent State Tracker** (`agent_tracker.py`) — movement-correlation + composite-agent detection + state propagation for click games. 94% detection rate across 14,798 actions.
- **Vortex reader** (`vtx_reader.py`) — parses `.vtx` files into clean `Replay` objects. Handles both level-based and score-based games, empty-frame backfill, mid-file summary records.
- **Frame renderer** (`frame_renderer.py`) — 64×64 grid → PNG with official ARC 16-color palette.
- **Replay viewer CLI** (`replay_viewer.py`) — step through any replay with rich diagnostics.
- **Test scripts** for reader, perception, cascade, agent tracker.
- **Project scaffolding**: `pyproject.toml`, `.gitignore`, `LICENSE`, `CONTRIBUTING.md`, GitHub CI workflow, issue templates.

### Changed
- Architecture updated from v4 → v4.1. See `AlphaMoo_v4.1_Delta.md` for the full delta.

### Known issues
- Agent State Tracker detection rate drops on click-only games (s5i5: 47%, vc33: 58%) — movement-correlation doesn't apply when the agent doesn't move. Fix planned for v4.2.
- `m0r0` and `lp85` show high shape-change counts (811, 137) which may indicate tracker instability on certain mechanics. Needs investigation.

## [0.4.0] — 2026-06-28

### Added
- v4 architecture design document.
- v4.1 delta document with all discoveries from play sessions and data mining.
