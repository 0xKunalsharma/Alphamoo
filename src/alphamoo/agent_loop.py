"""
AlphaMoo v4.1 — Agent Loop (Phase 0 stub version).

Wires together Perception + CascadeInterpreter + AgentStateTracker + LLM
into a real agent loop. For Phase 0 measurement, uses a replay-driven
simulator (we walk through a ground-truth replay frame-by-frame, measuring
pipeline cost at each step).

In production (Phase 5+) this same loop would connect to the real ARC-AGI-3
engine via the `arc-agi` package. The interface is the same; only the
"environment" changes.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .agent_tracker import AgentStateTracker
from .cascade_interpreter import interpret_cascade
from .llm_stub import (
    StubLLM,
    agent_state_to_summary,
    build_prompt,
    scene_graph_to_summary,
)
from .perception import detect_background_color, perceive
from .schemas import GameState
from .vtx_reader import Replay

# =============================================================================
# Step-level timing
# =============================================================================

@dataclass
class StepTiming:
    """Per-step timing breakdown."""
    step_index: int = 0
    perception_ms: float = 0.0
    cascade_ms: float = 0.0
    tracker_ms: float = 0.0
    prompt_build_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0
    n_subframes: int = 1
    n_objects: int = 0
    n_prompt_tokens: int = 0
    n_output_tokens: int = 0
    action_id: int = 0
    state_after: str = ""
    levels_completed: int = 0


@dataclass
class LoopStats:
    """Aggregate loop statistics."""
    n_steps: int = 0
    total_wall_clock_sec: float = 0.0
    avg_step_ms: float = 0.0
    p50_step_ms: float = 0.0
    p95_step_ms: float = 0.0
    p99_step_ms: float = 0.0
    max_step_ms: float = 0.0
    total_perception_ms: float = 0.0
    total_cascade_ms: float = 0.0
    total_tracker_ms: float = 0.0
    total_prompt_build_ms: float = 0.0
    total_llm_ms: float = 0.0
    avg_perception_pct: float = 0.0
    avg_cascade_pct: float = 0.0
    avg_tracker_pct: float = 0.0
    avg_prompt_build_pct: float = 0.0
    avg_llm_pct: float = 0.0
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    avg_prompt_tokens: int = 0
    avg_output_tokens: int = 0
    n_level_completions: int = 0
    n_terminal_states: int = 0
    per_step_timings: list[StepTiming] = field(default_factory=list)

    def finalize(self):
        """Compute aggregate stats from per-step timings."""
        if not self.per_step_timings:
            return
        self.n_steps = len(self.per_step_timings)
        self.total_wall_clock_sec = sum(t.total_ms for t in self.per_step_timings) / 1000
        self.avg_step_ms = self.total_wall_clock_sec * 1000 / self.n_steps

        totals = [t.total_ms for t in self.per_step_timings]
        sorted_totals = sorted(totals)
        n = len(sorted_totals)
        self.p50_step_ms = sorted_totals[n // 2]
        self.p95_step_ms = sorted_totals[int(n * 0.95)]
        self.p99_step_ms = sorted_totals[min(int(n * 0.99), n - 1)]
        self.max_step_ms = sorted_totals[-1]

        self.total_perception_ms = sum(t.perception_ms for t in self.per_step_timings)
        self.total_cascade_ms = sum(t.cascade_ms for t in self.per_step_timings)
        self.total_tracker_ms = sum(t.tracker_ms for t in self.per_step_timings)
        self.total_prompt_build_ms = sum(t.prompt_build_ms for t in self.per_step_timings)
        self.total_llm_ms = sum(t.llm_ms for t in self.per_step_timings)

        total_ms = sum(t.total_ms for t in self.per_step_timings)
        if total_ms > 0:
            self.avg_perception_pct = self.total_perception_ms / total_ms * 100
            self.avg_cascade_pct = self.total_cascade_ms / total_ms * 100
            self.avg_tracker_pct = self.total_tracker_ms / total_ms * 100
            self.avg_prompt_build_pct = self.total_prompt_build_ms / total_ms * 100
            self.avg_llm_pct = self.total_llm_ms / total_ms * 100

        self.total_prompt_tokens = sum(t.n_prompt_tokens for t in self.per_step_timings)
        self.total_output_tokens = sum(t.n_output_tokens for t in self.per_step_timings)
        self.avg_prompt_tokens = self.total_prompt_tokens / self.n_steps
        self.avg_output_tokens = self.total_output_tokens / self.n_steps


# =============================================================================
# Replay-driven simulator (Phase 0 environment)
# =============================================================================

class ReplaySimulator:
    """
    Phase 0 "environment" — walks through a ground-truth replay frame by frame.

    The agent sees the same frames the human saw, in order. We measure
    pipeline cost (perception + cascade + tracker + LLM). The agent's
    actions don't actually affect which frame comes next — we just step
    through the replay.

    This is sufficient for Phase 0 because we're measuring PIPELINE cost,
    not agent quality.
    """

    def __init__(self, replay: Replay):
        self.replay = replay
        self.step_idx = 0
        self.prev_record = None

    def reset(self) -> tuple:
        """Return the initial frame."""
        self.step_idx = 0
        self.prev_record = None
        record = self.replay.records[0]
        return record, self.replay.available_actions

    def step(self) -> tuple:
        """Advance one record. Returns (record, available_actions, done)."""
        self.step_idx += 1
        if self.step_idx >= len(self.replay.records):
            return None, self.replay.available_actions, True
        self.prev_record = self.replay.records[self.step_idx - 1]
        record = self.replay.records[self.step_idx]
        done = record.state in (GameState.WIN, GameState.GAME_OVER)
        return record, self.replay.available_actions, done


# =============================================================================
# Main agent loop
# =============================================================================

def run_phase0_loop(
    replay: Replay,
    llm: StubLLM,
    max_steps: int | None = None,
    verbose: bool = False,
) -> LoopStats:
    """
    Run the Phase 0 agent loop over a single replay.

    Args:
        replay: loaded Replay object
        llm: StubLLM instance (or real LLM with same interface)
        max_steps: cap on steps (for testing). None = run full replay.
        verbose: print per-step info

    Returns:
        LoopStats with timing breakdown.
    """
    sim = ReplaySimulator(replay)
    tracker = AgentStateTracker()
    stats = LoopStats()
    action_history: list[int] = []

    # Initial reset
    record, available_actions = sim.reset()

    step_count = 0
    while True:
        if max_steps is not None and step_count >= max_steps:
            break

        timing = StepTiming(step_index=step_count)
        step_start = time.perf_counter()

        # --- 1. Perception ---
        t0 = time.perf_counter()
        grid = np.array(record.final_grid, dtype=np.int8)
        bg_color = detect_background_color(grid)
        scene = perceive(grid.tolist(), background_color=bg_color)
        timing.perception_ms = (time.perf_counter() - t0) * 1000
        timing.n_objects = len(scene.objects)

        # --- 2. Cascade interpretation (only if N > 1) ---
        t0 = time.perf_counter()
        if record.n_subframes > 1:
            with contextlib.suppress(Exception):
                _, events = interpret_cascade(record.frame)
                # In production we'd update the hypothesis generator with events.
                # For Phase 0 we just measure the cost.
        timing.cascade_ms = (time.perf_counter() - t0) * 1000
        timing.n_subframes = record.n_subframes

        # --- 3. Agent state tracking ---
        t0 = time.perf_counter()
        agent_state, tracker_diag = tracker.update(
            grid,
            record.action_input.id,
            background_color=bg_color,
            available_actions=available_actions,
            action_input=record.action_input,
        )
        timing.tracker_ms = (time.perf_counter() - t0) * 1000

        # --- 4. Prompt building ---
        t0 = time.perf_counter()
        scene_summary = scene_graph_to_summary(scene)
        scene_summary["background_color"] = bg_color
        agent_summary = agent_state_to_summary(agent_state)
        prompt = build_prompt(
            scene_graph_summary=scene_summary,
            agent_state=agent_summary,
            available_actions=available_actions,
            action_history=action_history,
            levels_completed=record.levels_completed,
            win_levels=record.win_levels,
            game_id=replay.game_id,
        )
        timing.prompt_build_ms = (time.perf_counter() - t0) * 1000

        # --- 5. LLM call ---
        t0 = time.perf_counter()
        # Convert scene objects to the dict format the LLM expects
        perceived_objects = [
            {
                "id": obj.id,
                "color": obj.color,
                "n_cells": len(obj.cells),
                "bbox": obj.bounding_box,
                "topology": obj.topology,
            }
            for obj in scene.objects.values()
        ]
        llm_response = llm.generate(
            prompt=prompt,
            available_actions=available_actions,
            agent_position=agent_state.position if agent_state else None,
            perceived_objects=perceived_objects,
        )
        timing.llm_ms = (time.perf_counter() - t0) * 1000
        timing.n_prompt_tokens = llm_response.prompt_tokens
        timing.n_output_tokens = llm_response.output_tokens
        timing.action_id = llm_response.action_id

        # --- 6. Total step time ---
        timing.total_ms = (time.perf_counter() - step_start) * 1000
        timing.state_after = record.state
        timing.levels_completed = record.levels_completed
        stats.per_step_timings.append(timing)

        action_history.append(llm_response.action_id)

        if verbose and step_count % 50 == 0:
            print(f"  [step {step_count}] total={timing.total_ms:.1f}ms "
                  f"perc={timing.perception_ms:.1f}ms "
                  f"cascade={timing.cascade_ms:.1f}ms "
                  f"tracker={timing.tracker_ms:.1f}ms "
                  f"prompt={timing.prompt_build_ms:.1f}ms "
                  f"llm={timing.llm_ms:.1f}ms "
                  f"objs={timing.n_objects}")

        # --- Advance to next record ---
        record, available_actions, done = sim.step()
        if done or record is None:
            break

        step_count += 1

    stats.finalize()
    return stats


# =============================================================================
# Convenience: run on a specific game
# =============================================================================

def run_phase0_on_game(
    game_id: str,
    data_dir: str | Path = "/home/z/my-project/alphamoo/data",
    latency_model: str = "qwen2.5-0.5b-4bit",
    max_output_tokens: int = 80,
    max_steps: int | None = None,
    verbose: bool = False,
) -> tuple[LoopStats, dict]:
    """Run Phase 0 on a single game. Returns (loop_stats, llm_stats)."""
    # Load replay
    from .vtx_reader import load_replays_from_dir
    replays = load_replays_from_dir(data_dir)
    if game_id not in replays:
        raise ValueError(f"Game {game_id} not found. Available: {sorted(replays.keys())}")
    replay = replays[game_id]

    # Initialize LLM
    llm = StubLLM(
        latency_model=latency_model,
        max_output_tokens=max_output_tokens,
        seed=42,
    )

    # Run loop
    if verbose:
        print(f"\n=== Phase 0 on {game_id} ({replay.n_actions} actions, "
              f"latency_model={latency_model}) ===")
    stats = run_phase0_loop(replay, llm, max_steps=max_steps, verbose=verbose)

    return stats, llm.get_stats()
