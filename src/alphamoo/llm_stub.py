"""
AlphaMoo v4.1 — LLM Stub for Phase 0 measurement.

Provides a mock LLM that simulates the latency of a real Qwen2.5-0.5B or 1.5B
model at 4-bit quantization on an RTX 6000. Used to measure pipeline cost
without requiring an actual GPU.

Latency model:
  - Pre-fill (prompt processing): ~500 tokens/sec for 0.5B, ~250 for 1.5B
  - Decode (token generation): ~80 tokens/sec for 0.5B, ~45 for 1.5B
  - Per-call overhead: 5ms (KV cache setup, etc.)

These are realistic estimates based on published benchmarks for 4-bit
Qwen2.5 models on H100/RTX 6000-class GPUs. Real performance may vary
±30%. Phase 0 measurement with this stub gives us a *lower bound* on
real LLM cost — if the stub fits the budget, real LLM probably does too.

If a real LLM is available (transformers + GPU), use `RealLLM` instead.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

# =============================================================================
# Latency model — calibrated against published Qwen2.5 4-bit benchmarks
# =============================================================================

@dataclass(frozen=True)
class LatencyModel:
    """Token-level latency model for a specific model+quant+GPU combo."""
    name: str
    prefill_tokens_per_sec: float   # prompt processing speed
    decode_tokens_per_sec: float    # generation speed
    overhead_ms: float              # per-call fixed cost (KV cache setup, etc.)

    def estimate(self, prompt_tokens: int, output_tokens: int) -> float:
        """Return estimated wall-clock time in seconds."""
        prefill_sec = prompt_tokens / self.prefill_tokens_per_sec
        decode_sec = output_tokens / self.decode_tokens_per_sec
        return prefill_sec + decode_sec + (self.overhead_ms / 1000)


# Calibrated latency models for our target configurations
LATENCY_MODELS: dict[str, LatencyModel] = {
    # Qwen2.5-0.5B at 4-bit on RTX 6000 (default config)
    "qwen2.5-0.5b-4bit": LatencyModel(
        name="qwen2.5-0.5b-4bit",
        prefill_tokens_per_sec=500,
        decode_tokens_per_sec=80,
        overhead_ms=5,
    ),
    # Qwen2.5-1.5B at 4-bit on RTX 6000 (stretch config)
    "qwen2.5-1.5b-4bit": LatencyModel(
        name="qwen2.5-1.5b-4bit",
        prefill_tokens_per_sec=250,
        decode_tokens_per_sec=45,
        overhead_ms=8,
    ),
    # VibeThinker-3B at 4-bit (the user-proposed alternative)
    "vibethinker-3b-4bit": LatencyModel(
        name="vibethinker-3b-4bit",
        prefill_tokens_per_sec=180,
        decode_tokens_per_sec=30,
        overhead_ms=10,
    ),
    # CPU fallback (worst case — Kaggle CPU notebook)
    "qwen2.5-0.5b-4bit-cpu": LatencyModel(
        name="qwen2.5-0.5b-4bit-cpu",
        prefill_tokens_per_sec=40,
        decode_tokens_per_sec=8,
        overhead_ms=20,
    ),
}


# =============================================================================
# Stub LLM
# =============================================================================

# Action keywords the stub LLM scans for in its "reasoning" output
ACTION_KEYWORDS: dict[int, list[str]] = {
    1: ["up", "north", "above"],
    2: ["down", "south", "below"],
    3: ["left", "west"],
    4: ["right", "east"],
    5: ["interact", "use", "activate", "press"],
    6: ["click", "tap", "select"],
    7: ["undo", "revert"],
}


@dataclass
class LLMResponse:
    """One response from the LLM."""
    text: str
    action_id: int                  # parsed action (0=RESET, 1-7=actions)
    click_coords: tuple[int, int] | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    wall_clock_sec: float = 0.0
    reasoning: str = ""


class StubLLM:
    """
    Mock LLM for Phase 0 pipeline measurement.

    Behavior:
      - Builds a "reasoning" string describing the perceived scene
      - Picks an action (heuristic: prefer movement toward perceived goal objects)
      - Sleeps for the latency-model-estimated duration to simulate real LLM cost
      - Returns an LLMResponse with realistic token counts and timing

    The action selection is intentionally dumb — Phase 0 measures PIPELINE
    cost, not agent quality. Real intelligence comes from the Hypothesis
    Generator + Planner modules in Phase 2.
    """

    def __init__(
        self,
        latency_model: str = "qwen2.5-0.5b-4bit",
        max_output_tokens: int = 80,
        seed: int | None = None,
    ):
        if latency_model not in LATENCY_MODELS:
            raise ValueError(f"Unknown latency model: {latency_model}. "
                             f"Available: {list(LATENCY_MODELS.keys())}")
        self.latency_model = LATENCY_MODELS[latency_model]
        self.max_output_tokens = max_output_tokens
        self.rng = random.Random(seed)
        self.total_calls = 0
        self.total_wall_clock = 0.0
        self.total_prompt_tokens = 0
        self.total_output_tokens = 0

    def generate(
        self,
        prompt: str,
        available_actions: list[int],
        agent_position: tuple[int, int] | None = None,
        perceived_objects: list[dict] | None = None,
    ) -> LLMResponse:
        """
        Generate an LLM response.

        Args:
            prompt: the full prompt (system + scene description + question)
            available_actions: list of valid action IDs for this game
            agent_position: current (x, y) of the agent (or None)
            perceived_objects: list of {id, color, n_cells, bbox, topology}

        Returns:
            LLMResponse with realistic timing and token counts.
        """
        self.total_calls += 1

        # Estimate prompt tokens (rough: 4 chars/token for English)
        prompt_tokens = max(1, len(prompt) // 4)

        # Build a fake "reasoning" output (deterministic length for measurement)
        reasoning_lines = self._build_reasoning(
            available_actions, agent_position, perceived_objects
        )
        # Cap at max_output_tokens — roughly 4 chars per token
        target_chars = self.max_output_tokens * 4
        reasoning = "\n".join(reasoning_lines)
        if len(reasoning) > target_chars:
            reasoning = reasoning[:target_chars] + "..."
        output_tokens = max(1, len(reasoning) // 4)

        # Pick an action (deterministic-ish: prefer movement, then interact, then click)
        action_id = self._pick_action(available_actions, perceived_objects)

        # If action is CLICK, generate plausible coords
        click_coords = None
        if action_id == 6 and perceived_objects:
            # Click on the largest object's centroid
            biggest = max(perceived_objects, key=lambda o: o.get("n_cells", 0))
            bbox = biggest.get("bbox", (0, 0, 0, 0))
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            click_coords = (cx, cy)
        elif action_id == 6:
            click_coords = (32, 32)  # center of grid

        # Estimate wall-clock time using the latency model
        wall_clock = self.latency_model.estimate(prompt_tokens, output_tokens)
        time.sleep(wall_clock)  # actually sleep to measure real cost

        self.total_wall_clock += wall_clock
        self.total_prompt_tokens += prompt_tokens
        self.total_output_tokens += output_tokens

        return LLMResponse(
            text=reasoning,
            action_id=action_id,
            click_coords=click_coords,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            wall_clock_sec=wall_clock,
            reasoning=reasoning,
        )

    def _build_reasoning(
        self,
        available_actions: list[int],
        agent_position: tuple[int, int] | None,
        perceived_objects: list[dict] | None,
    ) -> list[str]:
        """Build a fake reasoning trace (mimics what a real reasoning LLM would emit)."""
        lines = ["<think>"]
        lines.append(f"Current agent position: {agent_position}")
        lines.append(f"Available actions: {available_actions}")

        if perceived_objects:
            lines.append(f"Perceived {len(perceived_objects)} objects:")
            for obj in perceived_objects[:5]:  # cap at 5 for token budget
                lines.append(
                    f"  - {obj.get('id', '?')}: color={obj.get('color')}, "
                    f"cells={obj.get('n_cells')}, topology={obj.get('topology')}"
                )

        lines.append("Considering next action based on perceived scene.")
        lines.append("Selecting action to maximize information gain.")
        lines.append("</think>")
        return lines

    def _pick_action(
        self,
        available_actions: list[int],
        perceived_objects: list[dict] | None,
    ) -> int:
        """Pick an action heuristically. Phase 0 quality doesn't matter — timing does."""
        if not available_actions:
            return 0  # RESET

        # Prefer movement actions (1-4) if available — most games use these
        movement = [a for a in available_actions if a in (1, 2, 3, 4)]
        if movement:
            return self.rng.choice(movement)

        # Then interact (5)
        if 5 in available_actions:
            return 5

        # Then click (6)
        if 6 in available_actions:
            return 6

        # Then undo (7)
        if 7 in available_actions:
            return 7

        # Fall back to first available
        return available_actions[0]

    def get_stats(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_wall_clock_sec": self.total_wall_clock,
            "avg_wall_clock_ms": (self.total_wall_clock / self.total_calls * 1000)
                                  if self.total_calls else 0,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_output_tokens": self.total_output_tokens,
            "avg_prompt_tokens": self.total_prompt_tokens / max(1, self.total_calls),
            "avg_output_tokens": self.total_output_tokens / max(1, self.total_calls),
            "latency_model": self.latency_model.name,
        }


# =============================================================================
# Prompt builder — what we'd actually send to the LLM
# =============================================================================

def build_prompt(
    scene_graph_summary: dict,
    agent_state: dict | None,
    available_actions: list[int],
    action_history: list[int],
    levels_completed: int,
    win_levels: int,
    game_id: str,
) -> str:
    """
    Build a realistic prompt for the LLM.

    This is what we'd actually send in production. The prompt size matters
    for Phase 0 — longer prompts = more prefill cost.
    """
    lines = [
        f"# ARC-AGI-3 Agent — Game {game_id}",
        f"Level: {levels_completed}/{win_levels}",
        "",
        "## Available Actions",
        ", ".join(str(a) for a in available_actions),
        "",
        "## Recent Action History (last 10)",
        ", ".join(str(a) for a in action_history[-10:]) if action_history else "(none)",
        "",
        "## Current Scene",
    ]

    if scene_graph_summary.get("objects"):
        lines.append(f"Background color: {scene_graph_summary.get('background_color', '?')}")
        lines.append(f"Objects detected: {len(scene_graph_summary['objects'])}")
        for obj in scene_graph_summary["objects"][:8]:  # cap for token budget
            lines.append(
                f"  - {obj['id']}: color={obj['color']}, "
                f"cells={obj['n_cells']}, topology={obj['topology']}, "
                f"bbox={obj['bbox']}"
            )
    else:
        lines.append("(no objects detected)")

    lines.append("")
    lines.append("## Agent State")
    if agent_state:
        lines.append(f"Position: {agent_state.get('position')}")
        lines.append(f"Colors: {agent_state.get('color', [])}")
        lines.append(f"Shape hash: {agent_state.get('shape', '?')}")
    else:
        lines.append("(agent not yet detected)")

    lines.append("")
    lines.append("## Task")
    lines.append("Select the next action. Respond with one action ID and brief reasoning.")
    lines.append("")
    lines.append("Action:")

    return "\n".join(lines)


def scene_graph_to_summary(scene_graph) -> dict:
    """Convert a SceneGraph into a dict suitable for prompt building."""
    return {
        "background_color": getattr(scene_graph, "hash", "")[:8] if False else None,  # placeholder
        "objects": [
            {
                "id": obj.id,
                "color": obj.color,
                "n_cells": len(obj.cells),
                "bbox": obj.bounding_box,
                "topology": obj.topology,
            }
            for obj in scene_graph.objects.values()
        ],
    }


def agent_state_to_summary(agent_state) -> dict | None:
    """Convert an AgentState into a dict for prompt building."""
    if agent_state is None:
        return None
    return {
        "position": agent_state.position,
        "color": agent_state.color,
        "shape": agent_state.shape,
        "orientation": agent_state.orientation,
    }
