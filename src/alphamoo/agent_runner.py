"""
AlphaMoo v4.1 — Full Agent Runner (Phase 5).

The complete agent that runs on Kaggle. Wires together all modules:

  Perception → Tracker → Cascade → Hypothesis → Goal → NearMiss →
  [Explore: ExperimentPlanner] or [Exploit: Planner Interface] →
  WorldModel → Verifier → Context Compressor → Semantic Memory

Usage on Kaggle:
    from alphamoo.agent_runner import AlphaMooAgent
    agent = AlphaMooAgent(use_real_llm=True, model_name="Qwen/Qwen2.5-0.5B-Instruct-AWQ")
    agent.run_game(env, game_id="ls20")

The agent handles:
  - Level transitions (compress context, retrieve ICL)
  - Win/Lose detection (update goal inference)
  - Exploration vs exploitation switching
  - Cross-level transfer via semantic memory
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .agent_tracker import AgentStateTracker
from .cascade_interpreter import classify_event, diff_grids, interpret_cascade
from .context_compressor import ContextCompressor, SemanticMemory
from .experiment_planner import ExperimentPlanner
from .goal_inference import GoalInferenceModule
from .hypothesis_generator import HypothesisGenerator
from .llm_stub import StubLLM
from .near_miss_tracker import NearMissTracker
from .perception import detect_background_color, perceive
from .planner_interface import plan_action
from .schemas import (
    GameState,
)
from .verifier import Verifier
from .world_model import WorldModel

# =============================================================================
# Agent configuration
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for the AlphaMoo agent."""
    # LLM
    use_real_llm: bool = False
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
    backend: str = "vllm"
    latency_model: str = "qwen2.5-0.5b-4bit"
    max_output_tokens: int = 80

    # Module thresholds
    hypothesis_confidence_threshold: float = 0.7
    planning_threshold: float = 0.6
    ig_floor: float = 0.05
    epsilon: float = 0.1

    # Budget
    budget_aware: bool = True
    budget_safety_margin: float = 0.4  # don't use more than 60% of remaining

    # Semantic memory
    semantic_memory_path: Path | None = None
    icl_top_k: int = 5

    # Ablation flags (Phase 5)
    enable_hypothesis_generator: bool = True
    enable_goal_inference: bool = True
    enable_near_miss_tracker: bool = True
    enable_world_model: bool = True
    enable_verifier: bool = True
    enable_context_compressor: bool = True
    enable_planner_interface: bool = True
    enable_experiment_planner: bool = True


# =============================================================================
# Agent step result
# =============================================================================

@dataclass
class StepResult:
    """Result of one agent step."""
    action_id: int
    click_coords: tuple[int, int] | None = None
    mode: str = "exploration"  # "exploration" or "planning"
    planner_name: str = ""
    n_objects_perceived: int = 0
    n_hypotheses: int = 0
    n_confirmed_hypotheses: int = 0
    goal_confidence: float = 0.0
    world_model_rules: int = 0
    verifier_match: bool | None = None
    wall_clock_ms: float = 0.0
    notes: str = ""


# =============================================================================
# Full Agent
# =============================================================================

class AlphaMooAgent:
    """
    The complete AlphaMoo agent.

    Usage:
        agent = AlphaMooAgent(AgentConfig(use_real_llm=True))
        # Run against the ARC-AGI-3 environment:
        for game_id in games:
            agent.start_game(game_id)
            while not done:
                obs = env.get_observation()
                action = agent.step(obs)
                env.execute(action)
            agent.end_game()
    """

    def __init__(self, config: AgentConfig = AgentConfig()):
        self.config = config

        # Core modules
        self.tracker = AgentStateTracker()
        self.hyp_gen = HypothesisGenerator() if config.enable_hypothesis_generator else None
        self.goal_module = GoalInferenceModule(
            planning_threshold=config.planning_threshold
        ) if config.enable_goal_inference else None
        self.near_miss = NearMissTracker() if config.enable_near_miss_tracker else None
        self.exploration_planner = ExperimentPlanner(
            ig_floor=config.ig_floor,
            epsilon=config.epsilon,
            rng_seed=42,
        ) if config.enable_experiment_planner else None
        self.world_model = WorldModel(
            confidence_threshold=config.hypothesis_confidence_threshold
        ) if config.enable_world_model else None
        self.verifier = Verifier(self.world_model) if (
            config.enable_verifier and self.world_model is not None
        ) else None

        # Semantic memory + context compressor
        self.semantic_memory = SemanticMemory(config.semantic_memory_path)
        self.context_compressor = ContextCompressor(self.semantic_memory) if (
            config.enable_context_compressor
        ) else None

        # LLM (lazy-loaded)
        self._llm = None

        # State
        self._current_game_id: str | None = None
        self._current_level: int = 0
        self._step_count: int = 0
        self._level_step_count: int = 0
        self._level_action_count: int = 0
        self._level_wins: int = 0
        self._level_loses: int = 0
        self._prev_grid: np.ndarray | None = None
        self._prev_record = None
        self._last_prediction = None
        self._icl_context: str = ""

        # Stats
        self._total_steps: int = 0
        self._exploration_steps: int = 0
        self._planning_steps: int = 0
        self._planning_successes: int = 0
        self._planner_types_used: dict[str, int] = {}

    def _get_llm(self):
        """Lazy-load the LLM."""
        if self._llm is None:
            if self.config.use_real_llm:
                from .llm_real import RealLLM
                self._llm = RealLLM(
                    model_name=self.config.model_name,
                    backend=self.config.backend,
                    max_output_tokens=self.config.max_output_tokens,
                )
            else:
                self._llm = StubLLM(
                    latency_model=self.config.latency_model,
                    max_output_tokens=self.config.max_output_tokens,
                    seed=42,
                )
        return self._llm

    def start_game(self, game_id: str) -> None:
        """Called when starting a new game."""
        self._current_game_id = game_id
        self._current_level = 0
        self._step_count = 0
        self._level_step_count = 0
        self._level_action_count = 0
        self._level_wins = 0
        self._level_loses = 0
        self._prev_grid = None
        self._prev_record = None
        self._last_prediction = None

        # Reset modules for new game
        self.tracker = AgentStateTracker()
        if self.hyp_gen:
            self.hyp_gen = HypothesisGenerator()
        if self.goal_module:
            self.goal_module = GoalInferenceModule(
                planning_threshold=self.config.planning_threshold
            )
        if self.near_miss:
            self.near_miss = NearMissTracker()
        if self.world_model:
            self.world_model = WorldModel(
                confidence_threshold=self.config.hypothesis_confidence_threshold
            )
        if self.verifier:
            self.verifier = Verifier(self.world_model)

        # Retrieve ICL context from semantic memory
        if self.context_compressor:
            self._icl_context = self.context_compressor.retrieve_for_next_level(
                game_id=game_id,
                k=self.config.icl_top_k,
            )

    def step(
        self,
        grid: list[list[int]],
        action_id_taken: int,
        available_actions: list[int],
        n_subframes: int = 1,
        subframes: list[list[list[int]]] | None = None,
        state: str = "NOT_FINISHED",
        levels_completed: int = 0,
        win_levels: int = 1,
    ) -> StepResult:
        """
        Process one observation and choose the next action.

        Args:
            grid: current 64x64 grid (final frame)
            action_id_taken: the action that produced this grid
            available_actions: list of valid action IDs
            n_subframes: number of subframes in this observation
            subframes: the full [N][64][64] subframe list (if n_subframes > 1)
            state: NOT_FINISHED / WIN / GAME_OVER
            levels_completed: current level count
            win_levels: total levels to win

        Returns:
            StepResult with the chosen next action.
        """
        t0 = time.perf_counter()
        self._step_count += 1
        self._level_step_count += 1
        self._total_steps += 1

        grid_np = np.array(grid, dtype=np.int8)
        bg = detect_background_color(grid_np)
        scene = perceive(grid, background_color=bg)
        agent_state, _ = self.tracker.update(
            grid_np, action_id_taken,
            background_color=bg,
            available_actions=available_actions,
            action_input=None,  # would need full ActionInput
        )

        # Detect events
        events = []
        if n_subframes > 1 and subframes:
            with __import__("contextlib").suppress(Exception):
                _, events = interpret_cascade(subframes)
        elif self._prev_grid is not None:
            diff = diff_grids(self._prev_grid, grid_np)
            events = classify_event(diff, None, None, timestep=0)

        # 1. Update world model from confirmed hypotheses
        if self.world_model and self.hyp_gen:
            confirmed = [h for h in self.hyp_gen.hypotheses
                         if h.confidence > self.config.hypothesis_confidence_threshold]
            self.world_model.update_from_hypotheses(confirmed)

        # 2. Verify previous prediction
        if self.verifier and self._last_prediction is not None:
            self.verifier.verify(
                self._last_prediction, scene, events, agent_state,
            )
            self._last_prediction = None

        # 3. Update hypothesis generator
        if self.hyp_gen:
            self.hyp_gen.observe(scene, agent_state, events, action_id_taken)

        # 4. Update goal inference
        if self.goal_module:
            self.goal_module.observe_step(scene, agent_state, action_id_taken)

            # Check for level transition
            if self._prev_record and levels_completed > (self._prev_record if isinstance(self._prev_record, int) else 0):
                self.goal_module.observe_win(scene, agent_state)
                self._level_wins += 1
                self._on_level_complete(levels_completed)
            elif state == GameState.WIN:
                self.goal_module.observe_win(scene, agent_state)
                self._level_wins += 1
            elif state == GameState.GAME_OVER:
                if self.near_miss:
                    near_miss_preds = self.near_miss.on_episode_end("GAME_OVER")
                    if near_miss_preds:
                        self.goal_module.observe_lose(scene, agent_state, near_miss_preds)
                self._level_loses += 1
            else:
                if self.near_miss:
                    self.near_miss.record_step(scene, agent_state)

        # 5. Decide: explore or exploit?
        mode = "exploration"
        planner_name = ""
        chosen_action_id = available_actions[0] if available_actions else 0
        chosen_click_coords = None

        ready_to_plan = (
            self.goal_module
            and self.goal_module.is_ready_to_plan()
            and self.world_model
            and len(self.world_model.rules) > 0
            and self.config.enable_planner_interface
        )

        if ready_to_plan:
            # Try to plan
            planning_result = plan_action(
                scene=scene,
                agent_state=agent_state,
                goal_module=self.goal_module,
                world_model=self.world_model,
                budget_remaining=1000,  # would be actual budget in production
                available_actions=available_actions,
            )
            self._planning_steps += 1
            planner_name = planning_result.planner_name
            self._planner_types_used[planner_name] = (
                self._planner_types_used.get(planner_name, 0) + 1
            )

            if planning_result.plan is not None:
                self._planning_successes += 1
                mode = "planning"
                chosen_action_id = planning_result.plan.actions[0]
                if planning_result.plan.click_coords:
                    chosen_click_coords = planning_result.plan.click_coords[0]
            else:
                # Plan failed — fall back to exploration
                mode = "exploration"

        if mode == "exploration" and self.exploration_planner and self.hyp_gen:
            top_hyps = self.hyp_gen.get_top_hypotheses(k=10)
            chosen = self.exploration_planner.select_action(
                available_actions=available_actions,
                scene=scene,
                agent_state=agent_state,
                hypotheses=top_hyps,
            )
            chosen_action_id = chosen.action_id
            chosen_click_coords = chosen.click_coords
            self._exploration_steps += 1
        elif mode == "exploration":
            # No exploration planner — just pick first available
            self._exploration_steps += 1

        # 6. Predict outcome for verification
        if self.world_model:
            self._last_prediction = self.world_model.predict(
                scene, agent_state, chosen_action_id
            )

        # 7. Build the result
        wall_clock_ms = (time.perf_counter() - t0) * 1000
        result = StepResult(
            action_id=chosen_action_id,
            click_coords=chosen_click_coords,
            mode=mode,
            planner_name=planner_name,
            n_objects_perceived=len(scene.objects),
            n_hypotheses=len(self.hyp_gen.hypotheses) if self.hyp_gen else 0,
            n_confirmed_hypotheses=len(self.world_model.rules) if self.world_model else 0,
            goal_confidence=(
                self.goal_module.get_top_goal().confidence
                if self.goal_module and self.goal_module.get_top_goal()
                else 0.0
            ),
            world_model_rules=len(self.world_model.rules) if self.world_model else 0,
            wall_clock_ms=wall_clock_ms,
        )

        # Update history
        self._prev_grid = grid_np
        self._prev_record = levels_completed
        self._level_action_count += 1

        return result

    def _on_level_complete(self, new_level: int) -> None:
        """Called when a level is completed. Compresses context."""
        if self.context_compressor:
            summary = self.context_compressor.compress(
                game_id=self._current_game_id or "unknown",
                level=self._current_level,
                confirmed_rules=self.world_model.rules if self.world_model else [],
                goal_module=self.goal_module,
                near_miss_tracker=self.near_miss,
                n_actions=self._level_action_count,
                n_wins=self._level_wins,
                n_loses=self._level_loses,
                action_efficiency=0.0,  # would compute from human baseline
            )
            self.context_compressor.write(summary)

        # Reset for next level
        self._current_level = new_level
        self._level_step_count = 0
        self._level_action_count = 0
        if self.near_miss:
            self.near_miss.reset()

        # Retrieve ICL for new level
        if self.context_compressor and self.goal_module:
            top_goal = self.goal_module.get_top_goal()
            goal_type = top_goal.terminal_condition if top_goal else None
            self._icl_context = self.context_compressor.retrieve_for_next_level(
                game_id=self._current_game_id or "",
                goal_type=goal_type,
                k=self.config.icl_top_k,
            )

    def end_game(self) -> None:
        """Called when the game ends (WIN or GAME_OVER)."""
        # Compress final level
        if self.context_compressor and self._current_level > 0:
            summary = self.context_compressor.compress(
                game_id=self._current_game_id or "unknown",
                level=self._current_level,
                confirmed_rules=self.world_model.rules if self.world_model else [],
                goal_module=self.goal_module,
                near_miss_tracker=self.near_miss,
                n_actions=self._level_action_count,
                n_wins=self._level_wins,
                n_loses=self._level_loses,
            )
            self.context_compressor.write(summary)

    def get_stats(self) -> dict:
        return {
            "total_steps": self._total_steps,
            "exploration_steps": self._exploration_steps,
            "planning_steps": self._planning_steps,
            "planning_successes": self._planning_successes,
            "planning_success_rate": (
                self._planning_successes / max(1, self._planning_steps)
            ),
            "planner_types_used": self._planner_types_used,
            "hypothesis_stats": self.hyp_gen.get_stats() if self.hyp_gen else {},
            "goal_stats": self.goal_module.get_stats() if self.goal_module else {},
            "world_model_stats": self.world_model.get_stats() if self.world_model else {},
            "verifier_stats": self.verifier.get_stats() if self.verifier else {},
            "semantic_memory_stats": self.semantic_memory.get_stats(),
            "llm_stats": self._llm.get_stats() if self._llm else {},
        }
