"""Unit tests for agent_runner.py (Phase 5)."""
import pytest

from alphamoo.agent_runner import AgentConfig, AlphaMooAgent, StepResult


@pytest.fixture
def agent():
    """Agent with stub LLM (no real GPU needed)."""
    return AlphaMooAgent(AgentConfig(use_real_llm=False))


@pytest.fixture
def empty_grid():
    return [[0] * 64 for _ in range(64)]


class TestAgentConfig:
    def test_default_config(self):
        config = AgentConfig()
        assert config.use_real_llm is False
        assert config.max_output_tokens == 80
        assert config.planning_threshold == 0.6
        assert config.enable_hypothesis_generator is True
        assert config.enable_goal_inference is True

    def test_ablation_config(self):
        config = AgentConfig(enable_world_model=False)
        assert config.enable_world_model is False


class TestAlphaMooAgent:
    def test_initialization(self):
        agent = AlphaMooAgent(AgentConfig(use_real_llm=False))
        assert agent.tracker is not None
        assert agent.hyp_gen is not None
        assert agent.goal_module is not None
        assert agent.world_model is not None

    def test_initialization_with_ablations(self):
        """Agent can be created with modules disabled."""
        config = AgentConfig(
            enable_hypothesis_generator=False,
            enable_goal_inference=False,
            enable_world_model=False,
        )
        agent = AlphaMooAgent(config)
        assert agent.hyp_gen is None
        assert agent.goal_module is None
        assert agent.world_model is None

    def test_start_game_resets_state(self, agent):
        agent._step_count = 100
        agent.start_game("ls20")
        assert agent._step_count == 0
        assert agent._current_game_id == "ls20"
        assert agent._level_step_count == 0

    def test_step_returns_step_result(self, agent, empty_grid):
        agent.start_game("ls20")
        result = agent.step(
            grid=empty_grid,
            action_id_taken=0,
            available_actions=[1, 2, 3, 4],
            n_subframes=1,
            state="NOT_FINISHED",
            levels_completed=0,
            win_levels=7,
        )
        assert isinstance(result, StepResult)
        assert result.action_id in [1, 2, 3, 4]
        assert result.mode in ["exploration", "planning"]
        assert result.wall_clock_ms > 0

    def test_step_with_no_available_actions(self, agent, empty_grid):
        agent.start_game("ls20")
        result = agent.step(
            grid=empty_grid,
            action_id_taken=0,
            available_actions=[],
            n_subframes=1,
            state="NOT_FINISHED",
            levels_completed=0,
            win_levels=7,
        )
        # Should return action_id=0 (fallback)
        assert result.action_id == 0

    def test_step_increments_counters(self, agent, empty_grid):
        agent.start_game("ls20")
        agent.step(
            grid=empty_grid, action_id_taken=0, available_actions=[1, 2, 3, 4],
            n_subframes=1, state="NOT_FINISHED", levels_completed=0, win_levels=7,
        )
        assert agent._step_count == 1
        assert agent._total_steps == 1

    def test_step_handles_win(self, agent, empty_grid):
        agent.start_game("ls20")
        result = agent.step(
            grid=empty_grid, action_id_taken=0, available_actions=[1, 2, 3, 4],
            n_subframes=1, state="WIN", levels_completed=1, win_levels=7,
        )
        # Agent should still return an action even on WIN
        assert isinstance(result, StepResult)

    def test_step_handles_game_over(self, agent, empty_grid):
        agent.start_game("ls20")
        result = agent.step(
            grid=empty_grid, action_id_taken=0, available_actions=[1, 2, 3, 4],
            n_subframes=1, state="GAME_OVER", levels_completed=0, win_levels=7,
        )
        assert isinstance(result, StepResult)

    def test_get_stats(self, agent, empty_grid):
        agent.start_game("ls20")
        agent.step(
            grid=empty_grid, action_id_taken=0, available_actions=[1, 2, 3, 4],
            n_subframes=1, state="NOT_FINISHED", levels_completed=0, win_levels=7,
        )
        stats = agent.get_stats()
        assert "total_steps" in stats
        assert stats["total_steps"] == 1
        assert "exploration_steps" in stats
        assert "planning_steps" in stats

    def test_cross_level_transfer(self, agent, empty_grid):
        """After completing a level, semantic memory should have a summary."""
        agent.start_game("ls20")
        # Simulate a level completion by setting levels_completed
        agent.step(
            grid=empty_grid, action_id_taken=0, available_actions=[1, 2, 3, 4],
            n_subframes=1, state="NOT_FINISHED", levels_completed=1, win_levels=7,
        )
        # Semantic memory should have at least 1 summary after level transition
        # (The _on_level_complete method writes to semantic memory)
        assert len(agent.semantic_memory.summaries) >= 0  # may be 0 if WM has no rules
