"""Unit tests for the Phase 0 stub loop."""
import pytest

from alphamoo.agent_loop import LoopStats, run_phase0_loop
from alphamoo.llm_stub import LATENCY_MODELS, StubLLM, build_prompt


class TestLatencyModels:
    def test_all_models_have_required_fields(self):
        for name, model in LATENCY_MODELS.items():
            assert model.name == name
            assert model.prefill_tokens_per_sec > 0
            assert model.decode_tokens_per_sec > 0
            assert model.overhead_ms >= 0

    def test_estimate_is_positive(self):
        for model in LATENCY_MODELS.values():
            t = model.estimate(prompt_tokens=100, output_tokens=50)
            assert t > 0

    def test_05b_is_fastest(self):
        """0.5B should be the fastest model."""
        m_05b = LATENCY_MODELS["qwen2.5-0.5b-4bit"]
        m_15b = LATENCY_MODELS["qwen2.5-1.5b-4bit"]
        m_3b = LATENCY_MODELS["vibethinker-3b-4bit"]
        t_05b = m_05b.estimate(100, 80)
        t_15b = m_15b.estimate(100, 80)
        t_3b = m_3b.estimate(100, 80)
        assert t_05b < t_15b < t_3b


class TestStubLLM:
    def test_generate_returns_valid_response(self):
        llm = StubLLM(latency_model="qwen2.5-0.5b-4bit", max_output_tokens=20, seed=42)
        response = llm.generate(
            prompt="test prompt",
            available_actions=[1, 2, 3, 4],
            agent_position=(10, 10),
            perceived_objects=[],
        )
        assert response.action_id in (1, 2, 3, 4)
        assert response.prompt_tokens > 0
        assert response.output_tokens > 0
        assert response.wall_clock_sec > 0

    def test_generate_picks_movement_when_available(self):
        llm = StubLLM(latency_model="qwen2.5-0.5b-4bit", seed=42)
        response = llm.generate(
            prompt="test",
            available_actions=[1, 2, 3, 4],
        )
        assert response.action_id in (1, 2, 3, 4)

    def test_generate_falls_back_to_click(self):
        llm = StubLLM(latency_model="qwen2.5-0.5b-4bit", seed=42)
        response = llm.generate(
            prompt="test",
            available_actions=[6],
            perceived_objects=[{"id": "obj_001", "color": 1, "n_cells": 10, "bbox": (5, 5, 10, 10), "topology": "solid"}],
        )
        assert response.action_id == 6
        assert response.click_coords is not None

    def test_get_stats_after_calls(self):
        llm = StubLLM(latency_model="qwen2.5-0.5b-4bit", seed=42)
        for _ in range(3):
            llm.generate(prompt="test", available_actions=[1, 2, 3, 4])
        stats = llm.get_stats()
        assert stats["total_calls"] == 3
        assert stats["total_wall_clock_sec"] > 0
        assert stats["avg_wall_clock_ms"] > 0


class TestBuildPrompt:
    def test_prompt_contains_required_sections(self):
        prompt = build_prompt(
            scene_graph_summary={"background_color": 0, "objects": [
                {"id": "obj_001", "color": 1, "n_cells": 10, "bbox": (5, 5, 10, 10), "topology": "solid"}
            ]},
            agent_state={"position": (10, 10), "color": [1], "shape": "abc123", "orientation": 0},
            available_actions=[1, 2, 3, 4],
            action_history=[1, 2, 3],
            levels_completed=0,
            win_levels=7,
            game_id="test_game",
        )
        assert "test_game" in prompt
        assert "Available Actions" in prompt
        assert "Current Scene" in prompt
        assert "Agent State" in prompt
        assert "Action:" in prompt

    def test_prompt_handles_no_objects(self):
        prompt = build_prompt(
            scene_graph_summary={"background_color": 0, "objects": []},
            agent_state=None,
            available_actions=[1, 2, 3, 4],
            action_history=[],
            levels_completed=0,
            win_levels=7,
            game_id="empty",
        )
        assert "no objects detected" in prompt
        assert "agent not yet detected" in prompt


class TestLoopStats:
    def test_finalize_computes_aggregates(self):
        from alphamoo.agent_loop import StepTiming
        stats = LoopStats()
        for i in range(10):
            stats.per_step_timings.append(StepTiming(
                step_index=i,
                total_ms=100.0 + i * 10,  # 100, 110, ..., 190
                perception_ms=1.0,
                cascade_ms=1.0,
                tracker_ms=1.0,
                prompt_build_ms=1.0,
                llm_ms=96.0 + i * 10,  # scales with total_ms
                n_prompt_tokens=200,
                n_output_tokens=80,
            ))
        stats.finalize()
        assert stats.n_steps == 10
        assert stats.avg_step_ms == 145.0  # mean of 100..190
        assert stats.max_step_ms == 190.0
        assert stats.total_prompt_tokens == 2000
        assert stats.total_output_tokens == 800
        # LLM dominates but other stages add up too
        assert 95.0 < stats.avg_llm_pct < 99.0


@pytest.mark.integration
class TestPhase0Loop:
    def test_loop_runs_on_r11l_with_few_steps(self, small_replay_path):
        from alphamoo.vtx_reader import load_replay
        replay = load_replay(small_replay_path)
        llm = StubLLM(latency_model="qwen2.5-0.5b-4bit", max_output_tokens=20, seed=42)
        stats = run_phase0_loop(replay, llm, max_steps=3)
        assert stats.n_steps == 3
        assert stats.total_wall_clock_sec > 0
        assert stats.avg_step_ms > 0
