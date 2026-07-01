"""Unit tests for context_compressor.py (Module 12)."""

import pytest

from alphamoo.context_compressor import (
    ContextCompressor,
    LevelSummary,
    SemanticMemory,
)
from alphamoo.schemas import Condition
from alphamoo.world_model import CausalRule


@pytest.fixture
def confirmed_rules():
    return [
        CausalRule(
            trigger_conditions=[
                Condition(predicate="agent_touches", args={"color": 5})
            ],
            trigger_temporal=None,
            effect={"type": "obj_appears", "args": {"color": 3}},
            confidence=0.85,
            support=10,
            mdl_cost=20,
        ),
        CausalRule(
            trigger_conditions=[
                Condition(predicate="agent_at", args={"position": (10, 10)})
            ],
            trigger_temporal=None,
            effect={"type": "episode_ends", "args": {"outcome": "WIN"}},
            confidence=0.95,
            support=5,
            mdl_cost=15,
        ),
    ]


@pytest.fixture
def mock_goal_module():
    """Mock goal module with a top goal."""
    class MockGoalModule:
        def get_top_goal(self):
            from alphamoo.goal_inference import GoalHypothesis, GoalPredicateType
            return GoalHypothesis(
                terminal_condition=GoalPredicateType.AGENT_AT.value,
                args={"position": (10, 10)},
                confidence=0.92,
                support=8,
            )
    return MockGoalModule()


@pytest.fixture
def mock_near_miss_tracker():
    """Mock near-miss tracker with progress data."""
    class MockNearMissTracker:
        def get_progress_summary(self):
            return {
                "distance_to_exit": 5.0,
                "color_match_count": 2.0,
                "enemies_remaining": 0.0,  # no progress
            }
    return MockNearMissTracker()


class TestLevelSummary:
    def test_creates_with_defaults(self):
        summary = LevelSummary(game_id="ls20", level=1)
        assert summary.game_id == "ls20"
        assert summary.level == 1
        assert summary.hash != ""

    def test_hash_is_deterministic(self):
        s1 = LevelSummary(game_id="ls20", level=1, goal_type="reach_exit")
        s2 = LevelSummary(game_id="ls20", level=1, goal_type="reach_exit")
        assert s1.hash == s2.hash

    def test_hash_differs_for_different_content(self):
        s1 = LevelSummary(game_id="ls20", level=1, goal_type="reach_exit")
        s2 = LevelSummary(game_id="ls20", level=1, goal_type="state_matching")
        assert s1.hash != s2.hash

    def test_to_prompt_text_contains_key_info(self):
        summary = LevelSummary(
            game_id="ls20",
            level=3,
            goal_type="state_matching",
            goal_args={"color": 5},
            goal_confidence=0.92,
            n_actions=78,
            n_wins=1,
            n_loses=2,
        )
        text = summary.to_prompt_text()
        assert "ls20" in text
        assert "Level 3" in text
        assert "state_matching" in text
        assert "0.92" in text


class TestSemanticMemory:
    def test_empty_memory_returns_empty_list(self):
        memory = SemanticMemory()
        assert memory.retrieve(k=5) == []

    def test_write_and_retrieve(self):
        memory = SemanticMemory()
        summary = LevelSummary(game_id="ls20", level=1)
        memory.write(summary)
        retrieved = memory.retrieve(k=5)
        assert len(retrieved) == 1

    def test_dedup_by_hash(self):
        memory = SemanticMemory()
        s1 = LevelSummary(game_id="ls20", level=1, goal_type="reach_exit")
        s2 = LevelSummary(game_id="ls20", level=1, goal_type="reach_exit")
        # Same content → same hash → second write should be deduped
        added1 = memory.write(s1)
        added2 = memory.write(s2)
        assert added1
        assert not added2
        assert len(memory.summaries) == 1

    def test_retrieve_by_game_id(self):
        memory = SemanticMemory()
        memory.write(LevelSummary(game_id="ls20", level=1))
        memory.write(LevelSummary(game_id="ft09", level=1))
        memory.write(LevelSummary(game_id="ls20", level=2))

        retrieved = memory.retrieve(game_id="ls20", k=5)
        assert len(retrieved) == 2
        assert all(s.game_id == "ls20" for s in retrieved)

    def test_retrieve_by_goal_type(self):
        memory = SemanticMemory()
        memory.write(LevelSummary(game_id="ls20", level=1, goal_type="reach_exit"))
        memory.write(LevelSummary(game_id="ft09", level=1, goal_type="state_matching"))
        memory.write(LevelSummary(game_id="ls20", level=2, goal_type="reach_exit"))

        retrieved = memory.retrieve(goal_type="reach_exit", k=5)
        assert len(retrieved) == 2
        assert all(s.goal_type == "reach_exit" for s in retrieved)

    def test_game_id_match_returns_only_matching(self):
        """When game_id is specified, only matching summaries are returned."""
        memory = SemanticMemory()
        memory.write(LevelSummary(game_id="ft09", level=1, goal_type="reach_exit"))
        memory.write(LevelSummary(game_id="ls20", level=1, goal_type="reach_exit"))

        retrieved = memory.retrieve(game_id="ls20", k=5)
        # Only ls20 summaries should be returned (filter)
        assert len(retrieved) == 1
        assert retrieved[0].game_id == "ls20"

    def test_retrieve_for_prompt_returns_text(self):
        memory = SemanticMemory()
        memory.write(LevelSummary(
            game_id="ls20", level=1, goal_type="reach_exit",
            goal_confidence=0.9, n_actions=50,
        ))
        text = memory.retrieve_for_prompt(game_id="ls20", k=5)
        assert "ls20" in text
        assert "Level 1" in text

    def test_retrieve_for_prompt_empty_memory(self):
        memory = SemanticMemory()
        text = memory.retrieve_for_prompt(k=5)
        assert "no prior" in text.lower()

    def test_persistence(self, tmp_path):
        path = tmp_path / "memory.json"
        memory = SemanticMemory(path)
        memory.write(LevelSummary(game_id="ls20", level=1))

        # New memory instance loads from same path
        memory2 = SemanticMemory(path)
        assert len(memory2.summaries) == 1
        assert memory2.summaries[0].game_id == "ls20"

    def test_get_stats(self):
        memory = SemanticMemory()
        memory.write(LevelSummary(game_id="ls20", level=1))
        memory.write(LevelSummary(game_id="ls20", level=2))
        memory.write(LevelSummary(game_id="ft09", level=1))
        stats = memory.get_stats()
        assert stats["total_summaries"] == 3
        assert stats["unique_games"] == 2


class TestContextCompressor:
    def test_compress_creates_summary(self, confirmed_rules, mock_goal_module, mock_near_miss_tracker):
        memory = SemanticMemory()
        compressor = ContextCompressor(memory)
        summary = compressor.compress(
            game_id="ls20",
            level=3,
            confirmed_rules=confirmed_rules,
            goal_module=mock_goal_module,
            near_miss_tracker=mock_near_miss_tracker,
            n_actions=78,
            n_wins=1,
            n_loses=2,
        )
        assert summary.game_id == "ls20"
        assert summary.level == 3
        assert len(summary.confirmed_rules) == 2
        assert summary.goal_type == "agent_at"
        assert summary.goal_confidence == 0.92
        assert summary.n_actions == 78

    def test_compress_serializes_rules(self, confirmed_rules, mock_goal_module):
        memory = SemanticMemory()
        compressor = ContextCompressor(memory)
        summary = compressor.compress(
            game_id="ls20",
            level=1,
            confirmed_rules=confirmed_rules,
            goal_module=mock_goal_module,
        )
        # Each rule should be serialized with trigger_desc
        for rule_dict in summary.confirmed_rules:
            assert "trigger_desc" in rule_dict
            assert "effect" in rule_dict
            assert "confidence" in rule_dict

    def test_compress_extracts_near_miss_predicates(self, confirmed_rules, mock_goal_module, mock_near_miss_tracker):
        memory = SemanticMemory()
        compressor = ContextCompressor(memory)
        summary = compressor.compress(
            game_id="ls20",
            level=1,
            confirmed_rules=confirmed_rules,
            goal_module=mock_goal_module,
            near_miss_tracker=mock_near_miss_tracker,
        )
        # distance_to_exit and color_match_count had non-zero progress
        assert "distance_to_exit" in summary.near_miss_predicates
        assert "color_match_count" in summary.near_miss_predicates
        # enemies_remaining had 0 progress → not included
        assert "enemies_remaining" not in summary.near_miss_predicates

    def test_write_to_memory(self, confirmed_rules, mock_goal_module):
        memory = SemanticMemory()
        compressor = ContextCompressor(memory)
        summary = compressor.compress(
            game_id="ls20",
            level=1,
            confirmed_rules=confirmed_rules,
            goal_module=mock_goal_module,
        )
        added = compressor.write(summary)
        assert added
        assert len(memory.summaries) == 1

    def test_retrieve_for_next_level(self, confirmed_rules, mock_goal_module):
        memory = SemanticMemory()
        compressor = ContextCompressor(memory)
        # Write a summary
        summary = compressor.compress(
            game_id="ls20",
            level=1,
            confirmed_rules=confirmed_rules,
            goal_module=mock_goal_module,
        )
        compressor.write(summary)

        # Retrieve for next level
        text = compressor.retrieve_for_next_level(game_id="ls20", k=5)
        assert "ls20" in text
        assert "Level 1" in text

    def test_extract_key_objects(self, confirmed_rules, mock_goal_module):
        memory = SemanticMemory()
        compressor = ContextCompressor(memory)
        summary = compressor.compress(
            game_id="ls20",
            level=1,
            confirmed_rules=confirmed_rules,
            goal_module=mock_goal_module,
        )
        # Rules reference color 5, color 3, and position (10,10)
        assert "color_5" in summary.key_objects
        assert "color_3" in summary.key_objects

    def test_get_stats(self, confirmed_rules, mock_goal_module):
        memory = SemanticMemory()
        compressor = ContextCompressor(memory)
        compressor.compress(
            game_id="ls20", level=1,
            confirmed_rules=confirmed_rules,
            goal_module=mock_goal_module,
        )
        stats = compressor.get_stats()
        assert stats["compression_count"] == 1
        assert "memory_stats" in stats
