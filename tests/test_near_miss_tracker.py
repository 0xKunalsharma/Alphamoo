"""Unit tests for near_miss_tracker.py (Module 7)."""
import pytest

from alphamoo.near_miss_tracker import (
    NearMissTracker,
    ProgressDirection,
    ProgressPredicate,
    compute_color_match_count,
    compute_distance_to_exit,
    compute_enemies_remaining,
    compute_objects_collected,
    compute_rooms_explored,
    compute_shape_match_count,
)
from alphamoo.schemas import AgentState, GameObject, SceneGraph


@pytest.fixture
def scene_with_one_object():
    obj = GameObject(
        id="obj_001", color=2, secondary_colors=[],
        cells=[(30, 30), (30, 31), (31, 30), (31, 31)],
        bounding_box=(30, 30, 31, 31),
        topology="solid", shape_hash="abc", is_agent=False,
    )
    return SceneGraph(objects={"obj_001": obj}, edges=set())


@pytest.fixture
def agent_at_origin():
    return AgentState(
        object_ids=[], position=(5, 5), orientation=0,
        shape="abc", color=[2], energy=None, inventory=[],
    )


class TestProgressPredicate:
    def test_minimize_near_miss(self):
        """Distance decreasing = near-miss for minimize predicate."""
        pred = ProgressPredicate("distance", ProgressDirection.MINIMIZE)
        pred.record(10.0)
        pred.record(8.0)
        pred.record(5.0)
        assert pred.is_near_miss()

    def test_maximize_near_miss(self):
        """Count increasing = near-miss for maximize predicate."""
        pred = ProgressPredicate("count", ProgressDirection.MAXIMIZE)
        pred.record(1.0)
        pred.record(2.0)
        pred.record(3.0)
        assert pred.is_near_miss()

    def test_no_near_miss_with_too_few_observations(self):
        pred = ProgressPredicate("x", ProgressDirection.MINIMIZE)
        pred.record(10.0)
        pred.record(8.0)
        assert not pred.is_near_miss()

    def test_no_near_miss_when_no_progress(self):
        """Distance increasing = not a near-miss for minimize predicate."""
        pred = ProgressPredicate("distance", ProgressDirection.MINIMIZE)
        pred.record(5.0)
        pred.record(8.0)
        pred.record(10.0)
        assert not pred.is_near_miss()

    def test_total_progress_minimize(self):
        pred = ProgressPredicate("distance", ProgressDirection.MINIMIZE)
        pred.record(10.0)
        pred.record(5.0)
        # Total progress = 10 - 5 = 5 (positive = good)
        assert pred.total_progress() == 5.0

    def test_total_progress_maximize(self):
        pred = ProgressPredicate("count", ProgressDirection.MAXIMIZE)
        pred.record(2.0)
        pred.record(7.0)
        # Total progress = 7 - 2 = 5
        assert pred.total_progress() == 5.0

    def test_reset(self):
        pred = ProgressPredicate("x", ProgressDirection.MINIMIZE)
        pred.record(10.0)
        pred.record(8.0)
        pred.reset()
        assert len(pred.trajectory) == 0


class TestProgressComputations:
    def test_distance_to_exit_with_agent(self, scene_with_one_object, agent_at_origin):
        # Agent at (5,5), object at (30-31, 30-31)
        # Distance to nearest cell: |30-5| + |30-5| = 50
        dist = compute_distance_to_exit(scene_with_one_object, agent_at_origin)
        assert dist == 50.0

    def test_distance_to_exit_no_agent(self, scene_with_one_object):
        dist = compute_distance_to_exit(scene_with_one_object, None)
        assert dist > 0  # some default large value

    def test_distance_to_exit_no_objects(self, agent_at_origin):
        empty_scene = SceneGraph(objects={}, edges=set())
        dist = compute_distance_to_exit(empty_scene, agent_at_origin)
        assert dist > 0

    def test_color_match_count_with_match(self, scene_with_one_object, agent_at_origin):
        # Agent color is 2, object color is 2 → match
        count = compute_color_match_count(scene_with_one_object, agent_at_origin)
        assert count == 1.0

    def test_color_match_count_no_match(self, scene_with_one_object):
        agent = AgentState(
            object_ids=[], position=(5, 5), orientation=0,
            shape="x", color=[99], energy=None, inventory=[],
        )
        count = compute_color_match_count(scene_with_one_object, agent)
        assert count == 0.0

    def test_shape_match_count(self, scene_with_one_object, agent_at_origin):
        # Agent shape="abc", object shape_hash="abc" → match
        count = compute_shape_match_count(scene_with_one_object, agent_at_origin)
        assert count == 1.0

    def test_objects_collected(self, agent_at_origin):
        # Initial: 5 objects, current: 1 object → 4 collected
        current_scene = SceneGraph(
            objects={"obj_001": GameObject(
                id="obj_001", color=2, secondary_colors=[],
                cells=[(10, 10)], bounding_box=(10, 10, 10, 10),
                topology="solid", shape_hash="x", is_agent=False,
            )},
            edges=set(),
        )
        count = compute_objects_collected(current_scene, initial_object_count=5)
        assert count == 4.0

    def test_enemies_remaining(self, scene_with_one_object, agent_at_origin):
        # 1 object with color 2 (agent color), so it's not an enemy
        count = compute_enemies_remaining(scene_with_one_object, agent_at_origin)
        assert count == 0.0

    def test_enemies_remaining_with_enemy(self):
        enemy = GameObject(
            id="enemy", color=99, secondary_colors=[],
            cells=[(20, 20), (20, 21)], bounding_box=(20, 20, 20, 21),
            topology="solid", shape_hash="x", is_agent=False,
        )
        scene = SceneGraph(objects={"enemy": enemy}, edges=set())
        agent = AgentState(
            object_ids=[], position=(5, 5), orientation=0,
            shape="x", color=[2], energy=None, inventory=[],
        )
        count = compute_enemies_remaining(scene, agent)
        assert count == 1.0

    def test_rooms_explored(self, agent_at_origin):
        # Agent at (5, 5) → quadrant 0 (top-left)
        count = compute_rooms_explored(SceneGraph(objects={}, edges=set()), agent_at_origin)
        assert count == 0.0

    def test_rooms_explored_bottom_right(self):
        agent = AgentState(
            object_ids=[], position=(50, 50), orientation=0,
            shape="x", color=[2], energy=None, inventory=[],
        )
        count = compute_rooms_explored(SceneGraph(objects={}, edges=set()), agent)
        assert count == 3.0  # bottom-right quadrant


class TestNearMissTracker:
    def test_initialization(self):
        tracker = NearMissTracker()
        assert len(tracker.predicates) == 8  # default predicates

    def test_record_step_increments_count(self, scene_with_one_object, agent_at_origin):
        tracker = NearMissTracker()
        tracker.record_step(scene_with_one_object, agent_at_origin)
        assert tracker._step_count == 1
        # All predicates should have 1 observation
        for pred in tracker.predicates.values():
            assert len(pred.trajectory) == 1

    def test_on_episode_end_win_returns_empty(self, scene_with_one_object, agent_at_origin):
        tracker = NearMissTracker()
        tracker.record_step(scene_with_one_object, agent_at_origin)
        result = tracker.on_episode_end("WIN")
        assert result == []

    def test_on_episode_end_lose_with_near_miss(self):
        """Simulate an episode where distance to exit decreases."""
        tracker = NearMissTracker()
        # Simulate agent moving toward an object
        obj = GameObject(
            id="exit", color=2, secondary_colors=[],
            cells=[(20, 20), (20, 21), (21, 20), (21, 21)],
            bounding_box=(20, 20, 21, 21),
            topology="solid", shape_hash="x", is_agent=False,
        )
        scene = SceneGraph(objects={"exit": obj}, edges=set())
        # Step 1: agent at (5, 5), distance = 30
        tracker.record_step(scene, AgentState(
            object_ids=[], position=(5, 5), orientation=0,
            shape="x", color=[2], energy=None, inventory=[],
        ))
        # Step 2: agent at (10, 10), distance = 20
        tracker.record_step(scene, AgentState(
            object_ids=[], position=(10, 10), orientation=0,
            shape="x", color=[2], energy=None, inventory=[],
        ))
        # Step 3: agent at (15, 15), distance = 10
        tracker.record_step(scene, AgentState(
            object_ids=[], position=(15, 15), orientation=0,
            shape="x", color=[2], energy=None, inventory=[],
        ))
        # Episode ends in LOSE
        near_misses = tracker.on_episode_end("GAME_OVER")
        assert "distance_to_exit" in near_misses

    def test_reset_clears_trajectories(self, scene_with_one_object, agent_at_origin):
        tracker = NearMissTracker()
        tracker.record_step(scene_with_one_object, agent_at_origin)
        tracker.reset()
        assert tracker._step_count == 0
        for pred in tracker.predicates.values():
            assert len(pred.trajectory) == 0

    def test_get_stats(self, scene_with_one_object, agent_at_origin):
        tracker = NearMissTracker()
        tracker.record_step(scene_with_one_object, agent_at_origin)
        tracker.on_episode_end("GAME_OVER")
        stats = tracker.get_stats()
        assert stats["step_count"] == 1
        assert stats["episode_count"] == 1

    def test_get_progress_summary(self, scene_with_one_object, agent_at_origin):
        tracker = NearMissTracker()
        tracker.record_step(scene_with_one_object, agent_at_origin)
        tracker.record_step(scene_with_one_object, agent_at_origin)
        summary = tracker.get_progress_summary()
        assert "distance_to_exit" in summary
        assert "color_match_count" in summary
