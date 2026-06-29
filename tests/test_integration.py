"""Integration tests that load real replay data.

These tests are marked `integration` and skipped if the data files
are not present (e.g. in CI without the data release artifact).
"""
from pathlib import Path

import pytest

# conftest.py at the tests/ root adds src/ to sys.path for us.
pytestmark = pytest.mark.integration

DATA_DIR = Path("/home/z/my-project/alphamoo/data")


@pytest.fixture(scope="module")
def replays():
    if not DATA_DIR.exists() or not list(DATA_DIR.glob("*.vtx")):
        pytest.skip("Data directory not available")
    from alphamoo.vtx_reader import load_replays_from_dir
    return load_replays_from_dir(DATA_DIR)


def test_all_25_replays_load(replays):
    assert len(replays) == 25


def test_all_replays_won(replays):
    for game_id, replay in replays.items():
        assert replay.won, f"{game_id} did not reach WIN state"


def test_all_replays_have_actions(replays):
    for game_id, replay in replays.items():
        assert replay.n_actions > 0, f"{game_id} has 0 actions"


def test_total_actions_match_known_count(replays):
    """We know there are 14,798 actions across all 25 replays."""
    total = sum(r.n_actions for r in replays.values())
    assert total == 14798


def test_each_replay_has_valid_first_record(replays):
    """First record should be either RESET (id=0) or, in some score-based
    games, the first gameplay action directly. Either is valid."""
    for game_id, replay in replays.items():
        first = replay.records[0]
        assert first.state == "NOT_FINISHED"
        # First action id is 0 (RESET) for level-based games, or a gameplay
        # action (1-6) for some score-based games that don't have an explicit reset.
        assert first.action_input.id in (0, 1, 2, 3, 4, 5, 6), \
            f"{game_id} unexpected first action id: {first.action_input.id}"


def test_each_replay_has_valid_final_state(replays):
    for game_id, replay in replays.items():
        last = replay.records[-1]
        assert last.state in ("WIN", "GAME_OVER"), f"{game_id} unexpected final state: {last.state}"


def test_perception_works_on_every_first_frame(replays):
    from alphamoo.perception import perceive_with_diagnostics
    for game_id, replay in replays.items():
        first_grid = replay.records[0].final_grid
        diag = perceive_with_diagnostics(first_grid)
        assert diag["n_objects"] > 0, f"{game_id} has 0 objects in first frame"
        assert diag["full_perception_ms"] < 100, f"{game_id} perception too slow"


def test_agent_tracker_runs_on_every_replay(replays):
    from alphamoo.agent_tracker import track_replay
    for game_id, replay in replays.items():
        _, stats = track_replay(replay)
        # Detection rate should be >50% for movement games, may be lower for click-only
        rate = stats["detection_count"] / max(1, len(replay.records))
        assert rate > 0.4, f"{game_id} detection rate too low: {rate:.0%}"
