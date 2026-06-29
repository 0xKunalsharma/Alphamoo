"""Unit tests for schemas.py."""

from alphamoo.schemas import (
    ARC_PALETTE,
    GRID_SIZE,
    MAX_COLOR,
    ActionId,
    ActionInput,
    FrameRecord,
)


class TestActionInput:
    def test_simple_action_no_click_coords(self):
        a = ActionInput(id=1, data={"game_id": "ls20"})
        assert a.is_click is False
        assert a.click_coords is None
        assert a.is_reset is False

    def test_click_action_has_coords(self):
        a = ActionInput(id=6, data={"game_id": "ls20", "x": 10, "y": 20})
        assert a.is_click is True
        assert a.click_coords == (10, 20)

    def test_reset_action(self):
        a = ActionInput(id=0, data={"game_id": "ls20"})
        assert a.is_reset is True
        assert a.is_click is False

    def test_undo_action(self):
        a = ActionInput(id=7, data={"game_id": "ls20"})
        assert a.id == ActionId.UNDO


class TestFrameRecord:
    def _make_record(self, state="NOT_FINISHED", n_subframes=1, levels=0):
        return FrameRecord(
            timestamp="2026-01-01T00:00:00+00:00",
            game_id="test",
            frame=[[[0] * 64 for _ in range(64)]] * n_subframes,
            state=state,
            levels_completed=levels,
            win_levels=7,
            action_input=ActionInput(id=1, data={}),
            guid="test",
            full_reset=False,
            available_actions=[1, 2, 3, 4],
        )

    def test_n_subframes_default(self):
        r = self._make_record()
        assert r.n_subframes == 1

    def test_n_subframes_cascade(self):
        r = self._make_record(n_subframes=5)
        assert r.n_subframes == 5

    def test_final_grid_returns_last_subframe(self):
        # Make a cascade where each subframe is filled with a different color
        frames = []
        for color in range(3):
            frames.append([[color] * 64 for _ in range(64)])
        r = FrameRecord(
            timestamp="t",
            game_id="g",
            frame=frames,
            state="NOT_FINISHED",
            levels_completed=0,
            win_levels=7,
            action_input=ActionInput(id=1, data={}),
            guid="g",
            full_reset=False,
            available_actions=[1, 2, 3, 4],
        )
        # Final grid should be color 2
        assert r.final_grid[0][0] == 2

    def test_is_terminal_win(self):
        r = self._make_record(state="WIN")
        assert r.is_terminal is True

    def test_is_terminal_game_over(self):
        r = self._make_record(state="GAME_OVER")
        assert r.is_terminal is True

    def test_not_terminal(self):
        r = self._make_record(state="NOT_FINISHED")
        assert r.is_terminal is False


class TestPalette:
    def test_palette_has_all_16_colors(self):
        assert len(ARC_PALETTE) == 16
        for i in range(16):
            assert i in ARC_PALETTE

    def test_palette_values_are_rgb_tuples(self):
        for _color, rgb in ARC_PALETTE.items():
            assert len(rgb) == 3
            for channel in rgb:
                assert 0 <= channel <= 255

    def test_black_is_zero(self):
        assert ARC_PALETTE[0] == (0, 0, 0)

    def test_white_is_15(self):
        assert ARC_PALETTE[15] == (255, 255, 255)


class TestConstants:
    def test_grid_size(self):
        assert GRID_SIZE == 64

    def test_max_color(self):
        assert MAX_COLOR == 15
