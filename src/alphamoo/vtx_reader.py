"""
AlphaMoo v4.1 — Vortex (.vtx) file reader.

Parses ARC-AGI-3 replay files stored in Vortex columnar format into clean
Python Replay objects.

Each .vtx file contains one complete game replay:
  - Line 0: initial game state (RESET action, id=0)
  - Lines 1..N-2: one record per action taken
  - Last line: summary/scorecard

Schema verified against 25 ground-truth replays (14,798 actions total).
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import vortex as vx

from .schemas import (
    ActionInput,
    FrameRecord,
    GameState,
    GameSummary,
    LevelSummary,
    Replay,
)


def _coerce_action_id(raw) -> int:
    """
    Coerce action ID to int. Some replays store it as int (1, 2, 3...),
    others as string ("ACTION1", "ACTION2"...), others as the bare name
    ("RESET", "ACTION3"). Handle all cases.
    """
    if raw is None:
        return 0
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if s.isdigit():
            return int(s)
        # Strip "ACTION" prefix or use known names
        if s.startswith("ACTION"):
            try:
                return int(s[len("ACTION"):])
            except ValueError:
                pass
        if s == "RESET":
            return 0
        if s == "UNDO":
            return 7
        # Last resort: try to parse the trailing digits
        digits = "".join(c for c in s if c.isdigit())
        return int(digits) if digits else 0
    return int(raw)


def _parse_action_input(raw: dict) -> ActionInput:
    """Parse the action_input sub-object."""
    return ActionInput(
        id=_coerce_action_id(raw["id"]),
        data=dict(raw.get("data", {})),
        reasoning=raw.get("reasoning"),
    )


def _parse_frame_record(timestamp: str, data: dict) -> FrameRecord:
    """Parse one play-record line (data field of the JSONL envelope).

    Handles two schemas observed in the wild:
      - Level-based games: levels_completed + win_levels
      - Score-based games: score + win_score (no levels_completed)
    """
    # Some games use score/win_score; normalize to levels_completed/win_levels
    levels_completed = int(data.get("levels_completed", 0) or 0)
    win_levels = int(data.get("win_levels", 0) or 0)

    # Score-based games: derive level-equivalent from score if no level data
    if win_levels == 0:
        score = data.get("score")
        win_score = data.get("win_score")
        if score is not None and win_score is not None:
            try:
                int(score) if score is not None else 0
                win_score_int = int(win_score) if win_score is not None else 0
                # Treat score as continuous progress; for now just store
                # 0 levels_completed and 1 "level" (the whole game).
                # The mining layer will use score separately.
                win_levels = 1 if win_score_int > 0 else 0
            except (TypeError, ValueError):
                pass

    return FrameRecord(
        timestamp=timestamp,
        game_id=data["game_id"],
        frame=data["frame"],  # leave as list[list[list[int]]]
        state=data["state"],
        levels_completed=levels_completed,
        win_levels=win_levels,
        action_input=_parse_action_input(data["action_input"]),
        guid=data["guid"],
        full_reset=bool(data.get("full_reset", False)),
        available_actions=list(data.get("available_actions", [])),
    )


def _parse_summary_record(data: dict) -> GameSummary:
    """Parse the final summary line."""
    # cards is a dict keyed by game_id; take the first (only) entry
    cards = data.get("cards") or {}
    if not cards:
        # Some replays may have null cards; construct minimal summary
        return GameSummary(
            game_id=data.get("game_id", "unknown"),
            total_plays=0,
            guids=[],
            final_levels_completed=[],
            final_states=[],
            total_actions_per_play=[],
            actions_by_level=[],
            resets_per_play=[],
            total_actions=int(data.get("total_actions", 0) or 0),
        )

    game_id = next(iter(cards))
    card = cards[game_id]

    # actions_by_level is list (per play) of lists of [level, cumulative_actions] pairs
    actions_by_level: list[list[LevelSummary]] = []
    for play_levels in card.get("actions_by_level", []):
        play_summaries = [
            LevelSummary(level=int(pair[0]), cumulative_actions=int(pair[1]))
            for pair in play_levels
        ]
        actions_by_level.append(play_summaries)

    return GameSummary(
        game_id=game_id,
        total_plays=int(card.get("total_plays", 1)),
        guids=list(card.get("guids", [])),
        final_levels_completed=[int(x) for x in card.get("levels_completed", [])],
        final_states=list(card.get("states", [])),
        total_actions_per_play=[int(x) for x in card.get("actions", [])],
        actions_by_level=actions_by_level,
        resets_per_play=[int(x) for x in card.get("resets", [])],
        total_actions=int(card.get("total_actions", 0)),
    )


def load_replay(path: str | Path) -> Replay:
    """
    Load a single .vtx file into a Replay object.

    Args:
        path: Path to the .vtx file.

    Returns:
        Replay object containing all FrameRecords and the GameSummary.

    Raises:
        FileNotFoundError: If path doesn't exist.
        ValueError: If the file is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Replay file not found: {path}")

    # Open via Vortex (lazy file handle) and materialize to Arrow table
    f = vx.open(str(path))
    table = f.to_arrow().read_all()

    # Two columns: timestamp (string), data (struct)
    n_rows = table.num_rows
    if n_rows < 2:
        raise ValueError(f"Replay file too short: {n_rows} rows (need >= 2)")

    # Convert to Python lists
    timestamps = table.column("timestamp").to_pylist()
    data_list = table.column("data").to_pylist()

    # Parse all records
    records: list[FrameRecord] = []
    summary: GameSummary | None = None
    for i in range(n_rows):
        data = data_list[i]
        # Summary records have cards != None AND no frame data.
        # These can appear anywhere in the file (some replays have them
        # mid-stream after a level transition; some only at the end).
        is_summary = data.get("cards") is not None and (
            data.get("frame") is None or
            (isinstance(data.get("frame"), list) and len(data.get("frame")) == 0)
        )
        if is_summary:
            # Parse as summary, keep the last one we see (some replays have
            # multiple summary records; the final one is authoritative).
            summary = _parse_summary_record(data)
            continue
        records.append(_parse_frame_record(timestamps[i], data))

    if not records:
        raise ValueError(f"No play records found in {path}")

    # If no summary was found, construct minimal one from last play record
    if summary is None:
        last = records[-1]
        summary = GameSummary(
            game_id=last.game_id,
            total_plays=1,
            guids=[last.guid],
            final_levels_completed=[last.levels_completed],
            final_states=[last.state],
            total_actions_per_play=[len(records) - 1],
            actions_by_level=[],
            resets_per_play=[],
            total_actions=len(records) - 1,
        )

    # Fix-up: GAME_OVER records (and some terminal records) can have empty
    # frames — the engine doesn't always emit a final grid on death.
    # Backfill with the previous record's final grid so downstream consumers
    # always have a valid grid.
    prev_grid: list[list[int]] | None = None
    fixed_records: list[FrameRecord] = []
    for r in records:
        if not r.frame:
            # Empty frame — backfill
            if prev_grid is not None:
                r = FrameRecord(
                    timestamp=r.timestamp,
                    game_id=r.game_id,
                    frame=[prev_grid],  # wrap as single-subframe
                    state=r.state,
                    levels_completed=r.levels_completed,
                    win_levels=r.win_levels,
                    action_input=r.action_input,
                    guid=r.guid,
                    full_reset=r.full_reset,
                    available_actions=r.available_actions,
                )
            else:
                # First record is empty — skip it (rare edge case)
                continue
        else:
            prev_grid = r.final_grid
        fixed_records.append(r)
    records = fixed_records

    # Fix-up: some replays have win_levels=0/None on play records but the
    # summary's final_levels_completed tells us the true count.
    true_win_levels = 0
    if summary.final_levels_completed:
        true_win_levels = max(summary.final_levels_completed)
    if true_win_levels == 0 and records[-1].state == GameState.WIN:
        # Last-ditch: if game was won but no level data, infer from
        # levels_completed progression.
        true_win_levels = max(r.levels_completed for r in records)

    if true_win_levels > 0:
        # Patch all records with the correct win_levels
        records = [
            FrameRecord(
                timestamp=r.timestamp,
                game_id=r.game_id,
                frame=r.frame,
                state=r.state,
                levels_completed=r.levels_completed,
                win_levels=true_win_levels,
                action_input=r.action_input,
                guid=r.guid,
                full_reset=r.full_reset,
                available_actions=r.available_actions,
            )
            for r in records
        ]

    # Extract game_id and guid from first record
    game_id = records[0].game_id
    guid = records[0].guid

    return Replay(
        game_id=game_id,
        guid=guid,
        records=records,
        summary=summary,
    )


def load_replays_from_dir(dir_path: str | Path,
                          pattern: str = "*.vtx") -> dict[str, Replay]:
    """
    Load all .vtx files in a directory.

    Args:
        dir_path: Directory containing .vtx files.
        pattern: Glob pattern (default "*.vtx").

    Returns:
        Dict mapping game_id (e.g. "ls20") to Replay object.
    """
    dir_path = Path(dir_path)
    replays: dict[str, Replay] = {}
    for vtx_file in sorted(dir_path.glob(pattern)):
        # Extract game_id from filename: "ls20-8aed7120-...vtx" -> "ls20"
        game_id = vtx_file.stem.split("-")[0]
        try:
            replays[game_id] = load_replay(vtx_file)
        except Exception as e:
            print(f"  [WARN] Failed to load {vtx_file.name}: {e}")
    return replays


def iter_records(replay: Replay) -> Iterator[FrameRecord]:
    """Iterate over all records in a replay (excluding the initial RESET)."""
    yield from replay.records[1:]


def iter_actions(replay: Replay) -> Iterator[tuple[FrameRecord, FrameRecord]]:
    """
    Iterate (before, after) pairs for each action.

    Yields tuples of (record_before_action, record_after_action) so callers
    can compute state deltas.
    """
    for i in range(1, len(replay.records)):
        yield (replay.records[i - 1], replay.records[i])


def iter_level_boundaries(replay: Replay) -> Iterator[tuple[int, int]]:
    """
    Yield (level_number, action_index) for each level completion.

    action_index is the index into replay.records (0-indexed from the
    initial RESET) where levels_completed first reached `level_number`.
    """
    prev_level = 0
    for i, record in enumerate(replay.records):
        if record.levels_completed > prev_level:
            for level_num in range(prev_level + 1, record.levels_completed + 1):
                yield (level_num, i)
            prev_level = record.levels_completed
