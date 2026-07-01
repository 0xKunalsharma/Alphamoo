"""
AlphaMoo v4.1 — Context Compressor (Module 12).

At episode/level end, compresses the full episodic memory into a fixed-size
LevelSummary and writes it to semantic memory. At the start of the next
level, retrieves the top-K most relevant summaries for in-context learning.

This is what makes multi-level progression work. Without it, context grows
unboundedly and reasoning degrades.

v4.1 change: ICL is the primary mechanism (not LoRA). 100 within-game level
summaries is too few for LoRA training; ICL works at that scale.

Components:
  1. LevelSummary — compressed representation of one level's lessons
  2. SemanticMemory — persistent store with relevance-scored retrieval
  3. ContextCompressor — orchestrates compression at level end
  4. ICL retrieval at level start — top-K most relevant summaries
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .world_model import CausalRule

# =============================================================================
# Level Summary — the compressed unit of cross-level knowledge
# =============================================================================

@dataclass
class TypeDefinition:
    """A type learned during a level (for Type Inferencer, Module 3)."""
    type_id: str
    primary_color: int
    avg_size: int
    topology: str
    behavior: str  # e.g. "consumable", "transformation_station", "exit"
    example_bboxes: list[tuple] = field(default_factory=list)


@dataclass
class LevelSummary:
    """
    Compressed representation of one level's lessons.

    Written to semantic memory at level end. Retrieved at next level start
    via relevance scoring.
    """
    game_id: str
    level: int
    confirmed_rules: list[dict] = field(default_factory=list)  # serialized CausalRules
    goal_type: str = ""                       # e.g. "reach_exit", "state_matching"
    goal_args: dict = field(default_factory=dict)
    goal_confidence: float = 0.0
    key_objects: list[str] = field(default_factory=list)  # object descriptions
    action_efficiency: float = 0.0            # actual_actions / human_baseline
    failure_modes: list[str] = field(default_factory=list)
    near_miss_predicates: list[str] = field(default_factory=list)
    type_discoveries: list[dict] = field(default_factory=list)  # serialized TypeDefinitions
    n_actions: int = 0
    n_wins: int = 0
    n_loses: int = 0
    timestamp: str = ""
    hash: str = ""                            # for dedup

    def __post_init__(self):
        if not self.hash:
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute a content-based hash for dedup."""
        content = (
            f"{self.game_id}|{self.level}|{self.goal_type}|"
            f"{json.dumps(self.confirmed_rules, sort_keys=True)}|"
            f"{json.dumps(self.goal_args, sort_keys=True)}"
        )
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def to_prompt_text(self) -> str:
        """
        Render this summary as text for ICL (in-context learning).
        Format is concise — fits within a few hundred tokens.
        """
        lines = [
            f"### Level {self.level} of {self.game_id}",
            f"Goal: {self.goal_type}({self.goal_args}) [conf={self.goal_confidence:.2f}]",
            f"Actions: {self.n_actions} ({self.n_wins}W/{self.n_loses}L, efficiency={self.action_efficiency:.2f})",
        ]
        if self.confirmed_rules:
            lines.append(f"Confirmed rules ({len(self.confirmed_rules)}):")
            for rule in self.confirmed_rules[:5]:  # cap at 5 for token budget
                eff = rule.get("effect", {})
                lines.append(f"  - {rule.get('trigger_desc', '?')} → {eff.get('type', '?')}({eff.get('args', {})})")
        if self.key_objects:
            lines.append(f"Key objects: {', '.join(self.key_objects[:5])}")
        if self.failure_modes:
            lines.append(f"Failure modes: {', '.join(self.failure_modes[:3])}")
        if self.type_discoveries:
            lines.append(f"Types learned: {len(self.type_discoveries)}")
        return "\n".join(lines)


# =============================================================================
# Semantic Memory — persistent store with relevance retrieval
# =============================================================================

class SemanticMemory:
    """
    Persistent store of LevelSummaries with relevance-scored retrieval.

    Relevance scoring (v4.1 spec):
      game_id match > goal_type match > type overlap > raw recency

    Usage:
        memory = SemanticMemory()
        # At level end:
        memory.write(summary)
        # At next level start:
        relevant = memory.retrieve(game_id="ls20", goal_type="state_matching",
                                    types_present=["yellow_cube"], k=5)
    """

    def __init__(self, persistence_path: Path | None = None):
        self.summaries: list[LevelSummary] = []
        self.persistence_path = persistence_path
        if persistence_path and persistence_path.exists():
            self.load()

    def write(self, summary: LevelSummary) -> bool:
        """
        Write a summary to memory. Returns True if added, False if deduped.

        Dedup: if a summary with the same hash already exists, skip.
        """
        for existing in self.summaries:
            if existing.hash == summary.hash:
                return False
        self.summaries.append(summary)
        if self.persistence_path:
            self.save()
        return True

    def retrieve(
        self,
        game_id: str | None = None,
        goal_type: str | None = None,
        types_present: list[str] | None = None,
        k: int = 5,
    ) -> list[LevelSummary]:
        """
        Retrieve the top-K most relevant summaries.

        When a filter criterion is specified, only summaries matching that
        criterion are considered. Within the filtered set, ranking is by:
          - goal_type match: +5
          - type overlap: +2 per shared type
          - recency: +1 per recent position

        If no filters are specified, all summaries are ranked by recency.
        """
        # Filter first
        candidates = self.summaries
        if game_id:
            candidates = [s for s in candidates if s.game_id == game_id]
        if goal_type:
            candidates = [s for s in candidates if s.goal_type == goal_type]
        # types_present is a soft filter (overlap scoring, not hard filter)

        # Score and rank
        scored: list[tuple[float, LevelSummary]] = []
        n = len(candidates)
        for i, summary in enumerate(candidates):
            score = 0.0
            # game_id and goal_type already filtered; no extra score needed
            if types_present and summary.type_discoveries:
                summary_type_ids = {t.get("type_id", "") for t in summary.type_discoveries}
                overlap = len(set(types_present) & summary_type_ids)
                score += overlap * 2
            # Recency: more recent = higher score (within the filtered set)
            recency = (i + 1) / max(1, n)
            score += recency
            scored.append((score, summary))

        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:k]]

    def retrieve_for_prompt(
        self,
        game_id: str | None = None,
        goal_type: str | None = None,
        types_present: list[str] | None = None,
        k: int = 5,
    ) -> str:
        """
        Retrieve top-K summaries and render as a single text block for ICL.
        """
        summaries = self.retrieve(game_id, goal_type, types_present, k)
        if not summaries:
            return "(no prior level summaries available)"
        blocks = [s.to_prompt_text() for s in summaries]
        return "\n\n".join(blocks)

    def get_stats(self) -> dict:
        return {
            "total_summaries": len(self.summaries),
            "unique_games": len({s.game_id for s in self.summaries}),
            "avg_summaries_per_game": (
                len(self.summaries) / max(1, len({s.game_id for s in self.summaries}))
            ),
        }

    def save(self) -> None:
        """Persist to disk as JSON."""
        if not self.persistence_path:
            return
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(s) for s in self.summaries]
        self.persistence_path.write_text(json.dumps(data, indent=2, default=str))

    def load(self) -> None:
        """Load from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return
        data = json.loads(self.persistence_path.read_text())
        self.summaries = []
        for item in data:
            # Reconstruct LevelSummary
            self.summaries.append(LevelSummary(**{
                k: v for k, v in item.items()
                if k in LevelSummary.__dataclass_fields__
            }))


# =============================================================================
# Context Compressor — orchestrates compression at level end
# =============================================================================

class ContextCompressor:
    """
    At episode/level end, compresses the full episodic memory into a
    LevelSummary and writes it to semantic memory.

    Usage:
        compressor = ContextCompressor(semantic_memory)
        # At level end:
        summary = compressor.compress(
            game_id="ls20",
            level=3,
            confirmed_rules=wm.rules,
            goal_module=goal_module,
            near_miss_tracker=near_miss,
            n_actions=78,
            n_wins=1,
            n_loses=2,
        )
        compressor.write(summary)
    """

    def __init__(self, semantic_memory: SemanticMemory):
        self.semantic_memory = semantic_memory
        self._compression_count: int = 0

    def compress(
        self,
        game_id: str,
        level: int,
        confirmed_rules: list[CausalRule],
        goal_module: object | None = None,
        near_miss_tracker: object | None = None,
        n_actions: int = 0,
        n_wins: int = 0,
        n_loses: int = 0,
        action_efficiency: float = 0.0,
        failure_modes: list[str] | None = None,
        type_discoveries: list[TypeDefinition] | None = None,
        timestamp: str = "",
    ) -> LevelSummary:
        """
        Compress one level's worth of experience into a LevelSummary.

        Args:
            game_id: which game this level was from
            level: level number (1-indexed)
            confirmed_rules: rules with confidence > 0.7 from World Model
            goal_module: GoalInferenceModule (for goal type extraction)
            near_miss_tracker: NearMissTracker (for near-miss predicates)
            n_actions: total actions taken this level
            n_wins: number of WIN events this level
            n_loses: number of GAME_OVER events this level
            action_efficiency: actual_actions / human_baseline
            failure_modes: list of failure descriptions
            type_discoveries: new types learned this level
            timestamp: ISO format timestamp

        Returns:
            LevelSummary ready to write to semantic memory.
        """
        self._compression_count += 1

        # Serialize confirmed rules
        serialized_rules = [self._serialize_rule(r) for r in confirmed_rules]

        # Extract goal info
        goal_type = ""
        goal_args = {}
        goal_confidence = 0.0
        if goal_module is not None:
            top_goal = goal_module.get_top_goal()
            if top_goal:
                goal_type = top_goal.terminal_condition
                goal_args = top_goal.args
                goal_confidence = top_goal.confidence

        # Extract near-miss predicates
        near_miss_predicates = []
        if near_miss_tracker is not None:
            progress = near_miss_tracker.get_progress_summary()
            near_miss_predicates = [
                name for name, val in progress.items() if val != 0
            ]

        # Extract key objects from rules
        key_objects = self._extract_key_objects(confirmed_rules)

        # Serialize type discoveries
        serialized_types = []
        if type_discoveries:
            for td in type_discoveries:
                serialized_types.append({
                    "type_id": td.type_id,
                    "primary_color": td.primary_color,
                    "avg_size": td.avg_size,
                    "topology": td.topology,
                    "behavior": td.behavior,
                })

        return LevelSummary(
            game_id=game_id,
            level=level,
            confirmed_rules=serialized_rules,
            goal_type=goal_type,
            goal_args=goal_args,
            goal_confidence=goal_confidence,
            key_objects=key_objects,
            action_efficiency=action_efficiency,
            failure_modes=failure_modes or [],
            near_miss_predicates=near_miss_predicates,
            type_discoveries=serialized_types,
            n_actions=n_actions,
            n_wins=n_wins,
            n_loses=n_loses,
            timestamp=timestamp,
        )

    def _serialize_rule(self, rule: CausalRule) -> dict:
        """Serialize a CausalRule for storage."""
        # Build a human-readable trigger description
        trigger_parts = []
        for cond in rule.trigger_conditions:
            neg = "NOT " if cond.negated else ""
            args_str = ", ".join(f"{k}={v}" for k, v in cond.args.items())
            trigger_parts.append(f"{neg}{cond.predicate}({args_str})")
        trigger_desc = " AND ".join(trigger_parts) if trigger_parts else "(always)"

        return {
            "trigger_desc": trigger_desc,
            "trigger_conditions": [
                {
                    "predicate": c.predicate,
                    "args": c.args,
                    "negated": c.negated,
                }
                for c in rule.trigger_conditions
            ],
            "effect": rule.effect,
            "confidence": rule.confidence,
            "support": rule.support,
        }

    def _extract_key_objects(self, rules: list[CausalRule]) -> list[str]:
        """Extract key object descriptions from rules."""
        objects = set()
        for rule in rules:
            for cond in rule.trigger_conditions:
                if "color" in cond.args:
                    objects.add(f"color_{cond.args['color']}")
                if "position" in cond.args:
                    objects.add(f"position_{cond.args['position']}")
            eff_args = rule.effect.get("args", {})
            if "color" in eff_args:
                objects.add(f"color_{eff_args['color']}")
        return sorted(objects)[:10]  # cap at 10

    def write(self, summary: LevelSummary) -> bool:
        """Write a summary to semantic memory."""
        return self.semantic_memory.write(summary)

    def retrieve_for_next_level(
        self,
        game_id: str,
        goal_type: str | None = None,
        types_present: list[str] | None = None,
        k: int = 5,
    ) -> str:
        """
        Retrieve relevant summaries for the next level's ICL prompt.
        Called at level start.
        """
        return self.semantic_memory.retrieve_for_prompt(
            game_id=game_id,
            goal_type=goal_type,
            types_present=types_present,
            k=k,
        )

    def get_stats(self) -> dict:
        return {
            "compression_count": self._compression_count,
            "memory_stats": self.semantic_memory.get_stats(),
        }
