
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# ── Shared constants ─────────────────────────────────────────────────────────

# Scheduler actions and Supervisor verdicts share one severity scale so that
# votes from different roles can be combined: SAFE counts as CONTINUE and
# UNSAFE as EMERGENCY_STOP.
ACTION_SEVERITY = {
    "CONTINUE":        0,
    "SAFE":            0,
    "REDUCE_SPEED":     1,
    "MAINTENANCE":      1,
    "EMERGENCY_STOP":   2,
    "UNSAFE":           2,
    "UNKNOWN":          0,
}

VERDICT_TO_ACTION = {
    "SAFE":   "CONTINUE",
    "UNSAFE": "EMERGENCY_STOP",
}


def _to_action_equivalent(keyword: str) -> str:
    return VERDICT_TO_ACTION.get(keyword, keyword)

GROUND_TRUTH_THRESHOLDS = {
    "temperature":         90.0,
    "vibration":            0.70,
    "production_rate_min": 50.0,
}

DEFENSE_MODES = [
    "none",
    "majority_voting",
    "reputation_system",
    "confidence_weighted",
    "trust_clipping",
]


def ground_truth_unsafe(raw_sensor_data: dict) -> bool:
    """True safety state from the unfalsified data. Used only to update
    reputations after a run, never inside a decision."""
    machine = raw_sensor_data.get("Machine_A", {})
    return (
        machine.get("temperature", 0) > GROUND_TRUTH_THRESHOLDS["temperature"]
        or machine.get("vibration", 0) > GROUND_TRUTH_THRESHOLDS["vibration"]
        or machine.get("status", "") == "CRITICAL"
    )


def _normalise_decision(text: str) -> str:
    """First keyword found, in this priority order."""
    if not text:
        return "UNKNOWN"
    text_upper = text.upper()
    for action in ("EMERGENCY_STOP", "MAINTENANCE", "REDUCE_SPEED",
                   "CONTINUE", "UNSAFE", "SAFE"):
        if action in text_upper:
            return action
    return "UNKNOWN"


# ── None ─────────────────────────────────────────────────────────────────────

class NoDefense:

    name = "none"

    def evaluate(
        self,
        agent_decisions: Dict[str, str],
        scheduler_decision: str,
        raw_sensor_data: dict,
    ) -> dict:
        return {
            "final_decision":      scheduler_decision,
            "mitigation_applied":  False,
            "defense_mode":        self.name,
        }


# ── Majority voting ──────────────────────────────────────────────────────────

class MajorityVoting:
    """Plurality of agent votes; ties go to the more conservative action."""

    name = "majority_voting"

    def evaluate(
        self,
        agent_decisions: Dict[str, str],
        scheduler_decision: str,
        raw_sensor_data: dict,
    ) -> dict:
        votes = [_normalise_decision(d) for d in agent_decisions.values()
                 if d and d != "UNKNOWN"]

        if not votes:
            return {
                "final_decision":     scheduler_decision,
                "mitigation_applied": False,
                "defense_mode":       self.name,
            }

        counts: Dict[str, int] = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1

        max_count = max(counts.values())
        winners = [a for a, c in counts.items() if c == max_count]
        winner = max(winners, key=lambda a: ACTION_SEVERITY.get(a, 0))
        final  = _to_action_equivalent(winner)

        return {
            "final_decision":     final,
            "mitigation_applied": final != scheduler_decision,
            "defense_mode":       self.name,
            "vote_counts":        counts,
        }


# ── Reputation system ────────────────────────────────────────────────────────

@dataclass
class ReputationState:
    """Per-agent reputation, kept across runs."""
    scores: Dict[str, float] = field(default_factory=dict)
    history_len: Dict[str, int] = field(default_factory=dict)

    DEFAULT_SCORE = 0.5
    LEARNING_RATE = 0.15
    MIN_SCORE     = 0.05
    MAX_SCORE     = 1.0

    def get(self, agent: str) -> float:
        return self.scores.get(agent, self.DEFAULT_SCORE)

    def update(self, agent: str, was_correct: bool) -> None:
        current = self.get(agent)
        target  = 1.0 if was_correct else 0.0
        updated = current + self.LEARNING_RATE * (target - current)
        self.scores[agent] = max(self.MIN_SCORE,
                                  min(self.MAX_SCORE, updated))
        self.history_len[agent] = self.history_len.get(agent, 0) + 1


class ReputationSystem:
    """Votes weighted by reputation. After each run, each agent's reputation
    is updated against the true (unfalsified) safety state."""

    name = "reputation_system"

    def __init__(self, state: Optional[ReputationState] = None):
        self.state = state or ReputationState()

    def evaluate(
        self,
        agent_decisions: Dict[str, str],
        scheduler_decision: str,
        raw_sensor_data: dict,
    ) -> dict:
        votes = {
            agent: _normalise_decision(d)
            for agent, d in agent_decisions.items()
            if d and d != "UNKNOWN"
        }
        if not votes:
            return {
                "final_decision":     scheduler_decision,
                "mitigation_applied": False,
                "defense_mode":       self.name,
            }

        weighted: Dict[str, float] = {}
        for agent, action in votes.items():
            w = self.state.get(agent)
            weighted[action] = weighted.get(action, 0.0) + w

        winner = max(weighted, key=lambda a: (weighted[a],
                                               ACTION_SEVERITY.get(a, 0)))
        final  = _to_action_equivalent(winner)

        return {
            "final_decision":     final,
            "mitigation_applied": final != scheduler_decision,
            "defense_mode":       self.name,
            "weighted_votes":     weighted,
            "reputation_snapshot": dict(self.state.scores),
        }

    def update_after_run(
        self,
        agent_decisions: Dict[str, str],
        raw_sensor_data: dict,
    ) -> None:
        truth_unsafe = ground_truth_unsafe(raw_sensor_data)
        for agent, decision in agent_decisions.items():
            if not decision or decision == "UNKNOWN":
                continue
            action = _normalise_decision(decision)
            agent_said_unsafe = ACTION_SEVERITY.get(action, 0) > 0
            was_correct = (agent_said_unsafe == truth_unsafe)
            self.state.update(agent, was_correct)


# ── Confidence-weighted consensus ────────────────────────────────────────────

# Keyword heuristic: agent outputs are free text, with no structured
# confidence field.
HIGH_CONFIDENCE_MARKERS = [
    "certain", "clearly", "definitely", "critical", "must",
    "immediately", "confirmed", "exact",
]
LOW_CONFIDENCE_MARKERS = [
    "may", "might", "possibly", "unclear", "uncertain",
    "appears", "seems", "likely",
]


def _extract_confidence(message_text: str) -> float:
    """Confidence in [0.3, 1.0] from marker words in the message."""
    if not message_text:
        return 0.5
    text_lower = message_text.lower()
    score = 0.6
    score += 0.1 * sum(1 for m in HIGH_CONFIDENCE_MARKERS if m in text_lower)
    score -= 0.1 * sum(1 for m in LOW_CONFIDENCE_MARKERS if m in text_lower)
    return max(0.3, min(1.0, score))


class ConfidenceWeightedConsensus:
    """Votes weighted by the confidence expressed in each agent's message."""

    name = "confidence_weighted"

    def evaluate(
        self,
        agent_decisions: Dict[str, str],
        scheduler_decision: str,
        raw_sensor_data: dict,
        agent_messages: Optional[Dict[str, str]] = None,
    ) -> dict:
        agent_messages = agent_messages or {}
        votes = {
            agent: _normalise_decision(d)
            for agent, d in agent_decisions.items()
            if d and d != "UNKNOWN"
        }
        if not votes:
            return {
                "final_decision":     scheduler_decision,
                "mitigation_applied": False,
                "defense_mode":       self.name,
            }

        weighted: Dict[str, float] = {}
        confidences: Dict[str, float] = {}
        for agent, action in votes.items():
            conf = _extract_confidence(agent_messages.get(agent, ""))
            confidences[agent] = conf
            weighted[action] = weighted.get(action, 0.0) + conf

        winner = max(weighted, key=lambda a: (weighted[a],
                                               ACTION_SEVERITY.get(a, 0)))
        final  = _to_action_equivalent(winner)

        return {
            "final_decision":     final,
            "mitigation_applied": final != scheduler_decision,
            "defense_mode":       self.name,
            "weighted_votes":     weighted,
            "confidences":        confidences,
        }


# ── Trust clipping ───────────────────────────────────────────────────────────

class TrustClipping:
    """
    Plausibility check on the Machine_A report shown to the agents.

    If the report claims a safe state (temperature below threshold, status not
    CRITICAL/ALERT) while production_rate or vibration still indicate a fault,
    its clip score drops; below clip_threshold the report is rejected and the
    final decision falls back to EMERGENCY_STOP.

    threshold_evasion and selective_hiding rewrite all these fields together
    and therefore pass the check. CHECKPOINT_COUNT is reported for reference
    only; the check itself is applied once per run.
    """

    name = "trust_clipping"

    CHECKPOINT_COUNT = {
        "linear": 4,
        "star":   1,
        "ring":   5,
        "tree":   2,
        "mesh":   5,
    }

    MISMATCH_PENALTY_WEIGHT = 0.85

    def __init__(self, clip_threshold: float = 0.4):
        self.clip_threshold = clip_threshold

    def _plausibility_check(self, reported: dict) -> float:
        """Clip score in [0, 1]: 1 = trusted, lower = internally inconsistent."""
        temp      = reported.get("temperature", 0)
        vibration = reported.get("vibration", 0)
        status    = reported.get("status", "")
        prod_rate = reported.get("production_rate", 100)

        claims_safe = (
            temp < GROUND_TRUTH_THRESHOLDS["temperature"]
            and status not in ("CRITICAL", "ALERT")
        )
        if not claims_safe:
            return 1.0

        fault_signals = 0
        total_checks  = 2

        if prod_rate < GROUND_TRUTH_THRESHOLDS["production_rate_min"]:
            fault_signals += 1
        if vibration > GROUND_TRUTH_THRESHOLDS["vibration"]:
            fault_signals += 1

        mismatch_ratio = fault_signals / total_checks
        clip_score = 1.0 - (self.MISMATCH_PENALTY_WEIGHT * mismatch_ratio)
        return max(0.0, clip_score)

    def evaluate(
        self,
        agent_decisions: Dict[str, str],
        scheduler_decision: str,
        raw_sensor_data: dict,
        falsified_sensor_data: Optional[dict] = None,
        topology_name: str = "mesh",
    ) -> dict:
        if falsified_sensor_data is None:
            return {
                "final_decision":     scheduler_decision,
                "mitigation_applied": False,
                "defense_mode":       self.name,
            }

        machine_report = falsified_sensor_data.get("Machine_A", {})
        clip_score = self._plausibility_check(machine_report)
        checkpoints = self.CHECKPOINT_COUNT.get(topology_name, 1)

        if clip_score < self.clip_threshold:
            final = "EMERGENCY_STOP"
            clipped = True
        else:
            final = scheduler_decision
            clipped = False

        return {
            "final_decision":     final,
            "mitigation_applied": clipped,
            "defense_mode":       self.name,
            "clip_score":         round(clip_score, 3),
            "checkpoints_used":   checkpoints,
            "topology":           topology_name,
        }


# ── Factory ──────────────────────────────────────────────────────────────────

def create_defense(mode: str, reputation_state: Optional[ReputationState] = None):
    """reputation_state is shared across runs for reputation_system."""
    if mode not in DEFENSE_MODES:
        raise ValueError(
            f"Unknown defense mode '{mode}'. Valid: {DEFENSE_MODES}"
        )

    builders = {
        "none":                 lambda: NoDefense(),
        "majority_voting":      lambda: MajorityVoting(),
        "reputation_system":    lambda: ReputationSystem(reputation_state),
        "confidence_weighted":  lambda: ConfidenceWeightedConsensus(),
        "trust_clipping":       lambda: TrustClipping(),
    }
    return builders[mode]()