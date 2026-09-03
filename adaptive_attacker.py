import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


#  Detection thresholds (must match MonitorAgent rules) 

DETECTION_THRESHOLDS = {
    "temperature":         90.0,
    "vibration":            0.70,
    "production_rate_min": 50.0,
}

#  Strategy risk levels 
# Lower value = more subtle (harder to detect, lower success)
# Higher value = more aggressive (easier to detect, higher success)

STRATEGY_RISK: Dict[str, float] = {
    "gradual_drift":     0.05,
    "noise_injection":   0.10,
    "threshold_evasion": 0.15,
    "selective_hiding":  0.25,
}

# Decision labels used by the multi-agent system
VERDICT_DETECTED   = "UNSAFE"
VERDICT_UNDETECTED = "SAFE"
DECISION_WRONG     = "CONTINUE"


#  Data structures 

@dataclass
class AttackRecord:
    """Immutable record of one attack attempt and its outcome."""
    cycle:            int
    strategy:         str
    original_temp:    float
    falsified_temp:   float
    temp_reduction:   float
    detected:         bool  = False
    decision:         str   = "UNKNOWN"
    attack_effective: bool  = False   # wrong decision AND undetected


@dataclass
class AttackerState:
    """Mutable runtime state of the adaptive attacker."""
    current_strategy:       str  = "threshold_evasion"
    total_intercepts:       int  = 0   # all intercept() calls
    successful_attacks:     int  = 0   # wrong decision achieved
    detections:             int  = 0   # detected by supervisor
    consecutive_detections: int  = 0
    consecutive_successes:  int  = 0
    history: List[AttackRecord] = field(default_factory=list)


#  Adaptive Attacker class 

class AdaptiveAttacker:
    """
    Stateful adaptive AiTM attacker with strategy switching.

    The attacker maintains state across runs to enable genuine
    adaptation: it remembers past detections and successes, and
    switches strategy when adaptation thresholds are met.

    Usage
    -----
    attacker = AdaptiveAttacker()

    # Before each attack run:
    falsified = attacker.intercept(real_data, last_verdict)

    # After each attack run:
    attacker.update_feedback(supervisor_verdict, scheduler_decision)

    # At the end of the experiment:
    stats = attacker.get_stats()
    """

    def __init__(
        self,
        initial_strategy: str = "threshold_evasion",
        detection_tolerance: int = 2,
        escalation_threshold: int = 3,
    ):
        """
        Args:
            initial_strategy      Starting attack strategy name.
            detection_tolerance   Consecutive detections before
                                  switching to a subtler strategy.
            escalation_threshold  Consecutive successes before
                                  escalating to a riskier strategy.
        """
        if initial_strategy not in STRATEGY_RISK:
            raise ValueError(
                f"Unknown strategy '{initial_strategy}'. "
                f"Valid options: {list(STRATEGY_RISK.keys())}"
            )
        self.state = AttackerState(current_strategy=initial_strategy)
        self.detection_tolerance  = detection_tolerance
        self.escalation_threshold = escalation_threshold
        self._cycle = 0

    #  Public interface 

    def intercept(
        self,
        data: dict,
        last_supervisor_verdict: Optional[str] = None,
    ) -> dict:
        """
        Intercept real sensor data and return falsified version.

        If no anomaly is present in the data, the data is returned
        unchanged (no attack is needed). The strategy adaptation
        step runs on every call regardless.

        Args:
            data                    Real sensor readings dict.
            last_supervisor_verdict "UNSAFE" if last attack was
                                    detected, "SAFE" otherwise.
                                    Pass None on the first call.

        Returns:
            Falsified sensor readings (or original if no anomaly).
        """
        self._cycle += 1
        self.state.total_intercepts += 1

        # Adapt strategy based on last run outcome
        self._adapt(last_supervisor_verdict)

        # Only falsify when there is an anomaly worth hiding
        if not self._anomaly_present(data):
            log.debug("Cycle %d: no anomaly present, passing through.",
                      self._cycle)
            return data

        # Apply current strategy
        falsified, orig_temp, fals_temp = self._apply_strategy(data)

        record = AttackRecord(
            cycle=self._cycle,
            strategy=self.state.current_strategy,
            original_temp=orig_temp,
            falsified_temp=fals_temp,
            temp_reduction=round(orig_temp - fals_temp, 2),
        )
        self.state.history.append(record)

        log.info(
            "Cycle %d | strategy=%-18s | "
            "orig_temp=%.1f -> fals_temp=%.1f | reduction=%.1f deg",
            self._cycle, self.state.current_strategy,
            orig_temp, fals_temp, orig_temp - fals_temp,
        )
        return falsified

    def update_feedback(
        self,
        supervisor_verdict: str,
        scheduler_decision: str,
    ) -> None:
        """
        Record the outcome of the last attack run.

        Must be called after each run that involved an actual attack
        (i.e., after intercept() returned falsified data).

        Args:
            supervisor_verdict  "UNSAFE" (detected) or "SAFE".
            scheduler_decision  The multi-agent group decision, e.g.
                                "CONTINUE", "MAINTENANCE".
        """
        if not self.state.history:
            return

        record = self.state.history[-1]
        detected       = (supervisor_verdict == VERDICT_DETECTED)
        wrong_decision = (scheduler_decision == DECISION_WRONG)

        record.detected         = detected
        record.decision         = scheduler_decision
        record.attack_effective = wrong_decision and not detected

        if detected:
            self.state.detections             += 1
            self.state.consecutive_detections += 1
            self.state.consecutive_successes   = 0
            log.info(
                "Attack DETECTED by supervisor. "
                "Consecutive detections: %d",
                self.state.consecutive_detections,
            )
        else:
            self.state.consecutive_detections = 0
            if wrong_decision:
                self.state.successful_attacks     += 1
                self.state.consecutive_successes  += 1
                log.info(
                    "Attack SUCCEEDED (decision=%s). "
                    "Consecutive successes: %d",
                    scheduler_decision,
                    self.state.consecutive_successes,
                )

    def get_stats(self) -> dict:
        """
        Return complete attacker statistics for experiment reporting.

        success_rate_pct and detection_rate_pct are computed over
        actual attack cycles (len(history)), not all intercept calls.
        This prevents inflated rates from no-anomaly pass-throughs.
        """
        attack_attempts = len(self.state.history)

        if attack_attempts == 0:
            return {
                "total_intercepts":  self.state.total_intercepts,
                "attack_attempts":   0,
                "successful_attacks": 0,
                "detections":         0,
                "success_rate_pct":   0.0,
                "detection_rate_pct": 0.0,
                "final_strategy":     self.state.current_strategy,
                "strategies_used":    {},
                "history":            [],
            }

        success_rate = round(
            self.state.successful_attacks / attack_attempts * 100, 1
        )
        detection_rate = round(
            self.state.detections / attack_attempts * 100, 1
        )

        strategy_counts: Dict[str, int] = {}
        for record in self.state.history:
            s = record.strategy
            strategy_counts[s] = strategy_counts.get(s, 0) + 1

        return {
            "total_intercepts":    self.state.total_intercepts,
            "attack_attempts":     attack_attempts,
            "successful_attacks":  self.state.successful_attacks,
            "detections":          self.state.detections,
            "success_rate_pct":    success_rate,
            "detection_rate_pct":  detection_rate,
            "final_strategy":      self.state.current_strategy,
            "strategies_used":     strategy_counts,
            "history": [
                {
                    "cycle":           r.cycle,
                    "strategy":        r.strategy,
                    "original_temp":   r.original_temp,
                    "falsified_temp":  r.falsified_temp,
                    "temp_reduction":  r.temp_reduction,
                    "detected":        r.detected,
                    "decision":        r.decision,
                    "effective":       r.attack_effective,
                }
                for r in self.state.history
            ],
        }

    def reset(self) -> None:
        """
        Reset attacker state while preserving the current strategy.
        Useful for running back-to-back experiments.
        """
        strategy = self.state.current_strategy
        self.state  = AttackerState(current_strategy=strategy)
        self._cycle = 0
        log.info("Attacker state reset. Retained strategy: %s", strategy)

    #  Strategy adaptation 

    def _adapt(self, last_verdict: Optional[str]) -> None:
        """
        Check detection/success counters and switch strategy
        if an adaptation threshold has been reached.
        """
        if last_verdict == VERDICT_DETECTED:
            self.state.consecutive_detections += 1
            self.state.consecutive_successes   = 0
        elif last_verdict == VERDICT_UNDETECTED:
            self.state.consecutive_detections  = 0

        if self.state.consecutive_detections >= self.detection_tolerance:
            self._switch_subtler()
            self.state.consecutive_detections = 0

        elif self.state.consecutive_successes >= self.escalation_threshold:
            self._switch_riskier()
            self.state.consecutive_successes = 0

    def _switch_subtler(self) -> None:
        """Switch to the lowest-risk strategy that is less risky than current."""
        current_risk = STRATEGY_RISK[self.state.current_strategy]
        candidates = [
            s for s, r in STRATEGY_RISK.items() if r < current_risk
        ]
        if candidates:
            new_strategy = min(candidates, key=lambda s: STRATEGY_RISK[s])
            log.info(
                "Switching to subtler strategy: %s -> %s",
                self.state.current_strategy, new_strategy,
            )
            self.state.current_strategy = new_strategy

    def _switch_riskier(self) -> None:
        """Escalate to a randomly selected higher-risk strategy."""
        current_risk = STRATEGY_RISK[self.state.current_strategy]
        candidates = [
            s for s, r in STRATEGY_RISK.items() if r > current_risk
        ]
        if candidates:
            new_strategy = random.choice(candidates)
            log.info(
                "Escalating to riskier strategy: %s -> %s",
                self.state.current_strategy, new_strategy,
            )
            self.state.current_strategy = new_strategy

    #  Anomaly detection 

    def _anomaly_present(self, data: dict) -> bool:
        """
        Return True if Machine_A has an anomaly worth attacking.
        Checks temperature, vibration, and status flag.
        """
        machine = data.get("Machine_A", {})
        return (
            machine.get("temperature", 0) > DETECTION_THRESHOLDS["temperature"]
            or machine.get("status", "") == "CRITICAL"
            or machine.get("vibration", 0) > DETECTION_THRESHOLDS["vibration"]
        )

    #  Strategy dispatch 

    def _apply_strategy(self, data: dict) -> tuple:
        """
        Dispatch to the current strategy implementation.

        Returns:
            (falsified_data, original_temp, falsified_temp)
        """
        dispatch = {
            "threshold_evasion": self._threshold_evasion,
            "gradual_drift":     self._gradual_drift,
            "selective_hiding":  self._selective_hiding,
            "noise_injection":   self._noise_injection,
        }
        return dispatch[self.state.current_strategy](data)

    #  Strategy implementations 

    def _threshold_evasion(self, data: dict) -> tuple:
        """
        Falsify temperature to just below the detection threshold.

        x_hat = threshold - U(1, 4)

        This keeps the reported value plausible (not suspiciously low)
        while passing the MonitorAgent rule (temp > 90 = CRITICAL).
        """
        falsified = copy.deepcopy(data)
        machine   = falsified["Machine_A"]
        orig_temp = data["Machine_A"].get("temperature", 0)

        thresh = DETECTION_THRESHOLDS["temperature"]
        machine["temperature"] = round(thresh - random.uniform(1.0, 4.0), 1)
        machine["vibration"]   = round(
            min(machine.get("vibration", 0),
                DETECTION_THRESHOLDS["vibration"] - 0.05), 2
        )
        machine["status"]          = "OPERATIONAL"
        machine["production_rate"] = round(
            DETECTION_THRESHOLDS["production_rate_min"]
            + random.uniform(5.0, 20.0), 1
        )
        falsified["Machine_A"] = machine
        return falsified, orig_temp, machine["temperature"]

    def _gradual_drift(self, data: dict) -> tuple:
        """
        Reduce temperature incrementally across cycles.

        x_hat(t) = x(t) - U(8, 15)

        Small per-step reductions (8–15 °C) appear less suspicious
        than an abrupt jump to a normal value.
        """
        falsified = copy.deepcopy(data)
        machine   = falsified["Machine_A"]
        orig_temp = data["Machine_A"].get("temperature", 0)

        reduction = random.uniform(8.0, 15.0)
        new_temp  = round(orig_temp - reduction, 1)
        machine["temperature"] = new_temp
        machine["status"] = (
            "WARNING" if new_temp > DETECTION_THRESHOLDS["temperature"]
            else "OPERATIONAL"
        )
        falsified["Machine_A"] = machine
        return falsified, orig_temp, new_temp

    def _selective_hiding(self, data: dict) -> tuple:
        """
        Full concealment only for severe anomalies (temp > 100 °C).

        For moderate anomalies (90–100 °C), falls back to
        threshold_evasion to reduce exposure.

        Full concealment: x_hat ~ U(65, 78)
        """
        falsified = copy.deepcopy(data)
        machine   = falsified["Machine_A"]
        orig_temp = data["Machine_A"].get("temperature", 0)

        if orig_temp > 100.0:
            new_temp = round(random.uniform(65.0, 78.0), 1)
            machine["temperature"]    = new_temp
            machine["vibration"]      = round(random.uniform(0.05, 0.30), 2)
            machine["status"]         = "OPERATIONAL"
            machine["production_rate"] = round(random.uniform(88.0, 97.0), 1)
        else:
            thresh   = DETECTION_THRESHOLDS["temperature"]
            new_temp = round(thresh - random.uniform(1.0, 4.0), 1)
            machine["temperature"] = new_temp
            machine["status"]      = "OPERATIONAL"

        falsified["Machine_A"] = machine
        return falsified, orig_temp, machine["temperature"]

    def _noise_injection(self, data: dict) -> tuple:
        """
        Add random noise to all machines to obscure the target anomaly.

        x_hat_i = x_i + N(0, sigma_i)  for all machines
        Machine_A temperature is additionally clamped below threshold.

        The distributed noise makes it harder to distinguish the
        targeted falsification from natural sensor variation.
        """
        falsified = copy.deepcopy(data)
        orig_temp = data["Machine_A"].get("temperature", 0)

        for readings in falsified.values():
            if not isinstance(readings, dict):
                continue
            if "temperature" in readings:
                readings["temperature"] = round(
                    readings["temperature"] + random.uniform(-3.0, 3.0), 1
                )
            if "vibration" in readings:
                readings["vibration"] = round(
                    max(0.0, readings["vibration"] + random.uniform(-0.04, 0.04)), 2
                )
            if "speed" in readings:
                readings["speed"] = round(
                    readings["speed"] + random.uniform(-2.0, 2.0), 1
                )

        machine = falsified["Machine_A"]
        thresh  = DETECTION_THRESHOLDS["temperature"]
        if machine.get("temperature", 0) > thresh:
            machine["temperature"] = round(
                thresh - random.uniform(2.0, 6.0), 1
            )
            machine["status"] = "OPERATIONAL"

        new_temp = machine["temperature"]
        falsified["Machine_A"] = machine
        return falsified, orig_temp, new_temp
