import json
import logging
import os
import time

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ── ThingsBoard MQTT configuration ────────────────────────
TB_HOST = os.getenv("THINGSBOARD_HOST", "localhost")
TB_PORT = int(os.getenv("THINGSBOARD_PORT", "1883"))
TB_TOPIC = "v1/devices/me/telemetry"

# Device tokens for dashboard streams
FALSIFIED_TOKEN  = os.getenv("MACHINE_A_FALSIFIED_TOKEN", "")
CASCADE_TOKEN    = os.getenv("CASCADE_STATUS_TOKEN", "")
ATTACK_TOKEN     = os.getenv("ATTACK_MONITOR_TOKEN", "")

# Agent contamination weights for visualisation
AGENT_POSITIONS = {
    "SensorAgent":    1,
    "MonitorAgent":   2,
    "SchedulerAgent": 3,
    "ActuatorAgent":  4,
    "SupervisorAgent": 5,
}


def _build_client(token: str, client_id: str) -> mqtt.Client:
    """Build and connect one MQTT client for a device token"""

    def on_connect(client, userdata, flags,
                   reason_code, properties):
        if reason_code == 0:
            log.debug(
                "Dashboard publisher connected: %s", client_id
            )
        else:
            log.warning(
                "Dashboard publisher connect failed: "
                "rc=%s id=%s", reason_code, client_id
            )

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=client_id,
    )
    client.username_pw_set(token)
    client.on_connect = on_connect

    try:
        client.connect(TB_HOST, TB_PORT, keepalive=60)
        client.loop_start()
        return client
    except Exception as exc:
        log.error(
            "Failed to connect dashboard client %s: %s",
            client_id, exc,
        )
        return None


class DashboardPublisher:
    """
    Publishes attack and cascade state to ThingsBoard
    for real-time dashboard visualisation.

    All publish methods are safe to call even if ThingsBoard
    is not running — they fail silently and log a warning,
    so the experiment continues uninterrupted.
    """

    def __init__(self):
        self._enabled = bool(
            FALSIFIED_TOKEN and CASCADE_TOKEN and ATTACK_TOKEN
        )

        if not self._enabled:
            log.info(
                "DashboardPublisher disabled: "
                "MACHINE_A_FALSIFIED_TOKEN, "
                "CASCADE_STATUS_TOKEN, or "
                "ATTACK_MONITOR_TOKEN not set in .env"
            )
            self._falsified_client = None
            self._cascade_client   = None
            self._attack_client    = None
            return

        self._falsified_client = _build_client(
            FALSIFIED_TOKEN, "falsified_publisher"
        )
        self._cascade_client = _build_client(
            CASCADE_TOKEN, "cascade_publisher"
        )
        self._attack_client = _build_client(
            ATTACK_TOKEN, "attack_publisher"
        )

        log.info("DashboardPublisher connected to ThingsBoard.")

    #  Public publish methods 

    def publish_falsified_data(
        self,
        original: dict,
        falsified: dict,
    ) -> None:
        """
        Publish real vs falsified Machine_A values.
        Allows dashboard to show the manipulation side by side.

        Args:
            original  : real sensor readings from simulator
            falsified : values sent to agents by attacker
        """
        if not self._enabled or not self._falsified_client:
            return

        orig_machine  = original.get("Machine_A", {})
        fals_machine  = falsified.get("Machine_A", {})

        payload = {
            # Real values
            "real_temperature":     orig_machine.get(
                "temperature", 0
            ),
            "real_vibration":       orig_machine.get(
                "vibration", 0
            ),
            "real_status":          orig_machine.get(
                "status", "UNKNOWN"
            ),
            "real_production_rate": orig_machine.get(
                "production_rate", 0
            ),
            # Falsified values seen by agents
            "falsified_temperature":     fals_machine.get(
                "temperature", 0
            ),
            "falsified_vibration":       fals_machine.get(
                "vibration", 0
            ),
            "falsified_status":          fals_machine.get(
                "status", "UNKNOWN"
            ),
            "falsified_production_rate": fals_machine.get(
                "production_rate", 0
            ),
            # Delta — how much was changed
            "temp_delta": round(
                orig_machine.get("temperature", 0) -
                fals_machine.get("temperature", 0),
                2,
            ),
            "attack_active": 1,
        }

        self._publish(self._falsified_client, payload)
        log.debug(
            "Published falsified data: real=%.1f "
            "falsified=%.1f delta=%.1f",
            payload["real_temperature"],
            payload["falsified_temperature"],
            payload["temp_delta"],
        )

    def publish_normal_state(self, data: dict) -> None:
        """
        Publish normal (no attack) state to dashboard.
        Resets falsified values to match real values.
        """
        if not self._enabled or not self._falsified_client:
            return

        machine = data.get("Machine_A", {})
        temp    = machine.get("temperature", 0)

        payload = {
            "real_temperature":          temp,
            "real_vibration":            machine.get(
                "vibration", 0
            ),
            "real_status":               machine.get(
                "status", "UNKNOWN"
            ),
            "real_production_rate":      machine.get(
                "production_rate", 0
            ),
            "falsified_temperature":     temp,
            "falsified_vibration":       machine.get(
                "vibration", 0
            ),
            "falsified_status":          machine.get(
                "status", "UNKNOWN"
            ),
            "falsified_production_rate": machine.get(
                "production_rate", 0
            ),
            "temp_delta":    0.0,
            "attack_active": 0,
        }

        self._publish(self._falsified_client, payload)

    def publish_cascade_state(
        self,
        decisions:    dict,
        topology:     str,
        run_number:   int,
        has_anomaly:  bool,
        under_attack: bool,
    ) -> None:
        """
        Publish per-agent contamination state to dashboard.

        Each agent is represented as a numeric field:
            0 = correct belief
            1 = false belief (contaminated)

        The dashboard can display this as a colour-coded
        agent pipeline showing cascade propagation in real time.

        Args:
            decisions    : dict of {agent_name: decision}
            topology     : current topology name
            run_number   : current run index
            has_anomaly  : whether anomaly was injected
            under_attack : whether AiTM attack was active
        """
        if not self._enabled or not self._cascade_client:
            return

        # h_i(t) per agent — 1 = false belief adopted
        contamination = {}
        for agent, decision in decisions.items():
            if has_anomaly and under_attack:
                h = 1 if decision == "CONTINUE" else 0
            else:
                h = 0
            contamination[agent] = h

        # Count contaminated agents
        n_contaminated = sum(contamination.values())
        n_total        = len(AGENT_POSITIONS)
        amplitude      = round(n_contaminated / n_total, 2)

        # Build propagation path string for display
        path_agents = [
            a for a in AGENT_POSITIONS
            if contamination.get(a, 0) == 1
        ]
        propagation_path = " -> ".join(path_agents) \
            if path_agents else "none"

        payload = {
            # Per-agent contamination state
            "sensor_agent_state":    contamination.get(
                "SensorAgent", 0
            ),
            "monitor_agent_state":   contamination.get(
                "MonitorAgent", 0
            ),
            "scheduler_agent_state": contamination.get(
                "SchedulerAgent", 0
            ),
            "actuator_agent_state":  contamination.get(
                "ActuatorAgent", 0
            ),
            "supervisor_agent_state": contamination.get(
                "SupervisorAgent", 0
            ),
            # Aggregate cascade metrics
            "cascade_amplitude":    amplitude,
            "n_contaminated":       n_contaminated,
            "propagation_path":     propagation_path,
            # Run metadata
            "topology":     topology,
            "run_number":   run_number,
            "has_anomaly":  int(has_anomaly),
            "under_attack": int(under_attack),
        }

        self._publish(self._cascade_client, payload)
        log.debug(
            "Published cascade state: topology=%s "
            "amplitude=%.2f path=%s",
            topology, amplitude, propagation_path,
        )

    def publish_attack_metadata(
        self,
        attacker,
        run_number: int,
        topology:   str,
        attack_succeeded: bool,
    ) -> None:
        """
        Publish attack performance metadata to dashboard.

        Args:
            attacker         : AdaptiveAttacker instance
            run_number       : current run index
            topology         : current topology name
            attack_succeeded : whether the attack worked
        """
        if not self._enabled or not self._attack_client:
            return

        stats = attacker.get_stats()

        # Map strategy name to numeric for dashboard gauge
        strategy_map = {
            "gradual_drift":     1,
            "noise_injection":   2,
            "threshold_evasion": 3,
            "selective_hiding":  4,
        }
        strategy_num = strategy_map.get(
            attacker.state.current_strategy, 0
        )

        payload = {
            "current_strategy":      attacker.state.current_strategy,
            "strategy_level":        strategy_num,
            "success_rate_pct":      stats.get(
                "success_rate_pct", 0
            ),
            "detection_rate_pct":    stats.get(
                "detection_rate_pct", 0
            ),
            "total_attempts":        stats.get(
                "attack_attempts", 0
            ),
            "successful_attacks":    stats.get(
                "successful_attacks", 0
            ),
            "detections":            stats.get("detections", 0),
            "attack_succeeded_now":  int(attack_succeeded),
            "run_number":            run_number,
            "topology":              topology,
            "consecutive_successes": (
                attacker.state.consecutive_successes
            ),
        }

        self._publish(self._attack_client, payload)
        log.debug(
            "Published attack metadata: "
            "strategy=%s success_rate=%.1f%%",
            attacker.state.current_strategy,
            payload["success_rate_pct"],
        )

    def disconnect(self) -> None:
        """Gracefully disconnect all MQTT clients"""
        for client in [
            self._falsified_client,
            self._cascade_client,
            self._attack_client,
        ]:
            if client:
                try:
                    client.loop_stop()
                    client.disconnect()
                except Exception:
                    pass
        log.info("DashboardPublisher disconnected.")

    #  Private helpers 

    def _publish(
        self,
        client: mqtt.Client,
        payload: dict,
    ) -> None:
        """Publish one payload, failing silently on error"""
        if not client:
            return
        try:
            result = client.publish(
                TB_TOPIC, json.dumps(payload)
            )
            if result.rc != 0:
                log.warning(
                    "MQTT publish failed rc=%s", result.rc
                )
        except Exception as exc:
            log.warning("Dashboard publish error: %s", exc)