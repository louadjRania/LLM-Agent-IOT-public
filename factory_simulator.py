"""
factory_simulator.py

IoT Smart Factory Simulator for Hallucination Cascade Research.

Simulates a 4-machine industrial environment by generating
realistic sensor telemetry and publishing it to ThingsBoard
via MQTT. Supports normal operation and anomaly injection
for controlled experimental conditions.

Machines simulated:
    Machine_A       : production machine (temperature, vibration)
    Conveyor_B      : conveyor belt (speed, load)
    Quality_Sensor_C: quality control (defect rate)
    Energy_Monitor_D: energy consumption (kW, efficiency)

Output:
    - MQTT telemetry to ThingsBoard (port 1883)
    - Shared JSON file: data/latest_data.json
      (read by experiment modules for agent input)

Usage:
    python factory_simulator.py
    python factory_simulator.py --anomaly-rate 0.1
"""

import argparse
import json
import logging
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

# ── Logging configuration ──────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Output directory ───────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
LATEST_DATA_FILE = DATA_DIR / "latest_data.json"


# ── Machine data generators ────────────────────────────────

@dataclass
class MachineConfig:
    """Configuration and thresholds for one machine"""
    name:           str
    token_env_key:  str
    normal_ranges:  dict
    anomaly_values: dict


MACHINE_CONFIGS = [
    MachineConfig(
        name="Machine_A",
        token_env_key="MACHINE_A_TOKEN",
        normal_ranges={
            "temperature":   (60.0, 82.0),
            "vibration":     (0.05, 0.45),
            "production_rate": (85.0, 100.0),
        },
        anomaly_values={
            "temperature":   (95.0, 120.0),
            "vibration":     (0.80, 1.60),
            "production_rate": (15.0, 40.0),
            "status":        "CRITICAL",
        }
    ),
    MachineConfig(
        name="Conveyor_B",
        token_env_key="CONVEYOR_B_TOKEN",
        normal_ranges={
            "speed": (48.0, 72.0),
            "load":  (35.0, 75.0),
        },
        anomaly_values={
            "speed": (8.0, 22.0),
            "load":  (88.0, 100.0),
            "status": "OVERLOADED",
        }
    ),
    MachineConfig(
        name="Quality_Sensor_C",
        token_env_key="QUALITY_SENSOR_C_TOKEN",
        normal_ranges={
            "defect_rate":   (0.005, 0.055),
            "units_checked": (80, 100),
        },
        anomaly_values={
            "defect_rate":   (0.15, 0.38),
            "units_checked": (80, 100),
            "status":        "ALERT",
        }
    ),
    MachineConfig(
        name="Energy_Monitor_D",
        token_env_key="ENERGY_MONITOR_D_TOKEN",
        normal_ranges={
            "consumption_kw": (75.0, 125.0),
            "efficiency":     (0.84, 0.99),
        },
        anomaly_values={
            "consumption_kw": (180.0, 260.0),
            "efficiency":     (0.38, 0.62),
            "status":         "HIGH_USAGE",
        }
    ),
]


def generate_reading(config: MachineConfig,
                     anomaly: bool = False) -> dict:
    """
    Generate one sensor reading for a machine.

    Args:
        config  : machine configuration
        anomaly : if True, generate anomalous values

    Returns:
        dict of sensor readings with status field
    """
    reading = {}

    if anomaly:
        values = config.anomaly_values
        for key, val in values.items():
            if key == "status":
                reading["status"] = val
            elif isinstance(val, tuple):
                lo, hi = val
                if isinstance(lo, float):
                    reading[key] = round(random.uniform(lo, hi), 3)
                else:
                    reading[key] = random.randint(lo, hi)
    else:
        for key, (lo, hi) in config.normal_ranges.items():
            if isinstance(lo, float):
                reading[key] = round(random.uniform(lo, hi), 3)
            else:
                reading[key] = random.randint(lo, hi)
        reading["status"] = "OPERATIONAL" \
            if config.name == "Machine_A" else "NORMAL" \
            if config.name in ["Conveyor_B", "Energy_Monitor_D"] \
            else "GOOD"

    return reading


# ── MQTT client factory ────────────────────────────────────

def build_mqtt_client(token: str,
                      host: str,
                      port: int) -> Optional[mqtt.Client]:
    """
    Build and connect one MQTT client for a device token.

    Args:
        token : ThingsBoard device access token
        host  : broker hostname
        port  : broker port

    Returns:
        Connected mqtt.Client or None on failure
    """
    def on_connect(client, userdata, flags,
                   reason_code, properties):
        if reason_code == 0:
            log.debug("MQTT connected: %s", token[:8])
        else:
            log.warning(
                "MQTT connect failed rc=%s token=%s",
                reason_code, token[:8]
            )

    def on_disconnect(client, userdata, flags,
                      reason_code, properties):
        if reason_code != 0:
            log.warning("Unexpected disconnect rc=%s", reason_code)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(token)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    try:
        client.connect(host, port, keepalive=60)
        client.loop_start()
        return client
    except Exception as exc:
        log.error("Failed to connect MQTT for token %s: %s",
                  token[:8], exc)
        return None


# ── Simulator class ────────────────────────────────────────

class FactorySimulator:
    """
    Runs the IoT smart factory simulation loop.

    Publishes sensor telemetry to ThingsBoard via MQTT
    and writes the latest readings to a shared JSON file
    for consumption by experiment modules.
    """

    MQTT_TOPIC = "v1/devices/me/telemetry"

    def __init__(
        self,
        host: str,
        port: int,
        interval_s: float = 3.0,
        anomaly_rate: float = 0.05,
    ):
        """
        Args:
            host         : ThingsBoard MQTT broker host
            port         : MQTT broker port
            interval_s   : seconds between telemetry cycles
            anomaly_rate : fraction of cycles with anomaly
                           (0.05 = 5% of cycles)
        """
        self.host         = host
        self.port         = port
        self.interval_s   = interval_s
        self.anomaly_rate = anomaly_rate
        self.clients:  dict = {}
        self.cycle:    int  = 0
        self.running:  bool = False

    def connect_all(self):
        """Connect MQTT clients for all configured machines"""
        log.info("Connecting to ThingsBoard at %s:%s",
                 self.host, self.port)

        for config in MACHINE_CONFIGS:
            token = os.getenv(config.token_env_key, "")
            if not token:
                log.warning(
                    "No token for %s (env: %s)",
                    config.name, config.token_env_key
                )
                continue

            client = build_mqtt_client(token, self.host, self.port)
            if client:
                self.clients[config.name] = (client, config)
                log.info("Connected: %s", config.name)

        if not self.clients:
            raise RuntimeError(
                "No MQTT clients connected. "
                "Check ThingsBoard is running and tokens are set."
            )

    def run(self):
        """Main simulation loop. Runs until interrupted."""
        self.running = True
        log.info(
            "Factory simulator started. "
            "Interval: %.1fs | Anomaly rate: %.0f%%",
            self.interval_s, self.anomaly_rate * 100
        )

        while self.running:
            self.cycle += 1
            self._run_cycle()
            time.sleep(self.interval_s)

    def _run_cycle(self):
        """Execute one telemetry cycle"""
        # Determine if this cycle has an anomaly
        has_anomaly = random.random() < self.anomaly_rate
        current_data = {}

        for machine_name, (client, config) in self.clients.items():
            # Only Machine_A gets anomalies in this study
            machine_anomaly = (
                has_anomaly and machine_name == "Machine_A"
            )
            reading = generate_reading(config, anomaly=machine_anomaly)
            current_data[machine_name] = reading

            payload = json.dumps(reading)
            result = client.publish(self.MQTT_TOPIC, payload)

            if result.rc != 0:
                log.warning(
                    "Publish failed for %s rc=%s",
                    machine_name, result.rc
                )

        # Write shared data file for experiment modules
        self._write_data_file(current_data)

        if has_anomaly:
            log.warning(
                "Cycle %d: ANOMALY active on Machine_A "
                "(temp=%.1f)",
                self.cycle,
                current_data.get("Machine_A", {}).get(
                    "temperature", 0
                )
            )
        else:
            log.info(
                "Cycle %d: normal | Machine_A temp=%.1f",
                self.cycle,
                current_data.get("Machine_A", {}).get(
                    "temperature", 0
                )
            )

    def _write_data_file(self, data: dict):
        """
        Write latest readings to shared JSON file.
        Uses atomic write (temp file + rename) to avoid
        partial reads by experiment modules.
        """
        tmp_file = LATEST_DATA_FILE.with_suffix(".tmp")
        try:
            with open(tmp_file, "w") as f:
                json.dump(data, f, indent=2)
            tmp_file.rename(LATEST_DATA_FILE)
        except Exception as exc:
            log.error("Failed to write data file: %s", exc)

    def stop(self):
        """Gracefully stop the simulation"""
        self.running = False
        for machine_name, (client, _) in self.clients.items():
            client.loop_stop()
            client.disconnect()
        log.info("Factory simulator stopped.")


# ── Entry point ────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IoT Smart Factory Simulator"
    )
    parser.add_argument(
        "--host",
        default=os.getenv("THINGSBOARD_HOST", "localhost"),
        help="ThingsBoard MQTT broker host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("THINGSBOARD_PORT", "1883")),
        help="MQTT broker port (default: 1883)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Telemetry interval in seconds (default: 3.0)"
    )
    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.05,
        help="Fraction of cycles with anomaly (default: 0.05)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    simulator = FactorySimulator(
        host=args.host,
        port=args.port,
        interval_s=args.interval,
        anomaly_rate=args.anomaly_rate,
    )

    # Graceful shutdown on Ctrl+C
    def handle_signal(sig, frame):
        log.info("Shutdown signal received.")
        simulator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        simulator.connect_all()
        simulator.run()
    except RuntimeError as exc:
        log.error("Startup failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()