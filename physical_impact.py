"""
physical_impact.py
=================

Recomputes the `physical_impact` field of every attack run from
`propagation_path` instead of `contaminated_agents`.

Rationale
---------
`contaminated_agents` records one entry per relayed message, so an agent
that forwards several messages is counted several times. In the star
topology the SupervisorAgent is the hub and therefore appears up to four
times in a single run, inflating the role-weighted impact score P.
`propagation_path`, produced by the same simulator, lists each agent once
and is the correct basis for P.

Verified on the released dataset: for all 1500 attack runs,
set(propagation_path) == set(contaminated_agents), and propagation_path
contains no duplicates.

Usage
-----
    python physical_impact.py --in data/results --out data/results_fixed
"""

import argparse
import json
import shutil
from pathlib import Path

# Safety-authority weights (Table II, IEC 61508 / ISA-95 ordering)
WEIGHTS = {
    "SensorAgent":     0.5,
    "MonitorAgent":    1.0,
    "SchedulerAgent":  2.0,
    "ActuatorAgent":   3.0,
    "SupervisorAgent": 1.5,
}


def recompute(record: dict) -> tuple[float, float]:
    """Return (old_P, new_P) for one run."""
    old = record.get("physical_impact", 0.0) or 0.0
    path = record.get("propagation_path") or []
    new = sum(WEIGHTS.get(a, 0.0) for a in path)
    return old, new


def process_file(src: Path, dst: Path) -> dict:
    data = json.loads(src.read_text())
    records = data["results"] if isinstance(data, dict) else data

    changed = 0
    for r in records:
        if not r.get("under_attack"):
            continue
        old, new = recompute(r)
        if abs(old - new) > 1e-9:
            changed += 1
        r["physical_impact_original"] = old   # keep an audit trail
        r["physical_impact"] = new

    if isinstance(data, dict):
        data.setdefault("metadata", {})["physical_impact_source"] = "propagation_path"

    dst.write_text(json.dumps(data, indent=1))
    return {"file": src.name, "changed": changed, "total": len(records)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", type=Path, default=Path("data/results"))
    ap.add_argument("--out", dest="dst", type=Path, default=Path("data/results_fixed"))
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    report = []
    for f in sorted(args.src.glob("cascade_results_*.json")):
        report.append(process_file(f, args.dst / f.name))

    # copy any non-result files through untouched
    for extra in args.src.glob("*.json"):
        if not extra.name.startswith("cascade_results_"):
            shutil.copy(extra, args.dst / extra.name)

    total = sum(r["changed"] for r in report)
    for r in report:
        print(f"  {r['file']:42s} {r['changed']:4d} runs corrected")
    print(f"\n{total} runs corrected in total -> {args.dst}")


if __name__ == "__main__":
    main()