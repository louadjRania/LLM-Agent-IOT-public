# LLM-Agent-IoT

Code and data for the paper **"Misinformation Propagation and Operational Risk in LLM-Based Multi-Agent IoT Systems"** (R. Louadj, D. Djenouri, M. Mumtaz, UWE Bristol).

The benchmark simulates an industrial IoT factory run by five role-specialised LLM agents (Sensor, Monitor, Scheduler, Actuator, Supervisor) under five communication topologies (linear, star, ring, tree, mesh). An adaptive Adversary-in-the-Middle attacker falsifies sensor telemetry before it reaches the first agent. The attack succeeds when the Scheduler selects `CONTINUE` while the machine is in a critical state.

> **Terminology note:** the cascades studied here originate from externally injected false sensor data (via the AiTM attacker), not from model-generated fabrications.

## Reproducing the paper's numbers and figures

No API keys are needed: every result in the paper is computed from the released run data.

```bash
pip install -r requirements.txt
./rebuild_analysis.sh        # all_runs.zip -> data/analysis/ (12 curated executions)
python paper_analysis.py     # data/analysis/ -> paper/ (tables, statistics, figures)
```

`paper/SUMMARY.md` lists every number reported in the paper. `data/analysis/README.md` documents which executions are used, which are excluded, and why.

**Failed runs.** A run in which both SchedulerAgent and SupervisorAgent returned no recognisable keyword is a failed model call (API error or timeout). These runs are excluded rather than counted as unsuccessful attacks.

**Repeated GPT-5.5 execution.** The original second GPT-5.5 execution produced no valid runs under tree and mesh (all model calls failed). It was repeated in full with the same code on 2026-09-25 and the repeat replaces it; the original is kept in `all_runs.zip`.

## Known limitations of the released code

These are described in the paper (Section IX) and will be corrected in a future release:

- In the tree topology, `TopologyRunner._next_receiver` returns only the first outgoing edge, so the Scheduler's decision reaches the Supervisor but not the Actuator.
- Agent decisions are extracted by keyword matching in a fixed priority order, which can misattribute an agent that quotes an action it rejects (notably the Supervisor).
- The task prompt given to SensorAgent includes a condition header.

## Project structure

| File | Role |
|---|---|
| `misinformation_cascade_benchmark.py` | Main experiment entry point |
| `adaptive_attacker.py` | Adaptive AiTM attacker (4 strategies) |
| `topology_manager.py` | Topology-constrained agent communication |
| `cascade_metrics.py` | Cascade metrics (contamination, impact P, distortion D) |
| `safeguard.py` | Defence mechanisms |
| `factory_simulator.py` | Virtual IoT factory (ThingsBoard + MQTT) |
| `model_factory.py` | Multi-LLM client factory |
| `physical_impact.py` | Recomputes P from `propagation_path` (removes star-hub double counting) |
| `rebuild_analysis.sh` | Builds the curated dataset from `all_runs.zip` |
| `paper_analysis.py` | All statistics, tables and figures in the paper |
| `all_runs.zip` | Raw output of every execution |

## Running new experiments (optional)

Copy `.env.example` to `.env` and fill in the API keys for the models you intend to run (`.env` is git-ignored). Then:

```bash
docker start thingsboard
python factory_simulator.py                                   # terminal 1
python misinformation_cascade_benchmark.py \
    --runs 50 --model gpt-5.5 --defense none                  # terminal 2
```

| `--model` | Model |
|---|---|
| `gpt-5.5` | OpenAI GPT-5.5 |
| `claude` | Anthropic Claude Sonnet 4.5 |
| `gemini-2.5` | Google Gemini 2.5 Flash |
| `gemini-3.5` | Google Gemini 3.5 Flash |

Defences (`--defense`): `none`, `majority_voting`, `reputation_system`, `confidence_weighted`, `trust_clipping`.
