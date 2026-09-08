# LLM-Agent-IoT 
# Misinformation Propagation and Operational Risk in LLM-Based Multi-Agent IoT Systems

## Research Question
How does network topology affect the speed, amplitude, and physical impact of misinformation cascades in LLM-based multi-agent IoT systems — and which defense mechanisms, if any, actually reduce attack success?

> **Terminology note:** the cascades studied here originate from
> externally injected false sensor data (via the adaptive AiTM attacker), not from model-generated fabrications. 

## Project Structure
misinformation_cascade_benchmark.py   — main experiment entry point (use this)
adaptive_attacker.py    — adaptive AiTM attacker (4 strategies)
cascade_metrics.py      — formal cascade metrics (speed, amplitude, distortion)
topology_manager.py     — structurally correct topology implementations
safeguard.py            — defense mechanisms (majority voting, reputation, etc.)
factory_simulator.py    — IoT virtual factory (ThingsBoard + MQTT)
model_factory.py        — unified multi-LLM client factory
analysis.py             — generates paper figures from results

## Models supported
| --model key    | Provider / model                            |
|----------------|---------------------------------------------|
| gpt-5.5        | OpenAI GPT-5.5                              |
| claude         | Anthropic Claude Sonnet 4.5                 |
| gemini-3.5     | Google Gemini 3.5 Flash                     |
| gemini-2.5     | Google Gemini 2.5 Flash.                    |

Set the matching API key in `.env`: `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`.

## Defense mechanisms (--defense)
- `none` — no defense (baseline)
- `majority_voting` — simple majority vote across agent decisions
- `reputation_system` — weights votes by per-agent historical reliability
- `confidence_weighted` — weights votes by agent-reported confidence
- `trust_clipping` — caps the influence any single agent can exert per round

## Startup sequence (every session)
1. Open Docker Desktop — wait for green engine
2. `docker start thingsboard`
3. `cd ~/Desktop/LLM-Agent-IOT && source venv/bin/activate`
4. Terminal 1: `python factory_simulator.py`
5. Terminal 2: `python misinformation_cascade_benchmark.py`
6. After experiment: `python analysis.py`

## Common run examples
```bash
# Default: gemini-3.5, no defense, all 5 topologies, N=4 runs/condition
python misinformation_cascade_benchmark.py

# Full paper run: N=30, Claude, trust clipping defense
python misinformation_cascade_benchmark.py \
    --runs 50 --model claude --defense trust_clipping

# Single topology sweep with a specific attack strategy
python misinformation_cascade_benchmark.py \
    --runs 50 --strategy gradual_drift --topologies star mesh --model gemini-3.5
```

## Docker commands
docker start thingsboard   # start
docker stop thingsboard    # stop
docker ps                  # check status
