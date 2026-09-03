#!/usr/bin/env bash
#
# rebuild_analysis.sh
# ===================
# Rebuilds a clean analysis dataset for LLM-Agent-IOT from all_runs.zip.
#
#   ./rebuild_analysis.sh
#
# Run it from the repository root (the folder containing all_runs.zip
# and physical_impact.py). Nothing existing is modified or deleted:
# data/results/ is left exactly as it is.
#
# What it produces:
#   data/analysis/*.json     12 curated files, physical_impact corrected
#   data/analysis/README.md  what was included, excluded, and why
#   data/master_table.csv    Model x Topology x Execution summary
#
set -euo pipefail

[[ -f all_runs.zip ]]        || { echo "ERROR: all_runs.zip not found. Run this from the repo root."; exit 1; }
[[ -f physical_impact.py ]]  || { echo "ERROR: physical_impact.py not found. Run this from the repo root."; exit 1; }

WORK=".rebuild_tmp"
rm -rf "$WORK" data/analysis
mkdir -p "$WORK/raw" data/analysis

echo "[1/4] Extracting the 12 genuine executions from all_runs.zip"
FILES=(
  20260714_195521   # GPT-5.5            exec 1
  20260716_214008   # GPT-5.5            exec 2
  20260715_155122   # Claude Sonnet 4.5  exec 1
  20260715_181031   # Claude Sonnet 4.5  exec 2
  20260714_185900   # Gemini 2.5 Flash   exec 1   <-- recovered, see README
  20260714_200243   # Gemini 2.5 Flash   exec 2
  20260715_142509   # Gemini 3.5 Flash   exec 1
  20260715_153624   # Gemini 3.5 Flash   exec 2
  20260716_163215   # defense: majority voting
  20260715_164233   # defense: reputation system
  20260715_180744   # defense: confidence weighted
  20260716_173637   # defense: trust clipping
)
for f in "${FILES[@]}"; do
  unzip -o -j -q all_runs.zip "cascade_results_$f.json" -d "$WORK/raw"
done
echo "      $(ls -1 "$WORK/raw" | wc -l) files extracted"

echo "[2/4] Correcting physical_impact (star-hub double counting)"
# NOTE: physical_impact.py only globs cascade_results_*.json, so the rename
# has to happen AFTER this step, never before.
python3 physical_impact.py --in "$WORK/raw" --out "$WORK/fixed"

echo "[3/4] Renaming into data/analysis/"
python3 - "$WORK/fixed" << 'PY'
import shutil, sys, pathlib
src = pathlib.Path(sys.argv[1])
M = {'20260714_195521':'gpt55_exec1',      '20260716_214008':'gpt55_exec2',
     '20260715_155122':'claude45_exec1',   '20260715_181031':'claude45_exec2',
     '20260714_185900':'gemini25_exec1',   '20260714_200243':'gemini25_exec2',
     '20260715_142509':'gemini35_exec1',   '20260715_153624':'gemini35_exec2',
     '20260716_163215':'defense_majority', '20260715_164233':'defense_reputation',
     '20260715_180744':'defense_confidence','20260716_173637':'defense_trust'}
for k, v in M.items():
    shutil.copy(src / f'cascade_results_{k}.json', f'data/analysis/{v}.json')
print(f'      {len(M)} files written')
PY

echo "[4/4] Building data/master_table.csv"
python3 - << 'PY'
import json, csv, statistics, math

TOPS = ['linear', 'star', 'ring', 'tree', 'mesh']
BASE = {'GPT-5.5':           ('gpt55_exec1',    'gpt55_exec2'),
        'Claude Sonnet 4.5': ('claude45_exec1', 'claude45_exec2'),
        'Gemini 2.5 Flash':  ('gemini25_exec1', 'gemini25_exec2'),
        'Gemini 3.5 Flash':  ('gemini35_exec1', 'gemini35_exec2')}
DEF  = {'Majority Voting': 'defense_majority', 'Reputation':     'defense_reputation',
        'Conf.-Weighted':  'defense_confidence','Trust Clipping': 'defense_trust'}

try:
    from scipy.stats import fisher_exact
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print('      WARNING: scipy not installed, fisher_p / holm_p left empty')
    print('               fix with: pip install scipy')

def load(name):
    with open(f'data/analysis/{name}.json') as fh:
        return json.load(fh)['results']

def cell(res, topo):
    a = [r for r in res if r['topology'] == topo and r['under_attack']]
    return (sum(1 for r in a if r['attack_succeeded']), len(a),
            statistics.mean(r['physical_impact'] for r in a))

def wilson(k, n, z=1.959963985):
    """Wilson score interval in percent (same formula as analysis.py:83)."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)

rows, pvals = [], []
for model, (f1, f2) in BASE.items():
    r1, r2 = load(f1), load(f2)
    for t in TOPS:
        s1, n1, p1 = cell(r1, t)
        s2, n2, p2 = cell(r2, t)
        lo, hi = wilson(s1 + s2, n1 + n2)
        pv = fisher_exact([[s1, n1 - s1], [s2, n2 - s2]])[1] if HAVE_SCIPY else ''
        pvals.append(pv)
        rows.append(dict(block='baseline', model=model, topology=t,
                         succ_e1=s1, n_e1=n1, succ_e2=s2, n_e2=n2,
                         rate_e1=round(100 * s1 / n1, 1), rate_e2=round(100 * s2 / n2, 1),
                         delta_pp=round(abs(100 * s1 / n1 - 100 * s2 / n2), 1),
                         rate_pooled=round(100 * (s1 + s2) / (n1 + n2), 1),
                         wilson_lo=round(lo, 1), wilson_hi=round(hi, 1),
                         P_e1=round(p1, 2), P_e2=round(p2, 2),
                         fisher_p=pv, holm_p=''))

# Holm step-down over the 20 baseline comparisons
if HAVE_SCIPY:
    order, running = sorted(range(len(pvals)), key=lambda i: pvals[i]), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        rows[i]['holm_p'] = running

for label, fname in DEF.items():
    res = load(fname)
    for t in TOPS:
        s, n, pi = cell(res, t)
        lo, hi = wilson(s, n)
        rows.append(dict(block='defense', model=label, topology=t,
                         succ_e1=s, n_e1=n, succ_e2='', n_e2='',
                         rate_e1=round(100 * s / n, 1), rate_e2='', delta_pp='',
                         rate_pooled=round(100 * s / n, 1),
                         wilson_lo=round(lo, 1), wilson_hi=round(hi, 1),
                         P_e1=round(pi, 2), P_e2='', fisher_p='', holm_p=''))

with open('data/master_table.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f'      {len(rows)} rows written')
PY

cat > data/analysis/README.md << 'EOF'
# Curated analysis dataset

Generated by `rebuild_analysis.sh` from `all_runs.zip`. Every file has been
passed through `physical_impact.py`: P is recomputed from `propagation_path`,
removing the star-hub double counting; `physical_impact_original` is kept as an
audit trail.

## Baseline: 2 independent executions per model, 25 attacked runs each

| File | Source timestamp | Model |
|---|---|---|
| gpt55_exec1.json | 20260714_195521 | GPT-5.5 |
| gpt55_exec2.json | 20260716_214008 | GPT-5.5 |
| claude45_exec1.json | 20260715_155122 | Claude Sonnet 4.5 |
| claude45_exec2.json | 20260715_181031 | Claude Sonnet 4.5 |
| gemini25_exec1.json | 20260714_185900 | Gemini 2.5 Flash (see OPEN ASSUMPTION) |
| gemini25_exec2.json | 20260714_200243 | Gemini 2.5 Flash |
| gemini35_exec1.json | 20260715_142509 | Gemini 3.5 Flash |
| gemini35_exec2.json | 20260715_153624 | Gemini 3.5 Flash |

## Defenses: Gemini 3.5 Flash, one execution, 25 attacked runs per topology

defense_majority.json, defense_reputation.json, defense_confidence.json,
defense_trust.json. The baseline for defense comparisons is
gemini35_exec1 + gemini35_exec2 pooled.

## OPEN ASSUMPTION

gemini25_exec1.json (20260714_185900) is treated as a valid independent
execution. It was recovered from all_runs.zip and had never been placed in
data/results/. Supporting evidence: 250 complete runs, 125 intercepts and 104
attack attempts (in range with every other file), no `checkpoint` flag, and a
47-minute duration ending 64 minutes before the next execution, matching the
end-then-relaunch pattern of Gemini 3.5 (6 min) and Claude (11 min).
This has NOT been confirmed against lab logs. If it proves to be a pilot run,
Gemini 2.5 has only ONE execution and must be excluded from the run-to-run
variability analysis.

## Excluded, and why

- cascade_results_gemini25_run1.json: byte-identical checkpoint copy of 20260714_200243
- cascade_results_gpt55_run1.json: checkpoint copy of 20260714_195521 (the original is used)
- statistics.json: stale per-batch artefact, contradicts data/figures/stats_tests.json
- all *.csv exports: incomplete (the Gemini 2.5 export stops at 36 of 250 rows)
EOF

rm -rf "$WORK"

echo
echo "Done."
echo "  data/analysis/       $(ls -1 data/analysis/*.json | wc -l) json files + README.md"
echo "  data/master_table.csv"
echo "  data/results/        untouched"