import argparse
import json
import logging
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except ImportError:  # graceful degradation: figures still work, tests skipped
    HAVE_SCIPY = False

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Output directory ───────────────────────────────────────────────────────
FIGURES_DIR = Path("data") / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Topology ordering ──────────────────────────────────────────────────────
TOPOLOGY_ORDER  = ["linear", "star", "ring", "tree", "mesh"]
TOPOLOGY_LABELS = ["Linear", "Star", "Ring", "Tree", "Mesh"]

TOPO_PALETTE = {
    "linear": "#2166AC",
    "star":   "#4DAC26",
    "ring":   "#D01C8B",
    "tree":   "#E66101",
    "mesh":   "#5E3C99",
}

MODEL_PALETTE = [
    "#1F4E79",   # Navy Blue
    "#5B6770",   # Slate Gray
    "#A6A6A6",   # Light Gray
    "#C44E52",   # Muted Red
    "#4DAC26",   # Muted Green
    "#E66101",   # Orange
    "#D01C8B",   # Magenta
    "#5E3C99",   # Purple
]

matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "legend.framealpha":  0.9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "grid.color":         "#cccccc",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})


# ══════════════════════════════════════════════════════════════════════════════
#  STATISTICS  (Wilson intervals, chi-square / Fisher tests)
# ══════════════════════════════════════════════════════════════════════════════

Z95 = 1.959963985  # two-sided 95% normal quantile


def wilson_ci(k: int, n: int, z: float = Z95) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion, returned in PERCENT.

    Preferred over the Wald interval because it never collapses to zero width
    at p = 0 or p = 1, and never produces bounds outside [0, 1].
    """
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom  = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half   = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return lo * 100.0, hi * 100.0


def bounded_err(values, stds, lo: float = 0.0,
                hi: Optional[float] = None) -> np.ndarray:
    """
    Asymmetric (2 x N) error bars for a bounded quantity.

    A symmetric mean +/- 1 SD bar is meaningless when it crosses a
    physical bound: cascade time and physical impact cannot be negative,
    and consensus distortion / cascade amplitude are defined on [0, 1].
    The whiskers are therefore truncated at the bounds.
    """
    v = np.asarray(values, dtype=float)
    s = np.asarray(stds,   dtype=float)
    lower = v - np.maximum(v - s, lo)
    upper = (np.minimum(v + s, hi) - v) if hi is not None else s
    return np.vstack([np.clip(lower, 0, None), np.clip(upper, 0, None)])


def yerr_wilson(agg: dict, keys: List[str]) -> np.ndarray:
    """Asymmetric (2 x N) error bars for matplotlib from Wilson bounds."""
    rates = np.array([agg[t]["attack_success_rate"] for t in keys], dtype=float)
    lo    = np.array([agg[t].get("ci_lo", r) for t, r in zip(keys, rates)], dtype=float)
    hi    = np.array([agg[t].get("ci_hi", r) for t, r in zip(keys, rates)], dtype=float)
    lower = np.clip(rates - lo, 0, None)
    upper = np.clip(hi - rates, 0, None)
    return np.vstack([lower, upper])


def contingency_from_agg(agg: dict, keys: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """Build a (topologies x [success, failure]) contingency table."""
    keys = keys or [t for t in TOPOLOGY_ORDER if t in agg]
    table = np.array(
        [[agg[t]["n_success"], agg[t]["n_attack"] - agg[t]["n_success"]]
         for t in keys],
        dtype=int,
    )
    return table, keys


def independence_test(table: np.ndarray) -> dict:
    """
    Test association in an R x 2 table.

    Uses chi-square when every EXPECTED count >= 5, otherwise falls back to
    Fisher's exact test (exact for 2x2, Monte-Carlo-free generalisation via
    scipy for larger tables when available).
    """
    out: dict = {
        "table": table.tolist(),
        "n_total": int(table.sum()),
        "test": None, "statistic": None, "dof": None, "p_value": None,
        "min_expected": None, "sparse": None,
    }
    if not HAVE_SCIPY:
        log.warning("scipy not installed - skipping significance tests "
                    "(pip install scipy)")
        return out

    # Drop all-zero rows, which break the chi-square expectation calculation
    table = table[table.sum(axis=1) > 0]
    if table.shape[0] < 2:
        return out

    chi2, p_chi, dof, expected = sps.chi2_contingency(table, correction=False)
    min_exp = float(expected.min())
    sparse  = min_exp < 5.0

    out.update(statistic=float(chi2), dof=int(dof),
               min_expected=min_exp, sparse=bool(sparse))

    if not sparse:
        out.update(test="chi2", p_value=float(p_chi))
    elif table.shape == (2, 2):
        odds, p_f = sps.fisher_exact(table)
        out.update(test="fisher_exact", p_value=float(p_f),
                   odds_ratio=float(odds), statistic=float(chi2))
    else:
        # R x 2 with sparse cells: Freeman-Halton style exact test
        try:
            res = sps.fisher_exact(table)          # scipy >= 1.15 supports RxC
            out.update(test="fisher_exact", p_value=float(res[1]))
        except (ValueError, NotImplementedError):
            # Fallback: chi-square with Yates-style caution flag
            out.update(test="chi2_sparse", p_value=float(p_chi))
    return out


def topology_independence_test(agg: dict, label: str = "") -> dict:
    """Does attack success depend on the network topology?"""
    table, keys = contingency_from_agg(agg)
    res = independence_test(table)
    res.update(model=label, topologies=keys)
    if res["p_value"] is not None:
        log.info("[%s] topology effect: %s stat=%.3f dof=%s p=%.4g%s",
                 label or "model", res["test"], res["statistic"],
                 res["dof"], res["p_value"],
                 "  (sparse cells -> exact test)" if res["sparse"] else "")
    return res


def pairwise_topology_tests(agg: dict, label: str = "") -> List[dict]:
    """All pairwise topology comparisons (2x2 Fisher), Holm-corrected."""
    keys = [t for t in TOPOLOGY_ORDER if t in agg]
    out: List[dict] = []
    if not HAVE_SCIPY:
        return out
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            table = np.array([
                [agg[a]["n_success"], agg[a]["n_attack"] - agg[a]["n_success"]],
                [agg[b]["n_success"], agg[b]["n_attack"] - agg[b]["n_success"]],
            ])
            odds, p = sps.fisher_exact(table)
            out.append({"model": label, "pair": [a, b],
                        "odds_ratio": float(odds), "p_value": float(p),
                        "table": table.tolist()})
    return holm_correct(out)


def defense_vs_baseline_test(base_agg: dict, def_agg: dict,
                             name: str) -> dict:
    """2x2 comparison of a defense against the no-defense baseline."""
    def totals(a):
        s = sum(a[t]["n_success"] for t in a)
        n = sum(a[t]["n_attack"] for t in a)
        return [s, n - s]

    table = np.array([totals(base_agg), totals(def_agg)])
    res = {"defense": name, "table": table.tolist()}
    if HAVE_SCIPY:
        odds, p = sps.fisher_exact(table)
        chi2, p_chi, dof, expected = sps.chi2_contingency(table, correction=False)
        sparse = bool(expected.min() < 5)
        res.update(odds_ratio=float(odds),
                   p_value=float(p) if sparse else float(p_chi),
                   test="fisher_exact" if sparse else "chi2",
                   p_fisher=float(p), p_chi2=float(p_chi),
                   sparse=sparse, min_expected=float(expected.min()))
        log.info("[defense: %-20s] %s p=%.4g  OR=%.3f",
                 name, res["test"], res["p_value"], res["odds_ratio"])
    return res


def holm_correct(results: List[dict], key: str = "p_value") -> List[dict]:
    """Holm-Bonferroni correction, added in place as 'p_holm' / 'significant'."""
    valid = [r for r in results if r.get(key) is not None]
    m = len(valid)
    if m == 0:
        return results
    order = sorted(valid, key=lambda r: r[key])
    running = 0.0
    for i, r in enumerate(order):
        adj = min(1.0, (m - i) * r[key])
        running = max(running, adj)          # enforce monotonicity
        r["p_holm"] = running
        r["significant"] = running < 0.05
    return results


def write_tables_csv(pooled_aggs: List[dict], model_names: List[str],
                     def_aggs: Optional[List[dict]] = None,
                     def_labels: Optional[List[str]] = None,
                     name: str = "tables.csv") -> Path:
    """
    Dump every per-condition number the paper reports into one CSV, so the
    text, the tables and the figures all quote the same source.
    """
    import csv
    path = FIGURES_DIR / name
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "model_or_defense", "topology", "n_attack",
                    "n_success", "success_rate_pct", "wilson_lo", "wilson_hi",
                    "cascade_time_s", "cascade_time_sd", "amplitude",
                    "physical_impact", "physical_impact_sd",
                    "consensus_distortion", "consensus_distortion_sd"])

        def rows(block, label, agg):
            for t in TOPOLOGY_ORDER:
                if t not in agg:
                    continue
                a = agg[t]
                w.writerow([block, label, t, a["n_attack"], a.get("n_success", ""),
                            f'{a["attack_success_rate"]:.1f}',
                            f'{a.get("ci_lo", 0):.1f}', f'{a.get("ci_hi", 0):.1f}',
                            f'{a["cascade_time_mean"]:.2f}', f'{a["cascade_time_std"]:.2f}',
                            f'{a["amplitude_mean"]:.3f}',
                            f'{a["impact_mean"]:.2f}', f'{a["impact_std"]:.2f}',
                            f'{a["distortion_mean"]:.3f}', f'{a["distortion_std"]:.3f}'])

        for agg, name_ in zip(pooled_aggs, model_names):
            rows("baseline", name_, agg)
        for agg, name_ in zip(def_aggs or [], def_labels or []):
            rows("defense", name_, agg)

    log.info("Per-condition table written to: %s", path)
    return path


def write_stats_report(payload: dict, name: str = "stats_tests.json") -> Path:
    """Dump every test result next to the figures, for the paper's tables."""
    path = FIGURES_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=float))
    log.info("Statistical tests written to: %s", path)
    return path


def run_all_tests(pooled_aggs: List[dict], model_names: List[str],
                  def_aggs: Optional[List[dict]] = None,
                  def_labels: Optional[List[str]] = None) -> dict:
    """Run topology + pairwise + defense tests and save them to JSON."""
    payload: dict = {"alpha": 0.05, "ci_method": "wilson_score",
                     "topology_tests": [], "pairwise_tests": [],
                     "defense_tests": []}

    omnibus = [topology_independence_test(a, n)
               for a, n in zip(pooled_aggs, model_names)]
    payload["topology_tests"] = holm_correct(omnibus)
    write_tables_csv(pooled_aggs, model_names, def_aggs, def_labels)

    for agg, name in zip(pooled_aggs, model_names):
        payload["pairwise_tests"] += pairwise_topology_tests(agg, name)

    if def_aggs and len(def_aggs) > 1:
        tests = [defense_vs_baseline_test(def_aggs[0], a, lab)
                 for a, lab in zip(def_aggs[1:], (def_labels or [])[1:])]
        payload["defense_tests"] = holm_correct(tests)

    write_stats_report(payload)
    return payload


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: Path) -> dict:
    """
    Load a JSON config file that declares models and their run files.

    Expected format:
    {
      "models": [
        {
          "name": "GPT-5.5",
          "runs": ["data/results/gpt55_run1.json",
                   "data/results/gpt55_run2.json"]
        },
        {
          "name": "Claude Sonnet 4.5",
          "runs": ["data/results/claude_run1.json",
                   "data/results/claude_run2.json"]
        }
      ],
      "defenses": [
        {
          "name": "Majority Voting",
          "file": "data/results/defense_majority.json"
        }
      ]
    }

    Every model must have 1 or 2 run files.
    If 2 runs are provided, stability figures are generated automatically.
    """
    raw = json.loads(config_path.read_text())
    return raw


def build_example_config(output_path: Path) -> None:
    """Write an example config file the user can fill in."""
    example = {
        "models": [
            {
                "name": "GPT-5.5",
                "runs": [
                    "data/results/gpt55_run1.json",
                    "data/results/gpt55_run2.json"
                ]
            },
            {
                "name": "Claude Sonnet 4.5",
                "runs": [
                    "data/results/claude_run1.json",
                    "data/results/claude_run2.json"
                ]
            },
            {
                "name": "Gemini 2.5 Flash",
                "runs": [
                    "data/results/gemini25_run1.json",
                    "data/results/gemini25_run2.json"
                ]
            },
            {
                "name": "Gemini 3.5 Flash",
                "runs": [
                    "data/results/gemini35_run1.json",
                    "data/results/gemini35_run2.json"
                ]
            }
        ],
        "defenses": [
            {"name": "None (baseline)",    "file": "data/results/gemini35_run1.json"},
            {"name": "Majority Voting",    "file": "data/results/defense_majority.json"},
            {"name": "Reputation System",  "file": "data/results/defense_reputation.json"},
            {"name": "Confidence Weighted","file": "data/results/defense_confidence.json"},
            {"name": "Trust Clipping",     "file": "data/results/defense_trust.json"}
        ]
    }
    output_path.write_text(json.dumps(example, indent=2))
    log.info("Example config written to: %s", output_path)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def load_results(path: Path) -> List[dict]:
    log.info("Loading: %s", path)
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "results" in raw:
        return raw["results"]
    return raw


def aggregate(results: List[dict]) -> Dict[str, dict]:
    """Per-topology statistics from attack runs only."""
    agg: Dict[str, dict] = {}
    for topo in TOPOLOGY_ORDER:
        attack = [r for r in results
                  if r["topology"] == topo and r["under_attack"]]
        normal = [r for r in results
                  if r["topology"] == topo and not r["under_attack"]]
        if not attack:
            continue

        def vals(key):
            return [r[key] for r in attack if r.get(key) is not None]

        def mean(key):
            v = vals(key)
            return statistics.mean(v) if v else 0.0

        def std(key):
            v = vals(key)
            return statistics.stdev(v) if len(v) > 1 else 0.0

        successes = [1 if r["attack_succeeded"] else 0 for r in attack]
        n = len(successes)
        n_success = sum(successes)
        rate = n_success / n * 100
        # 95% CI for the proportion — Wilson score interval (see wilson_ci)
        ci_lo, ci_hi = wilson_ci(n_success, n)

        agg[topo] = {
            "attack_success_rate":  rate,
            "n_success":            n_success,
            "ci_lo":                ci_lo,
            "ci_hi":                ci_hi,
            "attack_success_ci95":  (ci_hi - ci_lo) / 2,   # legacy symmetric field
            "attack_success_std":   statistics.stdev([x*100 for x in successes]) if len(attack) > 1 else 0.0,
            "n_attack":             n,
            "cascade_time_mean":    mean("cascade_time_s"),
            "cascade_time_std":     std("cascade_time_s"),
            "cascade_time_all":     vals("cascade_time_s"),
            "amplitude_all":        vals("cascade_amplitude"),
            "impact_all":           vals("physical_impact"),
            "distortion_all":       vals("consensus_distortion"),
            "amplitude_mean":       mean("cascade_amplitude"),
            "amplitude_std":        std("cascade_amplitude"),
            "impact_mean":          mean("physical_impact"),
            "impact_std":           std("physical_impact"),
            "distortion_mean":      mean("consensus_distortion"),
            "distortion_std":       std("consensus_distortion"),
            "false_rate_mean":      mean("false_action_rate"),
            "n_normal":             len(normal),
        }
    return agg


def pool_runs(run_aggs: List[Dict]) -> Dict[str, dict]:
    """
    Pool multiple runs of the SAME model by averaging per-topology metrics.
    Used to produce combined multi-model comparison figures.
    """
    if len(run_aggs) == 1:
        return run_aggs[0]

    pooled = {}
    for topo in TOPOLOGY_ORDER:
        entries = [a[topo] for a in run_aggs if topo in a]
        if not entries:
            continue
        # Pool the raw counts rather than averaging rates/intervals: the
        # Wilson interval is then recomputed on the full pooled sample.
        n_success_pooled = sum(e["n_success"] for e in entries)
        n_attack_pooled  = sum(e["n_attack"]  for e in entries)
        ci_lo, ci_hi = wilson_ci(n_success_pooled, n_attack_pooled)
        def cat(key):
            out = []
            for e in entries:
                out += e.get(key, [])
            return out

        def pooled_mean(key):
            v = cat(key)
            return statistics.mean(v) if v else 0.0

        def pooled_std(key):
            v = cat(key)
            return statistics.stdev(v) if len(v) > 1 else 0.0

        pooled[topo] = {
            "attack_success_rate": (n_success_pooled / n_attack_pooled * 100
                                    if n_attack_pooled else 0.0),
            "n_success":           n_success_pooled,
            "ci_lo":               ci_lo,
            "ci_hi":               ci_hi,
            "attack_success_ci95": (ci_hi - ci_lo) / 2,
            "attack_success_std": statistics.mean(
                [e["attack_success_std"] for e in entries]
            ),
            "n_attack": sum(e["n_attack"] for e in entries),
            "cascade_time_mean": pooled_mean("cascade_time_all"),
            "cascade_time_std":  pooled_std("cascade_time_all"),
            "cascade_time_all": cat("cascade_time_all"),
            "amplitude_mean": pooled_mean("amplitude_all"),
            "amplitude_std":  pooled_std("amplitude_all"),
            "amplitude_all": cat("amplitude_all"),
            "impact_mean": pooled_mean("impact_all"),
            "impact_std":  pooled_std("impact_all"),
            "impact_all": cat("impact_all"),
            "distortion_mean": pooled_mean("distortion_all"),
            "distortion_std":  pooled_std("distortion_all"),
            "distortion_all": cat("distortion_all"),
            "false_rate_mean": statistics.mean(
                [e["false_rate_mean"] for e in entries]
            ),
            "n_normal": sum(e["n_normal"] for e in entries),
        }
    return pooled


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def present_topologies(agg: dict) -> Tuple[list, list, list]:
    keys   = [t for t in TOPOLOGY_ORDER if t in agg]
    labels = [t.capitalize() for t in keys]
    colors = [TOPO_PALETTE[t] for t in keys]
    return keys, labels, colors


def label_bars(ax, bars, values, fmt="{:.1f}", fontsize=9, pad_frac=0.02):
    y_min, y_max = ax.get_ylim()
    pad = (y_max - y_min) * pad_frac
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + pad,
            fmt.format(val),
            ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold",
        )


def save_fig(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path)
    log.info("Saved: %s", path)
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-MODEL FIGURES  (figs 1–6)
# ══════════════════════════════════════════════════════════════════════════════

def fig1_attack_success(agg: dict, title_suffix: str = "") -> Path:
    keys, labels, colors = present_topologies(agg)
    values = [agg[t]["attack_success_rate"] for t in keys]
    errors = yerr_wilson(agg, keys)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors,
                  edgecolor="white", linewidth=0.8, width=0.55,
                  yerr=errors, capsize=5,
                  error_kw=dict(elinewidth=1.3, ecolor="#333333"))
    ax.axhline(50, color="#888888", linestyle="--",
               linewidth=1.2, alpha=0.7, label="chance level (50%)")
    ax.set_ylim(0, 112)
    label_bars(ax, bars, values, fmt="{:.0f}%", fontsize=9)
    title = "Attack Success Rate per Network Topology (Wilson 95% CI)"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_xlabel("Network Topology")
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()
    suffix = title_suffix.replace(" ", "_").lower()
    return save_fig(fig, f"fig1_attack_success{'_'+suffix if suffix else ''}.png")


def fig2_cascade_time(agg: dict, title_suffix: str = "") -> Path:
    keys, labels, colors = present_topologies(agg)
    means = [agg[t]["cascade_time_mean"] for t in keys]
    stds  = [agg[t]["cascade_time_std"]  for t in keys]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, means, color=colors, edgecolor="white", linewidth=0.8,
           width=0.55, yerr=bounded_err(means, stds, lo=0.0), capsize=5,
           error_kw=dict(elinewidth=1.3, ecolor="#333333"))
    ax.set_ylim(0, max(m+s for m,s in zip(means,stds))*1.25+1)
    title = "Mean Cascade Time per Topology (±1 SD)"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title)
    ax.set_ylabel("Cascade Time (seconds)")
    ax.set_xlabel("Network Topology")
    fig.tight_layout()
    suffix = title_suffix.replace(" ", "_").lower()
    return save_fig(fig, f"fig2_cascade_time{'_'+suffix if suffix else ''}.png")


def fig3_amplitude(agg: dict, title_suffix: str = "") -> Path:
    keys, labels, colors = present_topologies(agg)
    means = [agg[t]["amplitude_mean"] for t in keys]
    stds  = [agg[t]["amplitude_std"]  for t in keys]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, means, color=colors, edgecolor="white", linewidth=0.8,
           width=0.55, yerr=bounded_err(means, stds, lo=0.0, hi=1.0), capsize=5,
           error_kw=dict(elinewidth=1.3, ecolor="#333333"))
    ax.axhline(0.2, color="#888888", linestyle="--", linewidth=1.2,
               alpha=0.7, label="one agent (1/5)")
    ax.set_ylim(0, 0.5)
    title = "Cascade Amplitude A per Topology (±1 SD)"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title)
    ax.set_ylabel("Amplitude A (fraction of agents)")
    ax.set_xlabel("Network Topology")
    ax.legend(frameon=True)
    fig.tight_layout()
    suffix = title_suffix.replace(" ", "_").lower()
    return save_fig(fig, f"fig3_amplitude{'_'+suffix if suffix else ''}.png")


def fig4_physical_impact(agg: dict, title_suffix: str = "") -> Path:
    keys, labels, colors = present_topologies(agg)
    means = [agg[t]["impact_mean"] for t in keys]
    stds  = [agg[t]["impact_std"]  for t in keys]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y_pos = np.arange(len(keys))
    ax.barh(y_pos, means, color=colors, edgecolor="white", linewidth=0.8,
            height=0.5, xerr=bounded_err(means, stds, lo=0.0), capsize=5,
            error_kw=dict(elinewidth=1.3, ecolor="#333333"))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    x_max = max(m+s for m,s in zip(means,stds)) if means else 1
    for i, (m, y) in enumerate(zip(means, y_pos)):
        ax.text(m + x_max*0.03, y, f"{m:.2f}", va="center",
                fontsize=9, fontweight="bold")
    title = "Physical Impact Score P per Topology (±1 SD)"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title)
    ax.set_xlabel("P = Σ w_i · a_i(t)")
    ax.set_ylabel("Network Topology")
    fig.tight_layout()
    suffix = title_suffix.replace(" ", "_").lower()
    return save_fig(fig, f"fig4_physical_impact{'_'+suffix if suffix else ''}.png")


def fig5_distortion(agg: dict, title_suffix: str = "") -> Path:
    keys, labels, colors = present_topologies(agg)
    means = [agg[t]["distortion_mean"] for t in keys]
    stds  = [agg[t]["distortion_std"]  for t in keys]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, means, color=colors, edgecolor="white", linewidth=0.8,
           width=0.55, yerr=bounded_err(means, stds, lo=0.0, hi=1.0), capsize=5,
           error_kw=dict(elinewidth=1.3, ecolor="#333333"))
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="#CC2222", linestyle="--",
               linewidth=1.2, alpha=0.6, label="D=1.0 (CONTINUE)")
    title = "Consensus Distortion D per Topology (±1 SD)"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title)
    ax.set_ylabel("Distortion D  (0=safe, 1=CONTINUE)")
    ax.set_xlabel("Network Topology")
    ax.legend(frameon=True)
    fig.tight_layout()
    suffix = title_suffix.replace(" ", "_").lower()
    return save_fig(fig, f"fig5_distortion{'_'+suffix if suffix else ''}.png")


def fig6_propagation_grouped(results, title_suffix: str = "") -> Path:
    """
    Contamination count per agent role.

    `results` may be a single run (list of records) or a list of runs;
    all runs are pooled. Each agent is counted at most ONCE per run:
    contaminated_agents holds one entry per relayed message, so a hub
    agent would otherwise be counted several times for the same run.
    """
    from collections import Counter
    if results and isinstance(results[0], list):
        results = [rec for run in results for rec in run]
    topos = [t for t in TOPOLOGY_ORDER
             if any(r["topology"] == t and r["under_attack"] for r in results)]
    agent_order = ["SensorAgent","MonitorAgent","SchedulerAgent",
                   "ActuatorAgent","SupervisorAgent"]
    fig, axes = plt.subplots(1, len(topos), figsize=(3.5*len(topos), 4.5),
                             sharey=True)
    if len(topos) == 1:
        axes = [axes]
    for ax, topo in zip(axes, topos):
        attack_runs = [r for r in results
                       if r["topology"] == topo and r["under_attack"]
                       and r.get("attack_succeeded")]
        counter = Counter()
        for r in attack_runs:
            agents = r.get("propagation_path") or r.get("contaminated_agents", [])
            for agent in set(agents):          # once per run
                counter[agent] += 1
        counts = [counter.get(a, 0) for a in agent_order]
        short  = [a.replace("Agent", "") for a in agent_order]
        bars = ax.bar(short, counts, color=TOPO_PALETTE[topo],
                      edgecolor="white", linewidth=0.8, width=0.6)
        ax.set_title(topo.capitalize(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Agent Role", fontsize=9)
        for bar, c in zip(bars, counts):
            if c > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                        str(c), ha="center", va="bottom",
                        fontsize=8, fontweight="bold")
    axes[0].set_ylabel("Contamination Count")
    title = "Agent Contamination by Topology (successful attacks)"
    if title_suffix:
        title += f"\n{title_suffix}"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    suffix = title_suffix.replace(" ", "_").lower()
    return save_fig(fig, f"fig6_propagation{'_'+suffix if suffix else ''}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-MODEL COMPARISON FIGURES  (figs 7–11)
# ══════════════════════════════════════════════════════════════════════════════

def fig7_multimodel_success(all_agg, model_labels, common_keys) -> Path:
    x     = np.arange(len(common_keys))
    width = 0.8 / len(all_agg)
    clabels = [t.capitalize() for t in common_keys]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (agg, label) in enumerate(zip(all_agg, model_labels)):
        vals   = [agg[t]["attack_success_rate"] for t in common_keys]
        errors = yerr_wilson(agg, common_keys)
        offset = (i - len(all_agg)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width*0.9,
                      label=label, color=MODEL_PALETTE[i % len(MODEL_PALETTE)],
                      edgecolor="white", linewidth=0.7,
                      yerr=errors, capsize=3,
                      error_kw=dict(elinewidth=1.0, ecolor="#333333"))
        label_bars(ax, bars, vals, fmt="{:.0f}%", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(clabels, fontsize=10)
    ax.set_ylim(0, 118)
    ax.axhline(50, color="#888", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_xlabel("Network Topology")
    ax.set_title("Attack Success Rate: Model Comparison per Topology\n(Wilson 95% CI)")
    ax.legend(frameon=True, loc="upper right", fontsize=9,
              bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return save_fig(fig, "fig7_multimodel_success.png")


def fig8_multimodel_impact(all_agg, model_labels, common_keys) -> Path:
    x     = np.arange(len(common_keys))
    width = 0.8 / len(all_agg)
    clabels = [t.capitalize() for t in common_keys]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (agg, label) in enumerate(zip(all_agg, model_labels)):
        vals = [agg[t]["impact_mean"] for t in common_keys]
        errs = bounded_err(vals, [agg[t]["impact_std"] for t in common_keys], lo=0.0)
        offset = (i - len(all_agg)/2 + 0.5) * width
        ax.bar(x + offset, vals, width*0.9,
               label=label, color=MODEL_PALETTE[i % len(MODEL_PALETTE)],
               edgecolor="white", linewidth=0.7,
               yerr=errs, capsize=3,
               error_kw=dict(elinewidth=1.0, ecolor="#333333"))
    ax.set_xticks(x)
    ax.set_xticklabels(clabels, fontsize=10)
    ax.set_ylabel("Physical Impact P")
    ax.set_xlabel("Network Topology")
    ax.set_title("Physical Impact Score P: Model Comparison per Topology")
    ax.legend(frameon=True, loc="upper right", fontsize=9)
    fig.tight_layout()
    return save_fig(fig, "fig8_multimodel_impact.png")


def fig9_multimodel_time(all_agg, model_labels, common_keys) -> Path:
    x     = np.arange(len(common_keys))
    width = 0.8 / len(all_agg)
    clabels = [t.capitalize() for t in common_keys]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (agg, label) in enumerate(zip(all_agg, model_labels)):
        vals = [agg[t]["cascade_time_mean"] for t in common_keys]
        errs = bounded_err(vals, [agg[t]["cascade_time_std"] for t in common_keys], lo=0.0)
        offset = (i - len(all_agg)/2 + 0.5) * width
        ax.bar(x + offset, vals, width*0.9,
               label=label, color=MODEL_PALETTE[i % len(MODEL_PALETTE)],
               edgecolor="white", linewidth=0.7,
               yerr=errs, capsize=3,
               error_kw=dict(elinewidth=1.0, ecolor="#333333"))
    ax.set_xticks(x)
    ax.set_xticklabels(clabels, fontsize=10)
    ax.set_ylabel("Cascade Time (seconds)")
    ax.set_xlabel("Network Topology")
    ax.set_title("Cascade Time: Model Comparison per Topology")
    ax.legend(frameon=True, loc="upper right", fontsize=9)
    fig.tight_layout()
    return save_fig(fig, "fig9_multimodel_time.png")


def fig10_multimodel_distortion(all_agg, model_labels, common_keys) -> Path:
    x     = np.arange(len(common_keys))
    width = 0.8 / len(all_agg)
    clabels = [t.capitalize() for t in common_keys]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (agg, label) in enumerate(zip(all_agg, model_labels)):
        vals = [agg[t]["distortion_mean"] for t in common_keys]
        errs = bounded_err(vals, [agg[t]["distortion_std"] for t in common_keys], lo=0.0, hi=1.0)
        offset = (i - len(all_agg)/2 + 0.5) * width
        ax.bar(x + offset, vals, width*0.9,
               label=label, color=MODEL_PALETTE[i % len(MODEL_PALETTE)],
               edgecolor="white", linewidth=0.7,
               yerr=errs, capsize=3,
               error_kw=dict(elinewidth=1.0, ecolor="#333333"))
    ax.set_xticks(x)
    ax.set_xticklabels(clabels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Consensus Distortion D")
    ax.set_xlabel("Network Topology")
    ax.set_title("Consensus Distortion D: Model Comparison per Topology")
    ax.legend(frameon=True, loc="upper right", fontsize=9)
    fig.tight_layout()
    return save_fig(fig, "fig10_multimodel_distortion.png")


def fig11_radar(all_agg, model_labels) -> Path:
    metrics  = ["Attack\nSuccess", "Cascade\nAmplitude",
                 "Physical\nImpact", "Cascade\nTime", "Consensus\nDistortion"]
    n_metrics = len(metrics)
    angles = [n / float(n_metrics) * 2 * math.pi for n in range(n_metrics)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw=dict(polar=True))

    for i, (agg, label) in enumerate(zip(all_agg, model_labels)):
        topos_present = [t for t in TOPOLOGY_ORDER if t in agg]
        if not topos_present:
            continue

        def norm_mean(key, scale):
            v = [agg[t][key] for t in topos_present]
            return statistics.mean(v) / scale if v else 0.0

        vals = [
            norm_mean("attack_success_rate", 100),
            norm_mean("amplitude_mean",      1.0),
            norm_mean("impact_mean",         15.0),
            norm_mean("cascade_time_mean",   30.0),
            norm_mean("distortion_mean",     1.0),
        ]
        vals = [min(1.0, max(0.0, v)) for v in vals]
        vals += vals[:1]

        color = MODEL_PALETTE[i % len(MODEL_PALETTE)]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=label)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8,
                       color="#666666")
    ax.set_title(
        "Global Metric Profile per Model\n"
        "(each axis normalised to its own scale; time is not a vulnerability axis)",
        fontsize=12, fontweight="bold", pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10),
              frameon=True, fontsize=10)
    fig.tight_layout()
    return save_fig(fig, "fig11_radar.png")


# ══════════════════════════════════════════════════════════════════════════════
#  STABILITY FIGURES  (one per model + global summary)
# ══════════════════════════════════════════════════════════════════════════════

def fig_stability_per_model(
    agg_a: dict, agg_b: dict,
    model_name: str,
    fig_index: int,
) -> Path:
    """
    Two-panel stability figure for ONE model.
    Left : grouped bars run1 vs run2 per topology.
    Right: |delta| per topology.
    """
    keys = [t for t in TOPOLOGY_ORDER if t in agg_a and t in agg_b]
    labels = [t.capitalize() for t in keys]

    r1 = [agg_a[t]["attack_success_rate"] for t in keys]
    r2 = [agg_b[t]["attack_success_rate"] for t in keys]

    x = np.arange(len(keys))
    w = 0.32
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left — run1 vs run2
    ax = axes[0]
    b1 = ax.bar(x - w/2, r1, w, label="Run 1", color="#1F4E79",
                edgecolor="white", linewidth=0.7)
    b2 = ax.bar(x + w/2, r2, w, label="Run 2", color="#C44E52",
                edgecolor="white", linewidth=0.7)
    label_bars(ax, b1, r1, fmt="{:.0f}%", fontsize=8)
    label_bars(ax, b2, r2, fmt="{:.0f}%", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 112)
    ax.axhline(50, color="#888", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_xlabel("Network Topology")
    ax.set_title(f"Run-to-Run Comparison\n{model_name}")
    ax.legend(frameon=True, loc="upper right", fontsize=9)

    # Right — |delta|
    ax2 = axes[1]
    deltas = [abs(a - b) for a, b in zip(r1, r2)]
    mean_d = statistics.mean(deltas) if deltas else 0.0
    bars_d = ax2.bar(labels, deltas, color="#5B6770",
                     edgecolor="white", linewidth=0.7, width=0.55)
    label_bars(ax2, bars_d, deltas, fmt="{:.0f} pp", fontsize=8)
    ax2.axhline(mean_d, color="#E66101", linestyle="--",
                linewidth=1.2, alpha=0.8,
                label=f"Mean |Δ| = {mean_d:.1f} pp")
    # Stability threshold line at 10pp
    ax2.axhline(10, color="#2166AC", linestyle=":", linewidth=1.0,
                alpha=0.7, label="10 pp threshold")
    ax2.set_ylim(0, max(deltas + [12]) * 1.3)
    ax2.set_ylabel("|Run 1 − Run 2|  (percentage points)")
    ax2.set_xlabel("Network Topology")
    ax2.set_title("Run-to-Run Variance per Topology")
    ax2.legend(frameon=True, loc="upper right", fontsize=9)

    fig.tight_layout()
    fname = f"fig{fig_index}_stability_{model_name.replace(' ','_').lower()}.png"
    return save_fig(fig, fname)


def fig_stability_summary(model_names, all_run1_agg, all_run2_agg,
                           fig_index: int) -> Path:
    """
    Heatmap summary: |delta| (pp) for every model × topology combination.
    Immediately shows which models are stable and which are not.
    """
    topos  = TOPOLOGY_ORDER
    t_labels = [t.capitalize() for t in topos]
    n_models = len(model_names)

    # Build matrix  [model, topology]
    matrix = np.zeros((n_models, len(topos)))
    for mi, (a1, a2) in enumerate(zip(all_run1_agg, all_run2_agg)):
        for ti, topo in enumerate(topos):
            if topo in a1 and topo in a2:
                matrix[mi, ti] = abs(
                    a1[topo]["attack_success_rate"] -
                    a2[topo]["attack_success_rate"]
                )

    fig, ax = plt.subplots(figsize=(9, 0.8 + 0.9 * n_models))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=30)
    plt.colorbar(im, ax=ax, label="|Run 1 − Run 2|  (pp)")

    ax.set_xticks(range(len(topos)))
    ax.set_xticklabels(t_labels, fontsize=11)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_names, fontsize=11)

    # Annotate cells
    for mi in range(n_models):
        for ti in range(len(topos)):
            val = matrix[mi, ti]
            color = "white" if val > 18 else "black"
            ax.text(ti, mi, f"{val:.0f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    ax.set_title(
        "Stability Heatmap: |Run 1 − Run 2| per Model × Topology (pp)\n"
        "Darker = less stable  |  Values > 10 pp warrant caution",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return save_fig(fig, f"fig{fig_index}_stability_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
#  DEFENSE FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def fig_defense_comparison(defense_aggs: List[dict],
                            defense_labels: List[str],
                            fig_index: int) -> Path:
    """
    Grouped bar chart: attack success rate per topology per defense mechanism.
    Includes the baseline (no defense) as the first group.
    """
    common_keys = [t for t in TOPOLOGY_ORDER
                   if all(t in agg for agg in defense_aggs)]
    clabels = [t.capitalize() for t in common_keys]
    x     = np.arange(len(common_keys))
    width = 0.8 / len(defense_aggs)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (agg, label) in enumerate(zip(defense_aggs, defense_labels)):
        vals   = [agg[t]["attack_success_rate"] for t in common_keys]
        errors = yerr_wilson(agg, common_keys)
        offset = (i - len(defense_aggs)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width*0.9,
                      label=label,
                      color=MODEL_PALETTE[i % len(MODEL_PALETTE)],
                      edgecolor="white", linewidth=0.7,
                      yerr=errors, capsize=3,
                      error_kw=dict(elinewidth=1.0, ecolor="#333333"))
        label_bars(ax, bars, vals, fmt="{:.0f}%", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(clabels, fontsize=10)
    ax.set_ylim(0, 118)
    ax.axhline(50, color="#888", linestyle="--", linewidth=1.0,
               alpha=0.6, label="chance level (50%)")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_xlabel("Network Topology")
    ax.set_title(
        "Defense Mechanism Comparison: Attack Success Rate per Topology\n"
        "(Gemini 3.5 Flash, Wilson 95% CI)"
    )
    ax.legend(frameon=True, loc="upper right", fontsize=9,
              bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return save_fig(fig, f"fig{fig_index}_defense_comparison.png")


def fig_defense_mean_bar(defense_aggs: List[dict],
                          defense_labels: List[str],
                          fig_index: int) -> Path:
    """
    Horizontal bar: mean attack success across all topologies per defense.
    Makes the counter-intuitive defense result immediately visible.
    """
    means = []
    for agg in defense_aggs:
        rates = [agg[t]["attack_success_rate"]
                 for t in TOPOLOGY_ORDER if t in agg]
        means.append(statistics.mean(rates) if rates else 0.0)

    colors = [MODEL_PALETTE[i % len(MODEL_PALETTE)]
              for i in range(len(defense_labels))]

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(defense_labels))
    bars = ax.barh(y_pos, means, color=colors,
                   edgecolor="white", linewidth=0.8, height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(defense_labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.axvline(means[0], color="#E66101", linestyle="--",
               linewidth=1.5, alpha=0.8,
               label=f"Baseline = {means[0]:.1f}%")
    for bar, val in zip(bars, means):
        ax.text(val + 1.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Mean Attack Success Rate (%) across all topologies")
    ax.set_title(
        "Mean Attack Success Rate per Defense Mechanism\n"
        "(Gemini 3.5 Flash — lower is better)"
    )
    ax.legend(frameon=True)
    fig.tight_layout()
    return save_fig(fig, f"fig{fig_index}_defense_mean.png")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main_from_config(config_path: Path) -> None:
    """
    Main entry point when using a config file.
    Generates all figures automatically based on declared models and runs.
    """
    cfg = load_config(config_path)
    models   = cfg.get("models", [])
    defenses = cfg.get("defenses", [])

    if not models:
        log.error("Config has no 'models' entries.")
        return

    # ── Load and aggregate all runs ────────────────────────────────────────
    model_run_aggs = []  # list of lists: [model][run] = agg
    model_names    = []

    for m in models:
        name = m["name"]
        runs = m["runs"]
        model_names.append(name)
        run_aggs = []
        for rpath in runs:
            results = load_results(Path(rpath))
            run_aggs.append(aggregate(results))
        model_run_aggs.append(run_aggs)

    # Pooled agg per model (counts summed across runs) for comparison figures
    pooled_aggs = [pool_runs(run_aggs) for run_aggs in model_run_aggs]

    stats_def_aggs: Optional[List[dict]] = None
    stats_def_labels: Optional[List[str]] = None

    saved = []
    fig_idx = 1

    # ── Figs 1–6: single-model figures for FIRST model only ───────────────
    first_results = [load_results(Path(rp)) for rp in models[0]["runs"]]
    saved += [
        fig1_attack_success(pooled_aggs[0],   title_suffix=model_names[0]),
        fig2_cascade_time(pooled_aggs[0],      title_suffix=model_names[0]),
        fig3_amplitude(pooled_aggs[0],         title_suffix=model_names[0]),
        fig4_physical_impact(pooled_aggs[0],   title_suffix=model_names[0]),
        fig5_distortion(pooled_aggs[0],        title_suffix=model_names[0]),
        fig6_propagation_grouped(first_results, title_suffix=model_names[0]),
    ]
    fig_idx = 7

    # ── Figs 7–11: multi-model comparison using pooled runs ───────────────
    common_keys = [t for t in TOPOLOGY_ORDER
                   if all(t in agg for agg in pooled_aggs)]
    if common_keys:
        saved += [
            fig7_multimodel_success(pooled_aggs,    model_names, common_keys),
            fig8_multimodel_impact(pooled_aggs,     model_names, common_keys),
            fig9_multimodel_time(pooled_aggs,       model_names, common_keys),
            fig10_multimodel_distortion(pooled_aggs, model_names, common_keys),
            fig11_radar(pooled_aggs,                model_names),
        ]
    fig_idx = 12

    # ── Stability figures: one per model that has 2 runs ──────────────────
    has_stability = []
    for mi, (name, run_aggs) in enumerate(zip(model_names, model_run_aggs)):
        if len(run_aggs) >= 2:
            saved.append(fig_stability_per_model(
                run_aggs[0], run_aggs[1],
                model_name=name,
                fig_index=fig_idx,
            ))
            has_stability.append((name, run_aggs[0], run_aggs[1]))
            fig_idx += 1

    # ── Stability summary heatmap (all models with 2 runs) ────────────────
    if len(has_stability) >= 2:
        h_names = [h[0] for h in has_stability]
        h_r1    = [h[1] for h in has_stability]
        h_r2    = [h[2] for h in has_stability]
        saved.append(fig_stability_summary(h_names, h_r1, h_r2, fig_idx))
        fig_idx += 1

    # ── Defense figures ────────────────────────────────────────────────────
    if defenses:
        def_aggs   = []
        def_labels = []
        for d in defenses:
            results = load_results(Path(d["file"]))
            def_aggs.append(aggregate(results))
            def_labels.append(d["name"])

        stats_def_aggs, stats_def_labels = def_aggs, def_labels

        saved.append(fig_defense_comparison(def_aggs, def_labels, fig_idx))
        fig_idx += 1
        saved.append(fig_defense_mean_bar(def_aggs, def_labels, fig_idx))
        fig_idx += 1

    run_all_tests(pooled_aggs, model_names,
                  def_aggs=stats_def_aggs, def_labels=stats_def_labels)

    log.info("\nAll figures saved to: %s", FIGURES_DIR)
    for p in saved:
        log.info("  %s", p.name)


def main_legacy(
    results_paths=None,
    model_labels=None,
    stability_pair=None,
    stability_model_name="",
) -> None:
    """Legacy mode: same interface as before for backward compatibility."""
    if results_paths:
        all_results = [load_results(p) for p in results_paths]
    else:
        candidates = sorted(Path("data/results").glob("cascade_results_*.json"))
        if not candidates:
            raise FileNotFoundError("No results file found in data/results/")
        all_results = [load_results(candidates[-1])]

    all_agg = [aggregate(r) for r in all_results]
    if not all_agg[0]:
        log.error("No attack runs found.")
        return

    saved = []
    suffix = model_labels[0] if model_labels and len(model_labels) == 1 else ""
    saved += [
        fig1_attack_success(all_agg[0],        title_suffix=suffix),
        fig2_cascade_time(all_agg[0],           title_suffix=suffix),
        fig3_amplitude(all_agg[0],              title_suffix=suffix),
        fig4_physical_impact(all_agg[0],        title_suffix=suffix),
        fig5_distortion(all_agg[0],             title_suffix=suffix),
        fig6_propagation_grouped(all_results[0], title_suffix=suffix),
    ]
    if len(all_results) > 1:
        if not model_labels:
            model_labels = [f"Model {i+1}" for i in range(len(all_results))]
        common_keys = [t for t in TOPOLOGY_ORDER
                       if all(t in a for a in all_agg)]
        if common_keys:
            saved += [
                fig7_multimodel_success(all_agg,    model_labels, common_keys),
                fig8_multimodel_impact(all_agg,     model_labels, common_keys),
                fig9_multimodel_time(all_agg,       model_labels, common_keys),
                fig10_multimodel_distortion(all_agg, model_labels, common_keys),
                fig11_radar(all_agg,                model_labels),
            ]
    run_all_tests(all_agg,
                  model_labels or [f"Model {i+1}" for i in range(len(all_agg))])

    if stability_pair and len(stability_pair) == 2:
        a1 = aggregate(load_results(stability_pair[0]))
        a2 = aggregate(load_results(stability_pair[1]))
        saved.append(fig_stability_per_model(
            a1, a2,
            model_name=stability_model_name or "Model",
            fig_index=12,
        ))
    log.info("All figures saved to: %s", FIGURES_DIR)
    for p in saved:
        log.info("  %s", p.name)


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-DETECTION FROM data/results/ FOLDER
# ══════════════════════════════════════════════════════════════════════════════

# Display order for models in comparison figures
MODEL_ORDER = [
    "GPT-5.5",
    "Claude Sonnet 4.5",
    "Gemini 2.5 Flash",
    "Gemini 3.5 Flash",
]

# Defense display order
DEFENSE_ORDER = [
    "none",
    "majority_voting",
    "reputation_system",
    "confidence_weighted",
    "trust_clipping",
]

DEFENSE_DISPLAY = {
    "none":                "None (baseline)",
    "majority_voting":     "Majority Voting",
    "reputation_system":   "Reputation System",
    "confidence_weighted": "Confidence Weighted",
    "trust_clipping":      "Trust Clipping",
}


def read_metadata(path: Path) -> dict:
    """Read just the metadata block from a JSON result file."""
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        return raw.get("metadata", {})
    return {}


def auto_detect_files(results_dir: Path) -> dict:
    """
    Scan results_dir for all cascade_results_*.json files.
    Group them automatically by (model_display, defense_mode).
    Returns a structured dict ready for figure generation.

    Structure returned:
    {
      "baseline": {
          "GPT-5.5":         [Path, Path],   # run1, run2
          "Claude Sonnet 4.5": [Path, Path],
          ...
      },
      "defenses": {
          "none":             [Path],   # Gemini 3.5 baseline run
          "majority_voting":  [Path],
          ...
      }
    }
    """
    files = sorted(results_dir.glob("cascade_results_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No cascade_results_*.json files found in {results_dir}"
        )

    baseline = {}   # model_display -> list of paths (defense=none)
    defenses = {}   # defense_mode  -> list of paths (model=gemini-3.5)

    for f in files:
        meta = read_metadata(f)
        model   = meta.get("model_display", "Unknown")
        defense = meta.get("defense_mode",  "none")

        log.info("  Detected: %-25s  defense=%-20s  %s",
                 model, defense, f.name)

        if defense == "none":
            baseline.setdefault(model, []).append(f)
        else:
            # Defense runs — keep only Gemini 3.5 (or first model found)
            defenses.setdefault(defense, []).append(f)

    return {"baseline": baseline, "defenses": defenses}


def main_auto(results_dir: Path) -> None:
    """
    Fully automatic mode: scan results_dir, group by model and defense,
    generate all figures with zero user intervention.
    """
    log.info("Auto-detecting files in: %s", results_dir)
    detected = auto_detect_files(results_dir)

    baseline = detected["baseline"]
    defenses = detected["defenses"]

    if not baseline:
        log.error("No baseline (defense=none) result files found.")
        return

    log.info("\nFound %d model(s):", len(baseline))
    for m, paths in baseline.items():
        log.info("  %-25s : %d run(s)", m, len(paths))
    if defenses:
        log.info("Found %d defense type(s):", len(defenses))
        for d, paths in defenses.items():
            log.info("  %-25s : %d file(s)", d, len(paths))

    # ── Load and aggregate all baseline runs ──────────────────────────────
    # Sort models in canonical order
    ordered_models = [m for m in MODEL_ORDER if m in baseline]
    # Add any model not in MODEL_ORDER at the end
    ordered_models += [m for m in baseline if m not in ordered_models]

    model_names    = []
    model_run_aggs = []   # [model_idx][run_idx] = agg dict

    for model_name in ordered_models:
        paths = baseline[model_name]
        model_names.append(model_name)
        run_aggs = []
        for p in paths:
            results = load_results(p)
            run_aggs.append(aggregate(results))
        model_run_aggs.append(run_aggs)

    # Pooled (counts summed across runs) for multi-model comparison
    pooled_aggs = [pool_runs(ra) for ra in model_run_aggs]

    # Statistical tests are collected here and written after the defense
    # aggregates are built, so the JSON report contains everything at once.
    stats_def_aggs: Optional[List[dict]] = None
    stats_def_labels: Optional[List[str]] = None

    saved = []
    fig_idx = 1

    # ── Figs 1–6: single-model figures for first model ────────────────────
    first_results = [load_results(p) for p in baseline[ordered_models[0]]]
    saved += [
        fig1_attack_success(pooled_aggs[0],
                            title_suffix=ordered_models[0]),
        fig2_cascade_time(pooled_aggs[0],
                          title_suffix=ordered_models[0]),
        fig3_amplitude(pooled_aggs[0],
                       title_suffix=ordered_models[0]),
        fig4_physical_impact(pooled_aggs[0],
                             title_suffix=ordered_models[0]),
        fig5_distortion(pooled_aggs[0],
                        title_suffix=ordered_models[0]),
        fig6_propagation_grouped(first_results,
                                 title_suffix=ordered_models[0]),
    ]
    fig_idx = 7

    # ── Figs 7–11: multi-model comparison ─────────────────────────────────
    if len(pooled_aggs) > 1:
        common_keys = [t for t in TOPOLOGY_ORDER
                       if all(t in a for a in pooled_aggs)]
        if common_keys:
            saved += [
                fig7_multimodel_success(
                    pooled_aggs, model_names, common_keys),
                fig8_multimodel_impact(
                    pooled_aggs, model_names, common_keys),
                fig9_multimodel_time(
                    pooled_aggs, model_names, common_keys),
                fig10_multimodel_distortion(
                    pooled_aggs, model_names, common_keys),
                fig11_radar(pooled_aggs, model_names),
            ]
    fig_idx = 12

    # ── Stability figures: one per model with ≥2 runs ─────────────────────
    has_stability = []
    for name, run_aggs in zip(model_names, model_run_aggs):
        if len(run_aggs) >= 2:
            saved.append(fig_stability_per_model(
                run_aggs[0], run_aggs[1],
                model_name=name,
                fig_index=fig_idx,
            ))
            has_stability.append((name, run_aggs[0], run_aggs[1]))
            fig_idx += 1

    # ── Stability summary heatmap (all models with ≥2 runs) ───────────────
    if len(has_stability) >= 2:
        h_names = [h[0] for h in has_stability]
        h_r1    = [h[1] for h in has_stability]
        h_r2    = [h[2] for h in has_stability]
        saved.append(fig_stability_summary(h_names, h_r1, h_r2, fig_idx))
        fig_idx += 1

    # ── Defense figures ────────────────────────────────────────────────────
    if defenses:
        # Build baseline entry from Gemini 3.5 run 1
        gemini35_paths = baseline.get("Gemini 3.5 Flash", [])
        def_aggs   = []
        def_labels = []

        if gemini35_paths:
            # Pool every baseline run of the model, so the defense arms are
            # compared against the same baseline reported in Fig. 3 / Table VI
            # rather than against a single run.
            def_aggs.append(pool_runs(
                [aggregate(load_results(p)) for p in gemini35_paths]))
            def_labels.append("None (baseline)")
            log.info("Defense baseline pooled over %d run(s)", len(gemini35_paths))

        # Add defenses in canonical order
        ordered_defenses = [d for d in DEFENSE_ORDER if d in defenses
                            and d != "none"]
        ordered_defenses += [d for d in defenses
                             if d not in DEFENSE_ORDER and d != "none"]

        for defense_key in ordered_defenses:
            paths = defenses[defense_key]
            # Use first file for each defense type
            def_aggs.append(aggregate(load_results(paths[0])))
            def_labels.append(DEFENSE_DISPLAY.get(defense_key, defense_key))

        stats_def_aggs, stats_def_labels = def_aggs, def_labels

        if def_aggs:
            saved.append(
                fig_defense_comparison(def_aggs, def_labels, fig_idx))
            fig_idx += 1
            saved.append(
                fig_defense_mean_bar(def_aggs, def_labels, fig_idx))
            fig_idx += 1

    # ── Significance testing ───────────────────────────────────────────────
    run_all_tests(pooled_aggs, model_names,
                  def_aggs=stats_def_aggs, def_labels=stats_def_labels)

    log.info("\n✓ All figures saved to: %s", FIGURES_DIR)
    log.info("  Total: %d figures", len(saved))
    for p in saved:
        log.info("    %s", p.name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate IEEE-style figures from benchmark results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fully automatic (recommended) — reads all files in data/results/
  python analysis.py

  # Specify a different results folder
  python analysis.py --results-dir path/to/results/

  # Legacy mode with explicit file list
  python analysis.py --results file1.json file2.json --labels "GPT-5.5" "Claude"
        """
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("data/results"),
        help="Folder containing cascade_results_*.json files (default: data/results/)",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to JSON config file (optional — overrides auto-detection).",
    )
    parser.add_argument(
        "--generate-config", type=Path, default=None, metavar="OUTPUT",
        help="Write an example config file and exit.",
    )
    # Legacy flags kept for backward compatibility
    parser.add_argument("--results", nargs="+", type=Path, default=None)
    parser.add_argument("--labels",  nargs="+", default=None)
    parser.add_argument("--stability-pair", nargs=2, type=Path, default=None)
    parser.add_argument("--stability-model-name", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.generate_config:
        build_example_config(args.generate_config)

    elif args.config:
        main_from_config(args.config)

    elif args.results:
        # Legacy explicit file list
        if args.labels and len(args.labels) != len(args.results):
            raise ValueError(
                f"--labels count ({len(args.labels)}) must match "
                f"--results count ({len(args.results)})."
            )
        main_legacy(
            results_paths=args.results,
            model_labels=args.labels,
            stability_pair=args.stability_pair,
            stability_model_name=args.stability_model_name,
        )

    else:
        # DEFAULT: fully automatic mode
        main_auto(args.results_dir)