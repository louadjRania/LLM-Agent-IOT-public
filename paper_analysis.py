#!/usr/bin/env python3
"""
paper_analysis.py
=================

Everything the paper needs, and nothing else.

    python paper_analysis.py

Reads   : data/analysis/*.json   (produced by rebuild_analysis.sh)
Writes  : paper/table.csv        one row per condition, every reported number
          paper/stats.json       every statistical test, with its p-value
          paper/fig1_attack_success.png  (and .pdf)
          paper/fig2_run_variability.png (and .pdf)
          paper/fig3_physical_impact.png (and .pdf)
          paper/fig4_defenses.png        (and .pdf)
          + a console summary listing the values to paste into the LaTeX

Four tests, one per research question. No exploratory extras.

  RQ1  topology effect within each backbone      Pearson chi-square, 5x2, df=4
  RQ2  execution-to-execution heterogeneity      Fisher exact, 25 vs 25, Holm over 20
  RQ3  physical impact vs attack success         descriptive only (see NOTE)
  RQ4  each defense against the no-defense base  Pearson chi-square, Holm over 4

NOTE on RQ3: P is zero-inflated and multimodal, so it is reported as a mean
with the runs behind it, never tested. The paper says this explicitly.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact

# ── Configuration ────────────────────────────────────────────────────────────

DATA = Path("data/analysis")
OUT = Path("paper")
OUT.mkdir(exist_ok=True)

TOPOLOGIES = ["linear", "star", "ring", "tree", "mesh"]

# Two independent executions per backbone. Order matters: exec1, exec2.
MODELS = {
    "GPT-5.5":           ("gpt55_exec1",    "gpt55_exec2"),
    "Claude Sonnet 4.5": ("claude45_exec1", "claude45_exec2"),
    "Gemini 2.5 Flash":  ("gemini25_exec1", "gemini25_exec2"),
    "Gemini 3.5 Flash":  ("gemini35_exec1", "gemini35_exec2"),
}

# Defense study: Gemini 3.5 Flash only, one execution each.
DEFENSES = {
    "Majority voting":       "defense_majority",
    "Reputation":            "defense_reputation",
    "Confidence-weighted":   "defense_confidence",
    "Trust clipping":        "defense_trust",
}
DEFENSE_BASELINE = "Gemini 3.5 Flash"

# Colourblind-safe (Okabe-Ito), stable across all figures.
COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})

IEEE_WIDE = 7.16   # inches, full two-column width
IEEE_COL = 3.45    # inches, single column

# Output formats. PNG is easy to open and preview; PDF is vector and stays
# sharp at any zoom, which is what IEEE prefers. Both work with LaTeX.
# Keep only "png" here if you do not want the PDF copies.
FORMATS = ["png", "pdf"]
FIGURE_DPI = 400          # only affects the PNG copies


def save(fig, name: str) -> None:
    """Write one figure in every configured format."""
    for extension in FORMATS:
        fig.savefig(OUT / f"{name}.{extension}", bbox_inches="tight",
                    dpi=FIGURE_DPI)
    plt.close(fig)

# ── Statistics ───────────────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """95% Wilson score interval, in percent. Never collapses at p=0 or p=1."""
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjustment, monotonicity enforced."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


# ── Loading ──────────────────────────────────────────────────────────────────

def attacked_runs(name: str, topology: str) -> list[dict]:
    with open(DATA / f"{name}.json") as fh:
        results = json.load(fh)["results"]
    return [r for r in results
            if r["topology"] == topology and r["under_attack"]]


def condition(name: str, topology: str) -> dict:
    runs = attacked_runs(name, topology)
    successes = sum(1 for r in runs if r["attack_succeeded"])
    return {
        "successes": successes,
        "n": len(runs),
        "rate": 100 * successes / len(runs),
        "impact": statistics.mean(r["physical_impact"] for r in runs),
    }


# ── RQ1: does topology matter within a backbone? ─────────────────────────────

def rq1(cells: dict) -> list[dict]:
    out = []
    for model in MODELS:
        table = np.array([[cells[model][t]["successes"],
                           cells[model][t]["n"] - cells[model][t]["successes"]]
                          for t in TOPOLOGIES])
        chi2, p, dof, expected = chi2_contingency(table, correction=False)
        out.append({"model": model, "test": "pearson_chi2",
                    "chi2": round(float(chi2), 2), "df": int(dof),
                    "p": float(p), "min_expected": round(float(expected.min()), 1)})
    return out


# ── RQ2: do the two executions agree? ────────────────────────────────────────

def rq2(per_exec: dict) -> list[dict]:
    out, raw = [], []
    for model in MODELS:
        for t in TOPOLOGIES:
            a, b = per_exec[model][t]
            _, p = fisher_exact([[a["successes"], a["n"] - a["successes"]],
                                 [b["successes"], b["n"] - b["successes"]]])
            raw.append(float(p))
            out.append({"model": model, "topology": t,
                        "rate_exec1": round(a["rate"], 1),
                        "rate_exec2": round(b["rate"], 1),
                        "delta_pp": round(abs(a["rate"] - b["rate"]), 1),
                        "test": "fisher_exact", "p": float(p)})
    for row, adj in zip(out, holm(raw)):
        row["p_holm"] = adj
        row["significant"] = adj < 0.05
    return out


# ── RQ4: does each defense help? ─────────────────────────────────────────────

def rq4(cells: dict, defense_cells: dict) -> list[dict]:
    base_s = sum(cells[DEFENSE_BASELINE][t]["successes"] for t in TOPOLOGIES)
    base_n = sum(cells[DEFENSE_BASELINE][t]["n"] for t in TOPOLOGIES)

    out, raw = [], []
    for label in DEFENSES:
        s = sum(defense_cells[label][t]["successes"] for t in TOPOLOGIES)
        n = sum(defense_cells[label][t]["n"] for t in TOPOLOGIES)
        table = np.array([[base_s, base_n - base_s], [s, n - s]])
        chi2, p_chi, _, expected = chi2_contingency(table, correction=False)
        odds, p_fisher = fisher_exact(table)
        # chi-square is valid here; Fisher is the fallback for sparse tables
        sparse = expected.min() < 5
        raw.append(float(p_fisher if sparse else p_chi))
        out.append({"defense": label,
                    "rate": round(100 * s / n, 1), "n": n,
                    "baseline_rate": round(100 * base_s / base_n, 1),
                    "test": "fisher_exact" if sparse else "pearson_chi2",
                    "p": float(p_fisher if sparse else p_chi),
                    "odds_ratio_baseline_to_defense": round(float(odds), 3),
                    "min_expected": round(float(expected.min()), 1)})
    for row, adj in zip(out, holm(raw)):
        row["p_holm"] = adj
        row["significant"] = adj < 0.05
    return out


# ── Figures ──────────────────────────────────────────────────────────────────

def fig1_attack_success(cells: dict) -> None:
    """RQ1. Grouped bars, Wilson 95% CI, one group per topology."""
    fig, ax = plt.subplots(figsize=(IEEE_WIDE, 2.6))
    width = 0.2
    x = np.arange(len(TOPOLOGIES))

    for i, model in enumerate(MODELS):
        rates, lo, hi = [], [], []
        for t in TOPOLOGIES:
            c = cells[model][t]
            rates.append(c["rate"])
            a, b = wilson(c["successes"], c["n"])
            lo.append(c["rate"] - a)
            hi.append(b - c["rate"])
        ax.bar(x + (i - 1.5) * width, rates, width, label=model,
               color=COLOURS[i], edgecolor="white", linewidth=0.4)
        ax.errorbar(x + (i - 1.5) * width, rates, yerr=[lo, hi],
                    fmt="none", ecolor="#333333", elinewidth=0.7, capsize=1.6)

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in TOPOLOGIES])
    ax.set_ylabel("Attack success rate (%)")
    ax.set_ylim(0, 108)
    ax.legend(ncol=4, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.22))
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig1_attack_success")


def _spread(values: list[float], gap: float, lo: float, hi: float) -> list[float]:
    """Nudge label positions apart so that none overlap, keeping their order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    previous = lo - gap
    for i in order:
        placed[i] = max(placed[i], previous + gap)
        previous = placed[i]
    overflow = placed[order[-1]] - hi
    if overflow > 0:                       # pushed past the top: shift down
        for i in order:
            placed[i] -= overflow
    return placed


def fig2_run_variability(per_exec: dict, tests: list[dict]) -> None:
    """RQ2. One small panel per backbone. A flat line means the two
    executions agreed; a steep line means they did not. Lines that cross
    mean the topology ordering itself changed between executions."""
    sig = {(r["model"], r["topology"]): r["significant"] for r in tests}

    fig, axes = plt.subplots(1, 4, figsize=(IEEE_WIDE, 2.5), sharey=True)
    for ax, model in zip(axes, MODELS):
        starts = [per_exec[model][t][0]["rate"] for t in TOPOLOGIES]
        label_y = _spread(starts, gap=7.0, lo=0.0, hi=100.0)

        seen: set[tuple[float, float]] = set()
        for j, t in enumerate(TOPOLOGIES):
            a, b = per_exec[model][t]
            significant = sig[(model, t)]
            # Two topologies can trace exactly the same line. Dash the second
            # one so both remain visible, without moving either value.
            key = (a["rate"], b["rate"])
            style = (0, (4, 2)) if key in seen else "solid"
            seen.add(key)
            ax.plot([0, 1], [a["rate"], b["rate"]],
                    color=COLOURS[j], linestyle=style,
                    linewidth=2.0 if significant else 1.0,
                    alpha=1.0 if significant else 0.55,
                    marker="o", markersize=3.2, zorder=2)
            ax.text(-0.10, label_y[j], t, ha="right", va="center", fontsize=6,
                    color=COLOURS[j],
                    fontweight="bold" if significant else "normal")

        ax.set_title(model, fontsize=7.5, pad=4)
        ax.set_xlim(-0.62, 1.12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Exec 1", "Exec 2"], fontsize=7)
        ax.set_ylim(-5, 105)
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines["bottom"].set_linewidth(0.6)

    axes[0].set_ylabel("Attack success rate (%)")
    fig.text(0.5, -0.06,
             "Bold lines mark conditions whose two executions differ "
             "significantly (Fisher, Holm-adjusted, $p<0.05$).",
             ha="center", fontsize=6.5, color="#444444")
    fig.tight_layout()
    save(fig, "fig2_run_variability")


def fig3_physical_impact(cells: dict) -> None:
    """RQ3. Attack success against role-weighted impact, one marker per
    condition, coloured by backbone. The callouts mark the conditions the
    text discusses: identical success rates with unequal severity."""
    from scipy.stats import spearmanr

    fig, ax = plt.subplots(figsize=(IEEE_COL, 2.7))
    for i, model in enumerate(MODELS):
        xs = [cells[model][t]["rate"] for t in TOPOLOGIES]
        ys = [cells[model][t]["impact"] for t in TOPOLOGIES]
        ax.scatter(xs, ys, s=22, color=COLOURS[i], label=model,
                   edgecolor="white", linewidth=0.4, zorder=2)

    # Leader lines so a label cannot be read against a neighbouring marker.
    arrow = dict(arrowstyle="-", linewidth=0.5, color="#666666",
                 shrinkA=0, shrinkB=2)
    for model, t, label, xy_text in [
        ("GPT-5.5",           "tree",   "GPT-5.5 tree",   (20.0, 1.55)),
        ("GPT-5.5",           "mesh",   "GPT-5.5 mesh",   (18.0, 3.95)),
        ("Claude Sonnet 4.5", "linear", "Claude linear",  (40.0, 6.30)),
        ("Claude Sonnet 4.5", "tree",   "Claude tree",    (80.0, 1.90)),
    ]:
        c = cells[model][t]
        ax.annotate(label, xy=(c["rate"], c["impact"]), xytext=xy_text,
                    fontsize=6, color="#444444", va="center",
                    arrowprops=arrow, zorder=4)

    # Delete the next three lines if you do not want rho on the figure.
    rates = [cells[m][t]["rate"] for m in MODELS for t in TOPOLOGIES]
    impacts = [cells[m][t]["impact"] for m in MODELS for t in TOPOLOGIES]
    ax.text(0.97, 0.05, rf"Spearman $\rho$ = {spearmanr(rates, impacts)[0]:.2f}",
            transform=ax.transAxes, ha="right", fontsize=6, color="#666666")

    ax.set_xlabel("Attack success rate (%)")
    ax.set_ylabel("Physical impact $P$")
    ax.grid(linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    fig.tight_layout()
    save(fig, "fig3_physical_impact")


def fig4_defenses(cells: dict, defense_cells: dict) -> None:
    """RQ4. Baseline against each defense, per topology."""
    fig, ax = plt.subplots(figsize=(IEEE_WIDE, 2.6))
    series = [("No defense", {t: cells[DEFENSE_BASELINE][t] for t in TOPOLOGIES})]
    series += [(label, defense_cells[label]) for label in DEFENSES]

    width = 0.16
    x = np.arange(len(TOPOLOGIES))
    for i, (label, data) in enumerate(series):
        rates = [data[t]["rate"] for t in TOPOLOGIES]
        ax.bar(x + (i - 2) * width, rates, width, label=label,
               color=COLOURS[i], edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in TOPOLOGIES])
    ax.set_ylabel("Attack success rate (%)")
    ax.set_ylim(0, 108)
    ax.legend(ncol=5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.22))
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig4_defenses")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # Per execution, and pooled over the two executions.
    per_exec, cells = {}, {}
    for model, (f1, f2) in MODELS.items():
        per_exec[model] = {t: (condition(f1, t), condition(f2, t))
                           for t in TOPOLOGIES}
        cells[model] = {}
        for t in TOPOLOGIES:
            a, b = per_exec[model][t]
            s, n = a["successes"] + b["successes"], a["n"] + b["n"]
            runs = attacked_runs(f1, t) + attacked_runs(f2, t)
            cells[model][t] = {
                "successes": s, "n": n, "rate": 100 * s / n,
                "impact": statistics.mean(r["physical_impact"] for r in runs),
            }

    defense_cells = {label: {t: condition(f, t) for t in TOPOLOGIES}
                     for label, f in DEFENSES.items()}

    tests = {"rq1_topology_effect": rq1(cells),
             "rq2_execution_heterogeneity": rq2(per_exec),
             "rq4_defense_vs_baseline": rq4(cells, defense_cells)}
    (OUT / "stats.json").write_text(json.dumps(tests, indent=2))

    # One row per condition, holding every number the paper reports.
    rows = []
    for model in MODELS:
        for t in TOPOLOGIES:
            a, b = per_exec[model][t]
            c = cells[model][t]
            lo, hi = wilson(c["successes"], c["n"])
            rows.append({
                "block": "baseline", "model": model, "topology": t,
                "succ_exec1": a["successes"], "succ_exec2": b["successes"],
                "n_per_exec": a["n"],
                "rate_exec1": round(a["rate"], 1), "rate_exec2": round(b["rate"], 1),
                "delta_pp": round(abs(a["rate"] - b["rate"]), 1),
                "rate_pooled": round(c["rate"], 1),
                "wilson_lo": round(lo, 1), "wilson_hi": round(hi, 1),
                "impact_P": round(c["impact"], 2),
            })
    for label in DEFENSES:
        for t in TOPOLOGIES:
            c = defense_cells[label][t]
            lo, hi = wilson(c["successes"], c["n"])
            rows.append({
                "block": "defense", "model": label, "topology": t,
                "succ_exec1": c["successes"], "succ_exec2": "",
                "n_per_exec": c["n"],
                "rate_exec1": round(c["rate"], 1), "rate_exec2": "",
                "delta_pp": "", "rate_pooled": round(c["rate"], 1),
                "wilson_lo": round(lo, 1), "wilson_hi": round(hi, 1),
                "impact_P": round(c["impact"], 2),
            })
    with open(OUT / "table.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig1_attack_success(cells)
    fig2_run_variability(per_exec, tests["rq2_execution_heterogeneity"])
    fig3_physical_impact(cells)
    fig4_defenses(cells, defense_cells)

    # ── Console summary: the numbers to paste into the LaTeX ────────────────
    print("\nRQ1  topology effect (pooled, n=50 per condition)")
    for r in tests["rq1_topology_effect"]:
        print(f"     {r['model']:18s} chi2({r['df']}) = {r['chi2']:6.2f}   "
              f"p = {r['p']:.3g}   min expected = {r['min_expected']}")

    print("\nRQ2  execution-to-execution heterogeneity (Holm over 20 tests)")
    for r in tests["rq2_execution_heterogeneity"]:
        if r["significant"]:
            print(f"     {r['model']:18s} {r['topology']:7s} "
                  f"{r['rate_exec1']:5.0f}% vs {r['rate_exec2']:5.0f}%  "
                  f"|d| = {r['delta_pp']:5.0f} pp   p_holm = {r['p_holm']:.3g}")
    n_sig = sum(r["significant"] for r in tests["rq2_execution_heterogeneity"])
    print(f"     -> {n_sig} of 20 conditions differ significantly")

    print("\nRQ3  attack success and physical impact (descriptive)")
    for model in MODELS:
        line = "  ".join(f"{t[:4]} {cells[model][t]['rate']:3.0f}%/"
                         f"P={cells[model][t]['impact']:.2f}" for t in TOPOLOGIES)
        print(f"     {model:18s} {line}")

    print(f"\nRQ4  defenses vs no-defense baseline "
          f"({tests['rq4_defense_vs_baseline'][0]['baseline_rate']}%, "
          f"Holm over 4 tests)")
    for r in tests["rq4_defense_vs_baseline"]:
        verdict = "significant" if r["significant"] else "not significant"
        print(f"     {r['defense']:22s} {r['rate']:5.1f}%   "
              f"p_holm = {r['p_holm']:.3g}   OR = {r['odds_ratio_baseline_to_defense']:.3f}   {verdict}")

    print(f"\nWritten to {OUT}/ : table.csv, stats.json, "
          f"4 figures as {' and '.join(FORMATS)}\n")


if __name__ == "__main__":
    main()