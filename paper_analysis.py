
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
from scipy.stats import chi2_contingency, fisher_exact, spearmanr

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

FORMATS = ["png", "pdf"]   # PDF (vector) is preferred by IEEE
FIGURE_DPI = 400           # PNG only


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



# ── Analysis configuration ───────────────────────────────────────────────────

AGENTS = ["SensorAgent", "MonitorAgent", "SchedulerAgent",
          "ActuatorAgent", "SupervisorAgent"]
AGENT_SHORT = ["Sens.", "Mon.", "Sched.", "Act.", "Sup."]
W_PAPER = np.array([0.5, 1.0, 2.0, 3.0, 1.5])          # Table II

SCHEMES = {                          # order: S, M, Sc, A, Su
    "Table II (paper)":         [0.5, 1.0, 2.0, 3.0, 1.5],
    "Uniform":                  [1, 1, 1, 1, 1],
    "Actuator-dominant (w_A=5)": [0.5, 1.0, 2.0, 5.0, 1.5],
    "Rank-based":               [1, 2, 4, 5, 3],
    "Geometric":                [1, 2, 8, 16, 4],
}
N_RANDOM = 10_000
RNG = np.random.default_rng(2026)

# Conditions used as the RQ2 illustration (non-tree, same backbone).
EXAMPLE = ("Gemini 2.5 Flash", "star", "mesh")


# ── Loading ──────────────────────────────────────────────────────────────────

def is_failed(r: dict) -> bool:
    return (r.get("scheduler_decision") == "UNKNOWN"
            and r.get("supervisor_verdict") == "UNKNOWN")


def load(name: str) -> list[dict]:
    with open(DATA / f"{name}.json") as fh:
        return [r for r in json.load(fh)["results"] if r["under_attack"]]


def summarise(runs: list[dict], excluded: int) -> dict:
    n = len(runs)
    s = sum(1 for r in runs if r["attack_succeeded"])
    c = np.array([np.mean([a in (r.get("propagation_path") or []) for r in runs])
                  if n else np.nan for a in AGENTS])
    return {"successes": s, "n": n, "excluded": excluded,
            "rate": 100 * s / n if n else np.nan,
            "impact": statistics.mean(r["physical_impact"] for r in runs) if n else np.nan,
            "roles": c}


def cell(name: str, topology: str) -> tuple[dict, list[dict]]:
    all_runs = [r for r in load(name) if r["topology"] == topology]
    valid = [r for r in all_runs if not is_failed(r)]
    return summarise(valid, len(all_runs) - len(valid)), valid


# ── Tests ────────────────────────────────────────────────────────────────────

def rq1(cells):
    out = []
    for m in MODELS:
        tab = np.array([[cells[m][t]["successes"],
                         cells[m][t]["n"] - cells[m][t]["successes"]] for t in TOPOLOGIES])
        row = {"model": m, "test": "pearson_chi2"}
        if (tab.sum(axis=0) == 0).any():
            row.update(chi2=None, df=None, p=None,
                       note="not computable: identical outcome in every topology")
        else:
            chi2, p, dof, exp = chi2_contingency(tab, correction=False)
            row.update(chi2=round(float(chi2), 2), df=int(dof), p=float(p),
                       min_expected=round(float(exp.min()), 1))
        out.append(row)
    return out


def rq2(per_exec):
    out, raw, idx = [], [], []
    for m in MODELS:
        for t in TOPOLOGIES:
            a, b = per_exec[m][t]
            row = {"model": m, "topology": t, "n_exec1": a["n"], "n_exec2": b["n"],
                   "rate_exec1": None if not a["n"] else round(a["rate"], 1),
                   "rate_exec2": None if not b["n"] else round(b["rate"], 1)}
            if a["n"] and b["n"]:
                _, p = fisher_exact([[a["successes"], a["n"] - a["successes"]],
                                     [b["successes"], b["n"] - b["successes"]]])
                row.update(delta_pp=round(abs(a["rate"] - b["rate"]), 1), p=float(p))
                raw.append(float(p)); idx.append(len(out))
            else:
                row.update(delta_pp=None, p=None, p_holm=None, significant=False,
                           note="not testable: no valid runs in one execution")
            out.append(row)
    for i, adj in zip(idx, holm(raw)):
        out[i]["p_holm"] = adj
        out[i]["significant"] = adj < 0.05
    return out, len(raw)


def rq3_defences(cells, dcells):
    bs = sum(cells[DEFENSE_BASELINE][t]["successes"] for t in TOPOLOGIES)
    bn = sum(cells[DEFENSE_BASELINE][t]["n"] for t in TOPOLOGIES)
    out, raw = [], []
    for lab in DEFENSES:
        s = sum(dcells[lab][t]["successes"] for t in TOPOLOGIES)
        n = sum(dcells[lab][t]["n"] for t in TOPOLOGIES)
        tab = np.array([[bs, bn - bs], [s, n - s]])
        chi2, pc, _, exp = chi2_contingency(tab, correction=False)
        odds, pf = fisher_exact(tab)
        sparse = exp.min() < 5
        p = float(pf if sparse else pc); raw.append(p)
        out.append({"defense": lab, "rate": round(100 * s / n, 1), "n": n,
                    "baseline_rate": round(100 * bs / bn, 1), "baseline_n": bn,
                    "test": "fisher_exact" if sparse else "pearson_chi2", "p": p,
                    "odds_ratio_baseline_to_defense": round(float(odds), 3)})
    for r, adj in zip(out, holm(raw)):
        r["p_holm"] = adj; r["significant"] = adj < 0.05
    return out


def random_ordered_weights():
    """Random weights preserving Table II ordering S < M < Su < Sc < A."""
    v = np.sort(RNG.uniform(0, 1, 5))
    return np.array([v[0], v[1], v[3], v[4], v[2]])


def sensitivity(cells):
    keys = [(m, t) for m in MODELS for t in TOPOLOGIES]
    R = np.array([cells[m][t]["roles"] for m, t in keys])
    asr = np.array([cells[m][t]["rate"] for m, t in keys])
    p_paper = R @ W_PAPER
    m_, a, b = EXAMPLE
    ca, cb = cells[m_][a]["roles"], cells[m_][b]["roles"]

    def evaluate(w):
        P = R @ w
        return (float(spearmanr(asr, P).correlation),
                float(spearmanr(p_paper, P).correlation),
                float((cb @ w) / (ca @ w)))

    schemes = {}
    for name, w in SCHEMES.items():
        r1, r2, ratio = evaluate(np.array(w, float))
        schemes[name] = {"weights_S_M_Sc_A_Su": w, "rho_asr_P": round(r1, 3),
                         "rho_with_tableII_P": round(r2, 3),
                         "example_ratio": round(ratio, 2)}
    draws = np.array([evaluate(random_ordered_weights()) for _ in range(N_RANDOM)])
    q = lambda x: [round(float(v), 3) for v in np.percentile(x, [2.5, 50, 97.5])]
    return {"note": "failed runs excluded; P = sum_i w_i c_i",
            "example": f"{m_}: {b} vs {a}",
            "schemes": schemes,
            "random_order_preserving": {
                "n_draws": N_RANDOM,
                "rho_asr_P_2.5_50_97.5": q(draws[:, 0]),
                "rho_with_tableII_P_2.5_50_97.5": q(draws[:, 1]),
                "example_ratio_2.5_50_97.5": q(draws[:, 2]),
                "example_direction_holds_pct": round(100 * float(np.mean(draws[:, 2] > 1)), 1)}}


# ── Figures ──────────────────────────────────────────────────────────────────

def fig1(cells):
    fig, ax = plt.subplots(figsize=(IEEE_WIDE, 2.6))
    width, x = 0.2, np.arange(len(TOPOLOGIES))
    for i, m in enumerate(MODELS):
        r, lo, hi = [], [], []
        for t in TOPOLOGIES:
            c = cells[m][t]; a, b = wilson(c["successes"], c["n"])
            r.append(c["rate"]); lo.append(c["rate"] - a); hi.append(b - c["rate"])
        pos = x + (i - 1.5) * width
        ax.bar(pos, r, width, label=m, color=COLOURS[i], edgecolor="white", linewidth=0.4)
        ax.errorbar(pos, r, yerr=[lo, hi], fmt="none", ecolor="#333333",
                    elinewidth=0.7, capsize=1.6)
        for j, t in enumerate(TOPOLOGIES):
            if cells[m][t]["n"] < 50:
                ax.text(pos[j], 2, f"n={cells[m][t]['n']}", rotation=90, ha="center",
                        va="bottom", fontsize=4.8, color="white")
    ax.set_xticks(x); ax.set_xticklabels([t.capitalize() for t in TOPOLOGIES])
    ax.set_ylabel("Attack success rate (%)"); ax.set_ylim(0, 108)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.grid(axis="y", linewidth=0.3, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout(); save(fig, "fig1_attack_success")


def fig2(per_exec, tests):
    sig = {(r["model"], r["topology"]): r["significant"] for r in tests}
    fig, axes = plt.subplots(1, 4, figsize=(IEEE_WIDE, 2.5), sharey=True)
    for ax, m in zip(axes, MODELS):
        starts = [per_exec[m][t][0]["rate"] for t in TOPOLOGIES]
        label_y = _spread(starts, gap=7.0, lo=0.0, hi=100.0)
        seen = set()
        if max(starts) - min(starts) < 0.5:
            # All topologies share one value: one label, not a misleading stack.
            ax.plot([0, 1], [starts[0], starts[0]], color="#555555", linewidth=1.2,
                    marker="o", markersize=3.2)
            ax.text(-0.10, starts[0], "all five", ha="right", va="center",
                    fontsize=6, color="#333333")
            ax.text(0.5, 55, f"{starts[0]:.0f}% in every topology", ha="center",
                    va="center", fontsize=5.6, color="#555555")
            ax.set_title(m, fontsize=7.5, pad=4)
            ax.set_xlim(-0.62, 1.12); ax.set_xticks([0, 1])
            ax.set_xticklabels(["Exec 1", "Exec 2"], fontsize=7); ax.set_ylim(-5, 105)
            ax.grid(axis="y", linewidth=0.3, alpha=0.4); ax.set_axisbelow(True)
            continue
        for j, t in enumerate(TOPOLOGIES):
            a, b = per_exec[m][t]
            s = sig[(m, t)]
            ax.text(-0.10, label_y[j], t, ha="right", va="center", fontsize=6,
                    color=COLOURS[j], fontweight="bold" if s else "normal")
            key = (a["rate"], b["rate"])
            ax.plot([0, 1], [a["rate"], b["rate"]], color=COLOURS[j],
                    linestyle=(0, (4, 2)) if key in seen else "solid",
                    linewidth=2.0 if s else 1.0, alpha=1.0 if s else 0.55,
                    marker="o", markersize=3.2, zorder=2)
            seen.add(key)
        ax.set_title(m, fontsize=7.5, pad=4)
        ax.set_xlim(-0.62, 1.12); ax.set_xticks([0, 1])
        ax.set_xticklabels(["Exec 1", "Exec 2"], fontsize=7); ax.set_ylim(-5, 105)
        ax.grid(axis="y", linewidth=0.3, alpha=0.4); ax.set_axisbelow(True)
    axes[0].set_ylabel("Attack success rate (%)")
    fig.text(0.5, -0.06, "Failed runs excluded. Bold lines: executions differ "
             "significantly (Fisher, Holm-adjusted, $p<0.05$).",
             ha="center", fontsize=6.5, color="#444444")
    fig.tight_layout(); save(fig, "fig2_run_variability")


def fig3(cells):
    """Paired horizontal bars (paper Fig. 4 style), sorted by attack success.
    Tree rows are hatched: ActuatorAgent does not receive the scheduling
    decision in the implemented tree topology."""
    short = {"GPT-5.5": "GPT-5.5", "Claude Sonnet 4.5": "Claude 4.5",
             "Gemini 2.5 Flash": "Gemini 2.5", "Gemini 3.5 Flash": "Gemini 3.5"}
    rows = [(m, t) for m in MODELS for t in TOPOLOGIES]
    rows.sort(key=lambda k: (-cells[k[0]][k[1]]["rate"], -cells[k[0]][k[1]]["impact"]))
    colour = {m: COLOURS[i] for i, m in enumerate(MODELS)}
    y = np.arange(len(rows))[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(IEEE_COL, 3.3), sharey=True,
                                 gridspec_kw={"wspace": 0.05})
    for yi, (m, t) in zip(y, rows):
        c = cells[m][t]; hatch = "////" if t == "tree" else None
        kw = dict(color=colour[m], height=0.62, hatch=hatch,
                  edgecolor="white", linewidth=0.3)
        a1.barh(yi, c["rate"], **kw); a2.barh(yi, c["impact"], **kw)
    a1.set_yticks(y)
    a1.set_yticklabels([f"{short[m]} {t}" + ("$^\\dagger$" if t == "tree" else "")
                        for m, t in rows], fontsize=5.8)
    a1.invert_xaxis(); a1.set_xlim(100, 0)
    a1.set_xlabel("Attack success (%)", fontsize=7)
    a2.set_xlabel("Operational impact $P$", fontsize=7)
    for ax in (a1, a2):
        ax.tick_params(axis="x", labelsize=6); ax.grid(axis="x", linewidth=0.3, alpha=0.4)
        ax.set_axisbelow(True)
    a2.tick_params(axis="y", length=0)
    rates = [cells[m][t]["rate"] for m, t in rows]
    imp = [cells[m][t]["impact"] for m, t in rows]
    a2.text(0.97, 0.02, rf"Spearman $\rho$ = {spearmanr(rates, imp).correlation:.2f}",
            transform=a2.transAxes, ha="right", fontsize=5.8, color="#555555")
    save(fig, "fig3_impact_vs_success")


def fig4(cells, dcells):
    fig, ax = plt.subplots(figsize=(IEEE_WIDE, 2.6))
    series = [("No defence", {t: cells[DEFENSE_BASELINE][t] for t in TOPOLOGIES})]
    series += [(lab, dcells[lab]) for lab in DEFENSES]
    width, x = 0.16, np.arange(len(TOPOLOGIES))
    for i, (lab, d) in enumerate(series):
        ax.bar(x + (i - 2) * width, [d[t]["rate"] for t in TOPOLOGIES], width,
               label=lab, color=COLOURS[i], edgecolor="white", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([t.capitalize() for t in TOPOLOGIES])
    ax.set_ylabel("Attack success rate (%)"); ax.set_ylim(0, 108)
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.grid(axis="y", linewidth=0.3, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout(); save(fig, "fig4_defenses")


def fig5(cells):
    """Weight-free view: fraction of attacked runs in which each role was
    recorded as contaminated. P is a weighted sum of these columns."""
    short = {"GPT-5.5": "GPT-5.5", "Claude Sonnet 4.5": "Claude 4.5",
             "Gemini 2.5 Flash": "Gemini 2.5", "Gemini 3.5 Flash": "Gemini 3.5"}
    keys = [(m, t) for m in MODELS for t in TOPOLOGIES]
    M = np.array([cells[m][t]["roles"] for m, t in keys]) * 100
    fig, ax = plt.subplots(figsize=(IEEE_COL, 3.6))
    im = ax.imshow(M, cmap="Reds", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=5.2,
                    color="white" if M[i, j] > 60 else "#222222")
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"{a}\n($w$={w:g})" for a, w in zip(AGENT_SHORT, W_PAPER)],
                       fontsize=5.8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{short[m]} {t}" + ("$^\\dagger$" if t == "tree" else "")
                        for m, t in keys], fontsize=5.8)
    for k in range(5, len(keys), 5):
        ax.axhline(k - 0.5, color="white", linewidth=1.2)
    ax.tick_params(length=0); ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=5.5); cb.set_label("Contaminated runs (%)", fontsize=6)
    fig.tight_layout(); save(fig, "fig5_role_contamination")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    per_exec, cells, excluded_log = {}, {}, []
    for m, (f1, f2) in MODELS.items():
        per_exec[m], cells[m] = {}, {}
        for t in TOPOLOGIES:
            (a, ra), (b, rb) = cell(f1, t), cell(f2, t)
            per_exec[m][t] = (a, b)
            cells[m][t] = summarise(ra + rb, a["excluded"] + b["excluded"])
            for e, x in ((1, a), (2, b)):
                if x["excluded"]:
                    excluded_log.append((m, t, e, x["excluded"]))
    dcells = {lab: {t: cell(f, t)[0] for t in TOPOLOGIES} for lab, f in DEFENSES.items()}

    t1 = rq1(cells)
    t2, n_tests = rq2(per_exec)
    t3 = rq3_defences(cells, dcells)
    sens = sensitivity(cells)
    keys_all = [(m, t) for m in MODELS for t in TOPOLOGIES]
    rho = float(spearmanr([cells[m][t]["rate"] for m, t in keys_all],
                          [cells[m][t]["impact"] for m, t in keys_all]).correlation)
    (OUT / "stats.json").write_text(json.dumps(
        {"exclusion_rule": "scheduler_decision == supervisor_verdict == 'UNKNOWN'",
         "excluded_runs": [{"model": m, "topology": t, "exec": e, "n": n}
                           for m, t, e, n in excluded_log],
         "rq1_topology_effect": t1, "rq1_execution_heterogeneity": t2,
         "rq2_spearman_asr_P": rho, "rq3_defense_vs_baseline": t3}, indent=2))
    (OUT / "sensitivity.json").write_text(json.dumps(sens, indent=2))

    rows = []
    for m in MODELS:
        for t in TOPOLOGIES:
            a, b = per_exec[m][t]; c = cells[m][t]; lo, hi = wilson(c["successes"], c["n"])
            rows.append({"block": "baseline", "model": m, "topology": t,
                         "succ_exec1": a["successes"], "n_exec1": a["n"],
                         "succ_exec2": b["successes"], "n_exec2": b["n"],
                         "excluded_failed": c["excluded"],
                         "rate_exec1": round(a["rate"], 1) if a["n"] else "",
                         "rate_exec2": round(b["rate"], 1) if b["n"] else "",
                         "delta_pp": round(abs(a["rate"] - b["rate"]), 1) if a["n"] and b["n"] else "",
                         "rate_pooled": round(c["rate"], 1), "n_pooled": c["n"],
                         "wilson_lo": round(lo, 1), "wilson_hi": round(hi, 1),
                         "impact_P": round(c["impact"], 2)})
    for lab in DEFENSES:
        for t in TOPOLOGIES:
            c = dcells[lab][t]; lo, hi = wilson(c["successes"], c["n"])
            rows.append({"block": "defense", "model": lab, "topology": t,
                         "succ_exec1": c["successes"], "n_exec1": c["n"],
                         "succ_exec2": "", "n_exec2": "", "excluded_failed": c["excluded"],
                         "rate_exec1": round(c["rate"], 1), "rate_exec2": "", "delta_pp": "",
                         "rate_pooled": round(c["rate"], 1), "n_pooled": c["n"],
                         "wilson_lo": round(lo, 1), "wilson_hi": round(hi, 1),
                         "impact_P": round(c["impact"], 2)})
    with open(OUT / "table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(OUT / "role_contamination.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "topology", "n"] + [f"c_{a}" for a in AGENTS] + ["impact_P"])
        for m in MODELS:
            for t in TOPOLOGIES:
                c = cells[m][t]
                w.writerow([m, t, c["n"]] + [round(float(v), 3) for v in c["roles"]]
                           + [round(c["impact"], 3)])

    fig1(cells); fig2(per_exec, t2); fig3(cells); fig4(cells, dcells); fig5(cells)

    # ── Summary of numbers to paste ──────────────────────────────────────────
    S = ["# Numbers for the revised paper (failed runs excluded)\n"]
    S.append(f"Excluded failed runs: {sum(n for *_, n in excluded_log)} "
             f"(baseline). Per condition:")
    for m, t, e, n in excluded_log:
        S.append(f"- {m} {t} exec{e}: {n}")
    S.append("\n## RQ1 attack success (pooled, valid runs)")
    for m in MODELS:
        S.append(f"- {m}: " + ", ".join(
            f"{t} {cells[m][t]['rate']:.0f}% (n={cells[m][t]['n']})" for t in TOPOLOGIES))
    S.append("\n## RQ1 chi-square within backbone")
    for r in t1:
        S.append(f"- {r['model']}: " + (r["note"] if r["chi2"] is None else
                 f"chi2({r['df']}) = {r['chi2']}, p = {r['p']:.3g}"))
    S.append(f"\n## Execution differences (Fisher, Holm over {n_tests} testable conditions)")
    for r in sorted(t2, key=lambda r: -(r["delta_pp"] or -1)):
        if r["delta_pp"] is None:
            S.append(f"- {r['model']} {r['topology']}: {r['note']}")
        elif r["delta_pp"] >= 20 or r["significant"]:
            S.append(f"- {r['model']} {r['topology']}: {r['rate_exec1']}% (n={r['n_exec1']}) vs "
                     f"{r['rate_exec2']}% (n={r['n_exec2']}), |d| = {r['delta_pp']} pp, "
                     f"p_holm = {r['p_holm']:.3g}{'  SIGNIFICANT' if r['significant'] else ''}")
    S.append(f"\n## RQ2\n- Spearman rho(ASR, P) = {rho:.2f}")
    m_, a, b = EXAMPLE
    for t in (a, b):
        c = cells[m_][t]
        S.append(f"- {m_} {t}: ASR {c['rate']:.0f}%, P = {c['impact']:.2f}, "
                 f"actuator reached {100 * c['roles'][3]:.0f}%")
    rr = sens["random_order_preserving"]
    S.append(f"- Sensitivity (named schemes + 95% of random draws): rho(ASR,P) "
             f"{min(rr['rho_asr_P_2.5_50_97.5'][0], *[v['rho_asr_P'] for v in sens['schemes'].values()]):.2f}-"
             f"{max(rr['rho_asr_P_2.5_50_97.5'][2], *[v['rho_asr_P'] for v in sens['schemes'].values()]):.2f}; "
             f"rank agreement with Table II >= {min(rr['rho_with_tableII_P_2.5_50_97.5'][0], *[v['rho_with_tableII_P'] for v in sens['schemes'].values()]):.2f}; "
             f"example ratio {rr['example_ratio_2.5_50_97.5'][0]:.1f}-"
             f"{rr['example_ratio_2.5_50_97.5'][2]:.1f}x, direction holds "
             f"{rr['example_direction_holds_pct']}%")
    S.append("\n## RQ3 defences vs pooled baseline")
    for r in t3:
        S.append(f"- {r['defense']}: {r['rate']}% (n={r['n']}) vs baseline "
                 f"{r['baseline_rate']}% (n={r['baseline_n']}), p_holm = {r['p_holm']:.3g}"
                 f"{'  SIGNIFICANT' if r['significant'] else ''}")
    (OUT / "SUMMARY.md").write_text("\n".join(S) + "\n")
    print("\n".join(S))


if __name__ == "__main__":
    main()