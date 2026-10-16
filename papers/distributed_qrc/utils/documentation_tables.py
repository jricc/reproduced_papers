"""Generate the README result tables (phase 06) from the structured artifacts.

Writes ``results/results_summary.md`` with:

1. the paper-item table (paper value, reproduced value, delta, metric / trend
   agreement, claim support, seeds, comment) for Tables IV-VII (headline),
   Tables C (per-cell agreement by cell type), Table VIII (C7) and the C6
   cross-architecture comparison, plus the fair-baseline rows of phase 05;
2. the sweep index: every evidence sweep (R2, MerLin, phase 05) with its
   purpose, candidates, successful / expected runs, completeness, machine, code
   hash, declared selection rule and the result recomputed here from the raw
   ``metrics.json`` files, and the path of its full candidate table
   (``summary.json``, written by the sweep coordinator).

Inputs: ``results/headline_statistics.json``, ``results/E{1..4}-R2_vs_paper.csv``,
``results/fair_comparison.json``, ``results/merlin_comparison.json``,
``results/noise_distribution_comparison-R2.json`` and ``outdir/sweep_*``.

Usage (from the repository root)::

    python papers/distributed_qrc/utils/documentation_tables.py papers/distributed_qrc/outdir
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]
ARCHS = {"srsr": "E1", "mrsr": "E2", "srmr": "E3", "mrmr": "E4"}
TABLES = {
    "srsr": "IV / C1-C3",
    "mrsr": "V / C4-C6",
    "srmr": "VI / C7-C9",
    "mrmr": "VII / C10-C12",
}
# Evidence sweeps (superseded phase-03 sweeps without the -R2 suffix are excluded).
SWEEPS = {
    "P1-R6R7-R2": (
        "S1: observable and update-order choice (R6, R7)",
        "min mean eval RMSE over quantum_ideal|linear and quantum_ideal|quantum_kernel_ideal; 1 % ties -> first listed",
        ["quantum_ideal|linear", "quantum_ideal|quantum_kernel_ideal"],
    ),
    "E1-R2": (
        "E1 paper grid, SRSR (Table IV)",
        "no selection; seed aggregation",
        None,
    ),
    "E2-R2": ("E2 paper grid, MRSR (Table V)", "no selection; seed aggregation", None),
    "E3-R2": ("E3 paper grid, SRMR (Table VI)", "no selection; seed aggregation", None),
    "E4-R2": (
        "E4 paper grid, MRMR (Table VII)",
        "no selection; seed aggregation",
        None,
    ),
    "P1-ROBUST-R2": (
        "S2: robustness to normalization (per split) and artifact handling (interpolate); the merge rule R12 was resolved analytically, not swept",
        "no selection; materiality rule (results/r2_verdicts.json, S2-R2)",
        None,
    ),
    "GAP-SHOTS-R2": (
        "S3: finite-shot hypothesis for the noisy gap",
        "no selection; degradation-ratio categories (r2_verdicts.json, S3-R2)",
        None,
    ),
    "GAP-NOISE-SCALE-R2": (
        "S4: noise-strength scale for the noisy gap",
        "closest scale by min sum |ratio - paper|; 1 % ties -> smaller (r2_verdicts.json, S4-R2)",
        None,
    ),
    "GAP-INTERCEPT-R2": (
        "S5: missing-intercept hypothesis for the linear gap",
        "no selection; >= 13/24 cells rule (r2_verdicts.json, S5-R2)",
        None,
    ),
    "GAP-CLASSICAL-SRSR-R2": (
        "Classical-baseline seed spread, SRSR (20 seeds)",
        "no selection; paper value inside the 20-seed range in >= 5/8",
        None,
    ),
    "GAP-CLASSICAL-MRSR-R2": (
        "Classical-baseline seed spread, MRSR (20 seeds)",
        "as above",
        None,
    ),
    "GAP-CLASSICAL-SRMR-R2": (
        "Classical-baseline seed spread, SRMR (20 seeds)",
        "as above",
        None,
    ),
    "GAP-CLASSICAL-MRMR-R2": (
        "Classical-baseline seed spread, MRMR (20 seeds)",
        "as above",
        None,
    ),
    "XCHECK-E3-R2": (
        "Local CPU vs gpu-1 equivalence, SRMR 10/2",
        "no selection; max |difference| (results/xcheck-R2.json)",
        None,
    ),
    "XCHECK-E4-R2": ("Local CPU vs gpu-2 equivalence, MRMR 2x5", "as above", None),
    "P1-MERLIN-SCALES": (
        "Phase 04 M0: photonic encoding scales",
        "min mean eval RMSE over photonic|linear and photonic|photonic_kernel; 1 % ties -> first listed",
        ["photonic|linear", "photonic|photonic_kernel"],
    ),
    "E1-MERLIN": (
        "Phase 04 MerLin grid, SRSR",
        "no selection; preservation tolerance (results/merlin_comparison.json)",
        None,
    ),
    "E2-MERLIN": ("Phase 04 MerLin grid, MRSR", "as above", None),
    "E3-MERLIN": ("Phase 04 MerLin grid, SRMR", "as above", None),
    "E4-MERLIN": ("Phase 04 MerLin grid, MRMR", "as above", None),
    "B1B3-RAW": (
        "Phase 05 floors B1/B3 (no reservoir)",
        "no sweep selection; per-cell γ/λ on eval (results/fair_comparison.json)",
        None,
    ),
    "E1-CLASSICAL": (
        "Phase 05 fair baselines, SRSR",
        "no sweep selection; ESN, γ, λ selected on eval per cell",
        None,
    ),
    "E2-CLASSICAL": ("Phase 05 fair baselines, MRSR", "as above", None),
    "E3-CLASSICAL": ("Phase 05 fair baselines, SRMR", "as above", None),
    "E4-CLASSICAL": ("Phase 05 fair baselines, MRMR", "as above", None),
    "C6-SRSR": (
        "Phase 05 C6 control, SRSR, noise ×10",
        "no selection; decision rule of the Phase 5 plan",
        None,
    ),
    "C6-MRSR": ("Phase 05 C6 control, MRSR, noise ×10", "as above", None),
    "C6-MRMR": ("Phase 05 C6 control, MRMR, noise ×10", "as above", None),
}


def esc(text: str) -> str:
    return text.replace("|", "\\|")


def selection_result(sweep_dir: Path, cells: list[str]) -> str:
    """Recompute a declared selection: mean eval RMSE over seeds and cells; 1 % ties -> first listed."""

    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    order, scores = [], {}
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            continue
        key = json.dumps(run["candidate"], sort_keys=True)
        if key not in scores:
            order.append(key)
            scores[key] = []
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        by_cell = {
            f"{c['reservoir_type']}|{c['readout_type']}": c["metrics"]["eval"]["rmse"]
            for c in metrics["cells"]
        }
        scores[key].append(np.mean([by_cell[c] for c in cells]))
    means = {k: float(np.mean(v)) for k, v in scores.items()}
    best = min(means.values())
    tied = [k for k in order if means[k] <= best * 1.01]
    table = "; ".join(f"{k} {means[k]:.6f}" for k in order)
    return f"selected {tied[0]} (tied within 1 %: {', '.join(tied)}). All candidates: {table}"


def main() -> int:
    outdir = Path(sys.argv[1])
    head = json.loads((PAPER_DIR / "results" / "headline_statistics.json").read_text())
    fair = json.loads((PAPER_DIR / "results" / "fair_comparison.json").read_text())
    noise = json.loads(
        (PAPER_DIR / "results" / "noise_distribution_comparison-R2.json").read_text()
    )
    lines = [
        "# Results summary (generated by `utils/documentation_tables.py`)",
        "",
        "Test metrics on the normalized scale. Ours: mean over seeds 0-2 (full data, full-n kernels, simulation; `paper-accurate` protocol). Paper values: single runs from the paper's tables.",
        "",
        "## Paper items",
        "",
        "| Paper item | Claim tested | Paper value | Reproduced value | Delta | Metric agreement | Trend agreement | Claim support | Seeds | Comment |",
        "|---|---|---:|---:|---:|---|---|---|---:|---|",
    ]
    for arch, exp in ARCHS.items():
        p = head["paper"][arch]["mae"]
        o = head["ours"][arch]["paper_rule"]["mae"]
        for stat in ("average", "peak"):
            pv, ov, os = p[stat], o[f"{stat}_mean"], o[f"{stat}_std"]
            lines.append(
                f"| Table {TABLES[arch].split(' / ')[0]} ({exp}): {stat} MAE reduction, paper rule vs B0 | C1 | {pv:.1%} | {ov:.1%} ± {os:.1%} | {ov - pv:+.1%} | {'within 1 std' if abs(ov - pv) <= os else 'outside 1 std'} | yes (large reductions vs B0) | supported vs B0 only; unsupported vs fair classical (phase 05) | 3 | B0 is seed-unstable; paper value recomputed from its tables{' (reported 25.6 %)' if arch == 'mrsr' and stat == 'average' else ''} |"
            )
    for arch, exp in ARCHS.items():
        rows = list(
            csv.DictReader(open(PAPER_DIR / "results" / f"{exp}-R2_vs_paper.csv"))
        )
        groups: dict[str, list[dict]] = {}
        for r in rows:
            groups.setdefault(f"{r['reservoir']}|{r['readout']}", []).append(r)
        for cell in (
            "quantum_ideal|quantum_kernel_ideal",
            "classical|quantum_kernel_ideal",
            "classical|linear",
            "quantum_ideal|linear",
            "quantum_noisy|quantum_kernel_noisy",
        ):
            g = groups[cell]
            rel = np.array([float(r["mae_rel"]) for r in g])
            ours = np.array([float(r["mae_ours"]) for r in g])
            paper = np.array([float(r["mae_paper"]) for r in g])
            within = int(np.sum(np.abs(rel) <= 0.12))
            lines.append(
                f"| Tables {TABLES[arch].split(' / ')[1]} ({exp}): `{esc(cell)}` over 8 configurations | {'C1, C2, C4' if 'quantum_kernel' in cell else 'C1'}{', C5' if 'noisy' in cell else ''} | {paper.min():.4f}–{paper.max():.4f} | {ours.min():.4f}–{ours.max():.4f} | median {np.median(rel):+.0%} | {within}/8 within 12 % | {'yes' if within >= 6 else 'partial'} | see claim rows | 3 | per-cell values: `results/{exp}-R2_vs_paper.csv` |"
            )
    c7 = fair["C7"]
    lines.append(
        f"| Table VIII / Table V: best overall model, MRSR 3x20 classical reservoir + quantum kernel | C7 | 0.0113 | {c7['hybrid_3x20']['mean']:.4f} ± {c7['hybrid_3x20']['std']:.4f} | {c7['hybrid_3x20']['mean'] - 0.0113:+.4f} | no | no (not the best model in our runs) | unsupported | 3 | ties with the fair classical models ({c7['fair_classical_3x20']['mean']:.4f} ± {c7['fair_classical_3x20']['std']:.4f}) |"
    )
    same_sign = sum(
        v["same_sign"] for k, v in noise.items() if k.startswith("mrsr-vs-srsr@")
    )
    within = sum(
        abs(v["ours_diff_in_pooled_std"]) <= 1
        for k, v in noise.items()
        if k.startswith("mrsr-vs-srsr@")
    )
    c6 = fair["C6_x10"]
    lines.append(
        f"| Sec. III-B: noisy fully quantum MRSR vs SRSR at equal N (×1, paper mapping) | C6 | MRSR better at 8/8 | same sign as paper at {same_sign}/8, abs Δ ≤ 1 pooled std at {within}/8 | — | no | no | unsupported | 3 | `results/noise_distribution_comparison-R2.json` |"
    )
    lines.append(
        f"| Sec. III-B, controlled (phase 05): MRSR vs SRSR at noise ×10 | C6 | MRSR better at 8/8 | paper mapping {c6['paper']['mrsr_vs_srsr']['better']}/8; same device {c6['ibm_brisbane']['mrsr_vs_srsr']['better']}/8 | — | N/A (noise model differs) | only under the paper's device mapping | unsupported as a distribution effect | 3 | `results/fair_comparison.json` → `C6_x10` |"
    )
    for arch, exp in ARCHS.items():
        a = fair["architectures"][arch]
        cl = a["claims"]["C1_vs_fair_classical"]["counts"]
        h = a["E7_headline_vs_fair_classical"]["mae"]
        lines.append(
            f"| Phase 05 ({exp}): eval-selected quantum vs eval-selected fair classical (B0-B4) | C1 | N/A (no fair baseline in the paper) | {h['average']['mean']:+.1%} ± {h['average']['std']:.1%} average MAE reduction; better in {cl['better']}/8, tie {cl['tie']}/8, worse {cl['worse']}/8 | — | N/A | N/A | unsupported | 3 | `results/fair_comparison_table.md` |"
        )

    lines += [
        "",
        "## Sweep index",
        "",
        "Every evidence sweep. The full candidate-by-cell table of each sweep (mean and std over seeds of every split and metric) is its coordinator's `summary.json`. Superseded phase-03 sweeps (no `-R2` suffix) are kept on disk but are not evidence (`LOG.md#Re-run with complete pre-execution plans (R2)`).",
        "",
        "| Sweep | Purpose | Candidates | Successful / expected runs | Completeness | Machine | Code (`lib_sha256`) | Declared rule | Result (recomputed from raw metrics) | Full table |",
        "|---|---|---:|---:|---|---|---|---|---|---|",
    ]
    for sweep_id, (purpose, rule, cells) in SWEEPS.items():
        sweep_dir = outdir / f"sweep_{sweep_id}"
        status = json.loads((sweep_dir / "sweep_status.json").read_text())
        candidates = {
            json.dumps(r["candidate"], sort_keys=True) for r in status["runs"]
        }
        done = sum(r["status"] == "COMPLETED" for r in status["runs"])
        complete = (
            "COMPLETE"
            if status["status"] == "COMPLETED" and done == status["expected_runs"]
            else "PARTIAL"
        )
        machine = {"cuda:0": "NVIDIA L4"}.get(status.get("device"), "local CPU")
        result = selection_result(sweep_dir, cells) if cells else "see rule artifact"
        lines.append(
            f"| `{sweep_id}` | {purpose} | {len(candidates)} | {done} / {status['expected_runs']} | {complete} | {machine} | {status['code'].get('lib_sha256', '?')[:8]}… | {esc(rule)} | {esc(result)} | `outdir/sweep_{sweep_id}/summary.json` |"
        )
    path = PAPER_DIR / "results" / "results_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
