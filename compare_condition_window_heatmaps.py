#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qgl_matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qgl_reproduce as qgl


CONDITIONS = (
    {"condition": "ill", "label": "Ill-conditioned", "c": 0.0},
    {"condition": "well", "label": "Well-conditioned", "c": 0.1},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare first-mode Hamiltonian window checks for ill- and well-conditioned chains."
    )
    parser.add_argument("--seed", type=int, default=100, help="Seed for sampling.")
    parser.add_argument("--m", type=int, default=100, help="Number of modes.")
    parser.add_argument("--samples", type=int, default=100_000, help="Number of heterodyne samples.")
    parser.add_argument("--beta", type=float, default=0.5, help="Inverse-temperature prefactor.")
    parser.add_argument("--locality", type=int, default=4, help="Locality parameter for local reconstruction.")
    parser.add_argument("--window-modes", type=int, default=5, help="Number of consecutive modes to inspect.")
    parser.add_argument("--start-mode", type=int, default=1, help="One-based first mode of the inspected window.")
    parser.add_argument("--threshold", type=float, default=0.1, help="Pass/fail threshold for local max error.")
    parser.add_argument("--output-dir", default="condition_window_checks", help="Directory for generated outputs.")
    return parser.parse_args()


def mode_indices(start_mode: int, window_modes: int) -> list[int]:
    indices: list[int] = []
    for mode in range(start_mode, start_mode + window_modes):
        base = 2 * (mode - 1)
        indices.extend([base, base + 1])
    return indices


def coordinate_labels(start_mode: int, window_modes: int) -> list[str]:
    return [f"{mode}{quad}" for mode in range(start_mode, start_mode + window_modes) for quad in ("x", "p")]


def metrics(target: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    diff = estimate - target
    fro_target = np.linalg.norm(target, ord="fro")
    return {
        "max_abs_error": float(np.max(np.abs(diff))),
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "fro_error": float(np.linalg.norm(diff, ord="fro")),
        "relative_fro_error": float(np.linalg.norm(diff, ord="fro") / fro_target) if fro_target else float("nan"),
        "target_max_abs": float(np.max(np.abs(target))),
        "estimate_max_abs": float(np.max(np.abs(estimate))),
    }


def window(matrix: np.ndarray, indices: list[int]) -> np.ndarray:
    return matrix[np.ix_(indices, indices)]


def compute_condition(
    *,
    condition: str,
    label: str,
    c: float,
    seed: int,
    m: int,
    samples: int,
    beta: float,
    locality: int,
    start_mode: int,
    window_modes: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sample_matrix = qgl.sample_measurements(m, samples, c, rng, beta)
    cov_est = qgl.covariance_estimate(sample_matrix)
    target = qgl.hamiltonian_matrix(2 * m, c, beta)
    local_est = qgl.local_hamiltonian_estimate(cov_est, m, locality, c, beta)
    global_est = qgl.global_hamiltonian_estimate(cov_est, m, c, beta)
    indices = mode_indices(start_mode, window_modes)

    target_window = window(target, indices)
    local_window = window(local_est, indices)
    global_window = window(global_est, indices)

    return {
        "condition": condition,
        "label": label,
        "c": c,
        "target_window": target_window,
        "local_window": local_window,
        "global_window": global_window,
        "local_diff": local_window - target_window,
        "global_diff": global_window - target_window,
        "local_metrics": metrics(target_window, local_window),
        "global_metrics": metrics(target_window, global_window),
    }


def write_long_csv(path: Path, result: dict[str, Any], labels: list[str]) -> None:
    rows = []
    dim = len(labels)
    for i in range(dim):
        for j in range(dim):
            rows.append(
                {
                    "condition": result["condition"],
                    "c": result["c"],
                    "row": labels[i],
                    "col": labels[j],
                    "target": result["target_window"][i, j],
                    "local_estimate": result["local_window"][i, j],
                    "local_error": result["local_diff"][i, j],
                    "global_estimate": result["global_window"][i, j],
                    "global_error": result["global_diff"][i, j],
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def set_ticks(ax: plt.Axes, labels: list[str]) -> None:
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)


def write_error_heatmap(path: Path, results: list[dict[str, Any]], labels: list[str]) -> None:
    diffs = [result["local_diff"] for result in results] + [result["global_diff"] for result in results]
    vmax = max(float(np.max(np.abs(diff))) for diff in diffs)
    fig, axes = plt.subplots(len(results), 2, figsize=(11, 8), constrained_layout=True)
    if len(results) == 1:
        axes = np.array([axes])

    for row, result in enumerate(results):
        panels = [
            ("Local estimate - target", result["local_diff"], result["local_metrics"]),
            ("Global estimate - target", result["global_diff"], result["global_metrics"]),
        ]
        for col, (title, diff, panel_metrics) in enumerate(panels):
            ax = axes[row, col]
            image = ax.imshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(
                f"{result['label']} {title}\n"
                f"c={result['c']}, max |err|={panel_metrics['max_abs_error']:.3g}",
                fontsize=10,
            )
            set_ticks(ax, labels)
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.046, pad=0.03)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_target_heatmap(path: Path, results: list[dict[str, Any]], labels: list[str]) -> None:
    targets = [result["target_window"] for result in results]
    vmax = max(float(np.max(np.abs(target))) for target in targets)
    fig, axes = plt.subplots(1, len(results), figsize=(11, 4.5), constrained_layout=True)
    if len(results) == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        image = ax.imshow(result["target_window"], cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{result['label']} target window\nc={result['c']}", fontsize=10)
        set_ticks(ax, labels)
    fig.colorbar(image, ax=axes, fraction=0.046, pad=0.03)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_matrix_archive(path: Path, results: list[dict[str, Any]], labels: list[str], summary: dict[str, Any]) -> None:
    arrays: dict[str, np.ndarray] = {
        "coordinate_labels": np.array(labels),
        "labels_json": np.array(json.dumps(summary, indent=2, sort_keys=True)),
    }
    for result in results:
        prefix = result["condition"]
        arrays[f"{prefix}_target_window"] = result["target_window"]
        arrays[f"{prefix}_local_hamiltonian_window"] = result["local_window"]
        arrays[f"{prefix}_global_hamiltonian_window"] = result["global_window"]
        arrays[f"{prefix}_local_error_window"] = result["local_diff"]
        arrays[f"{prefix}_global_error_window"] = result["global_diff"]
    np.savez_compressed(path, **arrays)


def json_ready_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "condition": result["condition"],
        "label": result["label"],
        "c": result["c"],
        "local_metrics": result["local_metrics"],
        "global_metrics": result["global_metrics"],
    }


def run_condition_window_checks(
    *,
    seed: int = 100,
    m: int = 100,
    samples: int = 100_000,
    beta: float = 0.5,
    locality: int = 4,
    window_modes: int = 5,
    start_mode: int = 1,
    threshold: float = 0.1,
    output_dir: str | Path = "condition_window_checks",
) -> dict[str, Any]:
    if window_modes < 1:
        raise ValueError("window_modes must be positive.")
    if start_mode < 1 or start_mode + window_modes - 1 > m:
        raise ValueError("The requested window is outside the mode range.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    labels = coordinate_labels(start_mode, window_modes)
    results = [
        compute_condition(
            condition=item["condition"],
            label=item["label"],
            c=item["c"],
            seed=seed,
            m=m,
            samples=samples,
            beta=beta,
            locality=locality,
            start_mode=start_mode,
            window_modes=window_modes,
        )
        for item in CONDITIONS
    ]

    for result in results:
        write_long_csv(output_path / f"{result['condition']}_first_modes_window.csv", result, labels)

    error_heatmap = output_path / "condition_window_error_heatmaps.png"
    target_heatmap = output_path / "condition_window_target_heatmaps.png"
    matrix_archive = output_path / "condition_window_matrices.npz"
    write_error_heatmap(error_heatmap, results, labels)
    write_target_heatmap(target_heatmap, results, labels)

    summary = {
        "seed": seed,
        "m": m,
        "samples": samples,
        "beta": beta,
        "locality": locality,
        "window_modes": window_modes,
        "start_mode": start_mode,
        "modes": list(range(start_mode, start_mode + window_modes)),
        "coordinate_labels": labels,
        "threshold": threshold,
        "conditions": [json_ready_result(result) for result in results],
        "error_heatmap": str(error_heatmap),
        "target_heatmap": str(target_heatmap),
        "matrix_archive": str(matrix_archive),
        "passed": all(result["local_metrics"]["max_abs_error"] <= threshold for result in results),
    }
    write_matrix_archive(matrix_archive, results, labels, summary)
    (output_path / "condition_window_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    return summary


def main() -> None:
    args = parse_args()
    summary = run_condition_window_checks(
        seed=args.seed,
        m=args.m,
        samples=args.samples,
        beta=args.beta,
        locality=args.locality,
        window_modes=args.window_modes,
        start_mode=args.start_mode,
        threshold=args.threshold,
        output_dir=args.output_dir,
    )

    print("Condition window comparison")
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
