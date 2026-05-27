#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qgl_matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qgl_reproduce as qgl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a random small mode-window of the reconstructed Hamiltonian against the target."
    )
    parser.add_argument("--seed", type=int, default=100, help="Seed for sampling and window selection.")
    parser.add_argument("--m", type=int, default=100, help="Number of modes.")
    parser.add_argument("--samples", type=int, default=100_000, help="Number of heterodyne samples.")
    parser.add_argument("--c", type=float, default=0.1, help="Conditioning parameter in H = beta * (NormalPrecision + c I).")
    parser.add_argument("--beta", type=float, default=0.5, help="Inverse-temperature prefactor.")
    parser.add_argument("--locality", type=int, default=4, help="Locality parameter for local reconstruction.")
    parser.add_argument("--window-modes", type=int, default=5, help="Number of consecutive modes to inspect.")
    parser.add_argument(
        "--start-mode",
        type=int,
        default=None,
        help="Optional 1-based first mode of the window. If omitted, a seeded random window is chosen.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Pass/fail threshold for max absolute error of the local window.",
    )
    parser.add_argument("--output-dir", default="sanity_checks", help="Directory for CSV/PNG outputs.")
    parser.add_argument("--no-global", action="store_true", help="Skip the global estimator.")
    parser.add_argument("--no-matrices", action="store_true", help="Do not print the submatrices.")
    return parser.parse_args()


def mode_indices(start_mode: int, window_modes: int) -> list[int]:
    indices = []
    for mode in range(start_mode, start_mode + window_modes):
        base = 2 * (mode - 1)
        indices.extend([base, base + 1])
    return indices


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


def print_matrix(name: str, matrix: np.ndarray) -> None:
    print(f"\n{name}")
    print(np.array2string(matrix, precision=4, suppress_small=True, max_line_width=160))


def write_long_csv(path: Path, modes: list[int], target: np.ndarray, local: np.ndarray, global_est: np.ndarray | None) -> None:
    rows = []
    dim = len(modes) * 2
    labels = [f"mode{mode}_{quad}" for mode in modes for quad in ("x", "p")]
    for i in range(dim):
        for j in range(dim):
            row = {
                "row": labels[i],
                "col": labels[j],
                "target": target[i, j],
                "local_estimate": local[i, j],
                "local_error": local[i, j] - target[i, j],
            }
            if global_est is not None:
                row["global_estimate"] = global_est[i, j]
                row["global_error"] = global_est[i, j] - target[i, j]
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_heatmap(path: Path, local_diff: np.ndarray, global_diff: np.ndarray | None) -> None:
    if global_diff is None:
        fig, axes = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
        axes = [axes]
        diffs = [local_diff]
        titles = ["Local estimate - target"]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        diffs = [local_diff, global_diff]
        titles = ["Local estimate - target", "Global estimate - target"]

    vmax = max(float(np.max(np.abs(diff))) for diff in diffs)
    for ax, diff, title in zip(axes, diffs, titles):
        image = ax.imshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("submatrix index")
        ax.set_ylabel("submatrix index")
    fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.window_modes < 1:
        raise ValueError("--window-modes must be positive.")
    if args.window_modes > args.m:
        raise ValueError("--window-modes cannot exceed --m.")

    rng = np.random.default_rng(args.seed)
    if args.start_mode is None:
        window_rng = np.random.default_rng(args.seed + 999)
        start_mode = int(window_rng.integers(1, args.m - args.window_modes + 2))
    else:
        start_mode = args.start_mode
    if start_mode < 1 or start_mode + args.window_modes - 1 > args.m:
        raise ValueError("The requested window is outside the mode range.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Small-window Hamiltonian sanity check")
    print(
        json.dumps(
            {
                "seed": args.seed,
                "m": args.m,
                "samples": args.samples,
                "c": args.c,
                "beta": args.beta,
                "locality": args.locality,
                "window_modes": args.window_modes,
                "start_mode": start_mode,
                "threshold": args.threshold,
            },
            indent=2,
        )
    )

    print("\nSampling measurement data...")
    samples = qgl.sample_measurements(args.m, args.samples, args.c, rng, args.beta)
    cov_est = qgl.covariance_estimate(samples)

    print("Reconstructing local Hamiltonian estimate...")
    local_est = qgl.local_hamiltonian_estimate(cov_est, args.m, args.locality, args.c, args.beta)

    global_est = None
    if not args.no_global:
        print("Reconstructing global Hamiltonian estimate...")
        global_est = qgl.global_hamiltonian_estimate(cov_est, args.m, args.c, args.beta)

    target = qgl.hamiltonian_matrix(2 * args.m, args.c, args.beta)

    modes = list(range(start_mode, start_mode + args.window_modes))
    indices = mode_indices(start_mode, args.window_modes)
    target_window = target[np.ix_(indices, indices)]
    local_window = local_est[np.ix_(indices, indices)]
    global_window = global_est[np.ix_(indices, indices)] if global_est is not None else None

    local_metrics = metrics(target_window, local_window)
    global_metrics = metrics(target_window, global_window) if global_window is not None else None

    print(f"\nWindow modes: {modes}")
    print("\nLocal window metrics")
    print(json.dumps(local_metrics, indent=2))
    if global_metrics is not None:
        print("\nGlobal window metrics")
        print(json.dumps(global_metrics, indent=2))

    if not args.no_matrices:
        print_matrix("Target submatrix", target_window)
        print_matrix("Local estimate submatrix", local_window)
        print_matrix("Local estimate - target", local_window - target_window)
        if global_window is not None:
            print_matrix("Global estimate submatrix", global_window)
            print_matrix("Global estimate - target", global_window - target_window)

    csv_path = output_dir / "window_submatrix_check.csv"
    write_long_csv(csv_path, modes, target_window, local_window, global_window)
    heatmap_path = output_dir / "window_submatrix_errors.png"
    write_heatmap(heatmap_path, local_window - target_window, None if global_window is None else global_window - target_window)

    summary = {
        "modes": modes,
        "local": local_metrics,
        "global": global_metrics,
        "csv": str(csv_path),
        "heatmap": str(heatmap_path),
    }
    summary_path = output_dir / "window_submatrix_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    passed = local_metrics["max_abs_error"] <= args.threshold
    print("\nPASS" if passed else "\nFAIL")
    print(f"Local max abs error {local_metrics['max_abs_error']:.6g} vs threshold {args.threshold:.6g}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {heatmap_path}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
