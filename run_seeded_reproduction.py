#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qgl_matplotlib_cache"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

import qgl_reproduce as qgl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fully seeded reproduction of the NatComm numerical plots. "
            "By default this uses the manuscript-scale parameter choices and can be slow."
        )
    )
    parser.add_argument("--seed", type=int, default=20240527, help="Base random seed.")
    parser.add_argument("--beta", type=float, default=0.5, help="Inverse-temperature prefactor.")
    parser.add_argument("--ill-c", type=float, default=0.0, help="Conditioning parameter for the ill-conditioned chain.")
    parser.add_argument("--well-c", type=float, default=0.1, help="Conditioning parameter for the well-conditioned chain.")
    parser.add_argument("--mode-samples", type=int, default=10_000, help="Samples per mode-sweep realization.")
    parser.add_argument("--mode-repeats", type=int, default=5, help="Realizations averaged for each mode-sweep point.")
    parser.add_argument("--mode-locality", type=int, default=3, help="Locality parameter used in the mode-sweep local reconstruction.")
    parser.add_argument("--m-start", type=int, default=100, help="First m value in the mode sweep.")
    parser.add_argument("--m-stop", type=int, default=1150, help="Last m value in the mode sweep, inclusive.")
    parser.add_argument("--m-step", type=int, default=50, help="Step size for the mode sweep.")
    parser.add_argument("--l-m", type=int, default=100, help="System size for the locality sweep.")
    parser.add_argument("--l-samples", type=int, nargs="+", default=[10_000, 100_000], help="Sample counts for the locality sweep.")
    parser.add_argument("--l-values", type=int, nargs="+", default=[2, 4, 6, 8, 10], help="Locality parameters to plot.")
    parser.add_argument("--l-repeats", type=int, default=1, help="Realizations averaged for each locality-sweep point.")
    parser.add_argument(
        "--independent-l-samples",
        action="store_true",
        help="Use independent samples for every l instead of reusing one covariance estimate per sample count.",
    )
    parser.add_argument("--data-dir", default="reproduction_data_seeded", help="Directory for generated CSV files.")
    parser.add_argument(
        "--plot-dir",
        default="reproduced_plots_seeded",
        help="Directory for generated PNG and vector PDF figures.",
    )
    parser.add_argument(
        "--only",
        choices=["all", "mode", "locality"],
        default="all",
        help=(
            "Choose which part of the reproduction to run. "
            "'locality' regenerates only the m=100 locality-parameter plot."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a tiny smoke-test version of the pipeline instead of the manuscript-scale simulation.",
    )
    return parser.parse_args()


def resolve_dir(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def average_global_error(
    m: int,
    samples: int,
    repeats: int,
    c: float,
    beta: float,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    errors = []
    for repeat in range(repeats):
        sample_matrix = qgl.sample_measurements(m, samples, c, rng, beta)
        cov_est = qgl.covariance_estimate(sample_matrix)
        error = qgl.global_reconstruction_error(cov_est, m, c, beta)
        errors.append(error)
        print(f"global baseline N={samples} repeat={repeat + 1}: error={error:.6g}")
    return float(np.mean(errors)), float(np.std(errors, ddof=1)) if repeats > 1 else 0.0


def main() -> None:
    args = parse_args()
    repo_root = qgl.find_repo_root(Path.cwd())
    data_dir = resolve_dir(repo_root, args.data_dir)
    plot_dir = resolve_dir(repo_root, args.plot_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        args.mode_samples = 500
        args.mode_repeats = 1
        args.m_start = 10
        args.m_stop = 20
        args.m_step = 10
        args.l_m = 10
        args.l_samples = [500, 1000]
        args.l_values = [1, 2]
        args.l_repeats = 1

    m_values = tuple(range(args.m_start, args.m_stop + 1, args.m_step))
    if len(args.l_samples) < 2:
        raise ValueError("--l-samples must contain at least two sample counts.")

    run_config = {
        "seed": args.seed,
        "beta": args.beta,
        "ill_c": args.ill_c,
        "well_c": args.well_c,
        "mode_samples": args.mode_samples,
        "mode_repeats": args.mode_repeats,
        "mode_locality": args.mode_locality,
        "m_values": list(m_values),
        "l_m": args.l_m,
        "l_samples": args.l_samples,
        "l_values": args.l_values,
        "l_repeats": args.l_repeats,
        "reuse_samples_across_l": not args.independent_l_samples,
        "data_dir": str(Path(args.data_dir)),
        "plot_dir": str(Path(args.plot_dir)),
        "quick": args.quick,
        "only": args.only,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (data_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    print("Seeded reproduction run")
    print(json.dumps(run_config, indent=2))

    t0 = perf_counter()

    ill_config = qgl.ModeSweepConfig(
        condition="ill",
        c=args.ill_c,
        samples=args.mode_samples,
        locality=args.mode_locality,
        repeats=args.mode_repeats,
        beta=args.beta,
        seed=args.seed + 101,
        m_values=m_values,
    )
    well_config = qgl.ModeSweepConfig(
        condition="well",
        c=args.well_c,
        samples=args.mode_samples,
        locality=args.mode_locality,
        repeats=args.mode_repeats,
        beta=args.beta,
        seed=args.seed + 202,
        m_values=m_values,
    )
    l_config = qgl.LSweepConfig(
        c=args.well_c,
        m=args.l_m,
        sample_counts=tuple(args.l_samples),
        l_values=tuple(args.l_values),
        repeats=args.l_repeats,
        beta=args.beta,
        seed=args.seed + 303,
        reuse_samples_across_l=not args.independent_l_samples,
    )

    if args.only in {"all", "mode"}:
        print("\nRunning ill-conditioned mode sweep...")
        ill_df = qgl.run_mode_sweep(ill_config, data_dir / "mode_sweep_ill_seeded.csv", progress=True)

        print("\nRunning well-conditioned mode sweep...")
        well_df = qgl.run_mode_sweep(well_config, data_dir / "mode_sweep_well_seeded.csv", progress=True)

        print("\nWriting mode-sweep plots...")
        qgl.plot_mode_sweep(ill_df, "ill-conditioned", plot_dir / "simulation_errors_plot_ill")
        plt.close("all")
        qgl.plot_mode_sweep(well_df, "well-conditioned", plot_dir / "simulation_errors_plot_well")
        plt.close("all")

    if args.only in {"all", "locality"}:
        print("\nRunning sampled locality sweep...")
        sampled_l_df = qgl.run_l_sweep(l_config, data_dir / "l_sweep_sampled_seeded.csv", progress=True)

        print("\nRunning exact locality sweep...")
        exact_l_df = qgl.exact_l_sweep(l_config)
        exact_l_df.to_csv(data_dir / "l_sweep_exact_seeded.csv", index=False)

        high_sample_count = max(args.l_samples)
        print("\nRunning sampled global baseline for locality plot...")
        global_sampled_error, global_sampled_std = average_global_error(
            m=args.l_m,
            samples=high_sample_count,
            repeats=args.l_repeats,
            c=args.well_c,
            beta=args.beta,
            seed=args.seed + 404,
        )
        classical_exact_error = qgl.classical_exact_error(args.l_m, args.well_c, args.beta)
        pd.DataFrame(
            [
                {
                    "m": args.l_m,
                    "c": args.well_c,
                    "Beta": args.beta,
                    "GlobalSampledSamples": high_sample_count,
                    "GlobalSampledRepeats": args.l_repeats,
                    "GlobalSampledError": global_sampled_error,
                    "GlobalSampledErrorStd": global_sampled_std,
                    "ClassicalExactError": classical_exact_error,
                    "Seed": args.seed + 404,
                }
            ]
        ).to_csv(data_dir / "l_sweep_baselines_seeded.csv", index=False)

        print("\nWriting locality plot...")
        qgl.plot_l_sweep(
            sampled_l_df,
            exact_l_df,
            global_sampled_error=global_sampled_error,
            classical_exact=classical_exact_error,
            output_path=plot_dir / "improved_plot",
        )
        plt.close("all")

    elapsed = perf_counter() - t0
    summary = {
        "elapsed_seconds": elapsed,
        "csv_files": sorted(path.name for path in data_dir.glob("*.csv")),
        "plot_files": sorted(
            path.name for path in plot_dir.iterdir() if path.suffix.lower() in {".png", ".pdf"}
        ),
    }
    (data_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\nDone.")
    print(f"Data:  {data_dir}")
    print(f"Plots: {plot_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
