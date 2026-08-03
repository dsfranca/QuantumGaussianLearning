#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

import qgl_reproduce as qgl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dump labeled matrix archives for the seeded NatComm plot reproduction. "
            "Use --scope audit for a small colleague-facing diagnostic dump."
        )
    )
    parser.add_argument("--seed", type=int, default=100, help="Base random seed.")
    parser.add_argument(
        "--scope",
        choices=["audit", "locality", "mode", "all"],
        default="locality",
        help="Which plot family to dump. Use 'audit' for a compact window dump, or 'all' for the full large dump.",
    )
    parser.add_argument("--output-dir", default="matrix_dumps", help="Directory for matrix archives and manifests.")
    parser.add_argument("--beta", type=float, default=0.5, help="Inverse-temperature prefactor.")
    parser.add_argument("--ill-c", type=float, default=0.0, help="Conditioning parameter for the ill-conditioned chain.")
    parser.add_argument("--well-c", type=float, default=0.1, help="Conditioning parameter for the well-conditioned chain.")
    parser.add_argument("--mode-samples", type=int, default=10_000, help="Samples per mode-sweep realization.")
    parser.add_argument("--mode-repeats", type=int, default=5, help="Realizations for each mode-sweep point.")
    parser.add_argument("--mode-locality", type=int, default=3, help="Locality parameter used in the mode sweep.")
    parser.add_argument("--m-start", type=int, default=100, help="First m value in the mode sweep.")
    parser.add_argument("--m-stop", type=int, default=1150, help="Last m value in the mode sweep, inclusive.")
    parser.add_argument("--m-step", type=int, default=50, help="Step size for the mode sweep.")
    parser.add_argument("--l-m", type=int, default=100, help="System size for the locality sweep.")
    parser.add_argument("--l-samples", type=int, nargs="+", default=[10_000, 100_000], help="Sample counts for the locality sweep.")
    parser.add_argument("--l-values", type=int, nargs="+", default=[2, 4, 6, 8, 10], help="Locality values for the locality sweep.")
    parser.add_argument("--l-repeats", type=int, default=1, help="Realizations for each locality-sweep sample count.")
    parser.add_argument("--audit-window-modes", type=int, default=5, help="Number of modes in the compact audit submatrix.")
    parser.add_argument(
        "--audit-window-start",
        type=int,
        default=None,
        help="Optional 1-based first mode for the audit window. By default it is chosen deterministically from the seed.",
    )
    parser.add_argument(
        "--independent-l-samples",
        action="store_true",
        help="Use independent samples for every l instead of reusing one covariance estimate per sample count.",
    )
    parser.add_argument(
        "--include-samples",
        action="store_true",
        help="Also store raw Gaussian sample matrices. This can make dumps very large.",
    )
    parser.add_argument(
        "--uncompressed",
        action="store_true",
        help="Write uncompressed NPZ archives. Faster, but substantially larger.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a tiny smoke-test dump instead of manuscript-scale dimensions.",
    )
    return parser.parse_args()


def resolve_dir(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def slug_float(value: float) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def relpath(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def labels_array(metadata: dict[str, Any]) -> np.ndarray:
    return np.array(json.dumps(metadata, indent=2, sort_keys=True))


def save_npz(
    path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    compressed: bool,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload["labels_json"] = labels_array(metadata)
    if compressed:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)
    return path.stat().st_size


def error_against_target(target: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.max(np.abs(target - estimate)))


def hamiltonian_from_inverse(inv_cov_omega: np.ndarray, m: int, c: float, beta: float) -> np.ndarray:
    return qgl.mask_hamiltonian(qgl.hamiltonian_from_inverse(inv_cov_omega, m), c, beta)


def static_arrays(m: int, c: float, beta: float) -> dict[str, np.ndarray]:
    dim = 2 * m
    return {
        "target_hamiltonian": qgl.hamiltonian_matrix(dim, c, beta),
        "measurement_covariance": qgl.measurement_covariance(m, c, beta),
        "omega_matrix": qgl.omega_matrix(m),
        "hamiltonian_mask": qgl.hamiltonian_mask(dim, c, beta),
    }


def audit_window_indices(m: int, window_modes: int, start_mode: int | None, seed: int) -> tuple[list[int], np.ndarray]:
    if window_modes < 1:
        raise ValueError("--audit-window-modes must be at least 1.")
    if window_modes > m:
        raise ValueError("--audit-window-modes cannot exceed the system size m.")
    if start_mode is None:
        rng = np.random.default_rng(seed + 505)
        start_mode = int(rng.integers(1, m - window_modes + 2))
    if start_mode < 1 or start_mode + window_modes - 1 > m:
        raise ValueError("--audit-window-start must define a window contained in 1..m.")
    modes = list(range(start_mode, start_mode + window_modes))
    coord_indices = np.arange(2 * (start_mode - 1), 2 * (start_mode + window_modes - 1), dtype=int)
    return modes, coord_indices


def window_submatrix(matrix: np.ndarray, coord_indices: np.ndarray) -> np.ndarray:
    return matrix[np.ix_(coord_indices, coord_indices)]


def append_manifest_row(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    path: Path,
    file_size_bytes: int,
    arrays: dict[str, np.ndarray],
    array_labels: dict[str, str],
    metadata: dict[str, Any],
    estimator: str,
    error: float | None = None,
    locality: int | str | None = None,
    repeat: int | str | None = None,
    samples: int | str | None = None,
    sample_seed: int | str | None = None,
) -> None:
    rows.append(
        {
            "file": relpath(path, output_dir),
            "file_size_bytes": file_size_bytes,
            "scope": metadata.get("scope", ""),
            "case_id": metadata.get("case_id", ""),
            "plot_family": metadata.get("plot_family", ""),
            "condition": metadata.get("condition", ""),
            "estimator": estimator,
            "m": metadata.get("m", ""),
            "dim": metadata.get("dim", ""),
            "c": metadata.get("c", ""),
            "beta": metadata.get("beta", ""),
            "samples": samples if samples is not None else metadata.get("samples", ""),
            "locality": locality if locality is not None else metadata.get("locality", ""),
            "repeat": repeat if repeat is not None else metadata.get("repeat", ""),
            "base_seed": metadata.get("base_seed", ""),
            "sample_seed": sample_seed if sample_seed is not None else metadata.get("sample_seed", ""),
            "error_max_abs": "" if error is None else error,
            "arrays": ";".join(arrays.keys()),
            "array_labels": json.dumps(array_labels, sort_keys=True),
        }
    )


def write_static_bundle(
    output_dir: Path,
    manifest_rows: list[dict[str, Any]],
    *,
    scope: str,
    plot_family: str,
    case_id: str,
    condition: str,
    m: int,
    c: float,
    beta: float,
    base_seed: int,
    compressed: bool,
) -> Path:
    arrays = static_arrays(m, c, beta)
    array_labels = {
        "target_hamiltonian": "Exact target Hamiltonian H for this parameter set.",
        "measurement_covariance": "Exact measurement covariance used to sample Gaussian data.",
        "omega_matrix": "Canonical symplectic form Omega.",
        "hamiltonian_mask": "Sparsity mask used before computing the max-entry Hamiltonian error.",
    }
    metadata = {
        "scope": scope,
        "plot_family": plot_family,
        "case_id": case_id,
        "condition": condition,
        "m": m,
        "dim": 2 * m,
        "c": c,
        "beta": beta,
        "base_seed": base_seed,
        "arrays": array_labels,
    }
    path = output_dir / "static" / f"{case_id}_static_m{m:04d}_c{slug_float(c)}_beta{slug_float(beta)}.npz"
    file_size = save_npz(path, arrays, metadata, compressed)
    append_manifest_row(
        manifest_rows,
        output_dir=output_dir,
        path=path,
        file_size_bytes=file_size,
        arrays=arrays,
        array_labels=array_labels,
        metadata=metadata,
        estimator="static",
    )
    return path


def locality_exact_bundle(
    output_dir: Path,
    manifest_rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    compressed: bool,
) -> None:
    m = args.l_m
    c = args.well_c
    beta = args.beta
    dim = 2 * m
    target = qgl.hamiltonian_matrix(dim, c, beta)
    cov_exact = qgl.measurement_covariance(m, c, beta)
    exact_inv = qgl.inverse_cov_omega(cov_exact, m)
    classical_h = qgl.mask_hamiltonian(np.real(qgl.hermitize(exact_inv)), c, beta)

    arrays: dict[str, np.ndarray] = {
        "exact_inverse_cov_omega": exact_inv,
        "classical_exact_hamiltonian_estimate": classical_h,
    }
    array_labels = {
        "exact_inverse_cov_omega": "Inverse of 2*(Sigma-I/2)-i*Omega for the exact covariance.",
        "classical_exact_hamiltonian_estimate": "Classical exact-covariance baseline shown as the dotted line in the inset.",
    }
    row_errors: list[tuple[int, float]] = []
    for locality in args.l_values:
        local_inv = qgl.local_inverse_reconstruction(cov_exact, m, locality)
        local_h = hamiltonian_from_inverse(local_inv, m, c, beta)
        arrays[f"local_inverse_l{locality}"] = local_inv
        arrays[f"local_hamiltonian_l{locality}"] = local_h
        array_labels[f"local_inverse_l{locality}"] = f"Local inverse reconstruction from exact covariance with locality l={locality}."
        array_labels[f"local_hamiltonian_l{locality}"] = f"Local Hamiltonian estimate from exact covariance with locality l={locality}."
        row_errors.append((locality, error_against_target(target, local_h)))

    metadata = {
        "scope": args.scope,
        "plot_family": "locality",
        "case_id": "locality_exact",
        "condition": "well",
        "m": m,
        "dim": dim,
        "c": c,
        "beta": beta,
        "base_seed": args.seed,
        "arrays": array_labels,
    }
    path = output_dir / "locality" / f"locality_exact_m{m:04d}_c{slug_float(c)}_beta{slug_float(beta)}.npz"
    file_size = save_npz(path, arrays, metadata, compressed)
    for locality, error in row_errors:
        append_manifest_row(
            manifest_rows,
            output_dir=output_dir,
            path=path,
            file_size_bytes=file_size,
            arrays=arrays,
            array_labels=array_labels,
            metadata=metadata,
            estimator="local_exact",
            error=error,
            locality=locality,
        )
    append_manifest_row(
        manifest_rows,
        output_dir=output_dir,
        path=path,
        file_size_bytes=file_size,
        arrays=arrays,
        array_labels=array_labels,
        metadata=metadata,
        estimator="classical_exact",
        error=error_against_target(target, classical_h),
    )


def locality_sampled_bundles(
    output_dir: Path,
    manifest_rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    compressed: bool,
) -> None:
    m = args.l_m
    c = args.well_c
    beta = args.beta
    dim = 2 * m
    target = qgl.hamiltonian_matrix(dim, c, beta)
    rng = np.random.default_rng(args.seed + 303)

    for sample_count in args.l_samples:
        for repeat in range(1, args.l_repeats + 1):
            if not args.independent_l_samples:
                samples = qgl.sample_measurements(m, sample_count, c, rng, beta)
                cov_est = qgl.covariance_estimate(samples)
                arrays: dict[str, np.ndarray] = {"empirical_covariance": cov_est}
                array_labels = {
                    "empirical_covariance": (
                        "Sample covariance X^T X/N shared by all localities in this locality-sweep repeat."
                    ),
                }
                if args.include_samples:
                    arrays["sample_matrix"] = samples
                    array_labels["sample_matrix"] = "Raw Gaussian samples X used to form empirical_covariance."

                row_errors: list[tuple[int, float]] = []
                for locality in args.l_values:
                    local_inv = qgl.local_inverse_reconstruction(cov_est, m, locality)
                    local_h = hamiltonian_from_inverse(local_inv, m, c, beta)
                    arrays[f"local_inverse_l{locality}"] = local_inv
                    arrays[f"local_hamiltonian_l{locality}"] = local_h
                    array_labels[f"local_inverse_l{locality}"] = (
                        f"Local inverse reconstruction from sampled covariance with locality l={locality}."
                    )
                    array_labels[f"local_hamiltonian_l{locality}"] = (
                        f"Local Hamiltonian estimate from sampled covariance with locality l={locality}."
                    )
                    row_errors.append((locality, error_against_target(target, local_h)))

                metadata = {
                    "scope": args.scope,
                    "plot_family": "locality",
                    "case_id": f"locality_sampled_N{sample_count}_repeat{repeat:03d}",
                    "condition": "well",
                    "m": m,
                    "dim": dim,
                    "c": c,
                    "beta": beta,
                    "samples": sample_count,
                    "repeat": repeat,
                    "base_seed": args.seed,
                    "sample_seed": args.seed + 303,
                    "reuse_samples_across_l": True,
                    "arrays": array_labels,
                }
                path = (
                    output_dir
                    / "locality"
                    / f"locality_sampled_N{sample_count}_repeat{repeat:03d}_seed{args.seed + 303}.npz"
                )
                file_size = save_npz(path, arrays, metadata, compressed)
                for locality, error in row_errors:
                    append_manifest_row(
                        manifest_rows,
                        output_dir=output_dir,
                        path=path,
                        file_size_bytes=file_size,
                        arrays=arrays,
                        array_labels=array_labels,
                        metadata=metadata,
                        estimator="local_sampled",
                        error=error,
                        locality=locality,
                        repeat=repeat,
                        samples=sample_count,
                        sample_seed=args.seed + 303,
                    )
            else:
                for locality in args.l_values:
                    samples = qgl.sample_measurements(m, sample_count, c, rng, beta)
                    cov_est = qgl.covariance_estimate(samples)
                    local_inv = qgl.local_inverse_reconstruction(cov_est, m, locality)
                    local_h = hamiltonian_from_inverse(local_inv, m, c, beta)
                    arrays = {
                        "empirical_covariance": cov_est,
                        f"local_inverse_l{locality}": local_inv,
                        f"local_hamiltonian_l{locality}": local_h,
                    }
                    array_labels = {
                        "empirical_covariance": "Sample covariance X^T X/N for this independent locality run.",
                        f"local_inverse_l{locality}": (
                            f"Local inverse reconstruction from sampled covariance with locality l={locality}."
                        ),
                        f"local_hamiltonian_l{locality}": (
                            f"Local Hamiltonian estimate from sampled covariance with locality l={locality}."
                        ),
                    }
                    if args.include_samples:
                        arrays["sample_matrix"] = samples
                        array_labels["sample_matrix"] = "Raw Gaussian samples X used to form empirical_covariance."

                    metadata = {
                        "scope": args.scope,
                        "plot_family": "locality",
                        "case_id": f"locality_sampled_N{sample_count}_l{locality}_repeat{repeat:03d}",
                        "condition": "well",
                        "m": m,
                        "dim": dim,
                        "c": c,
                        "beta": beta,
                        "samples": sample_count,
                        "locality": locality,
                        "repeat": repeat,
                        "base_seed": args.seed,
                        "sample_seed": args.seed + 303,
                        "reuse_samples_across_l": False,
                        "arrays": array_labels,
                    }
                    path = (
                        output_dir
                        / "locality"
                        / (
                            f"locality_sampled_N{sample_count}_l{locality}_"
                            f"repeat{repeat:03d}_seed{args.seed + 303}.npz"
                        )
                    )
                    file_size = save_npz(path, arrays, metadata, compressed)
                    append_manifest_row(
                        manifest_rows,
                        output_dir=output_dir,
                        path=path,
                        file_size_bytes=file_size,
                        arrays=arrays,
                        array_labels=array_labels,
                        metadata=metadata,
                        estimator="local_sampled",
                        error=error_against_target(target, local_h),
                        locality=locality,
                        repeat=repeat,
                        samples=sample_count,
                        sample_seed=args.seed + 303,
                    )


def locality_global_baseline_bundles(
    output_dir: Path,
    manifest_rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    compressed: bool,
) -> None:
    m = args.l_m
    c = args.well_c
    beta = args.beta
    dim = 2 * m
    sample_count = max(args.l_samples)
    target = qgl.hamiltonian_matrix(dim, c, beta)
    rng = np.random.default_rng(args.seed + 404)

    for repeat in range(1, args.l_repeats + 1):
        samples = qgl.sample_measurements(m, sample_count, c, rng, beta)
        cov_est = qgl.covariance_estimate(samples)
        global_inv = qgl.inverse_cov_omega(cov_est, m)
        global_h = hamiltonian_from_inverse(global_inv, m, c, beta)
        arrays: dict[str, np.ndarray] = {
            "empirical_covariance": cov_est,
            "global_inverse_cov_omega": global_inv,
            "global_hamiltonian_estimate": global_h,
        }
        array_labels = {
            "empirical_covariance": "Sample covariance X^T X/N used for the sampled global baseline.",
            "global_inverse_cov_omega": "Global inverse of 2*(Sigma_hat-I/2)-i*Omega.",
            "global_hamiltonian_estimate": "Global plug-in Hamiltonian estimate shown as the sampled baseline.",
        }
        if args.include_samples:
            arrays["sample_matrix"] = samples
            array_labels["sample_matrix"] = "Raw Gaussian samples X used to form empirical_covariance."

        metadata = {
            "scope": args.scope,
            "plot_family": "locality",
            "case_id": f"locality_global_baseline_N{sample_count}_repeat{repeat:03d}",
            "condition": "well",
            "m": m,
            "dim": dim,
            "c": c,
            "beta": beta,
            "samples": sample_count,
            "repeat": repeat,
            "base_seed": args.seed,
            "sample_seed": args.seed + 404,
            "arrays": array_labels,
        }
        path = (
            output_dir
            / "locality"
            / f"locality_global_baseline_N{sample_count}_repeat{repeat:03d}_seed{args.seed + 404}.npz"
        )
        file_size = save_npz(path, arrays, metadata, compressed)
        append_manifest_row(
            manifest_rows,
            output_dir=output_dir,
            path=path,
            file_size_bytes=file_size,
            arrays=arrays,
            array_labels=array_labels,
            metadata=metadata,
            estimator="global_sampled",
            error=error_against_target(target, global_h),
            repeat=repeat,
            samples=sample_count,
            sample_seed=args.seed + 404,
        )


def dump_audit_scope(
    output_dir: Path,
    manifest_rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    compressed: bool,
) -> None:
    if args.independent_l_samples:
        raise ValueError("--scope audit expects the default shared sampled covariance across l.")

    print("Dumping compact audit matrices...")
    m = args.l_m
    c = args.well_c
    beta = args.beta
    dim = 2 * m
    high_sample_count = max(args.l_samples)
    modes, coord_indices = audit_window_indices(m, args.audit_window_modes, args.audit_window_start, args.seed)

    target = qgl.hamiltonian_matrix(dim, c, beta)
    cov_exact = qgl.measurement_covariance(m, c, beta)
    target_window = window_submatrix(target, coord_indices)
    arrays: dict[str, np.ndarray] = {
        "window_modes_one_based": np.array(modes, dtype=int),
        "window_coordinate_indices_zero_based": coord_indices,
        "target_hamiltonian_window": target_window,
        "exact_measurement_covariance_window": window_submatrix(cov_exact, coord_indices),
    }
    array_labels = {
        "window_modes_one_based": "One-based mode numbers included in the audit submatrix.",
        "window_coordinate_indices_zero_based": "Zero-based coordinate indices for the window in the full 2m-dimensional matrices.",
        "target_hamiltonian_window": "Target Hamiltonian restricted to the audit window.",
        "exact_measurement_covariance_window": "Exact measurement covariance restricted to the audit window.",
    }
    error_rows: list[dict[str, Any]] = []

    rng = np.random.default_rng(args.seed + 303)
    for sample_count in args.l_samples:
        for repeat in range(1, args.l_repeats + 1):
            samples = qgl.sample_measurements(m, sample_count, c, rng, beta)
            cov_est = qgl.covariance_estimate(samples)
            is_stored_window = sample_count == high_sample_count and repeat == 1
            if is_stored_window:
                arrays["sampled_measurement_covariance_window"] = window_submatrix(cov_est, coord_indices)
                array_labels["sampled_measurement_covariance_window"] = (
                    f"Sample covariance window for N={sample_count}, repeat {repeat}, shared across l."
                )
            for locality in args.l_values:
                local_inv = qgl.local_inverse_reconstruction(cov_est, m, locality)
                local_h = hamiltonian_from_inverse(local_inv, m, c, beta)
                local_window = window_submatrix(local_h, coord_indices)
                error_rows.append(
                    {
                        "estimator": "local_sampled",
                        "samples": sample_count,
                        "locality": locality,
                        "repeat": repeat,
                        "seed": args.seed + 303,
                        "error_max_abs_full": error_against_target(target, local_h),
                        "error_max_abs_window": error_against_target(target_window, local_window),
                    }
                )
                if is_stored_window:
                    arrays[f"local_sampled_hamiltonian_window_l{locality}"] = local_window
                    arrays[f"local_sampled_error_window_l{locality}"] = local_window - target_window
                    array_labels[f"local_sampled_hamiltonian_window_l{locality}"] = (
                        f"Sampled local Hamiltonian estimate restricted to the audit window, l={locality}."
                    )
                    array_labels[f"local_sampled_error_window_l{locality}"] = (
                        f"Sampled local Hamiltonian window minus target window, l={locality}."
                    )

    exact_inv = qgl.inverse_cov_omega(cov_exact, m)
    classical_h = qgl.mask_hamiltonian(np.real(qgl.hermitize(exact_inv)), c, beta)
    classical_window = window_submatrix(classical_h, coord_indices)
    arrays["classical_exact_hamiltonian_window"] = classical_window
    arrays["classical_exact_error_window"] = classical_window - target_window
    array_labels["classical_exact_hamiltonian_window"] = "Classical exact-covariance Hamiltonian baseline restricted to the audit window."
    array_labels["classical_exact_error_window"] = "Classical exact-covariance window minus target window."
    error_rows.append(
        {
            "estimator": "classical_exact",
            "samples": "",
            "locality": "",
            "repeat": "",
            "seed": "",
            "error_max_abs_full": error_against_target(target, classical_h),
            "error_max_abs_window": error_against_target(target_window, classical_window),
        }
    )

    for locality in args.l_values:
        exact_local_inv = qgl.local_inverse_reconstruction(cov_exact, m, locality)
        exact_local_h = hamiltonian_from_inverse(exact_local_inv, m, c, beta)
        exact_local_window = window_submatrix(exact_local_h, coord_indices)
        arrays[f"local_exact_hamiltonian_window_l{locality}"] = exact_local_window
        arrays[f"local_exact_error_window_l{locality}"] = exact_local_window - target_window
        array_labels[f"local_exact_hamiltonian_window_l{locality}"] = (
            f"Exact-covariance local Hamiltonian estimate restricted to the audit window, l={locality}."
        )
        array_labels[f"local_exact_error_window_l{locality}"] = (
            f"Exact-covariance local Hamiltonian window minus target window, l={locality}."
        )
        error_rows.append(
            {
                "estimator": "local_exact",
                "samples": "",
                "locality": locality,
                "repeat": "",
                "seed": "",
                "error_max_abs_full": error_against_target(target, exact_local_h),
                "error_max_abs_window": error_against_target(target_window, exact_local_window),
            }
        )

    global_rng = np.random.default_rng(args.seed + 404)
    for repeat in range(1, args.l_repeats + 1):
        samples = qgl.sample_measurements(m, high_sample_count, c, global_rng, beta)
        cov_est = qgl.covariance_estimate(samples)
        global_inv = qgl.inverse_cov_omega(cov_est, m)
        global_h = hamiltonian_from_inverse(global_inv, m, c, beta)
        global_window = window_submatrix(global_h, coord_indices)
        error_rows.append(
            {
                "estimator": "global_sampled",
                "samples": high_sample_count,
                "locality": "",
                "repeat": repeat,
                "seed": args.seed + 404,
                "error_max_abs_full": error_against_target(target, global_h),
                "error_max_abs_window": error_against_target(target_window, global_window),
            }
        )
        if repeat == 1:
            arrays["global_sampled_covariance_window"] = window_submatrix(cov_est, coord_indices)
            arrays["global_sampled_hamiltonian_window"] = global_window
            arrays["global_sampled_error_window"] = global_window - target_window
            array_labels["global_sampled_covariance_window"] = (
                f"Sample covariance window for the sampled global baseline, N={high_sample_count}."
            )
            array_labels["global_sampled_hamiltonian_window"] = (
                "Global sampled Hamiltonian estimate restricted to the audit window."
            )
            array_labels["global_sampled_error_window"] = "Global sampled Hamiltonian window minus target window."

    metadata = {
        "scope": args.scope,
        "plot_family": "locality",
        "case_id": "audit_window",
        "condition": "well",
        "m": m,
        "dim": dim,
        "c": c,
        "beta": beta,
        "samples": high_sample_count,
        "base_seed": args.seed,
        "sample_seed": args.seed + 303,
        "global_sample_seed": args.seed + 404,
        "window_modes_one_based": modes,
        "window_coordinate_indices_zero_based": coord_indices.tolist(),
        "reuse_samples_across_l": True,
        "arrays": array_labels,
    }
    path = output_dir / "audit" / f"audit_window_m{m:04d}_N{high_sample_count}_seed{args.seed}.npz"
    file_size = save_npz(path, arrays, metadata, compressed)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(error_rows).to_csv(output_dir / "audit_errors.csv", index=False)

    for row in error_rows:
        append_manifest_row(
            manifest_rows,
            output_dir=output_dir,
            path=path,
            file_size_bytes=file_size,
            arrays=arrays,
            array_labels=array_labels,
            metadata=metadata,
            estimator=row["estimator"],
            error=float(row["error_max_abs_full"]),
            locality=row["locality"],
            repeat=row["repeat"],
            samples=row["samples"],
            sample_seed=row["seed"],
        )


def dump_locality_scope(
    output_dir: Path,
    manifest_rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    compressed: bool,
) -> None:
    print("Dumping locality-sweep matrices...")
    write_static_bundle(
        output_dir,
        manifest_rows,
        scope=args.scope,
        plot_family="locality",
        case_id="locality",
        condition="well",
        m=args.l_m,
        c=args.well_c,
        beta=args.beta,
        base_seed=args.seed,
        compressed=compressed,
    )
    locality_sampled_bundles(output_dir, manifest_rows, args=args, compressed=compressed)
    locality_exact_bundle(output_dir, manifest_rows, args=args, compressed=compressed)
    locality_global_baseline_bundles(output_dir, manifest_rows, args=args, compressed=compressed)


def mode_case_bundle(
    output_dir: Path,
    manifest_rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    compressed: bool,
    condition: str,
    c: float,
    rng: np.random.Generator,
    config_seed: int,
    m: int,
    repeat: int,
    estimator: str,
) -> None:
    beta = args.beta
    dim = 2 * m
    target = qgl.hamiltonian_matrix(dim, c, beta)
    samples = qgl.sample_measurements(m, args.mode_samples, c, rng, beta)
    cov_est = qgl.covariance_estimate(samples)

    if estimator == "local":
        inv_name = "local_inverse_reconstruction"
        h_name = "local_hamiltonian_estimate"
        inv_est = qgl.local_inverse_reconstruction(cov_est, m, args.mode_locality)
        h_est = hamiltonian_from_inverse(inv_est, m, c, beta)
        array_labels = {
            "empirical_covariance": "Sample covariance X^T X/N used for the mode-sweep local reconstruction.",
            inv_name: f"Local inverse reconstruction with locality l={args.mode_locality}.",
            h_name: "Local Hamiltonian estimate entering the mode-sweep local error.",
        }
    elif estimator == "global":
        inv_name = "global_inverse_cov_omega"
        h_name = "global_hamiltonian_estimate"
        inv_est = qgl.inverse_cov_omega(cov_est, m)
        h_est = hamiltonian_from_inverse(inv_est, m, c, beta)
        array_labels = {
            "empirical_covariance": "Sample covariance X^T X/N used for the mode-sweep global reconstruction.",
            inv_name: "Global inverse of 2*(Sigma_hat-I/2)-i*Omega.",
            h_name: "Global plug-in Hamiltonian estimate entering the mode-sweep global error.",
        }
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    arrays: dict[str, np.ndarray] = {
        "empirical_covariance": cov_est,
        inv_name: inv_est,
        h_name: h_est,
    }
    if args.include_samples:
        arrays["sample_matrix"] = samples
        array_labels["sample_matrix"] = "Raw Gaussian samples X used to form empirical_covariance."

    metadata = {
        "scope": args.scope,
        "plot_family": "mode",
        "case_id": f"mode_{condition}_m{m:04d}_{estimator}_repeat{repeat:03d}",
        "condition": condition,
        "m": m,
        "dim": dim,
        "c": c,
        "beta": beta,
        "samples": args.mode_samples,
        "locality": args.mode_locality if estimator == "local" else "",
        "repeat": repeat,
        "base_seed": args.seed,
        "sample_seed": config_seed,
        "arrays": array_labels,
    }
    path = (
        output_dir
        / "mode"
        / condition
        / f"mode_{condition}_m{m:04d}_{estimator}_repeat{repeat:03d}_seed{config_seed}.npz"
    )
    file_size = save_npz(path, arrays, metadata, compressed)
    append_manifest_row(
        manifest_rows,
        output_dir=output_dir,
        path=path,
        file_size_bytes=file_size,
        arrays=arrays,
        array_labels=array_labels,
        metadata=metadata,
        estimator=f"{estimator}_sampled",
        error=error_against_target(target, h_est),
        locality=args.mode_locality if estimator == "local" else None,
        repeat=repeat,
        samples=args.mode_samples,
        sample_seed=config_seed,
    )


def dump_mode_scope(
    output_dir: Path,
    manifest_rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    compressed: bool,
) -> None:
    print("Dumping mode-sweep matrices...")
    m_values = tuple(range(args.m_start, args.m_stop + 1, args.m_step))
    conditions = [
        ("ill", args.ill_c, args.seed + 101),
        ("well", args.well_c, args.seed + 202),
    ]
    for condition, c, config_seed in conditions:
        rng = np.random.default_rng(config_seed)
        for m in m_values:
            write_static_bundle(
                output_dir,
                manifest_rows,
                scope=args.scope,
                plot_family="mode",
                case_id=f"mode_{condition}",
                condition=condition,
                m=m,
                c=c,
                beta=args.beta,
                base_seed=args.seed,
                compressed=compressed,
            )
            print(f"  {condition:>4} m={m:>4}: local repeats")
            for repeat in range(1, args.mode_repeats + 1):
                mode_case_bundle(
                    output_dir,
                    manifest_rows,
                    args=args,
                    compressed=compressed,
                    condition=condition,
                    c=c,
                    rng=rng,
                    config_seed=config_seed,
                    m=m,
                    repeat=repeat,
                    estimator="local",
                )
            print(f"  {condition:>4} m={m:>4}: global repeats")
            for repeat in range(1, args.mode_repeats + 1):
                mode_case_bundle(
                    output_dir,
                    manifest_rows,
                    args=args,
                    compressed=compressed,
                    condition=condition,
                    c=c,
                    rng=rng,
                    config_seed=config_seed,
                    m=m,
                    repeat=repeat,
                    estimator="global",
                )


def apply_quick_overrides(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.mode_samples = 500
    args.mode_repeats = 1
    args.m_start = 10
    args.m_stop = 20
    args.m_step = 10
    args.l_m = 10
    args.l_samples = [500, 1000]
    args.l_values = [1, 2]
    args.l_repeats = 1


def write_manifests(output_dir: Path, run_config: dict[str, Any], manifest_rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps(manifest_rows, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(manifest_rows).to_csv(output_dir / "manifest.csv", index=False)


def main() -> None:
    args = parse_args()
    apply_quick_overrides(args)
    repo_root = qgl.find_repo_root(Path.cwd())
    output_dir = resolve_dir(repo_root, args.output_dir)
    compressed = not args.uncompressed
    manifest_rows: list[dict[str, Any]] = []

    if len(args.l_samples) < 2:
        raise ValueError("--l-samples must contain at least two sample counts.")

    run_config = {
        "seed": args.seed,
        "scope": args.scope,
        "output_dir": str(Path(args.output_dir)),
        "beta": args.beta,
        "ill_c": args.ill_c,
        "well_c": args.well_c,
        "mode_samples": args.mode_samples,
        "mode_repeats": args.mode_repeats,
        "mode_locality": args.mode_locality,
        "m_values": list(range(args.m_start, args.m_stop + 1, args.m_step)),
        "l_m": args.l_m,
        "l_samples": args.l_samples,
        "l_values": args.l_values,
        "l_repeats": args.l_repeats,
        "audit_window_modes": args.audit_window_modes,
        "audit_window_start": args.audit_window_start,
        "reuse_samples_across_l": not args.independent_l_samples,
        "include_samples": args.include_samples,
        "compressed": compressed,
        "quick": args.quick,
        "format": "NPZ archives with a labels_json array in every archive plus manifest.csv/manifest.json.",
    }

    print("Matrix dump run")
    print(json.dumps(run_config, indent=2))
    t0 = perf_counter()

    if args.scope == "audit":
        dump_audit_scope(output_dir, manifest_rows, args=args, compressed=compressed)
    if args.scope in {"locality", "all"}:
        dump_locality_scope(output_dir, manifest_rows, args=args, compressed=compressed)
    if args.scope in {"mode", "all"}:
        dump_mode_scope(output_dir, manifest_rows, args=args, compressed=compressed)

    elapsed = perf_counter() - t0
    run_config["elapsed_seconds"] = elapsed
    archive_sizes = {row["file"]: row["file_size_bytes"] for row in manifest_rows}
    run_config["archive_count"] = len(archive_sizes)
    run_config["manifest_rows"] = len(manifest_rows)
    run_config["total_archive_bytes"] = int(sum(archive_sizes.values()))
    write_manifests(output_dir, run_config, manifest_rows)

    print("\nDone.")
    print(f"Matrix dump directory: {output_dir}")
    print(f"Manifest CSV:          {output_dir / 'manifest.csv'}")
    print(f"Manifest JSON:         {output_dir / 'manifest.json'}")
    print(f"Archives:              {run_config['archive_count']}")
    print(f"Manifest rows:         {run_config['manifest_rows']}")
    print(f"Elapsed seconds:       {elapsed:.2f}")


if __name__ == "__main__":
    main()
