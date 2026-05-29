from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
import io
import os
import shutil
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qgl_matplotlib_cache"))

import numpy as np
import pandas as pd
from scipy.linalg import expm, inv, logm
import matplotlib.pyplot as plt


DEFAULT_BETA = 0.5
DEFAULT_MODE_M_VALUES = list(range(100, 1151, 50))
DEFAULT_L_VALUES = [2, 4, 6, 8, 10]


ILL_PUBLISHED_CSV = """SystemSize_m,Avg_Naive_Error,Avg_Local_Error,Avg_Naive_Time_sec,Avg_Local_Time_sec
100.0,0.4010972029436672,0.2631239762907575,2.400156587,3.396325023333333
150.0,0.3928258789044998,0.2748748531673381,3.9310780783333334,7.532880201666667
200.0,0.35826884076290716,0.2337704908283255,7.7327406970000006,14.665984107
250.0,0.3972735892843282,0.2743049774475094,10.733624034,26.26754447233333
300.0,0.3711741059882267,0.4203029908480333,16.299742551666665,42.89746546166666
350.0,0.4274396113926449,0.29250774376483274,20.900993286,70.60134130133333
400.0,0.4357873274388382,0.26940144802738447,29.833314852666664,101.399556959
450.0,0.4331616119309301,0.3100299390573451,35.16779048633333,123.40153188699999
500.0,0.7585688466610149,0.34835006243051136,43.19673302333334,167.50661666933334
550.0,1.1310695177195065,0.29508586342544313,57.43090550466667,266.870392432
600.0,0.6208069626686499,0.3249324499348633,61.101498217,326.4959266453333
650.0,0.975680786042548,0.34648153479404115,74.33219965466667,407.70215670933334
700.0,2.3174495930532166,0.28310782666633477,92.06555236933333,497.03597990599997
750.0,2.286723046920183,0.2998280227799168,103.19488614199999,624.8410402056667
800.0,1.0395880803061852,0.3573651154071505,114.56099054066668,713.238369111
850.0,0.8598503437775632,0.32807269203488554,133.395484973,840.632176182
900.0,1.177811070972692,0.3630584450489603,140.63008168000002,943.5083508563333
950.0,2.7012240989545298,0.3595928106425783,161.49451646833333,1053.145244494
1000.0,1.133486292895195,0.3553531697554541,190.1686131063333,1168.433412371
1050.0,1.0682937282757805,0.3338623486107253,251.40017381533335,1386.7390368626666
1100.0,1.1312066720340683,0.29996120492653966,274.19389307733337,1847.9073768636665
1150.0,1.0951515303942891,0.35256753083656406,301.36816995099997,1884.167629747"""


L_PUBLISHED_ROWS = [
    {"Samples": 10_000, "l": 2, "Error": 0.2362797825586731},
    {"Samples": 10_000, "l": 4, "Error": 0.2114592090059657},
    {"Samples": 10_000, "l": 6, "Error": 0.29234335574237336},
    {"Samples": 10_000, "l": 8, "Error": 0.3067729129354426},
    {"Samples": 10_000, "l": 10, "Error": 0.36553700053862914},
    {"Samples": 100_000, "l": 2, "Error": 0.13646366775844276},
    {"Samples": 100_000, "l": 4, "Error": 0.045442938936133936},
    {"Samples": 100_000, "l": 6, "Error": 0.04475055220885116},
    {"Samples": 100_000, "l": 8, "Error": 0.045705644963831116},
    {"Samples": 100_000, "l": 10, "Error": 0.0478910149319256},
]


EXACT_L_PUBLISHED_ROWS = [
    {"l": 2, "Error": 0.11653511141870965},
    {"l": 4, "Error": 2.3904821522258146e-7},
    {"l": 6, "Error": 4.9713788641270185e-11},
    {"l": 8, "Error": 1.354472090042691e-14},
    {"l": 10, "Error": 9.769962616701378e-15},
]


@dataclass(frozen=True)
class ModeSweepConfig:
    condition: str
    c: float
    samples: int = 10_000
    locality: int = 3
    repeats: int = 5
    beta: float = DEFAULT_BETA
    seed: int = 20240527
    m_values: tuple[int, ...] = tuple(DEFAULT_MODE_M_VALUES)


@dataclass(frozen=True)
class LSweepConfig:
    c: float = 0.1
    m: int = 100
    sample_counts: tuple[int, ...] = (10_000, 100_000)
    l_values: tuple[int, ...] = tuple(DEFAULT_L_VALUES)
    repeats: int = 1
    beta: float = DEFAULT_BETA
    seed: int = 20240528
    reuse_samples_across_l: bool = True


def find_repo_root(start: Path | None = None) -> Path:
    path = Path.cwd() if start is None else Path(start).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "qgl_reproduce.py").exists() and (candidate / "plotterl.py").exists():
            return candidate
    raise RuntimeError("Could not find repository root from current working directory.")


def published_ill_mode_df() -> pd.DataFrame:
    return pd.read_csv(io.StringIO(ILL_PUBLISHED_CSV))


def published_l_sweep_df() -> pd.DataFrame:
    return pd.DataFrame(L_PUBLISHED_ROWS)


def published_exact_l_df() -> pd.DataFrame:
    return pd.DataFrame(EXACT_L_PUBLISHED_ROWS)


def write_published_tables(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    published_ill_mode_df().to_csv(data_dir / "mode_sweep_ill_published.csv", index=False)
    published_l_sweep_df().to_csv(data_dir / "l_sweep_sampled_published.csv", index=False)
    published_exact_l_df().to_csv(data_dir / "l_sweep_exact_published.csv", index=False)


def normal_precision(dim: int) -> np.ndarray:
    precision = np.zeros((dim, dim), dtype=float)
    np.fill_diagonal(precision, 2.0)
    precision[-1, -1] = 1.0
    idx = np.arange(dim - 1)
    precision[idx, idx + 1] = -1.0
    precision[idx + 1, idx] = -1.0
    return precision


def hamiltonian_matrix(dim: int, c: float, beta: float = DEFAULT_BETA) -> np.ndarray:
    return beta * (normal_precision(dim) + c * np.eye(dim))


def hamiltonian_mask(dim: int, c: float, beta: float = DEFAULT_BETA, tol: float = 1e-10) -> np.ndarray:
    return (np.abs(hamiltonian_matrix(dim, c, beta)) > tol).astype(float)


def omega_matrix(m: int) -> np.ndarray:
    omega_block = np.array([[0.0, 1.0], [-1.0, 0.0]])
    return np.kron(np.eye(m), omega_block)


def hermitize(a: np.ndarray) -> np.ndarray:
    return (a + a.conj().T) / 2


@lru_cache(maxsize=4)
def measurement_covariance(m: int, c: float, beta: float = DEFAULT_BETA) -> np.ndarray:
    dim = 2 * m
    h = hamiltonian_matrix(dim, c, beta)
    i_omega = 1j * omega_matrix(m)
    term_inv = inv((expm(2 * h @ i_omega) - np.eye(dim)) / 2)
    sigma = (i_omega @ term_inv + i_omega) / 2 + np.eye(dim) / 2
    return np.real(hermitize(sigma))


def sample_measurements(
    m: int,
    samples: int,
    c: float,
    rng: np.random.Generator,
    beta: float = DEFAULT_BETA,
) -> np.ndarray:
    sigma = measurement_covariance(m, c, beta)
    return rng.multivariate_normal(np.zeros(2 * m), sigma, size=samples, method="svd")


def covariance_estimate(sample_matrix: np.ndarray) -> np.ndarray:
    return sample_matrix.T @ sample_matrix / sample_matrix.shape[0]


def inverse_cov_omega(cov_matrix: np.ndarray, m: int) -> np.ndarray:
    dim = 2 * m
    return inv(2 * (cov_matrix - np.eye(dim) / 2) - 1j * omega_matrix(m))


def hamiltonian_from_inverse(inv_cov_omega: np.ndarray, m: int) -> np.ndarray:
    dim = 2 * m
    i_omega = 1j * omega_matrix(m)
    h = 0.5 * logm(np.eye(dim) + 2 * inv_cov_omega @ i_omega) @ i_omega
    return np.real(hermitize(h))


def mask_hamiltonian(h_est: np.ndarray, c: float, beta: float = DEFAULT_BETA) -> np.ndarray:
    return hamiltonian_mask(h_est.shape[0], c, beta) * h_est


def global_reconstruction_error(cov_matrix: np.ndarray, m: int, c: float, beta: float = DEFAULT_BETA) -> float:
    target = hamiltonian_matrix(2 * m, c, beta)
    h_est = mask_hamiltonian(hamiltonian_from_inverse(inverse_cov_omega(cov_matrix, m), m), c, beta)
    return float(np.max(np.abs(target - h_est)))


def global_hamiltonian_estimate(cov_matrix: np.ndarray, m: int, c: float, beta: float = DEFAULT_BETA) -> np.ndarray:
    return mask_hamiltonian(hamiltonian_from_inverse(inverse_cov_omega(cov_matrix, m), m), c, beta)


def classical_exact_error(m: int, c: float, beta: float = DEFAULT_BETA) -> float:
    target = hamiltonian_matrix(2 * m, c, beta)
    inv_cov = inverse_cov_omega(measurement_covariance(m, c, beta), m)
    h_linear = mask_hamiltonian(np.real(hermitize(inv_cov)), c, beta)
    return float(np.max(np.abs(target - h_linear)))


def local_inv_cov_omega_from_cov(cov_matrix: np.ndarray, m: int, locality: int, x: int) -> np.ndarray:
    dim = 2 * m
    j_min = max(2 * (x - 1) - 2 * locality, 0)
    j_max = min(2 * (x - 1) + 2 * locality + 2, dim)
    block = cov_matrix[j_min:j_max, j_min:j_max]
    block_dim = block.shape[0]
    i_omega_local = 1j * omega_matrix(block_dim // 2)
    return hermitize(inv(2 * (block - np.eye(block_dim) / 2) - i_omega_local))


def local_inverse_reconstruction(cov_matrix: np.ndarray, m: int, locality: int) -> np.ndarray:
    dim = 2 * m
    local_matrices = [
        local_inv_cov_omega_from_cov(cov_matrix, m, locality, x)
        for x in range(1, m + 1)
    ]
    reconstructed = np.zeros((dim, dim), dtype=complex)

    for j in range(1, m + 1):
        local_matrix = local_matrices[j - 1]
        base = max(2 * (j - 1) - 2 * locality, 0)
        for k in range(j, min(j + locality, m) + 1):
            r_start = 2 * (j - 1) - base
            c_start = 2 * (k - 1) - base
            block_2x2 = local_matrix[r_start:r_start + 2, c_start:c_start + 2]
            reconstructed[2 * j - 2:2 * j, 2 * k - 2:2 * k] = block_2x2
            if j != k:
                reconstructed[2 * k - 2:2 * k, 2 * j - 2:2 * j] = block_2x2.conj().T

    return hermitize(reconstructed)


def local_reconstruction_error(
    cov_matrix: np.ndarray,
    m: int,
    locality: int,
    c: float,
    beta: float = DEFAULT_BETA,
) -> float:
    target = hamiltonian_matrix(2 * m, c, beta)
    inv_cov_rec = local_inverse_reconstruction(cov_matrix, m, locality)
    h_est = mask_hamiltonian(hamiltonian_from_inverse(inv_cov_rec, m), c, beta)
    return float(np.max(np.abs(target - h_est)))


def local_hamiltonian_estimate(
    cov_matrix: np.ndarray,
    m: int,
    locality: int,
    c: float,
    beta: float = DEFAULT_BETA,
) -> np.ndarray:
    inv_cov_rec = local_inverse_reconstruction(cov_matrix, m, locality)
    return mask_hamiltonian(hamiltonian_from_inverse(inv_cov_rec, m), c, beta)


def exact_l_sweep(config: LSweepConfig) -> pd.DataFrame:
    cov_exact = measurement_covariance(config.m, config.c, config.beta)
    rows = []
    for locality in config.l_values:
        rows.append(
            {
                "l": locality,
                "Error": local_reconstruction_error(
                    cov_exact,
                    config.m,
                    locality,
                    config.c,
                    config.beta,
                ),
            }
        )
    return pd.DataFrame(rows)


def run_mode_sweep(
    config: ModeSweepConfig,
    csv_path: Path | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    rows = []

    for m in config.m_values:
        local_errors = []
        naive_errors = []
        local_start = perf_counter()
        for _ in range(config.repeats):
            samples = sample_measurements(m, config.samples, config.c, rng, config.beta)
            cov_est = covariance_estimate(samples)
            local_errors.append(local_reconstruction_error(cov_est, m, config.locality, config.c, config.beta))
        local_elapsed = perf_counter() - local_start

        naive_start = perf_counter()
        for _ in range(config.repeats):
            samples = sample_measurements(m, config.samples, config.c, rng, config.beta)
            cov_est = covariance_estimate(samples)
            naive_errors.append(global_reconstruction_error(cov_est, m, config.c, config.beta))
        naive_elapsed = perf_counter() - naive_start

        row = {
            "SystemSize_m": float(m),
            "Avg_Naive_Error": float(np.mean(naive_errors)),
            "Avg_Local_Error": float(np.mean(local_errors)),
            "Avg_Naive_Time_sec": naive_elapsed / config.repeats,
            "Avg_Local_Time_sec": local_elapsed / config.repeats,
            "Condition": config.condition,
            "c": config.c,
            "Samples": config.samples,
            "Locality": config.locality,
            "Repeats": config.repeats,
            "Beta": config.beta,
            "Seed": config.seed,
        }
        rows.append(row)
        if csv_path is not None:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
        if progress:
            print(
                f"{config.condition:>4} m={m:>4}: "
                f"global={row['Avg_Naive_Error']:.6g}, "
                f"local={row['Avg_Local_Error']:.6g}"
            )

    df = pd.DataFrame(rows)
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    return df


def run_l_sweep(
    config: LSweepConfig,
    csv_path: Path | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    rows = []

    for sample_count in config.sample_counts:
        for repeat in range(config.repeats):
            shared_cov = None
            if config.reuse_samples_across_l:
                samples = sample_measurements(config.m, sample_count, config.c, rng, config.beta)
                shared_cov = covariance_estimate(samples)
            for locality in config.l_values:
                if shared_cov is None:
                    samples = sample_measurements(config.m, sample_count, config.c, rng, config.beta)
                    cov_est = covariance_estimate(samples)
                else:
                    cov_est = shared_cov
                rows.append(
                    {
                        "Samples": sample_count,
                        "Repeat": repeat + 1,
                        "l": locality,
                        "Error": local_reconstruction_error(cov_est, config.m, locality, config.c, config.beta),
                        "c": config.c,
                        "m": config.m,
                        "Beta": config.beta,
                        "Seed": config.seed,
                        "ReuseSamplesAcrossL": config.reuse_samples_across_l,
                    }
                )
                if progress:
                    print(f"N={sample_count:>7} repeat={repeat + 1} l={locality:>2}: error={rows[-1]['Error']:.6g}")

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["Samples", "l"], as_index=False)
        .agg(Error=("Error", "mean"), ErrorStd=("Error", "std"), Repeats=("Error", "size"))
    )
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(csv_path, index=False)
        df.to_csv(csv_path.with_name(csv_path.stem + "_all_repeats.csv"), index=False)
    return summary


def plot_mode_sweep(df: pd.DataFrame, condition_label: str, output_path: Path) -> tuple[plt.Figure, plt.Axes]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        df["SystemSize_m"],
        df["Avg_Naive_Error"],
        marker="o",
        linestyle="-",
        linewidth=2,
        color="tab:blue",
        label="Average Error Global Reconstruction",
    )
    ax.plot(
        df["SystemSize_m"],
        df["Avg_Local_Error"],
        marker="s",
        linestyle="--",
        linewidth=2,
        color="tab:orange",
        label="Average Error Local Reconstruction",
    )
    ax.set_title(f"Reconstruction Errors vs Number of Modes for 1D, {condition_label} Hamiltonian", fontsize=16)
    ax.set_xlabel("Number of modes (m)", fontsize=12)
    ax.set_ylabel("Average Reconstruction Error", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend()
    ax.set_ylim(0, 3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    return fig, ax


def _sample_count_latex(n: int | float) -> str:
    n_int = int(n)
    if n_int == 10_000:
        return r"10^{4}"
    if n_int == 100_000:
        return r"10^5"
    exponent = int(np.log10(n_int)) if n_int > 0 and np.isclose(n_int, 10 ** int(np.log10(n_int))) else None
    if exponent is not None:
        return rf"10^{{{exponent}}}"
    return f"{n_int:,}"


def plot_l_sweep(
    sampled_df: pd.DataFrame,
    exact_df: pd.DataFrame,
    global_sampled_error: float,
    classical_exact: float,
    output_path: Path,
) -> tuple[plt.Figure, plt.Axes]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))

    sample_counts = sorted(sampled_df["Samples"].unique())
    if len(sample_counts) < 2:
        raise ValueError("plot_l_sweep needs at least two sample counts.")
    low_n = sample_counts[0]
    high_n = sample_counts[-1]
    low = sampled_df[sampled_df["Samples"] == low_n].sort_values("l")
    high = sampled_df[sampled_df["Samples"] == high_n].sort_values("l")
    exact = exact_df.sort_values("l")

    ax.plot(
        low["l"],
        low["Error"],
        marker="o",
        linestyle="-",
        linewidth=2,
        color="#d62728",
        label=rf"Local Rec. (Sampled, $N={_sample_count_latex(low_n)}$)",
    )
    ax.plot(
        high["l"],
        high["Error"],
        marker="o",
        linestyle="-",
        linewidth=2,
        color="#2ca02c",
        label=rf"Local Rec. (Sampled, $N={_sample_count_latex(high_n)}$)",
    )
    ax.axhline(
        y=global_sampled_error,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=rf"Global Rec. (Sampled, $N={_sample_count_latex(high_n)}$)",
    )
    ax.set_title(
        "Convergence of Local Reconstruction vs Locality Parameter $l$\n"
        "(Fixed System Size m=100)",
        fontsize=14,
        pad=15,
    )
    ax.set_xlabel("Locality Parameter ($l$)", fontsize=12)
    ax.set_ylabel("Reconstruction Error", fontsize=12)
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.02, top=4)
    ax.grid(True, which="major", linestyle="-", color="0.9")
    ax.set_xticks(list(exact["l"]))
    ax.legend(loc="lower right", frameon=True, framealpha=1, edgecolor="0.8")

    ax_ins = ax.inset_axes([0.08, 0.55, 0.35, 0.35])
    ax_ins.plot(
        exact["l"],
        exact["Error"],
        marker="o",
        markersize=4,
        linestyle="-",
        linewidth=1.5,
        color="#893093",
        label="Local rec. (Exact)",
    )
    ax_ins.axhline(
        y=classical_exact,
        color="black",
        linestyle=":",
        linewidth=1.5,
        alpha=0.8,
        label="Classical rec. (Exact)",
    )
    ax_ins.set_title("Exact Covariance Limit", fontsize=9)
    ax_ins.set_yscale("log")
    ax_ins.set_xlabel("$l$", fontsize=8)
    ax_ins.set_ylabel("Error", fontsize=8)
    ax_ins.set_ylim(bottom=1e-16, top=10)
    ax_ins.grid(True, which="major", linestyle="-", color="0.9", alpha=0.5)
    ax_ins.tick_params(axis="both", which="major", labelsize=8)
    ax_ins.set_xticks(list(exact["l"]))
    ax_ins.legend(fontsize=7, loc="lower left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    return fig, ax


def copy_existing_plot(source_dir: Path, filename: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename
    shutil.copy2(source_dir / filename, destination)
    return destination
