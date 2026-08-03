# Numerical experiments for Gaussian Hamiltonian learning

This repository contains the numerical implementation accompanying

> Marco Fanizza, Cambyse Rouzé, and Daniel Stilck França, “Efficient Hamiltonian, structure and trace distance learning of Gaussian states,” [arXiv:2411.03163](https://arxiv.org/abs/2411.03163).

The experiments compare global plug-in Hamiltonian reconstruction with local inversion for one-dimensional bosonic Gaussian Hamiltonians. The canonical workflow is the seeded Python command-line pipeline in `run_seeded_reproduction.py`.

## Numerical experiments

| Experiment | Default parameters | Figure output |
| --- | --- | --- |
| Mode sweep, ill-conditioned chain | `c = 0`, `N = 10^4`, `l = 3`, `m = 100, 150, ..., 1150`, five realizations | `simulation_errors_plot_ill.png` and `.pdf` |
| Mode sweep, well-conditioned chain | `c = 0.1`, `N = 10^4`, `l = 3`, `m = 100, 150, ..., 1150`, five realizations | `simulation_errors_plot_well.png` and `.pdf` |
| Locality sweep, well-conditioned chain | `m = 100`, `N = 10^4` and `10^5`, `l = 2, 4, 6, 8, 10`, one realization per sample count | `improved_plot.png` and `.pdf` |

All experiments use the inverse-temperature prefactor `beta = 0.5` and base random seed `20240527` by default.
The committed manuscript-scale figures are stored in `reproduced_plots_seeded/` in both formats.

## Repository layout

| Path | Purpose |
| --- | --- |
| `qgl_reproduce.py` | Covariance generation, sampling, global and local reconstruction, error metrics, and plotting. |
| `run_seeded_reproduction.py` | Canonical seeded pipeline for generating numerical tables and figures. |
| `requirements.txt` | Python dependencies. |
| `sanity_check_window.py` | Small-window reconstruction diagnostic. |
| `compare_condition_window_heatmaps.py` | Ill- and well-conditioned reconstruction diagnostics. |
| `dump_plot_matrices.py` | Export of labeled matrix data used in numerical audits. |

## Installation

From the repository root, create a virtual environment and install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick environment check

Run the reduced pipeline to verify the installation:

```bash
python run_seeded_reproduction.py --quick \
  --data-dir runs/quick/data \
  --plot-dir runs/quick/plots
```

This uses a small parameter grid and writes test outputs under `runs/quick/`. It is a functional check rather than a full-scale simulation.

## Run the seeded experiments

Run the complete seeded workflow with

```bash
python run_seeded_reproduction.py \
  --data-dir runs/full/data \
  --plot-dir runs/full/plots
```

The mode and locality experiments can also be run separately:

```bash
python run_seeded_reproduction.py --only mode \
  --data-dir runs/mode/data \
  --plot-dir runs/mode/plots

python run_seeded_reproduction.py --only locality \
  --data-dir runs/locality/data \
  --plot-dir runs/locality/plots
```

Each command uses a separate output directory so that artifacts from distinct runs are not mixed. The pipeline writes numerical tables and run metadata, including parameters and software versions, to the selected data directory and PNG and PDF figures to the selected plot directory.
The PDF files are vector graphics intended for manuscript preparation; the PNG files are matching high-resolution previews generated from the same figure objects.

### Benchmark evaluation convention

Before evaluating the reconstruction error, the matrix-log output is projected onto the real-symmetric matrices,

`H_real = Re[(H_raw + H_raw†) / 2]`.

Because the support of the target Hamiltonian is known in these benchmarks, the real-symmetric estimate is then restricted to that support,

`H_masked = 1_supp(H) ⊙ H_real`.

For both the global and local reconstructions, the reported metric is the maximum absolute entrywise error on the target support,

`max_{(i,j) in supp(H)} |H_ij - (H_masked)_ij|`.

## Validation and matrix exports

Run the small-window diagnostic with

```bash
python sanity_check_window.py
```

Compare the first five modes of the ill- and well-conditioned examples with

```bash
python compare_condition_window_heatmaps.py --seed 100
```

Export a compact labeled matrix dataset with

```bash
python dump_plot_matrices.py --scope audit --seed 100 --output-dir data
```

To export the full matrices associated with the seeded locality experiment at `m = 100`, run

```bash
python dump_plot_matrices.py --seed 100
```

The `--scope all` option also exports mode-sweep matrices. Add `--include-samples` only when the raw Gaussian samples are required, since these files can be large.

## Historical scripts

The Julia scripts and the plotting notebook are retained as records of earlier numerical workflows. The seeded Python command-line pipeline described above is the canonical implementation for generating new results.

## Citation

If you use this code, cite the accompanying paper:

```bibtex
@misc{fanizza2024efficienthamiltonianstructuretrace,
  title = {Efficient Hamiltonian, structure and trace distance learning of Gaussian states},
  author = {Marco Fanizza and Cambyse Rouzé and Daniel Stilck França},
  year = {2024},
  eprint = {2411.03163},
  archivePrefix = {arXiv},
  primaryClass = {quant-ph},
  doi = {10.48550/arXiv.2411.03163}
}
```
