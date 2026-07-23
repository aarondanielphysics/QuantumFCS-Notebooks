# Quantum FCS Notebooks

This repository accompanies an in-progress companion paper for
[QuantumFCS.jl](https://github.com/marcelojbp/QuantumFCS.jl). It is intended to
collect the reproducible notebooks, scripts, processed data, and figures used in
the publication.

Citation information and final paper metadata will be added after publication.

## What is here

| Manuscript | Notebook | Figure |
|---|---|---|
| Sec. 3.2 — minimal quantum dot | [`notebooks/usage_examples/minimal_quantum_dot_example.ipynb`](notebooks/usage_examples/minimal_quantum_dot_example.ipynb) | — |
| Sec. 4 — driven-dissipative Jaynes-Cummings | [`notebooks/paper_examples/driven_dissipative_jaynes_cummings.ipynb`](notebooks/paper_examples/driven_dissipative_jaynes_cummings.ipynb) | Fig. 2 |
| Sec. 5 — circuit-QED heat engine | [`notebooks/paper_examples/circuit_qed_heat_engine.ipynb`](notebooks/paper_examples/circuit_qed_heat_engine.ipynb) | Figs. 3, 4 |
| Sec. 6 — benchmark against MELT | `scripts/` + `make benchmark` | Fig. 5 |

Both paper-example notebooks are written to be read alongside the manuscript:
they show how the sweeps are set up and driven through the QuantumFCS API — how
the trace-constrained steady state is solved, how one incomplete-LU
factorization is reused across parameter points and handed to the FCS Drazin
solve, and which identities each point is checked against — and then rebuild the
published figure.

## Repository Layout

- `notebooks/paper_examples/`: notebooks reproducing the paper applications.
- `notebooks/usage_examples/`: notebooks focused on package usage from the text.
- `src/`: Julia helpers.
  - `jc_model.jl`, `qhe_model.jl`: the two paper-application pipelines.
  - `jc_diagnostics.jl`: Jaynes-Cummings truncation-reliability studies.
  - `paper_figures.jl`: the exact routines producing the manuscript figures.
  - `data_io.jl`: loaders for the checked-in datasets.
  - `models.jl`, `benchmarks.jl`, `plots.jl`, `paths.jl`, `config.jl`: benchmark half.
- `scripts/`: Julia and WolframScript entry points used by reproducible workflows.
- `data/`: checked-in datasets — benchmark CSVs, the Jaynes-Cummings production
  sweep, and the heat-engine sweeps under `data/qhe_paper_sweeps/`.
- `test/`: regression and parity suite (`make test`).
- `figures/`: checked-in figure outputs.
- `melt.m`: local MELT copy used for offline WolframScript benchmark reruns.

## Reproducing the paper applications

Everything below runs from the repository root with the checked-in environment.

```sh
make test            # regression + parity suite (~80 s)
make paper-figures   # rebuild Figs. 2-4 from checked-in data, no sweeps
```

The notebooks default to the checked-in data for the expensive parts and
recompute the rest live. To recompute the sweeps from scratch:

```sh
make qhe-sweeps      # heat-engine sweeps (~9 minutes)
make jc-production   # full Jaynes-Cummings sweep (~1 hour, several GB)
```

Inside the notebooks the same choice is a toggle: `RUN_FULL_JC` in the
Jaynes-Cummings notebook and `RUN_FINITE_AFFINITY_LIVE` in the heat-engine
notebook. The heat-engine antibunching sweeps always run live (about 15
seconds); only the 546-dimensional finite-affinity sweeps are gated.

Reduced runs for a quick check — these write elsewhere so they cannot overwrite
the checked-in datasets:

```sh
make qhe-sweeps QHE_SWEEPS=antibunching_g QHE_POINTS=25 QHE_OUTPUT_DIR=/tmp/qhe
```

### Cost and reliability at a glance

| Sweep | Hilbert dim | Points | Cost |
|---|---|---|---|
| JC production (3 detuning cuts) | up to 1002 (Fock cutoff 150–500) | 267 | ~1 hour |
| Heat engine, antibunching | 64 and 88 | 500 + 100 | ~15 s |
| Heat engine, finite affinity | 546 | 50 + 50 | ~8 min |

Every Jaynes-Cummings point is gated on the identity
`c₁ = κ⟨a†a⟩`; every heat-engine point is checked against tight coupling and
against currents computed directly from the steady state. The notebooks report
these, along with the truncation and rotating-wave diagnostics that appear in
the manuscript.

## Julia Environment

Install Julia, then instantiate the project:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The manifest was generated for Julia `1.12.6`. `Project.toml` points
`QuantumFCS` at the public GitHub repository and pins the commit used for the
current reproducible artifacts.

> **Note on the QuantumFCS pin.** The paper-example notebooks use the
> trace-constrained steady-state API (`trace_constrained_steadystate`,
> `prepare_fcs_context`, and preconditioner injection via `Pl`). That API is not
> yet on `main` of QuantumFCS.jl, so `Project.toml` currently pins the
> `feature/trace-constrained-steadystate` commit that produced the checked-in
> data. This pin should be moved to a tagged release once those changes are
> merged.

## Notebook Contributions

Place new notebooks under `notebooks/`, grouped by purpose. Prefer descriptive,
paper-oriented names such as:

```text
notebooks/paper_examples/linearised_heat_engine.ipynb
notebooks/usage_examples/basic_fcs_workflow.ipynb
```

Keep notebooks reproducible from the repository root using the checked-in Julia
environment. Move shared helper code into `src/` when more than one notebook or
script needs it. Avoid committing local scratch outputs, large raw data dumps,
or `.ipynb_checkpoints/` files.

## Current Benchmark Workflow

The existing benchmark workflow regenerates the combined benchmark figure:

- `figures/benchmark_combined_summary.png`
- `figures/benchmark_combined_summary.pdf`

For a quick check that avoids benchmark reruns:

```sh
make smoke
```

To regenerate figures from checked-in CSV data:

```sh
make plots
```

To rerun the full benchmark pipeline:

```sh
make benchmark
```

Full benchmark reruns may be expensive. Julia targets rebuild processed CSVs in
`data/`; Wolfram/MELT targets require `wolframscript` and use the checked-in
`melt.m` by default.

Useful narrower targets include:

```sh
make julia-benchmarks
make mathematica-benchmarks
make linearised-benchmark
make dense-benchmark
make melt-linearised-benchmark
make melt-dense-benchmark
```

For fast benchmark smoke checks, reduce sweep sizes:

```sh
make julia-benchmarks LINEARISED_N_VALUES=1 DENSE_N_VALUES=1 LINEARISED_SAMPLES=1 DENSE_SAMPLES=1
```
