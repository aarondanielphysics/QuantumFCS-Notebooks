# Quantum FCS Notebooks

> This repository accompanies a manuscript that is not yet published. Citation information and final paper metadata will be added after publication.

This repository contains the reproducible scripts, processed benchmark data, and figure outputs for the `benchmark_combined_summary` benchmarking figure of the QuantumFCS.jl manuscript.

The repository is intentionally narrow: it excludes unrelated exploratory notebooks, parameter-search archives, draft PDFs, and Marcelo's other notebooks. QuantumFCS.jl is the Julia package used for full-counting-statistics calculations: <https://github.com/marcelojbp/QuantumFCS.jl>.

## Contents

- `src/`: reusable Julia helpers for models, benchmark runs, paths, and plots.
- `scripts/`: Julia and Wolfram entry points used by the benchmark pipeline.
- `data/`: small checked-in CSVs used to regenerate the final figure.
- `figures/`: checked-in final figure outputs.
- `melt.m`: local MELT copy used for offline WolframScript benchmark reruns.

The final benchmark figure combines linearised runtime, dense runtime, linearised cumulant error, and dense cutoff-population diagnostics. It is generated as:

- `figures/benchmark_combined_summary.png`
- `figures/benchmark_combined_summary.pdf`

## Reproduce The Figure

Install Julia, then run:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
make plots
```

The source project manifest recorded Julia `1.12.6`. The original manifest developed `QuantumFCS` from a local sibling path, so this repository includes a fresh portable `Manifest.toml` generated from `Project.toml`. `Project.toml` points `QuantumFCS` at the public GitHub repository and pins the commit used by the local benchmark checkout.

For a quick check that avoids benchmark reruns:

```sh
make smoke
```

## Full Benchmark Rerun

To rerun the full benchmark pipeline:

```sh
make benchmark
```

Full reruns may be expensive. The Julia benchmark targets rebuild the processed CSVs in `data/`; the Wolfram/MELT targets require `wolframscript` and use the checked-in `melt.m` by default.

Useful narrower targets include:

```sh
make julia-benchmarks
make mathematica-benchmarks
make linearised-benchmark
make dense-benchmark
make melt-linearised-benchmark
make melt-dense-benchmark
```

Use tiny sweeps for quick benchmark checks, for example:

```sh
make julia-benchmarks LINEARISED_N_VALUES=1 DENSE_N_VALUES=1 LINEARISED_SAMPLES=1 DENSE_SAMPLES=1
```
