# Quantum FCS Notebooks

This repository accompanies an in-progress companion paper for
[QuantumFCS.jl](https://github.com/marcelojbp/QuantumFCS.jl). It is intended to
collect the reproducible notebooks, scripts, processed data, and figures used in
the publication.

The repository currently contains the benchmark workflow for the manuscript
summary figure. Future additions will include notebooks for the paper examples
and a usage-example notebook that demonstrates the main QuantumFCS.jl workflow
from the text.

Citation information and final paper metadata will be added after publication.

## Repository Layout

- `notebooks/`: publication companion notebooks. Use this for worked examples,
  usage walkthroughs, and narrative reproductions of paper results.
- `notebooks/paper_examples/`: notebooks for individual examples developed for
  the paper.
- `notebooks/usage_examples/`: notebooks focused on package usage from the
  manuscript.
- `src/`: reusable Julia helpers for models, benchmark runs, paths, and plots.
- `scripts/`: Julia and WolframScript entry points used by reproducible
  workflows.
- `data/`: small checked-in CSV outputs used to regenerate figures.
- `figures/`: checked-in final figure outputs.
- `melt.m`: local MELT copy used for offline WolframScript benchmark reruns.

## Julia Environment

Install Julia, then instantiate the project:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The manifest was generated for Julia `1.12.6`. `Project.toml` points
`QuantumFCS` at the public GitHub repository and pins the commit used for the
current reproducible artifacts.

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
