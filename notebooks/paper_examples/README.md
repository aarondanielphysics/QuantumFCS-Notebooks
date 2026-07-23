# Paper Examples

Notebooks reproducing the applications developed in the manuscript.

| Notebook | Manuscript | Figure |
|---|---|---|
| `driven_dissipative_jaynes_cummings.ipynb` | Sec. 4 | Fig. 2 |
| `circuit_qed_heat_engine.ipynb` | Sec. 5 | Figs. 3, 4 + validation |

Both are written for a reader checking the paper's results. Each one walks
through the QuantumFCS API on a single parameter point, then shows the sweep
that produces the figure — how the trace-constrained steady state is solved, how
the incomplete-LU factorization is reused across parameter points and handed to
the FCS Drazin solve, and which identities gate each point — before calling the
routine that produces the published figure.

Model construction and the publication plotting live in `src/`
(`jc_model.jl`, `qhe_model.jl`, `paper_figures.jl`); the sweep setup and the
QuantumFCS calls are deliberately in the notebooks.

Run from the repository root with the checked-in environment:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Expensive computations are behind toggles (`RUN_FULL_JC`,
`RUN_FINITE_AFFINITY_LIVE`), defaulting to the checked-in data so both notebooks
execute in a couple of minutes. `make test` verifies that recomputing reproduces
that data.
