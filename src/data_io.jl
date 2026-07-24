# Loading helpers for the checked-in paper-application datasets.
#
#   data/jc_fcs_production_g14_*        driven-dissipative Jaynes-Cummings
#                                       production sweep (paper Sec. 4, Fig. 2)
#   data/qhe_paper_sweeps/*.jld2        circuit-QED heat-engine sweeps
#                                       (paper Sec. 5, Figs. 3-4)
#
# These let a notebook reproduce the manuscript figures without rerunning the
# expensive sweeps, while the same notebooks can recompute everything on demand.

using CSV
using DataFrames
using JLD2

paper_data_dir() = joinpath(normpath(@__DIR__, ".."), "data")

# ---------------------------------------------------------------------------
# Driven-dissipative Jaynes-Cummings production sweep
# ---------------------------------------------------------------------------

jc_production_jld2_path() =
    joinpath(paper_data_dir(), "jc_fcs_production_g14_results.jld2")
jc_production_csv_path() =
    joinpath(paper_data_dir(), "jc_fcs_production_g14_rows.csv")
jc_production_metadata_path() =
    joinpath(paper_data_dir(), "jc_fcs_production_g14_metadata.md")

"""
    load_jc_production() -> NamedTuple

Load the checked-in JC production sweep: `rows` (one entry per drive point, the
same NamedTuples `run_jc_fcs_production_sweep` returns), `fock` (cavity Fock
distributions at the representative drive points shown in Fig. 2a-c),
`schedules` (the dynamic cutoff ladder per detuning cut), and the detuning cuts
and coupling actually used.

Only the keys the figures need are read. The stored `config` entry is skipped on
purpose: it contains an anonymous per-cut function that cannot be deserialised
into a fresh session, and nothing downstream uses it.
"""
function load_jc_production(path::AbstractString=jc_production_jld2_path())
    isfile(path) || error("Missing JC production data at $(path).")
    rows, fock, schedules, saved_at = jldopen(path, "r") do file
        (file["fcs_rows"], file["fock"], file["schedules"], file["saved_at"])
    end
    detuning_cuts = sort(unique(r.Δ for r in rows))
    g = first(rows).g
    return (; rows, fock, schedules, saved_at, detuning_cuts, g)
end

"""
    jc_production_dataframe(rows) -> DataFrame

Tabular view of the production rows, for compact display and column-wise
diagnostics. `cutoff_status` is stored as a `Symbol`; keep it as-is so the
reliability panels can filter on it.
"""
jc_production_dataframe(rows) = DataFrame(collect(rows))

"""
    load_jc_production_csv() -> DataFrame

The same production rows read from the human-readable CSV. Useful as an
independent check that the binary and text artefacts agree.
"""
load_jc_production_csv(path::AbstractString=jc_production_csv_path()) =
    CSV.read(path, DataFrame)

# ---------------------------------------------------------------------------
# Circuit-QED heat-engine sweeps
# ---------------------------------------------------------------------------

const QHE_SWEEP_NAMES = (
    "qhe_antibunching_g",
    "qhe_antibunching_lambda_c",
    "qhe_finite_affinity_g",
    "qhe_finite_affinity_lambda_c",
)

qhe_sweep_dir() = joinpath(paper_data_dir(), "qhe_paper_sweeps")
qhe_sweep_path(name::AbstractString) = joinpath(qhe_sweep_dir(), string(name, ".jld2"))

"""
    load_qhe_sweep(name) -> (df, params)

Load one checked-in heat-engine sweep by name (see `QHE_SWEEP_NAMES`), returning
the swept `DataFrame` and the parameter dictionary it was produced with.
"""
function load_qhe_sweep(name::AbstractString)
    path = qhe_sweep_path(name)
    isfile(path) || error("Missing QHE sweep '$(name)' at $(path).")
    df, params = jldopen(path, "r") do file
        (file["df"], file["params"])
    end
    return df, params
end

"""
    load_qhe_paper_sweeps() -> NamedTuple

Load all four sweeps backing Figs. 3 and 4 in one call.
"""
function load_qhe_paper_sweeps()
    antibunching_g, antibunching_g_params = load_qhe_sweep("qhe_antibunching_g")
    antibunching_λc, antibunching_λc_params = load_qhe_sweep("qhe_antibunching_lambda_c")
    finite_affinity_g, finite_affinity_g_params = load_qhe_sweep("qhe_finite_affinity_g")
    finite_affinity_λc, finite_affinity_λc_params =
        load_qhe_sweep("qhe_finite_affinity_lambda_c")
    return (;
        antibunching_g, antibunching_λc, finite_affinity_g, finite_affinity_λc,
        params=(;
            antibunching_g=antibunching_g_params,
            antibunching_λc=antibunching_λc_params,
            finite_affinity_g=finite_affinity_g_params,
            finite_affinity_λc=finite_affinity_λc_params,
        ),
    )
end
