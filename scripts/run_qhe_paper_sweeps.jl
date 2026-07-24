# Circuit-QED heat-engine paper sweeps (paper Sec. 5, Figs. 3 and 4).
#
# Recomputes the four sweeps backing the manuscript figures and writes them to
# data/qhe_paper_sweeps/. The antibunching sweeps take well under a minute; the
# finite-affinity sweeps use a 546-dimensional Hilbert space and take roughly
# four minutes each. Each point prints its swept value, Fano factors and
# truncation tails.
#
# Run:  julia --project=. scripts/run_qhe_paper_sweeps.jl
#       make qhe-sweeps
#
# Subsets and reduced grids:
#   QHE_SWEEPS="antibunching_g" QHE_POINTS=25 julia --project=. scripts/run_qhe_paper_sweeps.jl

using DrWatson
@quickactivate "QuantumFCSNotebooks"

using CSV
using DataFrames
using JLD2
using Printf

include(srcdir("qhe_model.jl"))
include(srcdir("config.jl"))
include(srcdir("data_io.jl"))

const ANTIBUNCHING = (; lambda_h = 0.47, Omega_c = 1000.0, Omega_ratio = π,
                        kappa_h = 2.0, kappa_c = 0.5, n_h = 0.5, n_c = 0.01,
                        solver = :iterative, trunc_tol = 5e-3, occupation_tol = 0.5)

const FINITE_AFFINITY = (; lambda_h = 0.47, Omega_c = 1000.0, Omega_ratio = π,
                           kappa_h = 2.0, kappa_c = 0.5, n_h = 2.5, n_c = 1.5,
                           solver = :iterative, trunc_tol = 3e-3, occupation_tol = 0.5)

const ANTIBUNCHING_G = 12.841683366733466

"""
Run one sweep, printing progress. `setter` maps a swept value to the keyword it
sets, so the same driver covers both g and λc sweeps.
"""
function run_sweep(values, setter; label, progress_every, params...)
    rows = NamedTuple[]
    t0 = time_ns()
    for (i, v) in enumerate(values)
        pt = qhe_point(; params..., setter(v)...)
        push!(rows, pt)
        if i == 1 || i == length(values) || i % progress_every == 0
            @printf("%-22s %3d/%3d  x=%9.4f | ℱ_h=%7.4f ℱ_c=%7.4f Q_h=%8.3f | tails %.1e/%.1e | %6.1fs\n",
                    label, i, length(values), v, pt.Fh, pt.Fc, pt.Qh,
                    pt.hot_tail, pt.cold_tail, (time_ns() - t0) / 1e9)
            flush(stdout)
        end
    end
    return DataFrame(rows)
end

npoints(default) = begin
    requested = parse_int_env("QHE_POINTS", 0)
    requested == 0 ? default : requested
end

requested_sweeps = let value = get(ENV, "QHE_SWEEPS", "")
    isempty(strip(value)) ? collect(QHE_SWEEP_NAMES) :
        ["qhe_" * strip(s) for s in split(replace(value, ";" => ","), ","; keepempty = false)]
end

# Reduced runs should not clobber the checked-in sweeps: point QHE_OUTPUT_DIR
# somewhere else when running a smoke check.
sweep_dir = get(ENV, "QHE_OUTPUT_DIR", qhe_sweep_dir())
mkpath(sweep_dir)

# name => (values, setter, extra parameters, stored parameter dictionary)
specs = Dict(
    "qhe_antibunching_g" => (
        LinRange(1.0, 20.0, npoints(500)), v -> (; g = v),
        (; Nmax_h = 7, Nmax_c = 7, lambda_c = 0.89, ANTIBUNCHING...),
        Dict{String,Any}("regime" => "antibunching", "sweep" => "g")),
    "qhe_antibunching_lambda_c" => (
        LinRange(0.1, 2.0, npoints(100)), v -> (; λc = v),
        (; Nmax_h = 7, Nmax_c = 10, g = ANTIBUNCHING_G, ANTIBUNCHING...),
        Dict{String,Any}("regime" => "antibunching", "sweep" => "lambda_c")),
    "qhe_finite_affinity_g" => (
        LinRange(0.1, 100.0, npoints(50)), v -> (; g = v),
        (; Nmax_h = 20, Nmax_c = 25, lambda_c = 0.7, FINITE_AFFINITY...),
        Dict{String,Any}("regime" => "finite_affinity_tur", "sweep" => "g")),
    "qhe_finite_affinity_lambda_c" => (
        LinRange(0.1, 2.0, npoints(50)), v -> (; λc = v),
        (; Nmax_h = 20, Nmax_c = 25, g = 10.0, FINITE_AFFINITY...),
        Dict{String,Any}("regime" => "finite_affinity_tur", "sweep" => "lambda_c")),
)

println("Circuit-QED heat-engine paper sweeps")
println("  sweeps: ", join(requested_sweeps, ", "))
flush(stdout)

for name in requested_sweeps
    haskey(specs, name) || error("Unknown sweep '$(name)'. Known: $(collect(keys(specs)))")
    values, setter, params, stored = specs[name]
    progress_every = max(1, length(values) ÷ 10)

    df = run_sweep(values, setter; label = name, progress_every = progress_every, params...)

    path = joinpath(sweep_dir, string(name, ".jld2"))
    merged = merge(stored, Dict{String,Any}(
        "grid_min" => first(values), "grid_max" => last(values),
        "grid_points" => length(values)))
    jldsave(path; df = df, params = merged, metadata = merged,
                  created_by = "scripts/run_qhe_paper_sweeps.jl")
    CSV.write(joinpath(sweep_dir, string(name, ".csv")), df)

    @printf("  wrote %s (%d rows) | worst ε_off %.2e | all cutoffs ok: %s\n\n",
            path, nrow(df), maximum(df.epsilon_off), all(df.cutoff_ok))
    flush(stdout)
end

println("Done.")
