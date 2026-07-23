# Full driven-dissipative Jaynes-Cummings production FCS sweep (paper Sec. 4).
#
# This is the command-line route to the same computation the notebook runs when
# RUN_FULL_JC = true. Expect roughly an hour and several GB of memory at the
# deepest bright-branch points; each point prints its parameters, cutoff and the
# steady-state and FCS timings as it completes.
#
# Writes data/jc_fcs_production_g14_{results.jld2,rows.csv,metadata.md}.
#
# Run:  julia --project=. scripts/run_jc_fcs_production_sweep.jl
#       make jc-production
#
# Reduced runs for a smoke check, via environment variables:
#   JC_DETUNINGS="0.0" JC_X_MIN=0.4 JC_X_MAX=0.6 JC_X_STEP=0.05 JC_N_OVERRIDE=40 \
#       julia --project=. scripts/run_jc_fcs_production_sweep.jl

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

using CSV
using DataFrames
using Dates
using JLD2

include(joinpath(PROJECT_ROOT, "src", "jc_model.jl"))
include(joinpath(PROJECT_ROOT, "src", "config.jl"))

parse_float_list_env(name, default) = begin
    value = get(ENV, name, "")
    isempty(strip(value)) && return collect(Float64, default)
    return parse.(Float64, split(replace(value, ";" => ","), ","; keepempty = false))
end

detuning_cuts = parse_float_list_env("JC_DETUNINGS", (0.0, 0.55, 0.70))
g             = parse_float_env("JC_G", 14.0)
κ             = parse_float_env("JC_KAPPA", 1.0)
nC            = parse_int_env("JC_NC", 3)
N_override    = parse_int_env("JC_N_OVERRIDE", 0)
N_override    = N_override == 0 ? nothing : N_override

# Drive grid: the graded production grid unless a coarse range is requested.
x_min  = parse_float_env("JC_X_MIN", NaN)
x_max  = parse_float_env("JC_X_MAX", NaN)
x_step = parse_float_env("JC_X_STEP", NaN)
x_values = (isnan(x_min) || isnan(x_max) || isnan(x_step)) ?
    jc_graded_drive_grid() : collect(x_min:x_step:x_max)

# Representative drives for the Fock-distribution panels of Fig. 2.
fock_x = parse_float_list_env("JC_FOCK_X", (0.8, 1.0, 1.3))

println("Driven-dissipative Jaynes-Cummings production sweep")
println("  g/κ = $g, κ = $κ, nC = $nC")
println("  detuning cuts Δ/κ = $detuning_cuts")
println("  drive points: $(length(x_values)) from $(minimum(x_values)) to $(maximum(x_values))")
println("  cutoff: ", N_override === nothing ? "dynamic schedule" : "forced N = $N_override")
flush(stdout)

started_at = string(now())
t0 = time_ns()

fcs_rows = run_jc_fcs_production_sweep(;
    detuning_cuts = detuning_cuts,
    x_values      = x_values,
    g             = g,
    κ             = κ,
    nC            = nC,
    N_override    = N_override,
    verbose       = true)

fock_rows = jc_collect_fock_distributions(;
    detuning_cuts = detuning_cuts,
    x_values      = fock_x,
    g             = g,
    κ             = κ,
    verbose       = true)

schedules = Dict{Float64,Any}()
if N_override === nothing
    for Δ in detuning_cuts
        schedules[Δ] = jc_dynamic_cutoff_schedule(x_values, Δ; g = g, κ = κ,
                                                  return_detail = true)
    end
end

elapsed    = (time_ns() - t0) / 1e9
finished_at = string(now())

# Reduced runs should not clobber the checked-in production dataset: point
# JC_OUTPUT_DIR somewhere else when running a smoke check.
output_dir = get(ENV, "JC_OUTPUT_DIR", joinpath(PROJECT_ROOT, "data"))
mkpath(output_dir)

results_path  = joinpath(output_dir, "jc_fcs_production_g14_results.jld2")
csv_path      = joinpath(output_dir, "jc_fcs_production_g14_rows.csv")
metadata_path = joinpath(output_dir, "jc_fcs_production_g14_metadata.md")

jldsave(results_path;
    fcs_rows        = fcs_rows,
    fock            = fock_rows,
    schedules       = schedules,
    validation_rows = Any[],
    started_at      = started_at,
    saved_at        = finished_at)
CSV.write(csv_path, DataFrame(fcs_rows))

config = (; g, κ, nC, x_values, detuning_cuts,
            tiers = JC_DEFAULT_FCS_PROD_TIERS,
            occ_max = JC_DEFAULT_FCS_PROD_OCC_MAX,
            pad_sigma = JC_DEFAULT_FCS_PROD_PAD_SIGMA,
            pad_abs = JC_DEFAULT_FCS_PROD_PAD_ABS,
            ilu_tau = JC_DEFAULT_SCOUT_SS_ILU_TAU,
            shift_factor = JC_DEFAULT_SS_ILU_SHIFT_FACTOR,
            rebuild_niter = JC_DEFAULT_SS_REUSE_REBUILD_NITER,
            itmax = JC_DEFAULT_SS_REUSE_ITMAX,
            fallback_itmax = JC_DEFAULT_SS_REUSE_FALLBACK_ITMAX,
            gmres_memory = JC_DEFAULT_SS_GMRES_MEMORY,
            gmres_rtol = JC_DEFAULT_SS_GMRES_RTOL,
            gmres_atol = JC_DEFAULT_SS_GMRES_ATOL,
            fcs_rtol = JC_DEFAULT_FCS_PROD_RTOL,
            fcs_itmax = JC_DEFAULT_FCS_PROD_ITMAX,
            fcs_memory = JC_DEFAULT_FCS_PROD_MEMORY,
            current_check_tol = JC_DEFAULT_FCS_PROD_CHECK_TOL)

open(metadata_path, "w") do io
    write(io, jc_fcs_production_metadata_markdown(;
        config      = config,
        rows        = fcs_rows,
        files       = [results_path, csv_path, metadata_path],
        started_at  = started_at,
        finished_at = finished_at))
end

worst_identity = maximum(abs(r.current_check - 1) for r in fcs_rows)
println()
println("Completed $(length(fcs_rows)) points in $(round(elapsed / 60; digits = 1)) min")
println("  worst |c₁/(κ⟨n⟩) − 1| = $(worst_identity)")
println("  FCS retries: $(count(r -> r.fcs_retry, fcs_rows))")
println("  wrote $results_path")
println("  wrote $csv_path")
println("  wrote $metadata_path")
