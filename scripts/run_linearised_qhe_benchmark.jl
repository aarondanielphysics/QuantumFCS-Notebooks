const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

using BenchmarkTools
using DelimitedFiles
using QuantumFCS
using QuantumOptics
using Statistics

include(joinpath(PROJECT_ROOT, "src", "paths.jl"))
include(joinpath(PROJECT_ROOT, "src", "config.jl"))
include(joinpath(PROJECT_ROOT, "src", "models.jl"))
include(joinpath(PROJECT_ROOT, "src", "benchmarks.jl"))

ensure_benchmark_dirs()

N_values = parse_int_list_env("LINEARISED_N_VALUES", 1:8)
nC_fixed = parse_int_env("NC", 2)
samples = parse_int_env("LINEARISED_SAMPLES", 100)
evals = parse_int_env("EVALS", 1)

# Keep the script defaults aligned with the full-pipeline Makefile parameters.
g = parse_float_env("LINEARISED_G", 0.35)
κh = parse_float_env("LINEARISED_KAPPA_H", 1.0)
κc = parse_float_env("LINEARISED_KAPPA_C", 1.0)
nh = parse_float_env("LINEARISED_NH", 0.5)
nc = parse_float_env("LINEARISED_NC", 0.05)

isapprox(κh, κc) || error("Linearised analytical cumulants require κh = κc")

println("Running linearised QHE benchmark")
println("N values: ", N_values)
# The runtime CSV stores the arithmetic mean across these BenchmarkTools samples.
println("nC: $nC_fixed, mean timing samples: $samples, evals: $evals")

dims, times_ms, cumulant_1, cumulant_2 = benchmark_vs_dimension_linearised(
    N_values,
    nC_fixed;
    g=g,
    κh=κh,
    κc=κc,
    nh=nh,
    nc=nc,
    samples=samples,
    evals=evals,
)

out = raw_data_path("benchmark_linearised_qhe_vs_dimension.csv")
write_two_column_csv(out, dims, times_ms)
println("Saved: ", out)

cumulants_out = raw_data_path("benchmark_linearised_qhe_cumulants_vs_dimension.csv")
analytic_c1, analytic_c2 = linearised_cumulants_analytic(g, κh, nh, nc)
write_columns_csv(
    cumulants_out,
    [
        "hilbert_dimension",
        "cumulant_1",
        "cumulant_2",
        "analytic_cumulant_1",
        "analytic_cumulant_2",
    ],
    dims,
    cumulant_1,
    cumulant_2,
    fill(analytic_c1, length(dims)),
    fill(analytic_c2, length(dims)),
)
println("Saved: ", cumulants_out)
