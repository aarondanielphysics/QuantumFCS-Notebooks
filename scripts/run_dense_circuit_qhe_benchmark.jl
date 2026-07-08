const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

using BenchmarkTools
using DelimitedFiles
using LinearAlgebra
using QuantumFCS
using QuantumOptics
using Statistics

include(joinpath(PROJECT_ROOT, "src", "paths.jl"))
include(joinpath(PROJECT_ROOT, "src", "config.jl"))
include(joinpath(PROJECT_ROOT, "src", "models.jl"))
include(joinpath(PROJECT_ROOT, "src", "benchmarks.jl"))

ensure_benchmark_dirs()

N_values = parse_int_list_env("DENSE_N_VALUES", 1:7)
nC_fixed = parse_int_env("NC", 2)
samples = parse_int_env("DENSE_SAMPLES", 10)
evals = parse_int_env("EVALS", 1)

Ωh = parse_float_env("DENSE_OMEGA_H", 5.0)
Ωc = parse_float_env("DENSE_OMEGA_C", 1.0)
# Keep the script defaults aligned with the full-pipeline Makefile parameters.
EJ = parse_float_env("DENSE_EJ", 1.75)
λh = parse_float_env("DENSE_LAMBDA_H", 0.20)
λc = parse_float_env("DENSE_LAMBDA_C", 0.25)
κh = parse_float_env("DENSE_KAPPA_H", 1.0)
κc = parse_float_env("DENSE_KAPPA_C", 1.0)
nbarh = parse_float_env("DENSE_NBAR_H", 0.50)
nbarc = parse_float_env("DENSE_NBAR_C", 0.05)

println("Running dense circuit-QED QHE benchmark")
println("N values: ", N_values)
# The runtime CSV stores the arithmetic mean across these BenchmarkTools samples.
println("nC: $nC_fixed, mean timing samples: $samples, evals: $evals")

dims, times_ms, cumulant_1, cumulant_2, highest_populations = benchmark_vs_dimension_dense(
    N_values,
    nC_fixed;
    samples=samples,
    evals=evals,
    Ωh=Ωh,
    Ωc=Ωc,
    EJ=EJ,
    λh=λh,
    λc=λc,
    κh=κh,
    κc=κc,
    nbarh=nbarh,
    nbarc=nbarc,
)

out = raw_data_path("benchmark_dense_circuit_qhe_vs_dimension.csv")
write_two_column_csv(out, dims, times_ms)
println("Saved: ", out)

cumulants_out = raw_data_path("benchmark_dense_circuit_qhe_cumulants_vs_dimension.csv")
write_columns_csv(
    cumulants_out,
    ["hilbert_dimension", "cumulant_1", "cumulant_2"],
    dims,
    cumulant_1,
    cumulant_2,
)
println("Saved: ", cumulants_out)

population_out = raw_data_path("benchmark_dense_circuit_qhe_highest_fock_population_vs_dimension.csv")
write_columns_csv(
    population_out,
    ["hilbert_dimension", "highest_joint_fock_population"],
    dims,
    highest_populations,
)
println("Saved: ", population_out)
