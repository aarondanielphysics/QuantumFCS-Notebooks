const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

using CairoMakie
using CSV
using DataFrames

include(joinpath(PROJECT_ROOT, "src", "paths.jl"))
include(joinpath(PROJECT_ROOT, "src", "plots.jl"))

ensure_benchmark_dirs()
out = plot_dense_comparison()
println("Saved: ", out)
population_out = plot_dense_highest_fock_population_scaling()
println("Saved: ", population_out)
combined_out = plot_combined_benchmark_summary()
println("Saved: ", combined_out)
