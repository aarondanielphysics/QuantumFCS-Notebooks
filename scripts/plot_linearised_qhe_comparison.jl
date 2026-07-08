const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

using CairoMakie
using CSV
using DataFrames

include(joinpath(PROJECT_ROOT, "src", "paths.jl"))
include(joinpath(PROJECT_ROOT, "src", "plots.jl"))

ensure_benchmark_dirs()
out = plot_linearised_comparison()
println("Saved: ", out)
error_out = plot_linearised_cumulant_errors()
println("Saved: ", error_out)
