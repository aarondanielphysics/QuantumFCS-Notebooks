# Test suite for the paper-application transplant.
#
#   model_helpers.jl                    fast unit tests + dataset integrity
#   qhe_parity.jl                       heat-engine sweeps reproduce the stored data
#   jc_production_sweep_regression.jl   JC sweep is bit-identical to the golden fixture
#
# Run:  julia --project=. test/runtests.jl     (or `make test`)

using Test

# Shared includes, so the individual files do not reload them.
include(joinpath(@__DIR__, "..", "src", "jc_model.jl"))
include(joinpath(@__DIR__, "..", "src", "qhe_model.jl"))
include(joinpath(@__DIR__, "..", "src", "data_io.jl"))

println("Running QuantumFCS-Notebooks paper-application tests")
t0 = time()

@testset "QuantumFCS-Notebooks paper applications" begin
    include(joinpath(@__DIR__, "model_helpers.jl"))
    include(joinpath(@__DIR__, "qhe_parity.jl"))
    include(joinpath(@__DIR__, "jc_production_sweep_regression.jl"))
end

println("Finished in $(round(time() - t0; digits = 1)) s")
