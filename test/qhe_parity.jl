# Parity guard for the circuit-QED heat-engine pipeline.
#
# Every checked-in sweep in data/qhe_paper_sweeps/ was produced by `qhe_point`.
# Recomputing a sampled row from its own stored parameters must return the same
# numbers, which is what makes the notebook's live sweeps trustworthy: the
# figures a reader regenerates are the figures in the manuscript.
#
# Run standalone:  julia --project=. test/qhe_parity.jl

using Test
using DataFrames

if !@isdefined(qhe_point)
    include(joinpath(@__DIR__, "..", "src", "qhe_model.jl"))
end
if !@isdefined(load_qhe_sweep)
    include(joinpath(@__DIR__, "..", "src", "data_io.jl"))
end

"""
Rebuild the exact parameter set a stored row was computed with. The row carries
every physical parameter, so no external bookkeeping is needed.
"""
function params_from_row(row)
    return NonlinearQHEParameters(
        Nmax_h = row.Nmax_h, Nmax_c = row.Nmax_c,
        lh = row.lh, lc = row.lc,
        λh = row.λh, λc = row.λc,
        Ωc = row.Ωc, Ωratio = row.Ωratio,
        κh = row.κh, κc = row.κc,
        nh = row.nh, nc = row.nc,
        g  = row.g,
    )
end

numerics_from_row(row) = nonlinear_qhe_numerics(
    solver         = :iterative,
    trunc_tol      = row.trunc_tol,
    occupation_tol = row.occupation_tol,
)

# Physics columns that must reproduce: the two currents, their noises, the Fano
# factors and the uncertainty products the manuscript reports.
const PARITY_COLS = (:Jh, :Dh, :Jc, :Dc, :Fh, :Fc, :Qh, :Qc, :A, :CRWA)

"""
Sample a few rows spanning the sweep rather than all of them: the finite-affinity
points use a 546-dimensional Hilbert space and cost seconds each.
"""
sample_indices(n, k = 3) = unique(round.(Int, range(1, n; length = min(k, n))))

@testset "QHE sweep parity vs checked-in data" begin
    for name in QHE_SWEEP_NAMES
        df, _ = load_qhe_sweep(name)
        @testset "$name ($(nrow(df)) rows)" begin
            @test nrow(df) > 0
            for i in sample_indices(nrow(df))
                row  = df[i, :]
                live = qhe_point(params_from_row(row), numerics_from_row(row))
                for col in PARITY_COLS
                    stored = getproperty(row, col)
                    got    = getproperty(live, col)
                    # Same code, same environment: this should be exact. Allow a
                    # tiny relative tolerance for platform floating-point drift.
                    @test isapprox(got, stored; rtol = 1e-10, atol = 1e-12)
                end
            end
        end
    end
end
