# Regression guard for run_jc_fcs_production_sweep (driven-dissipative JC production model).
#
# The clean-architecture rewrite of the sweep (continuation struct + direct QuantumFCS calls
# + extracted helpers) is pure restructuring, so it must be BIT-IDENTICAL. This test locks the
# deterministic result columns against a committed golden reference and guards efficiency with
# allocation bytes (deterministic; wall-clock is too noisy to assert). Wall-clock `*_seconds`
# columns are intentionally excluded from the value comparison.
#
# Run (verify):   julia --project=. test/jc_production_sweep_regression.jl
# Capture golden: JC_CAPTURE_GOLDEN=1 julia --project=. test/jc_production_sweep_regression.jl
#
# Included from test/runtests.jl; also runnable standalone.

using Test
using JLD2
using LinearAlgebra
using SparseArrays

include(joinpath(@__DIR__, "..", "src", "jc_model.jl"))

const GOLDEN = joinpath(@__DIR__, "fixtures", "jc_sweep_golden.jld2")

# Deterministic columns to compare (everything the row carries except wall-clock timings).
const DET_COLS = (
    :label, :probe_group, :Δ, :x, :g, :E, :N,
    :cutoff_budget_exceeded, :trunc_tol, :ss_ilu_tau, :ss_rebuild_niter,
    :ss_converged, :ss_iterations, :ss_rebuilds,
    :trace_error, :hermiticity_error, :liouvillian_residual, :relative_liouvillian_residual,
    :c1, :c2, :c3, :Fano, :current_check, :fcs_retry, :fcs_warnings,
    :n, :occupation_fraction, :cutoff_status, :cavity_boundary, :cavity_tail, :cavity_tail5,
)

# --- Fixtures (small, fast, fixed physics; adequate cutoffs so runs are clean) -------------
# (a) N_override single segment over low (dim-branch) drives: exercises cold start + warm ILU
#     reuse (rebuilds = [1,0,0]). Adequate cutoff ⇒ clean, fast. The rebuild-on-stall path is
#     exercised separately by the JCSteadyContinuation unit test.
fixture_a() = run_jc_fcs_production_sweep(;
    detuning_cuts = [0.0], x_values = [0.5, 0.7, 0.9],
    g = 10.0, κ = 1.0, nC = 3, N_override = 40, verbose = false)

# (b) Dynamic cutoff schedule producing TWO segments (N=12 for the dim-branch drives, N=48 for
#     the bright-branch point): exercises the schedule, cutoff segmentation, and per-segment
#     continuation reset. Reduced headroom keeps cutoffs small yet adequate.
fixture_b() = run_jc_fcs_production_sweep(;
    detuning_cuts = [0.0], x_values = [0.5, 0.7, 1.1],
    g = 10.0, κ = 1.0, nC = 3, tiers = (12, 48), pad_abs = 6.0, pad_sigma = 2.0, verbose = false)

project(rows) = [NamedTuple{DET_COLS}(map(c -> getproperty(r, c), DET_COLS)) for r in rows]

# Warmed allocation measurement (discard the compile-time first run).
function warmed_alloc(f)
    f()
    return @allocated f()
end

coleq(a, b) =
    (a isa AbstractFloat && b isa AbstractFloat) ?
        ((isnan(a) && isnan(b)) || isapprox(a, b; rtol = 1e-10, atol = 1e-12)) :
        isequal(a, b)

function summarize(tag, rows)
    println("[$tag] $(length(rows)) rows")
    for r in rows
        println("   N=$(r.N) x=$(r.x) iters=$(r.ss_iterations) rebuilds=$(r.ss_rebuilds) " *
                "c1=$(r.c1) cc=$(round(r.current_check; sigdigits=6)) retry=$(r.fcs_retry)")
    end
end

# Focused unit test: jc_continuation_solve! must reproduce a direct package solve on a cold
# point and reuse the ILU across a nearby point (the rebuild path is covered by fixture a).
function continuation_unit_test()
    @testset "JCSteadyContinuation vs direct trace_constrained_steadystate" begin
        N = 20; g = 10.0; κ = 1.0; Δ = 0.0
        ops = jc_operators_for_cutoff(N)
        cavity_loss = sqrt(κ) * ops.a
        settings = (; ilu_tau = JC_DEFAULT_SCOUT_SS_ILU_TAU,
            shift_factor = JC_DEFAULT_SS_ILU_SHIFT_FACTOR,
            rtol = JC_DEFAULT_SS_GMRES_RTOL, atol = JC_DEFAULT_SS_GMRES_ATOL,
            itmax = JC_DEFAULT_SS_REUSE_ITMAX, fallback_itmax = JC_DEFAULT_SS_REUSE_FALLBACK_ITMAX,
            memory = JC_DEFAULT_SS_GMRES_MEMORY, rebuild_niter = JC_DEFAULT_SS_REUSE_REBUILD_NITER)
        mkH(x) = -Δ * ops.detuning_op + g * ops.Hjc + (x * g / 2) * ops.Hdr
        sys1 = trace_constrained_system(QuantumToolbox.liouvillian(mkH(0.50), [cavity_loss]))

        cont = JCSteadyContinuation(settings)
        ss, sstats = jc_continuation_solve!(cont, sys1)
        ref = trace_constrained_steadystate(sys1; method = :iterative, Pl = nothing, u0 = nothing,
            τ = settings.ilu_tau, shift_factor = settings.shift_factor,
            rtol = settings.rtol, atol = settings.atol, itmax = settings.itmax, memory = settings.memory)
        @test Matrix(ss.rho_ss) ≈ Matrix(ref.rho_ss) rtol = 1e-10
        @test sstats.niters == ref.stats.iterations
        @test sstats.rebuilds == 1
        @test sstats.converged
        @test cont.Pl === ss.Pl                      # segment ILU retained on the continuation

        sys2 = trace_constrained_system(QuantumToolbox.liouvillian(mkH(0.55), [cavity_loss]))
        Pl_before = cont.Pl
        _, sstats2 = jc_continuation_solve!(cont, sys2)
        @test sstats2.converged
        @test sstats2.rebuilds == 0                  # warm reuse: no rebuild
        @test cont.Pl === Pl_before                  # same ILU object reused
    end
end

function segments_unit_test()
    @testset "jc_cutoff_segments" begin
        @test jc_cutoff_segments([5]) == [1:1]
        @test jc_cutoff_segments([5, 5, 5]) == [1:3]
        @test jc_cutoff_segments([1, 1, 2, 2, 2, 1]) == [1:2, 3:5, 6:6]
        @test jc_cutoff_segments([1, 2, 3]) == [1:1, 2:2, 3:3]
        @test jc_cutoff_segments(Int[]) == UnitRange{Int}[]
    end
end

if get(ENV, "JC_CAPTURE_GOLDEN", "0") == "1"
    mkpath(dirname(GOLDEN))
    a = fixture_a(); summarize("a", a)
    b = fixture_b(); summarize("b", b)
    alloc_a = warmed_alloc(fixture_a)
    alloc_b = warmed_alloc(fixture_b)
    println("alloc_a=$(alloc_a)  alloc_b=$(alloc_b)")
    jldsave(GOLDEN; a = project(a), b = project(b), alloc_a = alloc_a, alloc_b = alloc_b)
    println("Golden written to $GOLDEN")
else
    ref = JLD2.load(GOLDEN)
    segments_unit_test()
    continuation_unit_test()
    @testset "JC production sweep regression (bit-identical + allocations)" begin
        for (tag, fixture, refkey, allockey) in
                (("a", fixture_a, "a", "alloc_a"), ("b", fixture_b, "b", "alloc_b"))
            @testset "fixture $tag" begin
                got = project(fixture())
                want = ref[refkey]
                @test length(got) == length(want)
                for (gr, wr) in zip(got, want), c in DET_COLS
                    @test coleq(getproperty(gr, c), getproperty(wr, c))
                end
                # Efficiency: no allocation regression (deterministic; small headroom for
                # the continuation struct / extra NamedTuples introduced by the refactor).
                @test warmed_alloc(fixture) <= ref[allockey] * 1.10
            end
        end
    end
end
