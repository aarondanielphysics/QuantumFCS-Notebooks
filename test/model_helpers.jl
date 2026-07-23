# Focused unit tests for the ported model helpers, plus integrity checks on the
# checked-in datasets. These are fast; the heavy numerical parity guards live in
# jc_production_sweep_regression.jl and qhe_parity.jl.

using Test
using DataFrames
using LinearAlgebra

if !@isdefined(jc_operators_for_cutoff)
    include(joinpath(@__DIR__, "..", "src", "jc_model.jl"))
end
if !@isdefined(qhe_workspace)
    include(joinpath(@__DIR__, "..", "src", "qhe_model.jl"))
end
if !@isdefined(load_jc_production)
    include(joinpath(@__DIR__, "..", "src", "data_io.jl"))
end

@testset "JC model helpers" begin
    ops = jc_operators_for_cutoff(3)          # cavity dim 4 ⊗ atom dim 2 = 8
    @test size(ops.a.data) == (8, 8)
    @test size(ops.Hjc.data) == (8, 8)
    @test size(ops.detuning_op.data) == (8, 8)
    @test ops.adag.data ≈ ops.a.data'

    population = [0.1, 0.2, 0.3, 0.4]
    tail = jc_tail_diagnostics(population)
    @test tail.cavity_boundary == 0.4
    @test tail.cavity_tail ≈ 0.7
    @test tail.cavity_tail5 ≈ 1.0

    @test jc_cutoff_status_from_tails(1e-6, 1e-5; trunc_tol = 1e-4) == :safe
    @test jc_cutoff_status_from_tails(5e-4, 5e-3; trunc_tol = 1e-4) == :ambiguous
    @test jc_cutoff_status_from_tails(5e-2, 5e-1; trunc_tol = 1e-4) == :unsafe

    @testset "cutoff segmentation" begin
        @test jc_cutoff_segments([5]) == [1:1]
        @test jc_cutoff_segments([5, 5, 5]) == [1:3]
        @test jc_cutoff_segments([1, 1, 2, 2, 2, 1]) == [1:2, 3:5, 6:6]
        @test jc_cutoff_segments(Int[]) == UnitRange{Int}[]
    end

    @testset "semiclassical estimate and cutoff schedule" begin
        # Below the critical drive (x = 2E/g = 1) the resonant bright branch is empty.
        @test jc_semiclassical_bright_n(10.0, 0.4 * 10.0 / 2, 0.0) == 0.0
        # Above it the closed form is (4E² − g²)/κ².
        @test jc_semiclassical_bright_n(10.0, 1.5 * 10.0 / 2, 0.0) ≈ 4 * 7.5^2 - 100.0
        # The schedule is monotone in the drive and drawn from the tier ladder.
        tiers = (50, 100, 200)
        N = jc_dynamic_cutoff_schedule([0.2, 0.6, 1.2], 0.0; g = 10.0, tiers = tiers)
        @test issorted(N)
        @test all(n -> n in tiers, N)
    end
end

@testset "QHE model helpers" begin
    params = NonlinearQHEParameters(Nmax_h = 4, Nmax_c = 5)
    ws = qhe_workspace(params.Nmax_h, params.Nmax_c)
    @test length(ws.basis_h) == params.Nmax_h + 1
    @test length(ws.basis_c) == params.Nmax_c + 1
    dim = (params.Nmax_h + 1) * (params.Nmax_c + 1)
    @test size(ws.ah.data) == (dim, dim)

    tail = qhe_tail_diagnostics([0.1, 0.2, 0.3, 0.4])
    @test tail.boundary == 0.4
    @test tail.tail ≈ 0.7
    @test tail.tail5 ≈ 1.0
    @test qhe_cutoff_status_from_tails(1e-5, 1e-4; trunc_tol = 1e-3) == :safe
    @test qhe_cutoff_status_from_tails(5e-3, 5e-2; trunc_tol = 1e-3) == :ambiguous
    @test qhe_cutoff_status_from_tails(5e-2, 5e-1; trunc_tol = 1e-3) == :unsafe

    @testset "Josephson energy and Laguerre operators" begin
        # E_J = 2g / [(2λh)^lh (2λc)^lc]; for lh=1, lc=2 this is g/(4 λh λc²).
        @test effective_Ej(1.0, 0.5, 0.5) ≈ 1.0 / (4 * 0.5 * 0.25)
        op = laguerre_operator(0.47, 1, 0:3)
        @test size(op.data) == (4, 4)
        @test isdiag(Matrix(op.data))          # Laguerre weights are diagonal in Fock space
    end

    @testset "tight coupling holds at a cheap point" begin
        pt = qhe_point(; Nmax_h = 5, Nmax_c = 5, lambda_h = 0.47, lambda_c = 0.89,
                         Omega_c = 1000.0, Omega_ratio = π, kappa_h = 2.0, kappa_c = 0.5,
                         n_h = 0.5, n_c = 0.01, g = 8.0,
                         solver = :iterative, trunc_tol = 5e-3, occupation_tol = 0.5)
        # lc·Ωc·⟨J_h⟩ + lh·Ωh·⟨J_c⟩ = 0 for a tightly coupled engine.
        @test pt.tight_coupling_error < 1e-5
        # The FCS first cumulants agree with the currents read off the steady state.
        @test isapprox(pt.hot_current_check,  1.0; atol = 1e-8)
        @test isapprox(pt.cold_current_check, 1.0; atol = 1e-8)
        @test pt.A > 0                          # positive affinity: the engine is driven
    end
end

@testset "checked-in dataset integrity" begin
    jc = load_jc_production()
    @test length(jc.rows) == 267                     # 89 drive points × 3 detuning cuts
    @test jc.detuning_cuts == [0.0, 0.55, 0.7]
    @test jc.g == 14.0
    @test length(jc.fock) == 9                       # 3 representative drives per cut
    @test all(r -> isfinite(r.c1) && isfinite(r.c2) && isfinite(r.c3), jc.rows)
    # The acceptance gate that every production point had to pass.
    @test maximum(abs(r.current_check - 1) for r in jc.rows) < 1e-4

    # The CSV and the JLD2 describe the same run.
    csv = load_jc_production_csv()
    @test nrow(csv) == length(jc.rows)
    @test csv.c1 ≈ [r.c1 for r in jc.rows]
    @test csv.c2 ≈ [r.c2 for r in jc.rows]

    sweeps = load_qhe_paper_sweeps()
    @test nrow(sweeps.antibunching_g) == 500
    @test nrow(sweeps.antibunching_λc) == 100
    @test nrow(sweeps.finite_affinity_g) == 50
    @test nrow(sweeps.finite_affinity_λc) == 50
    # The manuscript's two headline claims, straight from the stored sweeps.
    @test minimum(sweeps.antibunching_g.Fh) < 1.0                       # antibunched
    @test all(sweeps.antibunching_g.Qh .> 2.0)                          # yet TUR-respecting
    @test minimum(abs.(sweeps.finite_affinity_g.Qh .- 2)) < 0.05        # approaches the bound
end
