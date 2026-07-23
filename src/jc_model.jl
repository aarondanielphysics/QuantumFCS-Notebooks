# Production full-counting-statistics pipeline for the driven-dissipative
# Jaynes-Cummings model (rotating frame, cavity loss only):
#
#   H = -Δ(a†a + σ₊σ₋) + g(a†σ₋ + aσ₊) - E(a + a†),   ρ̇ = -i[H,ρ] + κ𝒟[a]ρ,
#
# counting the cavity emission κ a ρ a†. `run_jc_fcs_production_sweep` sweeps the
# drive x = 2E/g at fixed g/κ for several detunings, choosing a per-point Fock
# cutoff from the semiclassical bright-branch estimate, solving the trace-
# constrained steady state by continuation (warm-started GMRES with an adaptive,
# reused shifted-ILU preconditioner), and computing the cumulants with that same
# ILU injected into QuantumFCS's Drazin solve. See the driven_dissipative_JC
# notebook to run it, and driven_dissipative_jc_diagnostics.jl for the
# steady-state reliability studies that are not part of this pipeline.

using LinearAlgebra
using SparseArrays
using LinearSolve
using Krylov #Necessary for iterative solvers
using IncompleteLU #Necessary for iterative solvers
using QuantumToolbox
using QuantumFCS
using Logging #Warning capture around the FCS Drazin solves
import Pkg #Provenance for the production metadata files



@isdefined(JC_DEFAULT_TRUNC_TOL) || const JC_DEFAULT_TRUNC_TOL = 1e-4
@isdefined(JC_DEFAULT_WIDE_TAIL_FACTOR) || const JC_DEFAULT_WIDE_TAIL_FACTOR = 10.0
@isdefined(JC_DEFAULT_AMBIGUOUS_TAIL_FACTOR) || const JC_DEFAULT_AMBIGUOUS_TAIL_FACTOR = 10.0
@isdefined(JC_DEFAULT_AMBIGUOUS_WIDE_TAIL_FACTOR) || const JC_DEFAULT_AMBIGUOUS_WIDE_TAIL_FACTOR = 100.0
@isdefined(JC_DEFAULT_SCOUT_REL_N_TOL) || const JC_DEFAULT_SCOUT_REL_N_TOL = 0.05
@isdefined(JC_DEFAULT_RESONANT_UNSAFE_X_MIN) || const JC_DEFAULT_RESONANT_UNSAFE_X_MIN = 4.1
@isdefined(JC_DEFAULT_SS_ILU_TAU) || const JC_DEFAULT_SS_ILU_TAU = 1e-3
@isdefined(JC_DEFAULT_SCOUT_SS_ILU_TAU) || const JC_DEFAULT_SCOUT_SS_ILU_TAU = 1e-1
@isdefined(JC_DEFAULT_SS_ILU_SHIFT_FACTOR) || const JC_DEFAULT_SS_ILU_SHIFT_FACTOR = 1e-6
@isdefined(JC_DEFAULT_FCS_TAU) || const JC_DEFAULT_FCS_TAU = 0.05
@isdefined(JC_DEFAULT_FCS_RTOL) || const JC_DEFAULT_FCS_RTOL = 1e-8
@isdefined(JC_DEFAULT_FCS_ITMAX) || const JC_DEFAULT_FCS_ITMAX = 200
@isdefined(JC_DEFAULT_FCS_MEMORY) || const JC_DEFAULT_FCS_MEMORY = 30
@isdefined(JC_DEFAULT_PROPAGATOR_DT) || const JC_DEFAULT_PROPAGATOR_DT = 50.0
@isdefined(JC_DEFAULT_PROPAGATOR_MAX_STEPS) || const JC_DEFAULT_PROPAGATOR_MAX_STEPS = 80
@isdefined(JC_DEFAULT_PROPAGATOR_KRYLOV_DIM) || const JC_DEFAULT_PROPAGATOR_KRYLOV_DIM = 50
@isdefined(JC_DEFAULT_PROPAGATOR_CHANGE_RTOL) || const JC_DEFAULT_PROPAGATOR_CHANGE_RTOL = 1e-6
@isdefined(JC_DEFAULT_PROPAGATOR_DERIVATIVE_RTOL) || const JC_DEFAULT_PROPAGATOR_DERIVATIVE_RTOL = 1e-7
@isdefined(JC_DEFAULT_SS_REUSE_REBUILD_NITER) || const JC_DEFAULT_SS_REUSE_REBUILD_NITER = 80
@isdefined(JC_DEFAULT_SS_REUSE_ITMAX) || const JC_DEFAULT_SS_REUSE_ITMAX = 120
@isdefined(JC_DEFAULT_SS_REUSE_FALLBACK_ITMAX) || const JC_DEFAULT_SS_REUSE_FALLBACK_ITMAX = 300
@isdefined(JC_DEFAULT_SS_GMRES_MEMORY) || const JC_DEFAULT_SS_GMRES_MEMORY = 60
@isdefined(JC_DEFAULT_SS_GMRES_RTOL) || const JC_DEFAULT_SS_GMRES_RTOL = 1e-10
@isdefined(JC_DEFAULT_SS_GMRES_ATOL) || const JC_DEFAULT_SS_GMRES_ATOL = 1e-14

jc_default_ss_alg() = KrylovJL_GMRES()

function jc_operators_for_cutoff(N_cutoff::Integer)
    a = QuantumToolbox.tensor(QuantumToolbox.destroy(N_cutoff + 1), QuantumToolbox.qeye(2))
    adag = a'
    sm = QuantumToolbox.tensor(QuantumToolbox.qeye(N_cutoff + 1), QuantumToolbox.sigmam())
    sp = sm'
    return (
        a = a,
        adag = adag,
        sm = sm,
        sp = sp,
        Hjc = adag * sm + a * sp,
        Hdr = -(a + adag),
        n_op = adag * a,
        detuning_op = adag * a + sp * sm,
    )
end

function jc_shifted_ilu_factor(A;
        ilu_tau=JC_DEFAULT_SS_ILU_TAU,
        shift_factor=JC_DEFAULT_SS_ILU_SHIFT_FACTOR)
    # Delegate to QuantumFCS.shifted_ilu_preconditioner: identical shift
    # (shift_factor·max(norm(A,1)/size(A,1), eps)) and identical factorization
    # (QuantumToolbox.ilu re-exports IncompleteLU.ilu). Coerce A to the sparse
    # ComplexF64 form the package method expects, without copying when it already
    # matches (the production hot path passes system.A, already this type).
    Asp = A isa SparseMatrixCSC{ComplexF64,Int} ? A : SparseMatrixCSC{ComplexF64,Int}(A)
    return shifted_ilu_preconditioner(Asp; τ=ilu_tau, shift_factor=shift_factor)
end

function jc_steady_solver_settings(;
        ss_ilu_tau=JC_DEFAULT_SS_ILU_TAU,
        ss_shift_factor=JC_DEFAULT_SS_ILU_SHIFT_FACTOR,
        ss_preconditioner=:shifted_ilu,
        ss_alg=jc_default_ss_alg(),
        ss_solver_kwargs=NamedTuple())
    return (
        ss_ilu_tau=ss_ilu_tau,
        ss_shift_factor=ss_shift_factor,
        ss_preconditioner=ss_preconditioner,
        ss_alg=ss_alg,
        ss_solver_kwargs=ss_solver_kwargs,
    )
end

function jc_fcs_solver_settings(;
        fcs_sigma=nothing,
        fcs_tau=JC_DEFAULT_FCS_TAU,
        fcs_rtol=JC_DEFAULT_FCS_RTOL,
        fcs_itmax=JC_DEFAULT_FCS_ITMAX,
        fcs_memory=JC_DEFAULT_FCS_MEMORY)
    return (
        fcs_sigma=fcs_sigma,
        fcs_tau=fcs_tau,
        fcs_rtol=fcs_rtol,
        fcs_itmax=fcs_itmax,
        fcs_memory=fcs_memory,
    )
end

function jc_recommended_fcs_solver_settings()
    return merge((solver_label="stronger preconditioner",), jc_fcs_solver_settings(
        fcs_tau=0.02,
        fcs_memory=50,
        fcs_rtol=1e-8,
        fcs_itmax=300,
        fcs_sigma=nothing,
    ))
end

function jc_cavity_population(ρss)
    cavity_state = QuantumToolbox.ptrace(ρss, 1)
    cavity_population = max.(real.(diag(cavity_state.data)), 0.0)
    total = sum(cavity_population)
    total > 0 || error("Cavity marginal has zero total population.")
    return cavity_population ./ total
end

function jc_tail_diagnostics(cavity_population)
    tail2_start = max(1, length(cavity_population) - 1)
    tail5_start = max(1, length(cavity_population) - 4)
    return (
        cavity_boundary = cavity_population[end],
        cavity_tail = sum(cavity_population[tail2_start:end]),
        cavity_tail5 = sum(cavity_population[tail5_start:end]),
    )
end

function jc_cutoff_status_from_tails(tail2, tail5;
        trunc_tol=JC_DEFAULT_TRUNC_TOL,
        wide_factor=JC_DEFAULT_WIDE_TAIL_FACTOR,
        ambiguous_tail_factor=JC_DEFAULT_AMBIGUOUS_TAIL_FACTOR,
        ambiguous_wide_tail_factor=JC_DEFAULT_AMBIGUOUS_WIDE_TAIL_FACTOR)
    if tail2 <= trunc_tol && tail5 <= wide_factor * trunc_tol
        return :safe
    elseif tail2 <= ambiguous_tail_factor * trunc_tol &&
           tail5 <= ambiguous_wide_tail_factor * trunc_tol
        return :ambiguous
    else
        return :unsafe
    end
end

function jc_max_threadid()
    return isdefined(Threads, :maxthreadid) ? Threads.maxthreadid() : Threads.nthreads()
end

function jc_parallel_sweeps_enabled(; min_jobs=2)
    return Threads.nthreads() > 1 && min_jobs >= 2
end

function jc_thread_summary(; min_jobs=2)
    return (
        threads=Threads.nthreads(),
        max_threadid=jc_max_threadid(),
        threaded=jc_parallel_sweeps_enabled(min_jobs=min_jobs),
    )
end

jc_seconds_since(t0::UInt64) = (time_ns() - t0) / 1e9

function jc_trace_constrained_linear_problem(L)
    # Build the trace-constrained system with QuantumFCS. The package reproduces the
    # previous local numerics exactly (weight = norm(L,1)/length(L); trace row on the
    # vectorized-identity diagonal), so A and b are identical. Keep the local
    # NamedTuple interface (A/b/L_data/N_hilbert/dimensions) and additionally carry the
    # package TraceConstrainedSystem.
    system = trace_constrained_system(L)
    return (
        A=system.A,
        b=system.b,
        L_data=system.L,
        N_hilbert=prod(L.dimensions),
        dimensions=L.dimensions,
        pkg_system=system,
    )
end

function jc_quantumobject_from_steady_vector(ρ_vec, N_hilbert, dimensions)
    ρ_mat = reshape(ρ_vec, N_hilbert, N_hilbert)
    ρ_mat = (ρ_mat + ρ_mat') / 2
    return QuantumToolbox.QuantumObject(ρ_mat, QuantumToolbox.Operator, dimensions), ρ_mat
end

function jc_steadystate_physics_metrics(ρss, n_op, N_cutoff, trunc_tol)
    # Application-specific physics / Fock-truncation diagnostics. These operate on the
    # QuantumObject steady state (expect / ptrace), so they are fast for a sparse or
    # dense state alike and never touch the vectorized density matrix directly.
    n = real(QuantumToolbox.expect(n_op, ρss))
    tail = jc_tail_diagnostics(jc_cavity_population(ρss))
    status = jc_cutoff_status_from_tails(tail.cavity_tail, tail.cavity_tail5;
        trunc_tol=trunc_tol)
    return merge((
        n=n,
        occupation_fraction=n / N_cutoff,
        cutoff_status=status,
    ), tail)
end

function jc_steadystate_quality_metrics(ρ_mat, ρss, L_data, n_op, N_cutoff, trunc_tol)
    # Full metrics = independently computed solver-quality diagnostics + the physics
    # diagnostics above. Used by the steady-state reliability study, which compares
    # solver methods and therefore recomputes trace / residual / hermiticity errors
    # itself. The production sweep instead takes solver quality from the package's
    # `ss.stats` and calls `jc_steadystate_physics_metrics` directly (see
    # run_jc_fcs_production_sweep). Densify the small n×n state once (no-op for dense
    # input) so the matvec and adjoint difference stay cheap.
    ρ_dense = Matrix(ρ_mat)
    ρ_vec = vec(ρ_dense)
    residual = norm(L_data * ρ_vec)
    residual_denominator = max(norm(L_data, 1) * norm(ρ_vec), eps(Float64))
    hermiticity_denominator = max(norm(ρ_dense), eps(Float64))
    return merge((
        trace_error=abs(tr(ρ_dense) - 1),
        liouvillian_residual=residual,
        relative_liouvillian_residual=residual / residual_denominator,
        hermiticity_error=norm(ρ_dense - ρ_dense') / hermiticity_denominator,
    ), jc_steadystate_physics_metrics(ρss, n_op, N_cutoff, trunc_tol))
end

# ---------------------------------------------------------------------------
# Steady-state continuation across a segment of equal-cutoff drive points.
#
# Holds the segment's shifted-ILU preconditioner and the previous point's dense
# solution, so each point warm-starts from its neighbour and reuses the ILU.
# `jc_continuation_solve!` calls QuantumFCS.trace_constrained_steadystate directly and
# rebuilds the ILU when a solve stalls (`!converged || iterations > rebuild_niter`).
# One instance per cutoff segment — construction is the reset. This is app policy
# (the reuse/rebuild strategy), built on the package solver; no numeric wrapper inside.
# ---------------------------------------------------------------------------
mutable struct JCSteadyContinuation{S}
    settings::S     # (; ilu_tau, shift_factor, rtol, atol, itmax, fallback_itmax, memory, rebuild_niter)
    Pl::Any         # current segment preconditioner (nothing ⇒ build fresh next solve)
    u_prev::Any     # previous dense solution as warm start (nothing ⇒ cold start)
end
JCSteadyContinuation(settings) = JCSteadyContinuation(settings, nothing, nothing)

"""
    jc_continuation_solve!(cont, sys) -> (ss, sstats)

Solve the trace-constrained steady state for the package system `sys`
(`QuantumFCS.TraceConstrainedSystem`), reusing `cont.Pl`/`cont.u_prev` and rebuilding the
ILU on a stall. Updates `cont` in place and returns the `TraceConstrainedSteadyState`
plus scalar `sstats` (`rebuilds, preconditioner_seconds, gmres_seconds, niters, converged`).
"""
function jc_continuation_solve!(cont::JCSteadyContinuation, sys)
    s = cont.settings
    preconditioner_seconds = 0.0
    rebuilds = 0

    cont.Pl === nothing && GC.gc()
    ss = trace_constrained_steadystate(sys;
        method=:iterative, Pl=cont.Pl, u0=cont.u_prev,
        τ=s.ilu_tau, shift_factor=s.shift_factor,
        rtol=s.rtol, atol=s.atol, itmax=s.itmax, memory=s.memory)
    if cont.Pl === nothing
        rebuilds += 1
        preconditioner_seconds += ss.stats.ilu_seconds
    end
    cont.Pl = ss.Pl
    niters = ss.stats.iterations
    converged = ss.stats.converged
    gmres_seconds = ss.stats.gmres_seconds

    if !converged || niters > s.rebuild_niter
        GC.gc()
        ss = trace_constrained_steadystate(sys;
            method=:iterative, Pl=nothing, u0=vec(Matrix(ss.rho_ss)),
            τ=s.ilu_tau, shift_factor=s.shift_factor,
            rtol=s.rtol, atol=s.atol, itmax=s.fallback_itmax, memory=s.memory)
        rebuilds += 1
        preconditioner_seconds += ss.stats.ilu_seconds
        cont.Pl = ss.Pl
        niters += ss.stats.iterations
        converged = ss.stats.converged
        gmres_seconds += ss.stats.gmres_seconds
    end

    cont.u_prev = vec(Matrix(ss.rho_ss))
    return ss, (; rebuilds, preconditioner_seconds, gmres_seconds, niters, converged)
end

function jc_retcode_text(solution)
    return (:retcode in propertynames(solution)) ? string(getproperty(solution, :retcode)) : ""
end

function jc_solution_niters(solution)
    if :stats in propertynames(solution)
        stats = getproperty(solution, :stats)
        if :niter in propertynames(stats)
            return getproperty(stats, :niter)
        elseif :niters in propertynames(stats)
            return getproperty(stats, :niters)
        end
    end
    return missing
end

function jc_scale_metadata(; scan_name, N, E_reference, g_reference,
        detuning_cuts, scale_count, scale_min, scale_max,
        trunc_tol=JC_DEFAULT_TRUNC_TOL,
        wide_tail_factor=JC_DEFAULT_WIDE_TAIL_FACTOR)
    return (
        scan=scan_name,
        cutoff=N,
        E_reference=E_reference,
        g_reference=g_reference,
        two_E_over_g=2E_reference / g_reference,
        detuning_cuts=collect(detuning_cuts),
        scale_min=scale_min,
        scale_max=scale_max,
        scale_count=scale_count,
        trunc_tol=trunc_tol,
        wide_tail_tol=wide_tail_factor * trunc_tol,
    )
end

function jc_warm_gmres_steady_solve(A, b, Pl;
        u0=nothing,
        gmres_memory=JC_DEFAULT_SS_GMRES_MEMORY,
        gmres_rtol=JC_DEFAULT_SS_GMRES_RTOL,
        gmres_atol=JC_DEFAULT_SS_GMRES_ATOL,
        itmax=JC_DEFAULT_SS_REUSE_ITMAX)
    t0 = time_ns()
    x, stats = u0 === nothing ?
        Krylov.gmres(A, b; M=Pl, ldiv=true, memory=gmres_memory,
            rtol=gmres_rtol, atol=gmres_atol, itmax=itmax) :
        Krylov.gmres(A, b, u0; M=Pl, ldiv=true, memory=gmres_memory,
            rtol=gmres_rtol, atol=gmres_atol, itmax=itmax)
    return x, stats.niter, stats.solved, jc_seconds_since(t0)
end




# Cutoff tiers. Fine steps (25 up to 300, 50 above) keep the jump between
# adjacent points' cutoffs small, so the tier-synced saw-tooth in the skewness
# `c₃/c₂^{3/2}` (residual c₃ truncation error that resets at each cutoff step)
# stays small and the curve reads smooth. Top tier 500 for the deep bright
# branch (quantum n up to ~343; ≥7σ headroom); N=500 points hold ~8-10 GB each
# — cleared machine, serial. Add a 600 tier to fully flatten the deepest
# detuned points (drops their occupation ~0.69 → ~0.61) at ~14 GB / +~30 min.
@isdefined(JC_DEFAULT_FCS_PROD_TIERS) || const JC_DEFAULT_FCS_PROD_TIERS =
    (150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500)
# Truncation headroom above the estimated bright-branch photon number. The
# bright state is coherent-like (width ≈ √n), so `pad_sigma·√n` standard
# deviations plus a small floor keeps the boundary tail negligible; the
# occupation cap is a loose backstop for the coexistence region (where the
# semiclassical estimate over-reports and the schedule is conservative anyway).
# Calibrated 2026-07-04/05 against measured quantum n at g/κ=14 (Notes/
# jc_n400_scan_reliability.md): the estimate matches quantum n to ~5% on the
# bright branch. `occ_max` is the aspirational occupation cap driving extra
# headroom for `c₃` — tightened 0.72 → 0.50 because the skewness saw-tooth
# showed `c₃` is truncation-biased already near occ ≈ 0.45. It only raises the
# chosen tier (never flags): where the ladder tops out the mean/Fano/tail stay
# resolved (that is the `pad_sigma·√n` "hard" criterion), only c₃ keeps a small
# bias. The 6σ hard headroom gives boundary tail ≈ 1e-9.
@isdefined(JC_DEFAULT_FCS_PROD_OCC_MAX) || const JC_DEFAULT_FCS_PROD_OCC_MAX = 0.50
@isdefined(JC_DEFAULT_FCS_PROD_PAD_SIGMA) || const JC_DEFAULT_FCS_PROD_PAD_SIGMA = 6.0
@isdefined(JC_DEFAULT_FCS_PROD_PAD_ABS) || const JC_DEFAULT_FCS_PROD_PAD_ABS = 25.0
@isdefined(JC_DEFAULT_FCS_PROD_RTOL) || const JC_DEFAULT_FCS_PROD_RTOL = 1e-8
@isdefined(JC_DEFAULT_FCS_PROD_ITMAX) || const JC_DEFAULT_FCS_PROD_ITMAX = 300
@isdefined(JC_DEFAULT_FCS_PROD_MEMORY) || const JC_DEFAULT_FCS_PROD_MEMORY = 60
@isdefined(JC_DEFAULT_FCS_PROD_CHECK_TOL) || const JC_DEFAULT_FCS_PROD_CHECK_TOL = 1e-4

"""
Semiclassical (Maxwell-Bloch) steady-state photon number of the driven JC model
in notebook units (κ𝒟[a], so Carmichael's κ_C = κ/2). At Δ = 0 the
above-critical branch has the closed form n = (4E² − g²)/κ² (zero below the
critical drive 2E/g = 1); for Δ ≠ 0 the largest root of the neoclassical state
equation is found by bisection over both dressing branches. Used only to
*estimate* the required Fock cutoff — it overestimates the quantum photon
number near and above the transition, which is the safe direction.
Validation anchors (quantum values): (g/κ=10, x=1.5, Δ=0) → 125.3;
(g/κ=28, x=5/7, Δ=0.55) → ≈129; (g/κ=28, x=5/7, Δ=0.70) → ≈357.
"""
function jc_semiclassical_bright_n(g, E, Δ; κ=1.0)
    κC = κ / 2
    iszero(Δ) && return max(4 * abs2(E) - abs2(g), 0.0) / abs2(κ)

    # Neoclassical state equation: u = E² / (κ_C² + D_s(u)²) with the effective
    # detuning D_s(u) = Δ − s·sgn(Δ)·g²/√(Δ² + 4g²u), branches s = ±1. The bright
    # state is the largest root over both branches.
    detuning_s(u, s) = Δ - s * sign(Δ) * abs2(g) / sqrt(abs2(Δ) + 4 * abs2(g) * u)
    residual_s(u, s) = u * (abs2(κC) + abs2(detuning_s(u, s))) - abs2(E)

    u_hi = 4 * abs2(E) / abs2(κ) + 1.0   # empty-cavity bound: F(u) > 0 beyond it
    n_best = 0.0
    for s in (1.0, -1.0)
        grid = range(0.0, u_hi; length=2001)
        last_sign_change = 0
        for i in 1:(length(grid) - 1)
            residual_s(grid[i], s) * residual_s(grid[i + 1], s) <= 0 && (last_sign_change = i)
        end
        last_sign_change == 0 && continue
        lo, hi = grid[last_sign_change], grid[last_sign_change + 1]
        for _ in 1:80
            mid = (lo + hi) / 2
            if residual_s(lo, s) * residual_s(mid, s) <= 0
                hi = mid
            else
                lo = mid
            end
        end
        n_best = max(n_best, (lo + hi) / 2)
    end
    return n_best
end

"""
Graded drive-sweep grid: a sorted, unique `x = 2E/g` vector built from
`segments`, each a `(start, stop, spacing)` triple sampled with `range(start,
stop; step=spacing)`. Shared endpoints between adjacent segments are deduplicated.

The default grades the sampling so the blockade-breakdown transition and the
giant-Fano spike (x ≈ 0.55–0.85 on the detuned cuts) are resolved finely, while
the dim tail and the smooth bright branch stay coarse: `(0.05,0.5,0.03)` +
`(0.5,1.0,0.01)` + `(1.0,1.45,0.02)` ≈ 89 points. The Δx = 0.01 window resolves
the ~0.05-wide first-order jump with ~5 points; tighten a segment (e.g. add
`(0.55,0.85,0.006)`) for a sharper discontinuity.
"""
function jc_graded_drive_grid(;
        segments=((0.05, 0.5, 0.03), (0.5, 1.0, 0.01), (1.0, 1.45, 0.02)))
    xs = Float64[]
    for (start, stop, spacing) in segments
        append!(xs, range(start, stop; step=spacing))
        push!(xs, stop)  # guarantee the endpoint even when step doesn't divide evenly
    end
    return sort(unique(round.(xs; digits=6)))
end

"""
Ordered point list for one production drive sweep: fixed `g`, detuning `Δ`,
`x = 2E/g` values ascending. Points feed `run_jc_fcs_production_sweep`.
"""
function jc_drive_sweep_points(Δ, x_values; g, κ=1.0)
    return [
        (
            label="fcs Δ=$(Δ), x=$(round(x; digits=4))",
            probe_group="fixed g drive sweep",
            Δ=Δ,
            x=x,
            g=g,
            E=x * g / 2,
        )
        for x in sort(collect(x_values))
    ]
end

"""
Resolve a schedule parameter that may be given either as a scalar (applied to
every cut) or as a per-cut callable `Δ -> value`.
"""
jc_resolve_cut_param(p, Δ) = p isa Function ? p(Δ) : p

"""
Hard (physical) Fock-cutoff requirement for a point with estimated bright-branch
photon number `n_est`: `n_est + pad_sigma·√n_est + pad_abs`. The √n term is the
coherent-state truncation criterion; a cutoff at or above this resolves the mean,
Fano, and tail. A point whose hard requirement exceeds the top tier is genuinely
under-resolved (this is what sets the `cutoff_budget_exceeded` flag).
"""
function jc_cutoff_hard(n_est;
        pad_sigma=JC_DEFAULT_FCS_PROD_PAD_SIGMA,
        pad_abs=JC_DEFAULT_FCS_PROD_PAD_ABS)
    n = max(n_est, 0.0)
    return n + pad_sigma * sqrt(n) + pad_abs
end

"""
Aspirational Fock-cutoff target: the larger of the hard requirement
([`jc_cutoff_hard`](@ref)) and the occupation aspiration `n_est / occ_max`. The
occupation term buys extra headroom for the *higher* cumulants (`c₃` is far more
truncation-sensitive than `c₁`; see the tier-synced saw-tooth in the skewness).
It only raises the chosen tier — it never flags a point, since a point that meets
the hard criterion is resolved even if the tighter occupation goal is not met at
the top of the tier ladder.
"""
function jc_cutoff_target(n_est;
        occ_max=JC_DEFAULT_FCS_PROD_OCC_MAX,
        pad_sigma=JC_DEFAULT_FCS_PROD_PAD_SIGMA,
        pad_abs=JC_DEFAULT_FCS_PROD_PAD_ABS)
    n = max(n_est, 0.0)
    return max(n / occ_max, jc_cutoff_hard(n; pad_sigma, pad_abs))
end

"""
Per-point Fock cutoff for one drive sweep: the smallest tier at or above the
aspirational [`jc_cutoff_target`](@ref) of the semiclassical estimate
(`jc_semiclassical_bright_n`), capped at the top tier.

A point is flagged `clamped` only when its **hard** requirement
([`jc_cutoff_hard`](@ref)) exceeds the top tier — i.e. it is genuinely
under-resolved. A point that meets the hard criterion but not the tighter
occupation aspiration (because the ladder tops out) is *not* flagged: it is
resolved for the mean/Fano/tail, only its `c₃` carries the residual truncation
bias made visible by the saw-tooth. Warn only on genuine clamps.

With `return_detail = true` returns a NamedTuple of vectors
`(x, n_est, target, N, clamped)` for previewing the ladder; otherwise returns
the `N` vector.
"""
function jc_dynamic_cutoff_schedule(x_values, Δ; g, κ=1.0,
        tiers=JC_DEFAULT_FCS_PROD_TIERS,
        occ_max=JC_DEFAULT_FCS_PROD_OCC_MAX,
        pad_sigma=JC_DEFAULT_FCS_PROD_PAD_SIGMA,
        pad_abs=JC_DEFAULT_FCS_PROD_PAD_ABS,
        return_detail=false)
    tier_list = sort(collect(tiers))
    top = tier_list[end]
    xs = sort(collect(x_values))
    n_ests = [jc_semiclassical_bright_n(g, x * g / 2, Δ; κ=κ) for x in xs]
    targets = [jc_cutoff_target(n; occ_max, pad_sigma, pad_abs) for n in n_ests]
    clamped = [jc_cutoff_hard(n; pad_sigma, pad_abs) > top for n in n_ests]
    N = [begin
            idx = findfirst(tier -> tier >= t, tier_list)
            idx === nothing ? top : tier_list[idx]
        end for t in targets]

    if any(clamped)
        offending = [(round(xs[i]; digits=3), ceil(Int, jc_cutoff_hard(n_ests[i]; pad_sigma, pad_abs)))
                     for i in eachindex(xs) if clamped[i]]
        @warn "Cutoff schedule clamped to the top tier $(top): the physical headroom " *
              "requirement exceeds it for some points on the Δ=$(Δ) cut — these are " *
              "genuinely under-resolved. Raise the tiers or trim the drive range." offending
    end
    return return_detail ? (x=xs, n_est=n_ests, target=targets, N=N, clamped=clamped) : N
end

# Count Warn-or-worse log records emitted by `f()` while still forwarding them
# to the current logger. Used to detect non-converged FCS Drazin solves (the
# iterative backend warns instead of throwing).
function jc_count_warnings(f)
    hits = Ref(0)
    parent = Logging.current_logger()
    logger = Logging.SimpleLogger(stderr, Logging.Warn)
    result = Logging.with_logger(JCWarnCountingLogger(hits, parent)) do
        f()
    end
    return result, hits[]
end

struct JCWarnCountingLogger <: Logging.AbstractLogger
    hits::Ref{Int}
    parent::Logging.AbstractLogger
end




Logging.min_enabled_level(logger::JCWarnCountingLogger) =
    Logging.min_enabled_level(logger.parent)
Logging.shouldlog(logger::JCWarnCountingLogger, level, _module, group, id) =
    Logging.shouldlog(logger.parent, level, _module, group, id)
Logging.catch_exceptions(logger::JCWarnCountingLogger) =
    Logging.catch_exceptions(logger.parent)
function Logging.handle_message(logger::JCWarnCountingLogger, level, message,
        _module, group, id, file, line; kwargs...)
    level >= Logging.Warn && (logger.hits[] += 1)
    return Logging.handle_message(logger.parent, level, message,
        _module, group, id, file, line; kwargs...)
end

"""
    jc_cutoff_segments(schedule) -> Vector{UnitRange{Int}}

Group consecutive indices of `schedule` sharing the same cutoff into contiguous ranges —
one steady-state continuation segment each (the preconditioner is reused within a range).
"""
function jc_cutoff_segments(schedule)
    segments = UnitRange{Int}[]
    n = length(schedule)
    start = 1
    while start <= n
        stop = start
        while stop < n && schedule[stop + 1] == schedule[start]
            stop += 1
        end
        push!(segments, start:stop)
        start = stop + 1
    end
    return segments
end

"""
    jc_fcs_with_current_check(L_sparse, ρss, jump, Pl, n; κ, nC, fcs_settings, current_check_tol)

Cumulants of the monitored `jump` with the steady-state ILU injected as `Pl`, validated against
the physical current `c₁ = κ⟨n⟩`. On warnings, non-finite cumulants, or a current-check mismatch,
retry once without `Pl` (QuantumFCS builds its own preconditioner). Returns
`(; cumulants, current_check, fcs_retry, fcs_warnings)`.
"""
function jc_fcs_with_current_check(L_sparse, ρss, jump, Pl, n;
        κ, nC, fcs_settings, current_check_tol)
    f = fcs_settings
    fcs_solve(pl) = jc_count_warnings() do
        fcscumulants_recursive(LindbladFCS(L=L_sparse, mJ=[jump], rho_ss=ρss, nu=[1.0],
            nC=nC, method=:iterative, Pl=pl, rtol=f.rtol, itmax=f.itmax, memory=f.memory))
    end
    cumulants, warns = fcs_solve(Pl)
    cc = cumulants[1] / (κ * n)
    retry = warns > 0 || !all(isfinite, cumulants) || abs(cc - 1) > current_check_tol
    if retry
        cumulants, warns = fcs_solve(nothing)   # rebuild the FCS preconditioner from scratch
        cc = cumulants[1] / (κ * n)
    end
    return (; cumulants, current_check=cc, fcs_retry=retry, fcs_warnings=warns)
end

"""
Production FCS sweep: for each detuning cut, sweep `x = 2E/g` at fixed `g`,
with the per-point cutoff from `jc_dynamic_cutoff_schedule`. Within each
contiguous cutoff segment the steady state is solved by the continuation
pattern (warm-started GMRES, shifted ILU reused and rebuilt adaptively), and
the cumulants are computed by QuantumFCS with the *same* ILU injected as the
Drazin preconditioner (`Pl`), so each point performs at most one factorization.

Acceptance gate per point: the identity `c₁ = κ⟨n⟩` (`current_check`), finite
c₂/c₃, and no solver warnings; on failure the point is retried once letting
QuantumFCS build its own fresh preconditioner (`fcs_retry = true`).
`N_override` bypasses the dynamic schedule and forces one cutoff for every
point — used by the lower-cutoff spot validation.

Returns one row per point: point fields, cutoff `N`, steady-state quality and
solver diagnostics, cumulants `c1, c2, c3` (κ units), `Fano = c2/c1`,
`current_check`, and timings.
"""
function run_jc_fcs_production_sweep(;
        detuning_cuts,
        x_values,
        g,
        κ=1.0,
        nC=3,
        trunc_tol=JC_DEFAULT_TRUNC_TOL,
        tiers=JC_DEFAULT_FCS_PROD_TIERS,
        occ_max=JC_DEFAULT_FCS_PROD_OCC_MAX,
        pad_sigma=JC_DEFAULT_FCS_PROD_PAD_SIGMA,
        pad_abs=JC_DEFAULT_FCS_PROD_PAD_ABS,
        ilu_tau=JC_DEFAULT_SCOUT_SS_ILU_TAU,
        shift_factor=JC_DEFAULT_SS_ILU_SHIFT_FACTOR,
        rebuild_niter=JC_DEFAULT_SS_REUSE_REBUILD_NITER,
        itmax=JC_DEFAULT_SS_REUSE_ITMAX,
        fallback_itmax=JC_DEFAULT_SS_REUSE_FALLBACK_ITMAX,
        gmres_memory=JC_DEFAULT_SS_GMRES_MEMORY,
        gmres_rtol=JC_DEFAULT_SS_GMRES_RTOL,
        gmres_atol=JC_DEFAULT_SS_GMRES_ATOL,
        fcs_rtol=JC_DEFAULT_FCS_PROD_RTOL,
        fcs_itmax=JC_DEFAULT_FCS_PROD_ITMAX,
        fcs_memory=JC_DEFAULT_FCS_PROD_MEMORY,
        current_check_tol=JC_DEFAULT_FCS_PROD_CHECK_TOL,
        N_override=nothing,
        verbose=true)
    rows = Vector{Any}()
    xs = sort(collect(x_values))
    steady_settings = (; ilu_tau, shift_factor, rtol=gmres_rtol, atol=gmres_atol,
                         itmax, fallback_itmax, memory=gmres_memory, rebuild_niter)
    fcs_settings = (; rtol=fcs_rtol, itmax=fcs_itmax, memory=fcs_memory)
    for Δ_val in detuning_cuts
        points = jc_drive_sweep_points(Δ_val, xs; g=g, κ=κ)
        # `N_override` forces one cutoff for every point (used by the
        # lower-cutoff spot validation); otherwise the dynamic schedule applies.
        if N_override === nothing
            # occ_max/pad_sigma/pad_abs may be per-cut callables (Δ -> value) so
            # a single cut (e.g. the small-photon-number Δ=0 resonant cut) can be
            # given extra σ-headroom for a clean c₃ without touching the others.
            detail = jc_dynamic_cutoff_schedule(xs, Δ_val; g=g, κ=κ,
                tiers=tiers,
                occ_max=jc_resolve_cut_param(occ_max, Δ_val),
                pad_sigma=jc_resolve_cut_param(pad_sigma, Δ_val),
                pad_abs=jc_resolve_cut_param(pad_abs, Δ_val),
                return_detail=true)
            schedule = detail.N
            clamped_flags = detail.clamped
        else
            schedule = fill(Int(N_override), length(xs))
            clamped_flags = falses(length(xs))
        end

        for seg in jc_cutoff_segments(schedule)
            N_cutoff = schedule[first(seg)]
            ops = jc_operators_for_cutoff(N_cutoff)
            cavity_loss = sqrt(κ) * ops.a
            cont = JCSteadyContinuation(steady_settings)   # fresh Pl / warm-start per segment

            for i in seg
                point = points[i]
                total_t0 = time_ns()
                H = -point.Δ * ops.detuning_op + point.g * ops.Hjc + point.E * ops.Hdr
                L = QuantumToolbox.liouvillian(H, [cavity_loss])
                sys = trace_constrained_system(L)          # QuantumFCS.TraceConstrainedSystem
                liouvillian_seconds = jc_seconds_since(total_t0)

                # Steady state: package solver with the segment's ILU reuse / adaptive rebuild.
                ss, sstats = jc_continuation_solve!(cont, sys)
                ρss = QuantumToolbox.QuantumObject(ss.rho_ss;
                    type=QuantumToolbox.Operator, dims=L.dimensions)

                # Solver quality from `ss.stats`; the app owns the physical ‖L·ρ‖ residual (one
                # matvec on the dense state `cont.u_prev`) and the Fock-truncation diagnostics.
                physics = jc_steadystate_physics_metrics(ρss, ops.n_op, N_cutoff, trunc_tol)
                ss_residual = norm(sys.L * cont.u_prev)
                ss_relative_residual =
                    ss_residual / max(norm(sys.L, 1) * norm(cont.u_prev), eps(Float64))

                # FCS cumulants, reusing the steady-state ILU as Pl, checked against c₁ = κ⟨n⟩.
                fcs_t0 = time_ns()
                fcs = jc_fcs_with_current_check(SparseMatrixCSC{ComplexF64,Int}(sys.L),
                    ρss, cavity_loss, cont.Pl, physics.n;
                    κ=κ, nC=nC, fcs_settings=fcs_settings, current_check_tol=current_check_tol)
                fcs_seconds = jc_seconds_since(fcs_t0)

                c1 = fcs.cumulants[1]
                c2 = nC >= 2 ? fcs.cumulants[2] : NaN
                c3 = nC >= 3 ? fcs.cumulants[3] : NaN
                push!(rows, merge(point, (
                    N=N_cutoff,
                    cutoff_budget_exceeded=clamped_flags[i],
                    trunc_tol=trunc_tol,
                    ss_ilu_tau=ilu_tau,
                    ss_rebuild_niter=rebuild_niter,
                    ss_converged=sstats.converged,
                    ss_iterations=sstats.niters,
                    ss_rebuilds=sstats.rebuilds,
                    trace_error=ss.stats.trace_error,
                    hermiticity_error=ss.stats.hermiticity_error,
                    liouvillian_residual=ss_residual,
                    relative_liouvillian_residual=ss_relative_residual,
                    c1=c1, c2=c2, c3=c3,
                    Fano=c2 / c1,
                    current_check=fcs.current_check,
                    fcs_retry=fcs.fcs_retry,
                    fcs_warnings=fcs.fcs_warnings,
                    liouvillian_seconds=liouvillian_seconds,
                    preconditioner_seconds=sstats.preconditioner_seconds,
                    ilu_seconds=sstats.preconditioner_seconds,
                    gmres_seconds=sstats.gmres_seconds,
                    fcs_seconds=fcs_seconds,
                    total_seconds=jc_seconds_since(total_t0),
                ), physics))
                verbose && println(
                    "fcs Δ=$(point.Δ) x=$(round(point.x; digits=3)) N=$(N_cutoff): " *
                    "n=$(round(physics.n; sigdigits=6)) F=$(round(c2 / c1; sigdigits=4)) " *
                    "cc=$(round(fcs.current_check; sigdigits=6)) " *
                    "[ss $(round(sstats.preconditioner_seconds + sstats.gmres_seconds; digits=1))s, " *
                    "fcs $(round(fcs_seconds; digits=1))s$(fcs.fcs_retry ? ", RETRY" : "")]")
                verbose && flush(stdout)
            end
            GC.gc()          # free the segment's preconditioner before the next segment
        end
    end
    return rows
end

"""
    jc_collect_fock_distributions(; detuning_cuts, x_values, g, κ=1.0, ...) -> Vector{NamedTuple}

Read-only companion to `run_jc_fcs_production_sweep`: for each detuning cut and
each drive `x = 2E/g` in `x_values`, solve the steady state with the *same*
operator / Liouvillian / continuation machinery and the *same*
`jc_dynamic_cutoff_schedule` cutoff, then record the cavity Fock-state
distribution `Pn = ⟨n|ρ_cavity|n⟩` (`n = 0:N`) via `jc_cavity_population`
instead of FCS cumulants. The production sweep is not modified.

Returns rows `(; Δ, x, g, E, N, Pn)` grouped by cut and sorted by `x`. Intended
for the handful of representative drive points shown in the Pₙ figure column,
not the full sweep grid.
"""
function jc_collect_fock_distributions(;
        detuning_cuts,
        x_values,
        g,
        κ=1.0,
        tiers=JC_DEFAULT_FCS_PROD_TIERS,
        occ_max=JC_DEFAULT_FCS_PROD_OCC_MAX,
        pad_sigma=JC_DEFAULT_FCS_PROD_PAD_SIGMA,
        pad_abs=JC_DEFAULT_FCS_PROD_PAD_ABS,
        ilu_tau=JC_DEFAULT_SCOUT_SS_ILU_TAU,
        shift_factor=JC_DEFAULT_SS_ILU_SHIFT_FACTOR,
        rebuild_niter=JC_DEFAULT_SS_REUSE_REBUILD_NITER,
        itmax=JC_DEFAULT_SS_REUSE_ITMAX,
        fallback_itmax=JC_DEFAULT_SS_REUSE_FALLBACK_ITMAX,
        gmres_memory=JC_DEFAULT_SS_GMRES_MEMORY,
        gmres_rtol=JC_DEFAULT_SS_GMRES_RTOL,
        gmres_atol=JC_DEFAULT_SS_GMRES_ATOL,
        verbose=true)
    rows = Vector{Any}()
    xs = sort(collect(x_values))
    steady_settings = (; ilu_tau, shift_factor, rtol=gmres_rtol, atol=gmres_atol,
                         itmax, fallback_itmax, memory=gmres_memory, rebuild_niter)
    for Δ_val in detuning_cuts
        points = jc_drive_sweep_points(Δ_val, xs; g=g, κ=κ)
        schedule = jc_dynamic_cutoff_schedule(xs, Δ_val; g=g, κ=κ,
            tiers=tiers,
            occ_max=jc_resolve_cut_param(occ_max, Δ_val),
            pad_sigma=jc_resolve_cut_param(pad_sigma, Δ_val),
            pad_abs=jc_resolve_cut_param(pad_abs, Δ_val))
        for seg in jc_cutoff_segments(schedule)
            N_cutoff = schedule[first(seg)]
            ops = jc_operators_for_cutoff(N_cutoff)
            cavity_loss = sqrt(κ) * ops.a
            cont = JCSteadyContinuation(steady_settings)   # warm-start within the segment
            for i in seg
                point = points[i]
                H = -point.Δ * ops.detuning_op + point.g * ops.Hjc + point.E * ops.Hdr
                L = QuantumToolbox.liouvillian(H, [cavity_loss])
                sys = trace_constrained_system(L)
                ss, _ = jc_continuation_solve!(cont, sys)
                ρss = QuantumToolbox.QuantumObject(ss.rho_ss;
                    type=QuantumToolbox.Operator, dims=L.dimensions)
                Pn = jc_cavity_population(ρss)
                push!(rows, (Δ=point.Δ, x=point.x, g=point.g, E=point.E,
                             N=N_cutoff, Pn=Pn))
                verbose && println(
                    "fock Δ=$(point.Δ) x=$(round(point.x; digits=3)) N=$(N_cutoff): " *
                    "peak n=$(argmax(Pn) - 1) Pmax=$(round(maximum(Pn); sigdigits=3))")
                verbose && flush(stdout)
            end
            GC.gc()          # free the segment's preconditioner before the next segment
        end
    end
    return rows
end

"""
Reproducibility metadata for a production FCS run, as a Markdown string. Pass
the run `config` (NamedTuple of physical and solver parameters), the result
`rows`, optional `validation_rows`, the produced `files`, and the wall-clock
window. Package provenance is read from the active environment
(`Pkg.dependencies`).
"""
function jc_fcs_production_metadata_markdown(;
        config,
        rows,
        validation_rows=Any[],
        files=String[],
        started_at,
        finished_at)
    dependency_info = Dict{String,Any}()
    for (_, info) in Pkg.dependencies()
        info.name in ("QuantumFCS", "QuantumToolbox", "Krylov", "IncompleteLU", "LinearSolve") || continue
        source = info.is_tracking_path ? "path: $(info.source)" :
            info.git_revision !== nothing ? "rev: $(info.git_revision) ($(info.git_source))" :
            "registry"
        dependency_info[info.name] = "v$(info.version) [$(source)]"
    end

    cuts = sort(unique([r.Δ for r in rows]))
    io = IOBuffer()
    println(io, "# Production FCS run: driven-dissipative JC, fixed g/κ drive sweeps")
    println(io)
    println(io, "Generated $(finished_at); started $(started_at).")
    println(io)
    println(io, "## Model")
    println(io)
    println(io, "Rotating-frame JC Hamiltonian `H = -Δ(a†a + σ₊σ₋) + g(a†σ₋ + aσ₊) - E(a + a†)`")
    println(io, "with cavity loss only: master equation `ρ̇ = -i[H,ρ] + κ𝒟[a]ρ`.")
    println(io, "Counted jump: cavity emission `κ a ρ a†` (`mJ = [√κ a]`, `ν = [1]`).")
    println(io, "Units: `κ = $(config.κ)`. Carmichael conversion: `κ_C = κ/2`.")
    println(io)
    println(io, "## Scan")
    println(io)
    println(io, "- `g/κ = $(config.g)` (fixed); drive sweep `x = 2E/g ∈ [$(minimum(config.x_values)), $(maximum(config.x_values))]`, $(length(config.x_values)) points per cut")
    println(io, "- detuning cuts `Δ/κ ∈ $(cuts)`")
    println(io, "- cumulants: `nC = $(config.nC)` (c₁, c₂, c₃ of the cavity-emission counting statistics)")
    println(io)
    println(io, "## Dynamic cutoff")
    println(io)
    pad_sigma_str = config.pad_sigma isa Function ?
        join(["Δ=$(Δ)→$(jc_resolve_cut_param(config.pad_sigma, Δ))" for Δ in cuts], ", ") :
        string(config.pad_sigma)
    println(io, "Per-point Fock cutoff from the semiclassical bright-branch estimate")
    println(io, "(`jc_semiclassical_bright_n`), tiers `$(config.tiers)`, occupation cap")
    println(io, "`occ_max = $(config.occ_max)`, headroom `pad_sigma·√n + $(config.pad_abs)` with")
    println(io, "pad_sigma = $(pad_sigma_str).")
    println(io, "Cutoffs used per cut:")
    println(io)
    for Δ_val in cuts
        cut = sort([r for r in rows if r.Δ == Δ_val], by=r -> r.x)
        used = join(["x∈[$(round(first(seg).x; digits=3)),$(round(last(seg).x; digits=3))]→N=$(first(seg).N)"
                     for seg in [cut[a:b] for (a, b) in jc_equal_run_ranges([r.N for r in cut])]], ", ")
        println(io, "- `Δ/κ = $(Δ_val)`: $(used)")
    end
    println(io)
    println(io, "## Solvers")
    println(io)
    println(io, "Steady state (per point, continuation within each cutoff segment):")
    println(io, "trace-constrained GMRES, warm-started, shifted crout ILU `τ = $(config.ilu_tau)`,")
    println(io, "shift factor `$(config.shift_factor)`, adaptive rebuild at `> $(config.rebuild_niter)`")
    println(io, "iterations (itmax $(config.itmax), fallback $(config.fallback_itmax)), Krylov memory")
    println(io, "$(config.gmres_memory), rtol $(config.gmres_rtol), atol $(config.gmres_atol).")
    println(io)
    println(io, "FCS (QuantumFCS.jl): `method = :iterative`, steady-state ILU injected as `Pl`")
    println(io, "(right-preconditioned, true-residual stopping), rtol $(config.fcs_rtol),")
    println(io, "itmax $(config.fcs_itmax), memory $(config.fcs_memory). Acceptance gate:")
    println(io, "`|c₁/(κ⟨n⟩) − 1| ≤ $(config.current_check_tol)`, finite c₂/c₃, no solver")
    println(io, "warnings; one retry with the internal preconditioner otherwise.")
    println(io)
    println(io, "## Run quality")
    println(io)
    retries = count(r -> r.fcs_retry, rows)
    worst_cc = maximum(abs(r.current_check - 1) for r in rows)
    worst_tail = maximum(r.cavity_tail for r in rows)
    unsafe = count(r -> r.cutoff_status !== :safe, rows)
    println(io, "- $(length(rows)) points; $(retries) FCS retries; $(unsafe) non-safe cutoff states")
    println(io, "- worst `|current_check − 1|` = $(round(worst_cc; sigdigits=3)); worst tail `p_{N-1}+p_N` = $(round(worst_tail; sigdigits=3))")
    if !isempty(validation_rows)
        println(io, "- lower-cutoff spot validation (relative drift, production vs one tier down):")
        for v in validation_rows
            println(io, "  - $(v.label): N=$(v.N_production) vs $(v.N_validation): " *
                "|Δc1|/c1 = $(round(v.rel_c1; sigdigits=3)), |Δc2|/c2 = $(round(v.rel_c2; sigdigits=3)), " *
                "|Δc3|/|c3| = $(round(v.rel_c3; sigdigits=3))")
        end
    end
    println(io)
    println(io, "## Provenance")
    println(io)
    for name in sort(collect(keys(dependency_info)))
        println(io, "- $(name): $(dependency_info[name])")
    end
    println(io, "- Julia $(VERSION), $(Threads.nthreads()) threads, host $(Base.Libc.gethostname())")
    println(io)
    if !isempty(files)
        println(io, "## Files")
        println(io)
        for f in files
            println(io, "- `$(f)`")
        end
    end
    return String(take!(io))
end

# Index ranges (a:b as tuples) of equal consecutive values — used to describe
# cutoff segments compactly.
function jc_equal_run_ranges(values)
    ranges = Tuple{Int,Int}[]
    isempty(values) && return ranges
    a = 1
    for i in 2:length(values)
        if values[i] != values[a]
            push!(ranges, (a, i - 1))
            a = i
        end
    end
    push!(ranges, (a, length(values)))
    return ranges
end
