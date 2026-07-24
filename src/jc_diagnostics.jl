# Diagnostics for the driven-dissipative Jaynes-Cummings model.
#
# NOT part of the production FCS drive-sweep pipeline (see jc_model.jl and the
# driven_dissipative_jaynes_cummings notebook). This module
# collects the reliability studies used while developing the numerics:
#   * the steady-state continuation scan over (g/κ, Δ/κ) with adaptive ILU reuse
#     (run_jc_steady_continuation_scan and friends),
#   * the lightweight cutoff scouts (run_jc_steady_scout, jc_scout_convergence),
#   * the ComplexF32 mixed-precision rescout that cross-checks the Fock cutoff at
#     large N (jc_f32_scout_steadystate_point, run_jc_f32_cutoff_rescout).
#
# It reuses the production module's operators, trace-constrained solve, warm
# GMRES, and quality metrics, so include that first (this file pulls it in).

# Idempotent: pull in the production pipeline only if it is not already loaded,
# so this file works standalone and alongside an explicit jc_model.jl include.
@isdefined(jc_operators_for_cutoff) || include("jc_model.jl")

using JLD2   # bright-branch tail audit persists p_n distributions
import Dates # timestamp for the audit metadata


function jc_shifted_diagonal_factor(A;
        shift_factor=JC_DEFAULT_SS_ILU_SHIFT_FACTOR)
    scale = norm(A, 1) / size(A, 1)
    shift = shift_factor * max(real(scale), eps(Float64))
    diagonal = Vector(diag(A))
    diagonal .+= shift
    floor = eps(Float64) * max(norm(A, 1), one(Float64))
    diagonal = [
        abs(value) > floor ? value : value + (iszero(value) ? floor : floor * value / abs(value))
        for value in diagonal
    ]
    return Diagonal(1 ./ diagonal)
end

function jc_steady_preconditioner_factor(A;
        preconditioner=:shifted_ilu,
        ilu_tau=JC_DEFAULT_SS_ILU_TAU,
        shift_factor=JC_DEFAULT_SS_ILU_SHIFT_FACTOR)
    preconditioner === :shifted_ilu && return jc_shifted_ilu_factor(A;
        ilu_tau=ilu_tau,
        shift_factor=shift_factor)
    preconditioner === :diagonal && return jc_shifted_diagonal_factor(A;
        shift_factor=shift_factor)
    preconditioner === :none && return nothing
    error("Unknown steady-state preconditioner $(preconditioner).")
end

function jc_steadystate(H, c_ops;
        ilu_tau=JC_DEFAULT_SS_ILU_TAU,
        shift_factor=JC_DEFAULT_SS_ILU_SHIFT_FACTOR,
        preconditioner=:shifted_ilu,
        ss_alg=jc_default_ss_alg(),
        solver_kwargs=NamedTuple())
    solver = SteadyStateLinearSolver(
        alg=ss_alg,
        Pl=A -> jc_steady_preconditioner_factor(A;
            preconditioner=preconditioner,
            ilu_tau=ilu_tau,
            shift_factor=shift_factor,
        ))
    return steadystate(H, c_ops; solver=solver, solver_kwargs...)
end

jc_skip_resonant_scout_point(g, E, Δ;
    resonant_unsafe_x_min=JC_DEFAULT_RESONANT_UNSAFE_X_MIN) =
    iszero(Δ) && 2E / g >= resonant_unsafe_x_min

function jc_skipped_steady_state_diagnostics(; N_cutoff, g, E, Δ,
        label, trunc_tol, skip_reason)
    return (
        label=label,
        N=N_cutoff,
        g=g,
        E=E,
        Δ=Δ,
        x=2E / g,
        n=NaN,
        occupation_fraction=NaN,
        trunc_tol=trunc_tol,
        cutoff_status=:unsafe,
        cavity_boundary=NaN,
        cavity_tail=NaN,
        cavity_tail5=NaN,
        skipped=true,
        skip_reason=skip_reason,
    )
end

function jc_scout_jobs(points, cutoffs)
    return [(point=point, N_cutoff=N_cutoff) for point in points for N_cutoff in cutoffs]
end

function jc_steady_state_diagnostics(; N_cutoff, g, E, Δ=0.0,
        κ=1.0,
        label="",
        trunc_tol=JC_DEFAULT_TRUNC_TOL,
        ilu_tau=JC_DEFAULT_SCOUT_SS_ILU_TAU,
        shift_factor=JC_DEFAULT_SS_ILU_SHIFT_FACTOR,
        preconditioner=:shifted_ilu,
        ss_alg=jc_default_ss_alg(),
        solver_kwargs=NamedTuple(),
        skip_resonant_unsafe=true,
        resonant_unsafe_x_min=JC_DEFAULT_RESONANT_UNSAFE_X_MIN)
    if skip_resonant_unsafe &&
       jc_skip_resonant_scout_point(g, E, Δ; resonant_unsafe_x_min)
        return jc_skipped_steady_state_diagnostics(
            N_cutoff=N_cutoff,
            g=g,
            E=E,
            Δ=Δ,
            label=label,
            trunc_tol=trunc_tol,
            skip_reason=:resonant_high_drive,
        )
    end

    ops = jc_operators_for_cutoff(N_cutoff)
    H = -Δ * ops.detuning_op + g * ops.Hjc + E * ops.Hdr
    cavity_loss = sqrt(κ) * ops.a
    ρss = jc_steadystate(H, [cavity_loss];
        ilu_tau=ilu_tau,
        shift_factor=shift_factor,
        preconditioner=preconditioner,
        ss_alg=ss_alg,
        solver_kwargs=solver_kwargs,
    )
    n = real(QuantumToolbox.expect(ops.n_op, ρss))
    tail = jc_tail_diagnostics(jc_cavity_population(ρss))
    status = jc_cutoff_status_from_tails(tail.cavity_tail, tail.cavity_tail5;
        trunc_tol=trunc_tol)
    return merge((
        label=label,
        N=N_cutoff,
        g=g,
        E=E,
        Δ=Δ,
        x=2E / g,
        n=n,
        occupation_fraction=n / N_cutoff,
        trunc_tol=trunc_tol,
        cutoff_status=status,
        skipped=false,
        skip_reason=nothing,
    ), tail)
end

function run_jc_steady_scout(points, cutoffs;
        κ=1.0,
        trunc_tol=JC_DEFAULT_TRUNC_TOL,
        ilu_tau=JC_DEFAULT_SCOUT_SS_ILU_TAU,
        shift_factor=JC_DEFAULT_SS_ILU_SHIFT_FACTOR,
        preconditioner=:shifted_ilu,
        ss_alg=jc_default_ss_alg(),
        solver_kwargs=NamedTuple(),
        skip_resonant_unsafe=true,
        resonant_unsafe_x_min=JC_DEFAULT_RESONANT_UNSAFE_X_MIN,
        threaded=false)
    jobs = jc_scout_jobs(points, cutoffs)
    results = Vector{Any}(undef, length(jobs))

    evaluate_job(i) = begin
        job = jobs[i]
        point = job.point
        jc_steady_state_diagnostics(
            N_cutoff=job.N_cutoff,
            g=point.g,
            E=point.E,
            Δ=point.Δ,
            κ=κ,
            label=point.label,
            trunc_tol=trunc_tol,
            ilu_tau=ilu_tau,
            shift_factor=shift_factor,
            preconditioner=preconditioner,
            ss_alg=ss_alg,
            solver_kwargs=solver_kwargs,
            skip_resonant_unsafe=skip_resonant_unsafe,
            resonant_unsafe_x_min=resonant_unsafe_x_min,
        )
    end

    if threaded && Threads.nthreads() > 1
        Threads.@threads for i in eachindex(jobs)
            results[i] = evaluate_job(i)
        end
    else
        for i in eachindex(jobs)
            results[i] = evaluate_job(i)
        end
    end

    return results
end

function jc_latest_cutoff_rows(rows)
    labels = unique([r.label for r in rows])
    return [sort(filter(r -> r.label == label, rows), by=r -> r.N)[end] for label in labels]
end

function jc_scout_convergence(rows; rel_n_tol=JC_DEFAULT_SCOUT_REL_N_TOL)
    labels = unique([r.label for r in rows])
    summaries = Vector{Any}()
    for label in labels
        cut = sort(filter(r -> r.label == label, rows), by=r -> r.N)
        reference = cut[end]
        previous = length(cut) >= 2 ? cut[end - 1] : cut[end]
        rel_n_change = if isfinite(reference.n) && isfinite(previous.n)
            abs(reference.n - previous.n) / max(abs(reference.n), eps(Float64))
        else
            NaN
        end
        status = reference.cutoff_status
        if isfinite(rel_n_change) && rel_n_change > rel_n_tol && status === :safe
            status = :ambiguous
        elseif isfinite(rel_n_change) && rel_n_change > rel_n_tol && status === :ambiguous
            status = :unsafe
        end
        push!(summaries, merge(reference, (
            rel_n_change=rel_n_change,
            previous_N=previous.N,
            scout_status=status,
        )))
    end
    return summaries
end

function jc_timed_steadystate_point(; N_cutoff, g, E, Δ=0.0,
        κ=1.0,
        label="",
        trunc_tol=JC_DEFAULT_TRUNC_TOL,
        ss_ilu_tau=JC_DEFAULT_SS_ILU_TAU,
        ss_shift_factor=JC_DEFAULT_SS_ILU_SHIFT_FACTOR,
        ss_preconditioner=:shifted_ilu,
        ss_alg=jc_default_ss_alg(),
        ss_solver_kwargs=NamedTuple())
    total_t0 = time_ns()

    operator_t0 = time_ns()
    ops = jc_operators_for_cutoff(N_cutoff)
    H = -Δ * ops.detuning_op + g * ops.Hjc + E * ops.Hdr
    cavity_loss = sqrt(κ) * ops.a
    c_ops = [cavity_loss]
    operator_seconds = jc_seconds_since(operator_t0)

    liouvillian_t0 = time_ns()
    L = QuantumToolbox.liouvillian(H, c_ops)
    system = jc_trace_constrained_linear_problem(L)
    liouvillian_seconds = jc_seconds_since(liouvillian_t0)

    preconditioner_t0 = time_ns()
    Pl = jc_steady_preconditioner_factor(system.A;
        preconditioner=ss_preconditioner,
        ilu_tau=ss_ilu_tau,
        shift_factor=ss_shift_factor,
    )
    preconditioner_seconds = jc_seconds_since(preconditioner_t0)

    gmres_t0 = time_ns()
    problem = LinearProblem(system.A, system.b)
    solution = Pl === nothing ?
        solve(problem, ss_alg; ss_solver_kwargs...) :
        solve(problem, ss_alg; Pl=Pl, ss_solver_kwargs...)
    gmres_seconds = jc_seconds_since(gmres_t0)

    postprocess_t0 = time_ns()
    ρss, ρ_mat = jc_quantumobject_from_steady_vector(
        solution.u,
        system.N_hilbert,
        system.dimensions,
    )
    quality = jc_steadystate_quality_metrics(
        ρ_mat,
        ρss,
        system.L_data,
        ops.n_op,
        N_cutoff,
        trunc_tol,
    )
    postprocess_seconds = jc_seconds_since(postprocess_t0)
    retcode = jc_retcode_text(solution)

    return merge((
        label=label,
        N=N_cutoff,
        g=g,
        E=E,
        Δ=Δ,
        x=2E / g,
        trunc_tol=trunc_tol,
        ss_ilu_tau=ss_ilu_tau,
        ss_shift_factor=ss_shift_factor,
        ss_preconditioner=ss_preconditioner,
        ss_solver_kwargs=ss_solver_kwargs,
        ss_retcode=retcode,
        ss_converged=isempty(retcode) ? missing : occursin("Success", retcode),
        ss_iterations=jc_solution_niters(solution),
        operator_seconds=operator_seconds,
        liouvillian_seconds=liouvillian_seconds,
        preconditioner_seconds=preconditioner_seconds,
        ilu_seconds=preconditioner_seconds,
        gmres_seconds=gmres_seconds,
        postprocess_seconds=postprocess_seconds,
        total_seconds=jc_seconds_since(total_t0),
    ), quality)
end

function jc_metadata_string(metadata)
    return "N=$(metadata.cutoff), E_ref=$(metadata.E_reference), g_ref=$(metadata.g_reference), " *
        "2E/g=$(round(metadata.two_E_over_g; digits=4)), " *
        "Δ/κ=$(metadata.detuning_cuts), scales=$(round(metadata.scale_min; digits=3))-" *
        "$(round(metadata.scale_max; digits=3)) ($(metadata.scale_count) pts), " *
        "tail tol=$(metadata.trunc_tol)"
end

# --- Continuation steady-state sweep with adaptive ILU reuse (2026-07 benchmark) ---
#
# Benchmarks (Notes/steady_state_solver_benchmark_results.md) showed that the
# shifted crout ILU at τ=0.1 acts as a near-complete factorization for this
# model: GMRES converges in <20 iterations everywhere, and the factorization
# cost dominates and grows steeply towards the bright/coexistence region.
# Reusing the previous point's ILU with a warm-started GMRES amortises that
# cost along a smoothly ordered scan; the preconditioner is rebuilt only when
# the reused solve needs more than `rebuild_niter` iterations.

"""
Run a steady-state sweep over `points` (named tuples with `g`, `E`, `Δ`, and
optionally `label`) reusing the shifted-ILU preconditioner and warm-starting
GMRES from the previous solution. `points` must be ordered smoothly (for
example one detuning cut at a time, ascending in scale); use
`jc_unique_points` and sort beforehand. Returns one diagnostics row per point.
"""
function run_jc_steady_continuation_sweep(points; N_cutoff, κ=1.0,
        trunc_tol=JC_DEFAULT_TRUNC_TOL,
        ilu_tau=JC_DEFAULT_SCOUT_SS_ILU_TAU,
        shift_factor=JC_DEFAULT_SS_ILU_SHIFT_FACTOR,
        rebuild_niter=JC_DEFAULT_SS_REUSE_REBUILD_NITER,
        itmax=JC_DEFAULT_SS_REUSE_ITMAX,
        fallback_itmax=JC_DEFAULT_SS_REUSE_FALLBACK_ITMAX,
        gmres_memory=JC_DEFAULT_SS_GMRES_MEMORY,
        gmres_rtol=JC_DEFAULT_SS_GMRES_RTOL,
        gmres_atol=JC_DEFAULT_SS_GMRES_ATOL,
        return_states=false)
    rows = Vector{Any}()
    states = Vector{Any}()
    ops = jc_operators_for_cutoff(N_cutoff)
    Pl = nothing
    u_prev = nothing
    for point in points
        total_t0 = time_ns()
        H = -point.Δ * ops.detuning_op + point.g * ops.Hjc + point.E * ops.Hdr
        L = QuantumToolbox.liouvillian(H, [sqrt(κ) * ops.a])
        system = jc_trace_constrained_linear_problem(L)
        liouvillian_seconds = jc_seconds_since(total_t0)

        preconditioner_seconds = 0.0
        rebuilds = 0
        if Pl === nothing
            GC.gc()
            t0 = time_ns()
            Pl = jc_shifted_ilu_factor(system.A; ilu_tau=ilu_tau, shift_factor=shift_factor)
            preconditioner_seconds += jc_seconds_since(t0)
            rebuilds += 1
        end

        u, niters, solved, gmres_seconds = jc_warm_gmres_steady_solve(
            system.A, system.b, Pl;
            u0=u_prev, gmres_memory, gmres_rtol, gmres_atol, itmax)
        if !solved || niters > rebuild_niter
            Pl = nothing
            GC.gc()
            t0 = time_ns()
            Pl = jc_shifted_ilu_factor(system.A; ilu_tau=ilu_tau, shift_factor=shift_factor)
            preconditioner_seconds += jc_seconds_since(t0)
            rebuilds += 1
            u, niters_retry, solved, gmres_retry_seconds = jc_warm_gmres_steady_solve(
                system.A, system.b, Pl;
                u0=u, gmres_memory, gmres_rtol, gmres_atol, itmax=fallback_itmax)
            niters += niters_retry
            gmres_seconds += gmres_retry_seconds
        end

        ρss, ρ_mat = jc_quantumobject_from_steady_vector(u, system.N_hilbert, system.dimensions)
        quality = jc_steadystate_quality_metrics(ρ_mat, ρss, system.L_data,
            ops.n_op, N_cutoff, trunc_tol)
        push!(rows, merge(point, (
            label=get(point, :label, ""),
            N=N_cutoff,
            g=point.g,
            E=point.E,
            Δ=point.Δ,
            x=2point.E / point.g,
            trunc_tol=trunc_tol,
            ss_ilu_tau=ilu_tau,
            ss_shift_factor=shift_factor,
            ss_rebuild_niter=rebuild_niter,
            ss_converged=solved,
            ss_iterations=niters,
            ss_rebuilds=rebuilds,
            liouvillian_seconds=liouvillian_seconds,
            preconditioner_seconds=preconditioner_seconds,
            ilu_seconds=preconditioner_seconds,
            gmres_seconds=gmres_seconds,
            total_seconds=jc_seconds_since(total_t0),
        ), quality))
        return_states && push!(states, ρss)
        u_prev = u
    end
    return return_states ? (rows=rows, states=states) : rows
end

function jc_continuation_cut_points(Δ_val, scale_values;
        E_reference=5.0,
        g_reference=14.0)
    return [
        (
            label="det Δ=$(Δ_val), scale=$(round(scale; digits=4))",
            probe_group="fixed ratio detuned",
            scale=scale,
            g=scale * g_reference,
            E=scale * E_reference,
            Δ=Δ_val,
            x=2E_reference / g_reference,
        )
        for scale in sort(collect(scale_values))
    ]
end

"""
Production steady-state scan: one continuation sweep per detuning cut, each
ordered by ascending scale so the ILU is built on the cheap dim side and
reused towards the bright side. `N_cutoff` may be an `Integer` or a function
`Δ -> Integer` (e.g. to run dim cuts at a smaller cutoff). With
`threaded=true` cuts run on separate threads — each active chain holds its own
ILU (~3 GB at N=300), so use at most 2 threads on a 16 GB machine and only
when nothing else heavy is resident.
"""
function run_jc_steady_continuation_scan(;
        detuning_cuts=(0.0, 0.55, 0.70),
        scale_values=collect(range(0.1, 2.0; length=40)),
        E_reference=5.0,
        g_reference=14.0,
        N_cutoff=300,
        κ=1.0,
        trunc_tol=JC_DEFAULT_TRUNC_TOL,
        ilu_tau=JC_DEFAULT_SCOUT_SS_ILU_TAU,
        shift_factor=JC_DEFAULT_SS_ILU_SHIFT_FACTOR,
        rebuild_niter=JC_DEFAULT_SS_REUSE_REBUILD_NITER,
        itmax=JC_DEFAULT_SS_REUSE_ITMAX,
        fallback_itmax=JC_DEFAULT_SS_REUSE_FALLBACK_ITMAX,
        gmres_memory=JC_DEFAULT_SS_GMRES_MEMORY,
        gmres_rtol=JC_DEFAULT_SS_GMRES_RTOL,
        gmres_atol=JC_DEFAULT_SS_GMRES_ATOL,
        threaded=false)
    cuts = collect(detuning_cuts)
    cut_rows = Vector{Any}(undef, length(cuts))
    cutoff_for_cut(Δ_val) = N_cutoff isa Integer ? N_cutoff : N_cutoff(Δ_val)

    run_cut(i) = begin
        Δ_val = cuts[i]
        points = jc_continuation_cut_points(Δ_val, scale_values;
            E_reference=E_reference, g_reference=g_reference)
        run_jc_steady_continuation_sweep(points;
            N_cutoff=cutoff_for_cut(Δ_val),
            κ=κ,
            trunc_tol=trunc_tol,
            ilu_tau=ilu_tau,
            shift_factor=shift_factor,
            rebuild_niter=rebuild_niter,
            itmax=itmax,
            fallback_itmax=fallback_itmax,
            gmres_memory=gmres_memory,
            gmres_rtol=gmres_rtol,
            gmres_atol=gmres_atol,
        )
    end

    if threaded && Threads.nthreads() > 1
        Threads.@threads for i in eachindex(cuts)
            cut_rows[i] = run_cut(i)
        end
    else
        for i in eachindex(cuts)
            cut_rows[i] = run_cut(i)
        end
    end
    return reduce(vcat, cut_rows)
end

"""
Flag scan points whose cutoff safety should be re-checked at a larger cutoff.
Tail diagnostics alone provably miss truncation error in the coexistence
region (see Notes/steady_state_solver_benchmark_results.md), so this flags,
per detuning cut: non-`safe` points, high-occupation points, and points at
sharp photon-number steps (the coexistence cliff), including one neighbour on
each side of a step.
"""
function jc_flag_continuation_suspects(rows;
        occupation_threshold=0.35,
        rel_n_step_tol=0.30)
    suspects = Vector{Any}()
    for Δ_val in unique([r.Δ for r in rows])
        cut = sort(filter(r -> r.Δ == Δ_val, rows), by=r -> r.g)
        flagged = falses(length(cut))
        for (i, r) in enumerate(cut)
            r.cutoff_status === :safe || (flagged[i] = true)
            r.occupation_fraction >= occupation_threshold && (flagged[i] = true)
            if i > 1
                n_prev, n_here = cut[i - 1].n, r.n
                step = abs(n_here - n_prev) / max(abs(n_prev), abs(n_here), eps(Float64))
                if step > rel_n_step_tol && max(abs(n_prev), abs(n_here)) > 1.0
                    flagged[max(i - 1, 1)] = true
                    flagged[i] = true
                end
            end
        end
        append!(suspects, cut[flagged])
    end
    return suspects
end

"""
Re-scout flagged points at a larger cutoff with the memory-lean F32 solver and
compare the photon number against the production rows. A point keeps `safe`
only if its relative n-change is below `rel_n_tol` (default matches
`JC_DEFAULT_SCOUT_REL_N_TOL`) and its own tail diagnostics pass.
"""
function run_jc_f32_cutoff_rescout(suspect_rows;
        N_cutoff=400,
        κ=1.0,
        trunc_tol=JC_DEFAULT_TRUNC_TOL,
        ilu_tau=JC_DEFAULT_SCOUT_SS_ILU_TAU,
        shift_factor=JC_DEFAULT_SS_ILU_SHIFT_FACTOR,
        rel_n_tol=JC_DEFAULT_SCOUT_REL_N_TOL)
    return [
        begin
            scout = jc_f32_scout_steadystate_point(
                N_cutoff=N_cutoff,
                g=row.g,
                E=row.E,
                Δ=row.Δ,
                κ=κ,
                label=row.label,
                trunc_tol=trunc_tol,
                ilu_tau=ilu_tau,
                shift_factor=shift_factor,
            )
            rel_n_change = abs(scout.n - row.n) / max(abs(row.n), eps(Float64))
            scout_status = if rel_n_change > rel_n_tol
                :unsafe
            elseif scout.cutoff_status === :safe
                :safe
            else
                scout.cutoff_status
            end
            merge(scout, (
                scale=get(row, :scale, NaN),
                reference_N=row.N,
                reference_n=row.n,
                rel_n_change=rel_n_change,
                scout_status=scout_status,
            ))
        end
        for row in suspect_rows
    ]
end

"""
Fold the rescout verdicts back into the production rows: any point whose
rescout came back non-`safe` is demoted (its `cutoff_status` is replaced by
the rescout status). Returns updated rows.
"""
function jc_apply_rescout_status(rows, rescout_rows)
    verdicts = Dict((r.g, r.E, r.Δ) => r.scout_status for r in rescout_rows)
    return [
        begin
            verdict = get(verdicts, (row.g, row.E, row.Δ), nothing)
            (verdict === nothing || verdict === :safe) ? row :
                merge(row, (cutoff_status=verdict,))
        end
        for row in rows
    ]
end

# --- Mixed-precision (ComplexF32) ILU scout for large cutoffs ---
#
# The F32 ILU builds roughly twice as fast and holds half the memory of the
# F64 one. Photon numbers agree with the F64 reference to 1e-5–5e-4 relative,
# but the F32 factor weakens with N (GMRES can stall near rel. residual 1e-12
# and trace errors reach ~1e-2 at N=400 bright points). That fails production
# tolerances but is far below the 5e-2 scout tolerance, so it is the
# memory-viable way to check cutoff convergence in n at N≈400 on a 16 GB
# machine. Right preconditioning keeps the GMRES stopping criterion on the
# true residual.

struct JCMixedPrecisionILU{TF,TV}
    factorization::TF
    buffer::TV
end

function LinearAlgebra.ldiv!(y::AbstractVector{ComplexF64}, P::JCMixedPrecisionILU,
        x::AbstractVector{ComplexF64})
    P.buffer .= ComplexF32.(x)
    ldiv!(P.factorization, P.buffer)
    y .= ComplexF64.(P.buffer)
    return y
end

function LinearAlgebra.ldiv!(P::JCMixedPrecisionILU, x::AbstractVector{ComplexF64})
    P.buffer .= ComplexF32.(x)
    ldiv!(P.factorization, P.buffer)
    x .= ComplexF64.(P.buffer)
    return x
end

function jc_f32_shifted_ilu_preconditioner(A;
        ilu_tau=JC_DEFAULT_SCOUT_SS_ILU_TAU,
        shift_factor=JC_DEFAULT_SS_ILU_SHIFT_FACTOR)
    scale = norm(A, 1) / size(A, 1)
    shift = shift_factor * max(real(scale), eps(Float64))
    A32 = SparseMatrixCSC{ComplexF32,Int}(A + shift * sparse(I, size(A)...))
    F32 = QuantumToolbox.ilu(A32; τ=Float32(ilu_tau))
    return JCMixedPrecisionILU(F32, zeros(ComplexF32, size(A, 1)))
end

"""
Memory-lean steady-state scout for large cutoffs (for example N=400): F32 ILU
with right-preconditioned GMRES. `n` is accurate to roughly 1e-5–5e-4
relative — sufficient for cutoff scouting, NOT for production FCS points
(trace errors can reach 1e-2 at N=400 bright points even when `n` is good).
"""
function jc_f32_scout_steadystate_point(; N_cutoff, g, E, Δ=0.0, κ=1.0,
        label="",
        trunc_tol=JC_DEFAULT_TRUNC_TOL,
        ilu_tau=JC_DEFAULT_SCOUT_SS_ILU_TAU,
        shift_factor=JC_DEFAULT_SS_ILU_SHIFT_FACTOR,
        gmres_memory=JC_DEFAULT_SS_GMRES_MEMORY,
        gmres_rtol=1e-11,
        itmax=JC_DEFAULT_SS_REUSE_FALLBACK_ITMAX,
        return_population=false)
    total_t0 = time_ns()
    ops = jc_operators_for_cutoff(N_cutoff)
    H = -Δ * ops.detuning_op + g * ops.Hjc + E * ops.Hdr
    L = QuantumToolbox.liouvillian(H, [sqrt(κ) * ops.a])
    system = jc_trace_constrained_linear_problem(L)

    GC.gc()
    t0 = time_ns()
    Pl = jc_f32_shifted_ilu_preconditioner(system.A;
        ilu_tau=ilu_tau, shift_factor=shift_factor)
    preconditioner_seconds = jc_seconds_since(t0)

    t0 = time_ns()
    u, stats = Krylov.gmres(system.A, system.b; N=Pl, ldiv=true,
        memory=gmres_memory, rtol=gmres_rtol, atol=1e-16, itmax=itmax)
    gmres_seconds = jc_seconds_since(t0)

    ρss, ρ_mat = jc_quantumobject_from_steady_vector(u, system.N_hilbert, system.dimensions)
    quality = jc_steadystate_quality_metrics(ρ_mat, ρss, system.L_data,
        ops.n_op, N_cutoff, trunc_tol)
    base = merge((
        label=label,
        N=N_cutoff,
        g=g,
        E=E,
        Δ=Δ,
        x=2E / g,
        trunc_tol=trunc_tol,
        ss_ilu_tau=ilu_tau,
        ss_precision=:f32_scout,
        ss_converged=stats.solved,
        ss_iterations=stats.niter,
        preconditioner_seconds=preconditioner_seconds,
        ilu_seconds=preconditioner_seconds,
        gmres_seconds=gmres_seconds,
        total_seconds=jc_seconds_since(total_t0),
    ), quality)
    # The full cavity marginal is needed by the bright-branch tail audit; it is
    # opt-in because the rescout pipeline only wants the scalar diagnostics.
    return return_population ? merge(base, (cavity_population=jc_cavity_population(ρss),)) : base
end

# --- Production FCS: fixed g/κ drive sweeps with dynamic cutoff (2026-07) ---
#
# Design (Notes/steady_state_solver_benchmark_results.md and
# Notes/jc_n400_scan_reliability.md): sweeping x = 2E/g at fixed g/κ crosses the
# blockade-breakdown transition at bounded photon number, so the cutoff can be
# chosen per point from the semiclassical (Maxwell-Bloch) bright-branch estimate
# and stays ≤ 400. The steady state is solved with the continuation solver and
# its shifted ILU is handed to QuantumFCS as the Drazin preconditioner (`Pl`),
# so each parameter point performs at most one factorization.

# --- Bright-branch cumulative-tail audit (2026-07) ---
#
# The production tail guardrail (`jc_tail_diagnostics`) sums only the top-2
# (`p_{N-1}+p_N`) and top-5 boundary Fock populations. On the deep bright branch
# (occupation n/N → 0.6-0.7 at N=500) that number is ~1e-13 while the state fills
# most of the ladder, so a *broad* cumulative tail — probability or, more
# importantly, second-/third-moment weight spread over the upper Fock range — is
# invisible to it. Fano (`c₂/c₁`) and skewness (`c₃/c₂^{3/2}`) weight that tail,
# so a non-negligible tail moment-fraction is the necessary condition for a
# truncation-biased cumulant. These functions quantify it and cross-check the
# distribution against larger cutoffs (F32 scout), to decide whether the
# high-drive Fano rise / negative-skew shoulder are physical or truncation
# artefacts. They are diagnostics only — the production pipeline is untouched.

"""
    jc_cumulative_tail_profile(p_n; frac_cuts, abs_cuts, bump_frac, bump_floor)

Cumulative-tail and shape diagnostics for a cavity photon-number distribution
`p_n` (index `i` ↔ Fock `n = i-1`; e.g. from [`jc_cavity_population`](@ref)).
Beyond the production boundary tail it reports, for each Fock threshold `m`
(given as fractions of the cutoff and as absolute cuts), the tail mass
`Σ_{n≥m} p_n` **and the fraction of `⟨n⟩`, `⟨n²⟩`, and `Σ|n−⟨n⟩|³ p_n` carried
above `m`** — the last two being what a truncated Fano and skewness are sensitive
to. Also returns the σ-headroom `(N−⟨n⟩)/σ` and `(N−⟨n⟩)/√⟨n⟩`, the
photon-number skewness `μ₃/σ³` (distinct from the FCS *counting* skewness), and a
shape read-out (peak location, number of interior local maxima, and whether a
secondary bump sits against the boundary — a bright peak pressed on the Fock
wall).
"""
function jc_cumulative_tail_profile(p_n::AbstractVector{<:Real};
        frac_cuts=(0.5, 0.6, 0.7, 0.8, 0.9),
        abs_cuts=(300, 400, 500),
        bump_frac=0.85,
        bump_floor=1e-8)
    L = length(p_n)
    L >= 2 || error("distribution too short")
    Ncut = L - 1                        # top Fock index == the Fock cutoff N
    total = sum(p_n)
    total > 0 || error("empty distribution")
    w = isapprox(total, 1.0; atol=1e-12) ? p_n : p_n ./ total

    mean_n = 0.0
    @inbounds for i in 1:L
        mean_n += (i - 1) * w[i]
    end
    var_n = 0.0; m3 = 0.0; raw2 = 0.0; absm3 = 0.0
    @inbounds for i in 1:L
        n = i - 1; d = n - mean_n
        var_n += d * d * w[i]
        m3    += d * d * d * w[i]
        raw2  += float(n) * n * w[i]
        absm3 += abs(d)^3 * w[i]
    end
    sigma = sqrt(max(var_n, 0.0))
    skew_pn = sigma > 0 ? m3 / sigma^3 : 0.0
    headroom_sigma = sigma > 0 ? (Ncut - mean_n) / sigma : Inf
    headroom_sqrtn = mean_n > 0 ? (Ncut - mean_n) / sqrt(mean_n) : Inf

    # Tail contributions above an absolute Fock threshold m (probability plus the
    # fractions of ⟨n⟩, ⟨n²⟩, and Σ|n−mean|³ that live at n ≥ m).
    function tail_above(m)
        mass = 0.0; meanc = 0.0; sqc = 0.0; absm3c = 0.0
        @inbounds for i in 1:L
            n = i - 1
            n >= m || continue
            d = n - mean_n
            mass   += w[i]
            meanc  += n * w[i]
            sqc    += float(n) * n * w[i]
            absm3c += abs(d)^3 * w[i]
        end
        return (;
            mass,
            mean_frac  = mean_n > 0 ? meanc / mean_n : 0.0,
            sq_frac    = raw2 > 0 ? sqc / raw2 : 0.0,
            absm3_frac = absm3 > 0 ? absm3c / absm3 : 0.0,
        )
    end

    frac_thresholds = [clamp(round(Int, f * Ncut), 0, Ncut) for f in frac_cuts]
    frac_profiles = [merge((; frac=frac_cuts[k], m=frac_thresholds[k]),
                           tail_above(frac_thresholds[k])) for k in eachindex(frac_cuts)]
    abs_profiles = [merge((; m=Int(m)), tail_above(Int(m))) for m in abs_cuts if m <= Ncut]

    # Shape: peak, interior local maxima, boundary bump.
    peak_index = argmax(w) - 1
    n_local_maxima = 0
    boundary_bump = false
    @inbounds for i in 2:(L - 1)
        if w[i] > w[i - 1] && w[i] >= w[i + 1] && w[i] > bump_floor
            n_local_maxima += 1
            (i - 1) >= bump_frac * Ncut && (boundary_bump = true)
        end
    end

    return (;
        N_cutoff = Ncut,
        total_mass = total,
        mean_n, sigma, skew_pn, m3_signed = m3,
        headroom_sigma, headroom_sqrtn,
        peak_index, n_local_maxima, boundary_bump, unimodal = n_local_maxima <= 1,
        # convenience scalars for quick gating / CSV
        boundary_mass = w[end] + w[end - 1],
        mass_above_half = tail_above(round(Int, 0.5 * Ncut)).mass,
        sqfrac_above_60 = tail_above(round(Int, 0.6 * Ncut)).sq_frac,
        absm3frac_above_80 = tail_above(round(Int, 0.8 * Ncut)).absm3_frac,
        frac_profiles, abs_profiles,
    )
end

"""
    run_jc_bright_tail_audit(specs; g, N=500, higher_Ns=(600,700), save_path, ...)

Minimal-first cumulative-tail audit of suspect bright-branch points. `specs` is a
vector of `(Δ=, x=)` NamedTuples. For each point it (1) re-solves the F64 steady
state at cutoff `N` (no FCS) with [`run_jc_steady_continuation_sweep`], takes the
cavity marginal, and profiles it with [`jc_cumulative_tail_profile`]; then (2)
cross-checks the distribution at each larger cutoff in `higher_Ns` with the
memory-lean F32 scout ([`jc_f32_scout_steadystate_point`], state-only), reporting
the relative `n` drift and, crucially, the probability mass that appears *above*
the production cutoff `N` on the larger grid (the direct "is there a long tail
beyond N?" test). Optionally saves all `p_n` arrays + profiles to `save_path`
(JLD2). Returns one NamedTuple per point.
"""
function run_jc_bright_tail_audit(specs;
        g,
        κ=1.0,
        N=500,
        higher_Ns=(600, 700),
        trunc_tol=JC_DEFAULT_TRUNC_TOL,
        save_path=nothing,
        verbose=true)
    results = Vector{Any}()
    populations = Dict{String,Vector{Float64}}()
    for spec in specs
        Δ = spec.Δ; x = spec.x
        label = "Δ=$(Δ), x=$(x)"
        E = x * g / 2
        pt = (label=label, probe_group="bright tail audit", Δ=Δ, x=x, g=g, E=E)

        verbose && println(">>> [$label] F64 steady state at N=$N ...")
        swp = run_jc_steady_continuation_sweep([pt]; N_cutoff=N, κ=κ,
            trunc_tol=trunc_tol, return_states=true)
        row = swp.rows[1]
        p_n = jc_cavity_population(swp.states[1])
        profile = jc_cumulative_tail_profile(p_n)
        populations["$(label)__N$(N)"] = collect(p_n)
        verbose && println("    n=$(round(row.n; sigdigits=6)) occ=$(round(row.occupation_fraction; digits=3)) " *
                           "top2=$(round(row.cavity_tail; sigdigits=3)) hσ=$(round(profile.headroom_sigma; digits=2)) " *
                           "mass≥0.6N=$(round(jc_cumulative_tail_profile(p_n; frac_cuts=(0.6,)).frac_profiles[1].mass; sigdigits=3))")

        higher = Vector{Any}()
        for Nh in higher_Ns
            verbose && println(">>> [$label] F32 scout at N=$Nh ...")
            GC.gc()
            sc = jc_f32_scout_steadystate_point(N_cutoff=Nh, g=g, E=E, Δ=Δ, κ=κ,
                label=label, trunc_tol=trunc_tol, return_population=true)
            p_h = sc.cavity_population
            populations["$(label)__N$(Nh)"] = collect(p_h)
            # Probability mass at Fock n > N (production cutoff) on the larger grid:
            # Fock n uses index i = n+1, so n > N ⇒ i ≥ N+2.
            mass_beyond_prodN = length(p_h) >= N + 2 ? sum(@view p_h[(N + 2):end]) : 0.0
            push!(higher, (;
                N = Nh,
                n = sc.n,
                rel_n_vs_prod = abs(sc.n - row.n) / max(abs(row.n), eps(Float64)),
                mass_beyond_prodN,
                trace_error = sc.trace_error,
                ss_converged = sc.ss_converged,
                profile = jc_cumulative_tail_profile(p_h),
                p_n = collect(p_h),
            ))
            verbose && println("    n=$(round(sc.n; sigdigits=6)) rel_n=$(round(abs(sc.n-row.n)/max(abs(row.n),eps());sigdigits=3)) " *
                               "mass(n>$(N))=$(round(mass_beyond_prodN; sigdigits=3)) trace_err=$(round(sc.trace_error; sigdigits=2))")
        end

        push!(results, (;
            label, Δ, x, g, E, N,
            n = row.n,
            occupation_fraction = row.occupation_fraction,
            cavity_boundary = row.cavity_boundary,
            cavity_tail = row.cavity_tail,
            cavity_tail5 = row.cavity_tail5,
            ss_converged = row.ss_converged,
            profile,
            p_n = collect(p_n),
            higher,
        ))
        GC.gc()
    end

    if save_path !== nothing
        JLD2.jldsave(save_path;
            results = results,
            populations = populations,
            meta = (; g, κ, N, higher_Ns=collect(higher_Ns),
                     generated = string(Dates.now())))
        verbose && println("saved audit to $(save_path)")
    end
    return results
end
