# Circuit-QED quantum heat engine based on photon-assisted Cooper-pair
# tunnelling (paper Sec. 5): two bosonic modes (hot/cold) coupled through a
# Josephson junction, with Laguerre matrix elements encoding the nonlinear
# k:l photon-exchange process (paper uses lh=1, lc=2).
#
# Ported from the research repository for the companion notebooks. The
# LU-vs-iterative runtime study and the exploratory zero-bias / linearised
# helpers are intentionally omitted: they are not used by any manuscript figure.
using CSV
using DataFrames
using Dates
using HypergeometricFunctions
using IncompleteLU
using JLD2
using Krylov
using LinearAlgebra
using Logging
using QuantumFCS
using QuantumToolbox
using SparseArrays
using Statistics

@isdefined(NONLINEAR_QHE_CURRENT_TOL) || const NONLINEAR_QHE_CURRENT_TOL = 1e-8
@isdefined(NONLINEAR_QHE_TRUNC_TOL) || const NONLINEAR_QHE_TRUNC_TOL = 1e-3
@isdefined(NONLINEAR_QHE_EPSILON_TOL) || const NONLINEAR_QHE_EPSILON_TOL = 0.05
@isdefined(NONLINEAR_QHE_OCCUPATION_TOL) || const NONLINEAR_QHE_OCCUPATION_TOL = 0.50
@isdefined(NONLINEAR_QHE_WIDE_TAIL_FACTOR) || const NONLINEAR_QHE_WIDE_TAIL_FACTOR = 10.0
@isdefined(NONLINEAR_QHE_AMBIGUOUS_TAIL_FACTOR) || const NONLINEAR_QHE_AMBIGUOUS_TAIL_FACTOR = 10.0
@isdefined(NONLINEAR_QHE_AMBIGUOUS_WIDE_TAIL_FACTOR) || const NONLINEAR_QHE_AMBIGUOUS_WIDE_TAIL_FACTOR = 100.0
@isdefined(NONLINEAR_QHE_SS_ILU_TAU) || const NONLINEAR_QHE_SS_ILU_TAU = 1e-3
@isdefined(NONLINEAR_QHE_SS_ILU_SHIFT_FACTOR) || const NONLINEAR_QHE_SS_ILU_SHIFT_FACTOR = 1e-6
@isdefined(NONLINEAR_QHE_SS_GMRES_RTOL) || const NONLINEAR_QHE_SS_GMRES_RTOL = 1e-10
@isdefined(NONLINEAR_QHE_SS_GMRES_ATOL) || const NONLINEAR_QHE_SS_GMRES_ATOL = 1e-14
@isdefined(NONLINEAR_QHE_SS_GMRES_MEMORY) || const NONLINEAR_QHE_SS_GMRES_MEMORY = 60
@isdefined(NONLINEAR_QHE_SS_ITMAX) || const NONLINEAR_QHE_SS_ITMAX = 200
@isdefined(NONLINEAR_QHE_FCS_RTOL) || const NONLINEAR_QHE_FCS_RTOL = 1e-8
@isdefined(NONLINEAR_QHE_FCS_ITMAX) || const NONLINEAR_QHE_FCS_ITMAX = 300
@isdefined(NONLINEAR_QHE_FCS_MEMORY) || const NONLINEAR_QHE_FCS_MEMORY = 60
@isdefined(NONLINEAR_QHE_FCS_TAU) || const NONLINEAR_QHE_FCS_TAU = 0.05
@isdefined(NONLINEAR_QHE_FCS_CHECK_TOL) || const NONLINEAR_QHE_FCS_CHECK_TOL = 1e-4

if !@isdefined(NonlinearQHEParameters)
    Base.@kwdef struct NonlinearQHEParameters
        Nmax_h::Int = 7
        Nmax_c::Int = 7
        lh::Int = 1
        lc::Int = 2
        λh::Float64 = 0.47
        λc::Float64 = 0.89
        Ωc::Float64 = 1000.0
        Ωratio::Float64 = π
        κh::Float64 = 2.0
        κc::Float64 = 0.5
        nh::Float64 = 0.5
        nc::Float64 = 0.01
        g::Float64 = 7.76
    end
end

if !@isdefined(NonlinearQHENumerics)
    Base.@kwdef struct NonlinearQHENumerics
        solver::Symbol = :iterative
        trunc_tol::Float64 = NONLINEAR_QHE_TRUNC_TOL
        epsilon_tol::Float64 = NONLINEAR_QHE_EPSILON_TOL
        current_tol::Float64 = NONLINEAR_QHE_CURRENT_TOL
        occupation_tol::Float64 = NONLINEAR_QHE_OCCUPATION_TOL
        ss_ilu_tau::Float64 = NONLINEAR_QHE_SS_ILU_TAU
        ss_shift_factor::Float64 = NONLINEAR_QHE_SS_ILU_SHIFT_FACTOR
        ss_gmres_rtol::Float64 = NONLINEAR_QHE_SS_GMRES_RTOL
        ss_gmres_atol::Float64 = NONLINEAR_QHE_SS_GMRES_ATOL
        ss_gmres_memory::Int = NONLINEAR_QHE_SS_GMRES_MEMORY
        ss_itmax::Int = NONLINEAR_QHE_SS_ITMAX
        fcs_tau::Float64 = NONLINEAR_QHE_FCS_TAU
        fcs_rtol::Float64 = NONLINEAR_QHE_FCS_RTOL
        fcs_itmax::Int = NONLINEAR_QHE_FCS_ITMAX
        fcs_memory::Int = NONLINEAR_QHE_FCS_MEMORY
        fcs_check_tol::Float64 = NONLINEAR_QHE_FCS_CHECK_TOL
    end
end

function nonlinear_qhe_parameters(; kwargs...)
    kw = Dict{Symbol,Any}(kwargs)
    for (alias, canonical) in (
            :l_h => :lh,
            :l_c => :lc,
            :lambda_h => :λh,
            :lambda_c => :λc,
            :Omega_c => :Ωc,
            :Omega_ratio => :Ωratio,
            :kappa_h => :κh,
            :kappa_c => :κc,
            :n_h => :nh,
            :n_c => :nc)
        haskey(kw, alias) && (kw[canonical] = pop!(kw, alias))
    end
    parameter_keys = fieldnames(NonlinearQHEParameters)
    unknown = setdiff(collect(keys(kw)), parameter_keys)
    isempty(unknown) || error("Unknown nonlinear QHE parameter keyword(s): $(unknown)")
    return NonlinearQHEParameters(;
        Nmax_h=get(kw, :Nmax_h, 7),
        Nmax_c=get(kw, :Nmax_c, 7),
        lh=get(kw, :lh, 1),
        lc=get(kw, :lc, 2),
        λh=get(kw, :λh, 0.47),
        λc=get(kw, :λc, 0.89),
        Ωc=get(kw, :Ωc, 1000.0),
        Ωratio=get(kw, :Ωratio, π),
        κh=get(kw, :κh, 2.0),
        κc=get(kw, :κc, 0.5),
        nh=get(kw, :nh, 0.5),
        nc=get(kw, :nc, 0.01),
        g=get(kw, :g, 7.76))
end

function nonlinear_qhe_numerics(; solver=:iterative,
        trunc_tol=NONLINEAR_QHE_TRUNC_TOL,
        epsilon_tol=NONLINEAR_QHE_EPSILON_TOL,
        current_tol=NONLINEAR_QHE_CURRENT_TOL,
        occupation_tol=NONLINEAR_QHE_OCCUPATION_TOL,
        ss_ilu_tau=NONLINEAR_QHE_SS_ILU_TAU,
        ss_shift_factor=NONLINEAR_QHE_SS_ILU_SHIFT_FACTOR,
        ss_gmres_rtol=NONLINEAR_QHE_SS_GMRES_RTOL,
        ss_gmres_atol=NONLINEAR_QHE_SS_GMRES_ATOL,
        ss_gmres_memory=NONLINEAR_QHE_SS_GMRES_MEMORY,
        ss_itmax=NONLINEAR_QHE_SS_ITMAX,
        fcs_tau=NONLINEAR_QHE_FCS_TAU,
        fcs_rtol=NONLINEAR_QHE_FCS_RTOL,
        fcs_itmax=NONLINEAR_QHE_FCS_ITMAX,
        fcs_memory=NONLINEAR_QHE_FCS_MEMORY,
        fcs_check_tol=NONLINEAR_QHE_FCS_CHECK_TOL)
    return NonlinearQHENumerics(; solver, trunc_tol, epsilon_tol, current_tol,
        occupation_tol, ss_ilu_tau, ss_shift_factor, ss_gmres_rtol,
        ss_gmres_atol, ss_gmres_memory, ss_itmax, fcs_tau, fcs_rtol,
        fcs_itmax, fcs_memory, fcs_check_tol)
end

function split_nonlinear_qhe_kwargs(kwargs)
    kw = Dict{Symbol,Any}(kwargs)
    numerics_keys = (:solver, :trunc_tol, :epsilon_tol, :current_tol,
        :occupation_tol, :ss_ilu_tau, :ss_shift_factor, :ss_gmres_rtol,
        :ss_gmres_atol, :ss_gmres_memory, :ss_itmax, :fcs_tau, :fcs_rtol,
        :fcs_itmax, :fcs_memory, :fcs_check_tol)
    numerics_kw = Dict{Symbol,Any}()
    for key in numerics_keys
        haskey(kw, key) && (numerics_kw[key] = pop!(kw, key))
    end
    return nonlinear_qhe_parameters(; kw...), nonlinear_qhe_numerics(; numerics_kw...)
end

if @isdefined(_nonlinear_qhe_workspace_cache)
    empty!(_nonlinear_qhe_workspace_cache)
else
    _nonlinear_qhe_workspace_cache = Dict{Tuple{Int,Int},NamedTuple}()
end

function qhe_workspace(Nmax_h::Int, Nmax_c::Int)
    key = (Nmax_h, Nmax_c)
    return get!(_nonlinear_qhe_workspace_cache, key) do
        Nh = Nmax_h + 1
        Nc = Nmax_c + 1
        basis_h = 0:Nmax_h
        basis_c = 0:Nmax_c
        Id_h = QuantumToolbox.qeye(Nh)
        Id_c = QuantumToolbox.qeye(Nc)
        ah = QuantumToolbox.tensor(QuantumToolbox.destroy(Nh), Id_c)
        ac = QuantumToolbox.tensor(Id_h, QuantumToolbox.destroy(Nc))
        adh = ah'
        adc = ac'
        (; basis_h, basis_c, dims=(Nh, Nc), dimensions=ah.dimensions, ah, ac, adh, adc)
    end
end

laguerre_poly(n, α, x) = binomial(n + α, n) * pFq((-n,), (α + 1,), x)
laguerre(n, α, x) = laguerre_poly(n, α, x)

function laguerre_prefactor(n::Int, k::Int, λ)
    return Float64((2λ)^k * exp(-2λ^2) * factorial(big(n)) /
        factorial(big(n+k)) * laguerre_poly(n, k, 4λ^2))
end

qhe_basis_dim(basis::Integer) = basis
qhe_basis_dim(basis) = length(basis)
qhe_basis_dim(basis::QuantumToolbox.QuantumObject) = size(basis.data, 1)

function laguerre_operator(λ, k::Int, basis)
    N = qhe_basis_dim(basis)
    diag_elements = ComplexF64[laguerre_prefactor(n, k, λ) for n in 0:(N - 1)]
    return QuantumToolbox.QuantumObject(spdiagm(0 => diag_elements);
        type=QuantumToolbox.Operator,
        dims=(N,))
end
# The effective Josephson energy for the RWA Hamiltonian, given the coupling g and the λ parameter;
# in general g = (2λh)^lh * (2λc)^lc * Ej/2, so Ej = 2g / ((2λh)^lh * (2λc)^lc). For the common case of lh=1, lc=2, this reduces to g / (4 * λh * λc^2).
effective_Ej(g, λh, λc) = g / (4 * λh * λc^2)

function liouv_RWA(params::NonlinearQHEParameters)
    ws = qhe_workspace(params.Nmax_h, params.Nmax_c)
    Ej = effective_Ej(params.g, params.λh, params.λc)
    AhAc = QuantumToolbox.tensor(laguerre_operator(params.λh, params.lh, ws.basis_h),
        laguerre_operator(params.λc, params.lc, ws.basis_c)
    )
    H = -Ej/2 * (1im^(params.lh + params.lc) *
        (ws.adc)^params.lc * AhAc * ws.ah^params.lh)
    H += H'
    J = [
        sqrt((params.nh + 1) * params.κh) * ws.ah,
        sqrt((params.nc + 1) * params.κc) * ws.ac,
        sqrt(params.nh * params.κh) * ws.adh,
        sqrt(params.nc * params.κc) * ws.adc,
    ]
    return H, J, ws, Ej
end

function liouv_RWA(; kwargs...)
    params = nonlinear_qhe_parameters(; kwargs...)
    return liouv_RWA(params)
end

function off_resonant_epsilon(ρss, Ej, ws, params::NonlinearQHEParameters)
    δ = abs((params.Ωratio - 3) * params.Ωc)
    δ == 0 && return Inf
    AhAc = QuantumToolbox.tensor(laguerre_operator(params.λh, 0, ws.basis_h),
        laguerre_operator(params.λc, 1, ws.basis_c)
    )
    F = -Ej/2 * (1im * ws.ac * AhAc)
    return sqrt(max(real(QuantumToolbox.expect(F * F' + F' * F, ρss)), 0.0)) / δ
end

"""
    rwa_coherence(ρss, params, ws=qhe_workspace(params.Nmax_h, params.Nmax_c))

Return the l1 weight of steady-state coherences on the resonant RWA links
`|n_h,n_c> <-> |n_h-lh,n_c+lc>`. This is a diagnostic for whether the same
coherent Josephson matrix elements that define the heat-engine cycle are
active in the steady state.
"""
function rwa_coherence(ρss, params::NonlinearQHEParameters,
        ws=qhe_workspace(params.Nmax_h, params.Nmax_c))
    coherence = 0.0
    for nh_level in params.lh:params.Nmax_h
        for nc_level in 0:(params.Nmax_c - params.lc)
            src = QuantumToolbox.tensor(QuantumToolbox.basis(params.Nmax_h + 1, nh_level),
                QuantumToolbox.basis(params.Nmax_c + 1, nc_level))
            dst = QuantumToolbox.tensor(
                QuantumToolbox.basis(params.Nmax_h + 1, nh_level - params.lh),
                QuantumToolbox.basis(params.Nmax_c + 1, nc_level + params.lc))
            coherence += abs(dst' * ρss * src)
        end
    end
    return coherence
end

function rwa_coherence(; kwargs...)
    params, numerics = split_nonlinear_qhe_kwargs(kwargs)
    H, J, ws, _ = liouv_RWA(params)
    ρss = if numerics.solver == :eigenvector
        QuantumToolbox.steadystate(H, J; solver=QuantumToolbox.SteadyStateEigenSolver())
    elseif numerics.solver == :quantumtoolbox_linear
        QuantumToolbox.steadystate(H, J; solver=QuantumToolbox.SteadyStateLinearSolver())
    else
        qhe_steadystate(H, J, ws, params, numerics).ρss
    end
    return rwa_coherence(ρss, params, ws)
end

qhe_seconds_since(t0::UInt64) = (time_ns() - t0) / 1e9

function qhe_tail_diagnostics(population)
    tail2_start = max(1, length(population) - 1)
    tail5_start = max(1, length(population) - 4)
    return (
        boundary = population[end],
        tail = sum(population[tail2_start:end]),
        tail5 = sum(population[tail5_start:end]),
    )
end

function qhe_cutoff_status_from_tails(tail2, tail5;
        trunc_tol=NONLINEAR_QHE_TRUNC_TOL,
        wide_factor=NONLINEAR_QHE_WIDE_TAIL_FACTOR,
        ambiguous_tail_factor=NONLINEAR_QHE_AMBIGUOUS_TAIL_FACTOR,
        ambiguous_wide_tail_factor=NONLINEAR_QHE_AMBIGUOUS_WIDE_TAIL_FACTOR)
    if tail2 <= trunc_tol && tail5 <= wide_factor * trunc_tol
        return :safe
    elseif tail2 <= ambiguous_tail_factor * trunc_tol &&
           tail5 <= ambiguous_wide_tail_factor * trunc_tol
        return :ambiguous
    else
        return :unsafe
    end
end

function qhe_marginal_diagnostics(ρss, params::NonlinearQHEParameters)
    hot_population = max.(real.(diag(QuantumToolbox.ptrace(ρss, 1).data)), 0.0)
    cold_population = max.(real.(diag(QuantumToolbox.ptrace(ρss, 2).data)), 0.0)
    hot_population ./= sum(hot_population)
    cold_population ./= sum(cold_population)
    hot_tail = qhe_tail_diagnostics(hot_population)
    cold_tail = qhe_tail_diagnostics(cold_population)
    hot_n = sum((0:params.Nmax_h) .* hot_population)
    cold_n = sum((0:params.Nmax_c) .* cold_population)
    hot_status = qhe_cutoff_status_from_tails(hot_tail.tail, hot_tail.tail5;
        trunc_tol=NONLINEAR_QHE_TRUNC_TOL)
    cold_status = qhe_cutoff_status_from_tails(cold_tail.tail, cold_tail.tail5;
        trunc_tol=NONLINEAR_QHE_TRUNC_TOL)
    return (
        hot_n=hot_n,
        cold_n=cold_n,
        hot_occupation_fraction=hot_n / max(params.Nmax_h, 1),
        cold_occupation_fraction=cold_n / max(params.Nmax_c, 1),
        hot_boundary=hot_tail.boundary,
        cold_boundary=cold_tail.boundary,
        hot_tail=hot_tail.tail,
        cold_tail=cold_tail.tail,
        hot_tail5=hot_tail.tail5,
        cold_tail5=cold_tail.tail5,
        hot_cutoff_status=hot_status,
        cold_cutoff_status=cold_status,
    )
end

function qhe_trace_constrained_linear_problem(L, N_hilbert::Integer)
    # Build the trace-constrained system with QuantumFCS. The package reproduces the
    # previous local numerics exactly (weight = norm(L,1)/length(L); trace row on the
    # vectorized-identity diagonal), so A and b are identical. We keep the local
    # NamedTuple interface (A/b/L_data/N_hilbert) and additionally carry the package
    # `TraceConstrainedSystem` so the iterative solve can reuse it without rebuilding A.
    system = trace_constrained_system(L)
    return (
        A=system.A,
        b=system.b,
        L_data=system.L,
        N_hilbert=N_hilbert,
        pkg_system=system,
    )
end

function qhe_shifted_ilu_factor(A;
        ilu_tau=NONLINEAR_QHE_SS_ILU_TAU,
        shift_factor=NONLINEAR_QHE_SS_ILU_SHIFT_FACTOR)
    scale = norm(A, 1) / size(A, 1)
    shift = shift_factor * max(real(scale), eps(Float64))
    return ilu(A + shift * sparse(I, size(A, 1), size(A, 2)); τ=ilu_tau)
end

function qhe_warm_gmres_steady_solve(A, b, Pl;
        u0=nothing,
        gmres_memory=NONLINEAR_QHE_SS_GMRES_MEMORY,
        gmres_rtol=NONLINEAR_QHE_SS_GMRES_RTOL,
        gmres_atol=NONLINEAR_QHE_SS_GMRES_ATOL,
        itmax=NONLINEAR_QHE_SS_ITMAX)
    t0 = time_ns()
    x, stats = u0 === nothing ?
        Krylov.gmres(A, b; M=Pl, ldiv=true, memory=gmres_memory,
            rtol=gmres_rtol, atol=gmres_atol, itmax=itmax) :
        Krylov.gmres(A, b, u0; M=Pl, ldiv=true, memory=gmres_memory,
            rtol=gmres_rtol, atol=gmres_atol, itmax=itmax)
    return x, stats.niter, stats.solved, qhe_seconds_since(t0)
end

function qhe_sparse_operator_from_vector(ρ_vec, ws, N_hilbert::Integer)
    ρ_mat = reshape(ρ_vec, N_hilbert, N_hilbert)
    ρ_mat = (ρ_mat + ρ_mat') / 2
    ρ_mat ./= tr(ρ_mat)
    return QuantumToolbox.QuantumObject(sparse(ρ_mat);
        type=QuantumToolbox.Operator,
        dims=ws.dimensions), sparse(ρ_mat)
end

function qhe_steadystate_quality_metrics(ρ_mat, ρss, L_data,
        params::NonlinearQHEParameters, numerics::NonlinearQHENumerics)
    ρ_vec = vec(Matrix(ρ_mat))
    residual = norm(L_data * ρ_vec)
    residual_denominator = max(norm(L_data, 1) * norm(ρ_vec), eps(Float64))
    hermiticity_denominator = max(norm(Matrix(ρ_mat)), eps(Float64))
    marginal = qhe_marginal_diagnostics(ρss, params)
    occupation_ok = marginal.hot_occupation_fraction <= numerics.occupation_tol &&
        marginal.cold_occupation_fraction <= numerics.occupation_tol
    trunc_ok = marginal.hot_tail <= numerics.trunc_tol &&
        marginal.cold_tail <= numerics.trunc_tol
    cutoff_ok = trunc_ok && occupation_ok
    return merge((
        trace_error=abs(tr(ρ_mat) - 1),
        liouvillian_residual=residual,
        relative_liouvillian_residual=residual / residual_denominator,
        hermiticity_error=norm(Matrix(ρ_mat) - Matrix(ρ_mat)') / hermiticity_denominator,
        occupation_tol=numerics.occupation_tol,
        trunc_tol=numerics.trunc_tol,
        occupation_ok=occupation_ok,
        trunc_ok=trunc_ok,
        cutoff_ok=cutoff_ok,
    ), marginal)
end

function qhe_steadystate(H, J, ws, params::NonlinearQHEParameters,
        numerics::NonlinearQHENumerics)
    L = QuantumToolbox.liouvillian(H, J)
    system = qhe_trace_constrained_linear_problem(L, size(H.data, 1))
    if numerics.solver == :eigenvector || numerics.solver == :quantumtoolbox_linear
        t0 = time_ns()
        ρss = numerics.solver == :eigenvector ?
            QuantumToolbox.steadystate(H, J; solver=QuantumToolbox.SteadyStateEigenSolver()) :
            QuantumToolbox.steadystate(H, J; solver=QuantumToolbox.SteadyStateLinearSolver())
        ρ_mat = sparse(ρss.data ./ tr(ρss.data))
        quality = qhe_steadystate_quality_metrics(ρ_mat, ρss, system.L_data,
            params, numerics)
        return merge((
            ρss=ρss,
            ρ_mat=ρ_mat,
            L=L,
            system=system,
            Pl=nothing,
            ss_backend=numerics.solver,
            ss_converged=true,
            ss_iterations=missing,
            ss_rebuilds=0,
            ilu_seconds=0.0,
            gmres_seconds=0.0,
            liouvillian_seconds=0.0,
            steady_state_seconds=qhe_seconds_since(t0),
        ), quality)
    end

    # Iterative steady state via QuantumFCS: builds the shifted ILU, runs GMRES on the
    # trace-constrained system, and returns rho_ss together with the reusable
    # preconditioner Pl. Settings map one-to-one onto the previous local chain
    # (ss_ilu_tau→τ, ss_shift_factor, ss_gmres_rtol/atol→rtol/atol, ss_itmax→itmax,
    # ss_gmres_memory→memory), so the numerics are unchanged. The returned
    # TraceConstrainedSteadyState is stored so the FCS context can reuse Pl through the
    # package bridge (see qhe_prepare_fcs_context_for_backend).
    fcs_ss = trace_constrained_steadystate(system.pkg_system;
        method=:iterative,
        τ=numerics.ss_ilu_tau,
        shift_factor=numerics.ss_shift_factor,
        rtol=numerics.ss_gmres_rtol,
        atol=numerics.ss_gmres_atol,
        itmax=numerics.ss_itmax,
        memory=numerics.ss_gmres_memory)
    ρ_mat = fcs_ss.rho_ss
    ρss = QuantumToolbox.QuantumObject(ρ_mat;
        type=QuantumToolbox.Operator, dims=ws.dimensions)
    quality = qhe_steadystate_quality_metrics(ρ_mat, ρss, system.L_data,
        params, numerics)
    return merge((
        ρss=ρss,
        ρ_mat=ρ_mat,
        L=L,
        system=system,
        Pl=fcs_ss.Pl,
        fcs_steadystate=fcs_ss,
        ss_backend=:iterative,
        ss_converged=fcs_ss.stats.converged,
        ss_iterations=fcs_ss.stats.iterations,
        ss_rebuilds=1,
        ss_ilu_tau=numerics.ss_ilu_tau,
        ss_shift_factor=numerics.ss_shift_factor,
        ilu_seconds=fcs_ss.stats.ilu_seconds,
        gmres_seconds=fcs_ss.stats.gmres_seconds,
        liouvillian_seconds=0.0,
        steady_state_seconds=fcs_ss.stats.ilu_seconds + fcs_ss.stats.gmres_seconds,
    ), quality)
end

function qhe_lu_factor_nnz(F)
    try
        return nnz(F.L) + nnz(F.U)
    catch
        return missing
    end
end

function qhe_direct_lu_steadystate(H, J, ws, params::NonlinearQHEParameters,
        numerics::NonlinearQHENumerics)
    liouvillian_timed = @timed QuantumToolbox.liouvillian(H, J)
    L = liouvillian_timed.value
    system_timed = @timed qhe_trace_constrained_linear_problem(L, size(H.data, 1))
    system = system_timed.value
    # Qualify `lu`: both LinearAlgebra and QuantumToolbox export it, so a bare call is
    # ambiguous (and errors) when the file is loaded into a scope that uses both.
    factor_timed = @timed LinearAlgebra.lu(system.A)
    F = factor_timed.value
    solve_timed = @timed F \ system.b
    ρss, ρ_mat = qhe_sparse_operator_from_vector(solve_timed.value, ws,
        system.N_hilbert)
    quality = qhe_steadystate_quality_metrics(ρ_mat, ρss, system.L_data,
        params, numerics)
    return merge((
        ρss=ρss,
        ρ_mat=ρ_mat,
        L=L,
        system=system,
        Pl=nothing,
        ss_backend=:lu,
        ss_converged=true,
        ss_iterations=0,
        ss_rebuilds=0,
        ss_ilu_tau=NaN,
        ss_shift_factor=NaN,
        lu_factor_seconds=factor_timed.time,
        lu_solve_seconds=solve_timed.time,
        lu_factor_allocated_bytes=factor_timed.bytes,
        lu_solve_allocated_bytes=solve_timed.bytes,
        lu_factor_nnz=qhe_lu_factor_nnz(F),
        ilu_seconds=0.0,
        gmres_seconds=0.0,
        liouvillian_seconds=liouvillian_timed.time,
        system_seconds=system_timed.time,
        steady_state_seconds=liouvillian_timed.time + system_timed.time +
            factor_timed.time + solve_timed.time,
    ), quality)
end

struct QHEWarnCountingLogger <: Logging.AbstractLogger
    hits::Ref{Int}
    parent::Logging.AbstractLogger
end

Logging.min_enabled_level(logger::QHEWarnCountingLogger) =
    Logging.min_enabled_level(logger.parent)
Logging.shouldlog(logger::QHEWarnCountingLogger, level, _module, group, id) =
    Logging.shouldlog(logger.parent, level, _module, group, id)
Logging.catch_exceptions(logger::QHEWarnCountingLogger) =
    Logging.catch_exceptions(logger.parent)
function Logging.handle_message(logger::QHEWarnCountingLogger, level, message,
        _module, group, id, file, line; kwargs...)
    level >= Logging.Warn && (logger.hits[] += 1)
    return Logging.handle_message(logger.parent, level, message,
        _module, group, id, file, line; kwargs...)
end

function qhe_count_warnings(f)
    hits = Ref(0)
    parent = Logging.current_logger()
    result = Logging.with_logger(QHEWarnCountingLogger(hits, parent)) do
        f()
    end
    return result, hits[]
end

function qhe_jump_current(ρss, jumps, weights)
    return sum(real(weights[i] * QuantumToolbox.expect(jumps[i]' * jumps[i], ρss))
        for i in eachindex(jumps))
end

function qhe_fcs_cumulants(L_data, ρss, jumps, weights, Pl,
        numerics::NonlinearQHENumerics; nC=2)
    fcs_problem = LindbladFCS(
        L=L_data,
        mJ=jumps,
        rho_ss=ρss,
        nu=weights,
        nC=nC,
        method=:iterative,
        Pl=Pl,
        τ=numerics.fcs_tau,
        rtol=numerics.fcs_rtol,
        itmax=numerics.fcs_itmax,
        memory=numerics.fcs_memory,
    )
    cumulants, warnings = qhe_count_warnings() do
        fcscumulants_recursive(fcs_problem)
    end
    retry = false
    if warnings > 0 || !all(isfinite, cumulants)
        retry = true
        retry_problem = LindbladFCS(
            L=L_data,
            mJ=jumps,
            rho_ss=ρss,
            nu=weights,
            nC=nC,
            method=:iterative,
            τ=numerics.fcs_tau,
            rtol=numerics.fcs_rtol,
            itmax=numerics.fcs_itmax,
            memory=numerics.fcs_memory,
        )
        cumulants, warnings = qhe_count_warnings() do
            fcscumulants_recursive(retry_problem)
        end
    end
    return real.(cumulants), warnings, retry
end

function qhe_runtime_backend_numerics(numerics::NonlinearQHENumerics, backend::Symbol)
    backend === :lu && return nonlinear_qhe_numerics(;
        solver=:lu,
        trunc_tol=numerics.trunc_tol,
        epsilon_tol=numerics.epsilon_tol,
        current_tol=numerics.current_tol,
        occupation_tol=numerics.occupation_tol,
        ss_ilu_tau=numerics.ss_ilu_tau,
        ss_shift_factor=numerics.ss_shift_factor,
        ss_gmres_rtol=numerics.ss_gmres_rtol,
        ss_gmres_atol=numerics.ss_gmres_atol,
        ss_gmres_memory=numerics.ss_gmres_memory,
        ss_itmax=numerics.ss_itmax,
        fcs_tau=numerics.fcs_tau,
        fcs_rtol=numerics.fcs_rtol,
        fcs_itmax=numerics.fcs_itmax,
        fcs_memory=numerics.fcs_memory,
        fcs_check_tol=numerics.fcs_check_tol)
    backend === :iterative && return nonlinear_qhe_numerics(;
        solver=:iterative,
        trunc_tol=numerics.trunc_tol,
        epsilon_tol=numerics.epsilon_tol,
        current_tol=numerics.current_tol,
        occupation_tol=numerics.occupation_tol,
        ss_ilu_tau=numerics.ss_ilu_tau,
        ss_shift_factor=numerics.ss_shift_factor,
        ss_gmres_rtol=numerics.ss_gmres_rtol,
        ss_gmres_atol=numerics.ss_gmres_atol,
        ss_gmres_memory=numerics.ss_gmres_memory,
        ss_itmax=numerics.ss_itmax,
        fcs_tau=numerics.fcs_tau,
        fcs_rtol=numerics.fcs_rtol,
        fcs_itmax=numerics.fcs_itmax,
        fcs_memory=numerics.fcs_memory,
        fcs_check_tol=numerics.fcs_check_tol)
    throw(ArgumentError("Unknown QHE runtime backend :$(backend); expected :lu or :iterative."))
end

function qhe_prepare_fcs_context_for_backend(steady, backend::Symbol,
        numerics::NonlinearQHENumerics)
    if backend === :lu
        return prepare_fcs_context(;
            L=steady.system.L_data,
            rho_ss=steady.ρss,
            method=:lu,
        )
    elseif backend === :iterative
        # Reuse the steady-state solve's preconditioner through the QuantumFCS bridge,
        # which forwards L, rho_ss, and Pl straight from the TraceConstrainedSteadyState
        # (no second ILU build). Fall back to the explicit form if the steady state was
        # produced without the package solver (e.g. an eigenvector/direct steady state).
        if hasproperty(steady, :fcs_steadystate)
            return prepare_fcs_context(steady.fcs_steadystate;
                method=:iterative,
                τ=numerics.fcs_tau,
                rtol=numerics.fcs_rtol,
                itmax=numerics.fcs_itmax,
                memory=numerics.fcs_memory,
            )
        end
        return prepare_fcs_context(;
            L=steady.system.L_data,
            rho_ss=steady.ρss,
            method=:iterative,
            Pl=steady.Pl,
            τ=numerics.fcs_tau,
            rtol=numerics.fcs_rtol,
            itmax=numerics.fcs_itmax,
            memory=numerics.fcs_memory,
        )
    end
    throw(ArgumentError("Unknown QHE runtime backend :$(backend); expected :lu or :iterative."))
end

function qhe_timed_warning_count(f)
    timed = @timed qhe_count_warnings(f)
    value, warnings = timed.value
    return value, warnings, timed.time, timed.bytes
end

function qhe_runtime_fcs_cumulants(steady, J, params::NonlinearQHEParameters,
        numerics::NonlinearQHENumerics, backend::Symbol; nC=2)
    Ωh = params.Ωratio * params.Ωc
    hot_jumps = [J[1], J[3]]
    cold_jumps = [J[2], J[4]]
    hot_weights = [-Ωh, Ωh]
    cold_weights = [-params.Ωc, params.Ωc]

    ctx, prepare_warnings, prepare_seconds, prepare_bytes =
        qhe_timed_warning_count() do
            qhe_prepare_fcs_context_for_backend(steady, backend, numerics)
        end
    hot_cumulants, hot_warnings, hot_seconds, hot_bytes =
        qhe_timed_warning_count() do
            fcscumulants_recursive(ctx; mJ=hot_jumps, nu=hot_weights, nC=nC)
        end
    cold_cumulants, cold_warnings, cold_seconds, cold_bytes =
        qhe_timed_warning_count() do
            fcscumulants_recursive(ctx; mJ=cold_jumps, nu=cold_weights, nC=nC)
        end

    return (
        hot_cumulants=real.(hot_cumulants),
        cold_cumulants=real.(cold_cumulants),
        hot_jumps=hot_jumps,
        cold_jumps=cold_jumps,
        hot_weights=hot_weights,
        cold_weights=cold_weights,
        fcs_prepare_warnings=prepare_warnings,
        hot_fcs_warnings=hot_warnings,
        cold_fcs_warnings=cold_warnings,
        fcs_prepare_seconds=prepare_seconds,
        fcs_hot_seconds=hot_seconds,
        fcs_cold_seconds=cold_seconds,
        fcs_seconds=prepare_seconds + hot_seconds + cold_seconds,
        fcs_prepare_allocated_bytes=prepare_bytes,
        fcs_hot_allocated_bytes=hot_bytes,
        fcs_cold_allocated_bytes=cold_bytes,
        fcs_allocated_bytes=prepare_bytes + hot_bytes + cold_bytes,
        fcs_context_reused=true,
        hot_fcs_retry=false,
        cold_fcs_retry=false,
    )
end

function qhe_relative_error(value, reference)
    return abs(value - reference) / max(abs(value), abs(reference), eps(Float64))
end

function qhe_runtime_physics_metrics(params::NonlinearQHEParameters, steady,
        fcs, Ej, ws)
    Ωh = params.Ωratio * params.Ωc
    Jh, Dh = fcs.hot_cumulants[1], fcs.hot_cumulants[2]
    Jc, Dc = fcs.cold_cumulants[1], fcs.cold_cumulants[2]
    A = log(1 / params.nc + 1) / params.lh - log(1 / params.nh + 1) / params.lc
    σh = params.lc * (Jh / Ωh) * A
    σc = -params.lh * (Jc / params.Ωc) * A
    Fh = abs(Dh / (Ωh * Jh))
    Fc = abs(Dc / (params.Ωc * Jc))
    Qh = (Dh / Jh^2) * σh
    Qc = (Dc / Jc^2) * σc
    W_out = -Jh + -Jc
    hot_direct_current = qhe_jump_current(steady.ρss, fcs.hot_jumps,
        fcs.hot_weights)
    cold_direct_current = qhe_jump_current(steady.ρss, fcs.cold_jumps,
        fcs.cold_weights)
    hot_current_check = abs(hot_direct_current) > eps() ?
        Jh / hot_direct_current : NaN
    cold_current_check = abs(cold_direct_current) > eps() ?
        Jc / cold_direct_current : NaN
    return (
        Ej=Ej,
        Ωh=Ωh,
        Jh=Jh,
        Dh=Dh,
        Jc=Jc,
        Dc=Dc,
        Fh=Fh,
        Fc=Fc,
        σh=σh,
        σc=σc,
        Qh=Qh,
        Qc=Qc,
        A=A,
        W_out=W_out,
        P_drive=-W_out,
        CRWA=rwa_coherence(steady.ρss, params, ws),
        epsilon_off=off_resonant_epsilon(steady.ρss, Ej, ws, params),
        hot_direct_current=hot_direct_current,
        cold_direct_current=cold_direct_current,
        hot_current_check=hot_current_check,
        cold_current_check=cold_current_check,
    )
end

function qhe_point(params::NonlinearQHEParameters=NonlinearQHEParameters(),
        numerics::NonlinearQHENumerics=NonlinearQHENumerics())
    Ωh = params.Ωratio * params.Ωc
    H, J, ws, Ej = liouv_RWA(params)
    steady = qhe_steadystate(H, J, ws, params, numerics)
    ρss = steady.ρss

    hot_jumps = [J[1], J[3]]
    cold_jumps = [J[2], J[4]]
    fcs_t0 = time_ns()
    hot_cumulants, hot_fcs_warnings, hot_fcs_retry = qhe_fcs_cumulants(
        steady.system.L_data, steady.ρss, hot_jumps, [-Ωh, Ωh], steady.Pl,
        numerics)
    cold_cumulants, cold_fcs_warnings, cold_fcs_retry = qhe_fcs_cumulants(
        steady.system.L_data, steady.ρss, cold_jumps, [-params.Ωc, params.Ωc],
        steady.Pl, numerics)
    fcs_seconds = qhe_seconds_since(fcs_t0)
    Jh, Dh = hot_cumulants[1], hot_cumulants[2]
    Jc, Dc = cold_cumulants[1], cold_cumulants[2]

    A = log(1 / params.nc + 1) / params.lh - log(1 / params.nh + 1) / params.lc
    σh = params.lc * (Jh / Ωh) * A
    σc = -params.lh * (Jc / params.Ωc) * A
    Fh = abs(Dh / (Ωh * Jh))
    Fc = abs(Dc / (params.Ωc * Jc))
    Qh = (Dh / Jh^2) * σh
    Qc = (Dc / Jc^2) * σc
    Qh_from_F = params.lc * A * Fh
    Qc_from_F = params.lh * A * Fc
    Qh_factor_error = abs(Qh - Qh_from_F) / max(abs(Qh), eps())
    Qc_factor_error = abs(Qc - Qc_from_F) / max(abs(Qc), eps())
    Fh_TUR_threshold = A > 0 ? 2 / (params.lc * A) : NaN
    Fc_TUR_threshold = A > 0 ? 2 / (params.lh * A) : NaN
    W_out = -Jh + -Jc
    P_drive = -W_out

    CRWA = rwa_coherence(ρss, params, ws)
    epsilon_off = off_resonant_epsilon(ρss, Ej, ws, params)
    tight_coupling_error = abs(params.lc * params.Ωc * Jh + params.lh * Ωh * Jc) /
        max(abs(params.lc * params.Ωc * Jh), abs(params.lh * Ωh * Jc), eps())
    hot_direct_current = qhe_jump_current(ρss, hot_jumps, [-Ωh, Ωh])
    cold_direct_current = qhe_jump_current(ρss, cold_jumps, [-params.Ωc, params.Ωc])
    hot_current_check = abs(hot_direct_current) > eps() ? Jh / hot_direct_current : NaN
    cold_current_check = abs(cold_direct_current) > eps() ? Jc / cold_direct_current : NaN
    fcs_current_ok = isfinite(hot_current_check) && isfinite(cold_current_check) &&
        abs(hot_current_check - 1) <= numerics.fcs_check_tol &&
        abs(cold_current_check - 1) <= numerics.fcs_check_tol
    current_ok = abs(Jh / Ωh) > numerics.current_tol && abs(Jc / params.Ωc) > numerics.current_tol
    rwa_ok = epsilon_off < numerics.epsilon_tol
    entropy_ok = σh > 0 && σc > 0 && A > 0
    physical_ok = current_ok && rwa_ok && entropy_ok && tight_coupling_error < 1e-5
    solver_ok = steady.ss_converged && isfinite(steady.relative_liouvillian_residual) &&
        isfinite(steady.trace_error) && isfinite(steady.hermiticity_error)
    fcs_ok = fcs_current_ok && hot_fcs_warnings == 0 && cold_fcs_warnings == 0 &&
        all(isfinite, hot_cumulants) && all(isfinite, cold_cumulants)
    viable = physical_ok && steady.cutoff_ok && solver_ok && fcs_ok

    return (; g=params.g, Ej, λh=params.λh, λc=params.λc, Ωc=params.Ωc, Ωh,
        Ωratio=params.Ωratio, κh=params.κh, κc=params.κc, nh=params.nh,
        nc=params.nc, lh=params.lh, lc=params.lc, Nmax_h=params.Nmax_h,
        Nmax_c=params.Nmax_c, Jh, Dh, Jc, Dc, P_drive, W_out, Fh, Fc, σh, σc,
        Qh, Qc, Qh_from_F, Qc_from_F, Qh_factor_error, Qc_factor_error, A,
        Fh_TUR_threshold, Fc_TUR_threshold, CRWA, epsilon_off,
        tight_coupling_error, hot_direct_current, cold_direct_current,
        hot_current_check, cold_current_check, fcs_current_ok,
        hot_fcs_warnings, cold_fcs_warnings, hot_fcs_retry, cold_fcs_retry,
        fcs_seconds, current_ok, rwa_ok, entropy_ok, physical_ok,
        solver_ok, fcs_ok, viable,
        ss_backend=steady.ss_backend, ss_converged=steady.ss_converged,
        ss_iterations=steady.ss_iterations, ss_rebuilds=steady.ss_rebuilds,
        ss_ilu_tau=get(steady, :ss_ilu_tau, NaN),
        ss_shift_factor=get(steady, :ss_shift_factor, NaN),
        ilu_seconds=steady.ilu_seconds, gmres_seconds=steady.gmres_seconds,
        steady_state_seconds=steady.steady_state_seconds,
        trace_error=steady.trace_error,
        liouvillian_residual=steady.liouvillian_residual,
        relative_liouvillian_residual=steady.relative_liouvillian_residual,
        hermiticity_error=steady.hermiticity_error,
        hot_n=steady.hot_n, cold_n=steady.cold_n,
        hot_occupation_fraction=steady.hot_occupation_fraction,
        cold_occupation_fraction=steady.cold_occupation_fraction,
        occupation_tol=steady.occupation_tol,
        hot_boundary=steady.hot_boundary, cold_boundary=steady.cold_boundary,
        hot_tail=steady.hot_tail, cold_tail=steady.cold_tail,
        hot_tail5=steady.hot_tail5, cold_tail5=steady.cold_tail5,
        hot_cutoff_status=steady.hot_cutoff_status,
        cold_cutoff_status=steady.cold_cutoff_status,
        trunc_tol=steady.trunc_tol, occupation_ok=steady.occupation_ok,
        trunc_ok=steady.trunc_ok, cutoff_ok=steady.cutoff_ok)
end

function qhe_point(; kwargs...)
    params, numerics = split_nonlinear_qhe_kwargs(kwargs)
    return qhe_point(params, numerics)
end

function sweep_g(g_values; kwargs...)
    return DataFrame([qhe_point(; kwargs..., g=g) for g in g_values])
end

function sweep_nc(nc_values; kwargs...)
    return DataFrame([qhe_point(; kwargs..., nc=nc) for nc in nc_values])
end

function sweep_λc(λc_values; kwargs...)
    return DataFrame([qhe_point(; kwargs..., λc=λc) for λc in λc_values])
end

function sweep_g_λc(g_values, λc_values; kwargs...)
    rows = NamedTuple[]
    for λc in λc_values, g in g_values
        push!(rows, qhe_point(; kwargs..., g=g, λc=λc))
    end
    return DataFrame(rows)
end

function sweep_λh_λc(λh_values, λc_values; kwargs...)
    rows = NamedTuple[]
    for λc in λc_values, λh in λh_values
        push!(rows, qhe_point(; kwargs..., λh=λh, λc=λc))
    end
    return DataFrame(rows)
end

function qhe_grid_matrix(df, x_values, y_values, column::Symbol)
    expected = length(x_values) * length(y_values)
    nrow(df) == expected || error("Grid dimensions imply $expected rows, but data frame has $(nrow(df)) rows")
    return reshape(df[!, column], length(x_values), length(y_values))'
end

function convergence_over_cutoffs(N_values; kwargs...)
    return DataFrame([qhe_point(; kwargs..., Nmax_h=N, Nmax_c=N) for N in N_values])
end

function convergence_over_cutoffs(cutoffs::AbstractVector{<:Tuple}; kwargs...)
    return DataFrame([qhe_point(; kwargs..., Nmax_h=Nh, Nmax_c=Nc)
        for (Nh, Nc) in cutoffs])
end

function qhe_cutoff_ladder(; Nmax_h_start=5, Nmax_c_start=5,
        Nmax_h_stop=15, Nmax_c_stop=15, step=2)
    hot = collect(Nmax_h_start:step:Nmax_h_stop)
    cold = collect(Nmax_c_start:step:Nmax_c_stop)
    n = min(length(hot), length(cold))
    return [(hot[i], cold[i]) for i in 1:n]
end

function qhe_cutoff_preview(params::NonlinearQHEParameters;
        occupation_tol=NONLINEAR_QHE_OCCUPATION_TOL,
        tail_tol=NONLINEAR_QHE_TRUNC_TOL)
    point = qhe_point(params, nonlinear_qhe_numerics(
        occupation_tol=occupation_tol,
        trunc_tol=tail_tol,
    ))
    recommended_h = max(params.Nmax_h,
        ceil(Int, point.hot_n / max(occupation_tol, eps(Float64))))
    recommended_c = max(params.Nmax_c,
        ceil(Int, point.cold_n / max(occupation_tol, eps(Float64))))
    return merge(point, (
        recommended_Nmax_h=recommended_h,
        recommended_Nmax_c=recommended_c,
        cutoff_budget_exceeded=point.hot_occupation_fraction > occupation_tol ||
            point.cold_occupation_fraction > occupation_tol ||
            point.hot_tail > tail_tol || point.cold_tail > tail_tol,
    ))
end

function qhe_relative_change(new, old)
    return abs(new - old) / max(abs(new), abs(old), eps(Float64))
end

function qhe_cumulant_convergence_table(cutoffs; kwargs...)
    df = convergence_over_cutoffs(cutoffs; kwargs...)
    if nrow(df) <= 1
        df.hot_Fh_rel_change = fill(missing, nrow(df))
        df.cold_Fc_rel_change = fill(missing, nrow(df))
        df.hot_current_rel_change = fill(missing, nrow(df))
        df.cold_current_rel_change = fill(missing, nrow(df))
        return df
    end
    hot_Fh_rel_change = Vector{Union{Missing,Float64}}(missing, nrow(df))
    cold_Fc_rel_change = Vector{Union{Missing,Float64}}(missing, nrow(df))
    hot_current_rel_change = Vector{Union{Missing,Float64}}(missing, nrow(df))
    cold_current_rel_change = Vector{Union{Missing,Float64}}(missing, nrow(df))
    for i in 2:nrow(df)
        hot_Fh_rel_change[i] = qhe_relative_change(df.Fh[i], df.Fh[i - 1])
        cold_Fc_rel_change[i] = qhe_relative_change(df.Fc[i], df.Fc[i - 1])
        hot_current_rel_change[i] = qhe_relative_change(df.Jh[i], df.Jh[i - 1])
        cold_current_rel_change[i] = qhe_relative_change(df.Jc[i], df.Jc[i - 1])
    end
    df.hot_Fh_rel_change = hot_Fh_rel_change
    df.cold_Fc_rel_change = cold_Fc_rel_change
    df.hot_current_rel_change = hot_current_rel_change
    df.cold_current_rel_change = cold_current_rel_change
    return df
end

function candidate_table(df; metric=:Qh, n=10, require=:physical)
    base_mask = isfinite.(df[!, metric])
    mask = if require == :viable
        base_mask .& df.viable
    elseif require == :physical
        base_mask .& df.physical_ok
    elseif require == :all
        base_mask
    else
        error("require must be :viable, :physical, or :all")
    end
    ok = df[mask, :]
    nrow(ok) == 0 && return ok
    cols = [:g, :λh, :λc, :nh, :nc, :κh, :κc, :A, :Qh, :Qc, :Fh, :Fc, :CRWA,
        :Fh_TUR_threshold, :Fc_TUR_threshold, :epsilon_off, :hot_tail,
        :cold_tail, :hot_tail5, :cold_tail5, :hot_occupation_fraction,
        :cold_occupation_fraction, :hot_cutoff_status, :cold_cutoff_status,
        :trace_error, :relative_liouvillian_residual, :hermiticity_error,
        :hot_current_check, :cold_current_check, :tight_coupling_error,
        :Qh_factor_error, :Qc_factor_error, :trunc_ok, :occupation_ok,
        :cutoff_ok, :solver_ok, :fcs_ok, :physical_ok, :viable]
    return sort(ok[:, intersect(cols, propertynames(ok))], metric)[1:min(n, nrow(ok)), :]
end

function resonant_solutions(; Ωratio=π, lh=1, lc=2, kmin=-120, kmax=120,
        lmin=-240, lmax=240, tol=1e-10)
    Ωc = 1.0
    Ωh = Ωratio * Ωc
    voltage = lh * Ωh - lc * Ωc
    rows = NamedTuple[]
    for k in kmin:kmax, l in lmin:lmax
        detuning = voltage - (k * Ωh - l * Ωc)
        abs(detuning) < tol && push!(rows, (; branch=:+, k, l, detuning))
        detuning_conj = voltage + (k * Ωh - l * Ωc)
        abs(detuning_conj) < tol && push!(rows, (; branch=:-, k, l, detuning=detuning_conj))
    end
    return DataFrame(rows)
end

function near_resonant_candidates(; Ωratio=π, lh=1, lc=2, kmax=8, lmax=12, top=10)
    Ωc = 1.0
    Ωh = Ωratio * Ωc
    voltage = lh * Ωh - lc * Ωc
    rows = NamedTuple[]
    for k in 0:kmax, l in 0:lmax
        k == lh && l == lc && continue
        photon_order = k + l
        photon_order == 0 && continue
        detuning = abs(voltage - (k * Ωh - l * Ωc))
        push!(rows, (; k, l, photon_order, detuning))
    end
    df = sort(DataFrame(rows), [:detuning, :photon_order])
    return df[1:min(top, nrow(df)), :]
end
