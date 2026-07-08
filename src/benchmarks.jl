using Statistics

function benchmark_vs_dimension_linearised(
    N_values,
    nC_fixed;
    g=0.35,
    κh=1.0,
    κc=1.0,
    nh=0.5,
    nc=0.05,
    samples=100,
    evals=1,
)
    dims = Int[]
    times_ms = Float64[]
    cumulant_1 = Float64[]
    cumulant_2 = Float64[]

    for N in N_values
        println("Benchmarking linearised cutoff N=$N, fixed nC=$nC_fixed")
        H, J, mJ, nu, ρss, basis, _ =
            build_linearised_model(; Nh=N, Nc=N, g=g, κh=κh, κc=κc, nh=nh, nc=nc)

        cumulants = fcscumulants_recursive(H, J, mJ, nC_fixed, ρss, nu)
        @assert length(cumulants) == nC_fixed
        @assert all(isfinite, cumulants)

        trial = @benchmark fcscumulants_recursive(
            $H, $J, $mJ, $nC_fixed, $ρss, $nu
        ) samples=samples evals=evals

        push!(dims, length(basis))
        # Average the requested BenchmarkTools sample times so each plotted
        # point reports the arithmetic mean runtime for this dimension.
        push!(times_ms, mean(trial.times) / 1e6)
        # Store the already-computed cumulants so diagnostics use the same run
        # as the timing validation call.
        push!(cumulant_1, cumulants[1])
        push!(cumulant_2, cumulants[2])
    end

    return dims, times_ms, cumulant_1, cumulant_2
end

function benchmark_vs_dimension_dense(
    N_values,
    nC_fixed;
    samples=10,
    evals=1,
    model_kwargs...,
)
    dims = Int[]
    times_ms = Float64[]
    cumulant_1 = Float64[]
    cumulant_2 = Float64[]
    highest_populations = Float64[]

    for N in N_values
        println("Benchmarking dense cutoff N=$N, fixed nC=$nC_fixed")
        H, J, mJ, nu, ρss, basis =
            build_dense_circuit_qhe_model(N, N; model_kwargs...)

        cumulants = fcscumulants_recursive(H, J, mJ, nC_fixed, ρss, nu)
        @assert length(cumulants) == nC_fixed
        @assert all(isfinite, cumulants)

        trial = @benchmark fcscumulants_recursive(
            $H, $J, $mJ, $nC_fixed, $ρss, $nu
        ) samples=samples evals=evals

        push!(dims, length(basis))
        # Average the requested BenchmarkTools sample times so each plotted
        # point reports the arithmetic mean runtime for this dimension.
        push!(times_ms, mean(trial.times) / 1e6)
        # Save cumulants and the cutoff-corner occupation for convergence
        # diagnostics without an extra model build.
        push!(cumulant_1, cumulants[1])
        push!(cumulant_2, cumulants[2])
        push!(highest_populations, highest_joint_fock_population(ρss))
    end

    return dims, times_ms, cumulant_1, cumulant_2, highest_populations
end
