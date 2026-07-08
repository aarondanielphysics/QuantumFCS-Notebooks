function build_linearised_model(; Nh=8, Nc=8, g=0.35, κh=1.0, κc=1.0, nh=0.5, nc=0.05)
    basis_h = FockBasis(Nh)
    basis_c = FockBasis(Nc)
    basis = basis_h ⊗ basis_c

    Id_h = one(basis_h)
    Id_c = one(basis_c)
    ah = destroy(basis_h) ⊗ Id_c
    ac = Id_h ⊗ destroy(basis_c)
    adh = create(basis_h) ⊗ Id_c
    adc = Id_h ⊗ create(basis_c)

    H = g * (adh * ac + adc * ah)
    J = [
        sqrt((nh + 1) * κh) * ah,
        sqrt((nc + 1) * κc) * ac,
        sqrt(nh * κh) * adh,
        sqrt(nc * κc) * adc,
    ]
    mJ = [J[2], J[4]]
    nu = [-1, 1]
    ρss = steadystate.iterative(H, J)

    return H, J, mJ, nu, ρss, basis, (ah=ah, ac=ac, adh=adh, adc=adc)
end

current_analytic(g, κ, nh, nc) = 2 * g^2 * κ / (4 * g^2 + κ^2) * (nh - nc)

function scaled_variance_analytic(g, κ, nh, nc)
    current = current_analytic(g, κ, nh, nc)
    E = current / (nh - nc)
    S = E * (1 - 2 * g^2 * (4 * g^2 + 5 * κ^2) / (4 * g^2 + κ^2)^2)
    return E * (nh * (nh + 1) + nc * (nc + 1)) - S * (nh - nc)^2
end

shot_noise_analytic(g, κ, nh, nc) = current_analytic(g, κ, nh, nc) / (nh - nc)

function thermal_noise_analytic(g, κ, nh, nc)
    E = shot_noise_analytic(g, κ, nh, nc)
    return E * (1 - 2 * g^2 * (4 * g^2 + 5 * κ^2) / (4 * g^2 + κ^2)^2)
end

function linearised_cumulants_analytic(g, κ, nh, nc)
    # The linearised benchmark monitors cold-bath jumps with weights [-1, 1],
    # so its first cumulant has the opposite sign from the positive heat current.
    return (-current_analytic(g, κ, nh, nc), scaled_variance_analytic(g, κ, nh, nc))
end

function numerical_moments_linearised(g; Nh=15, Nc=15, κh=1.0, κc=1.0, nh=0.5, nc=0.05)
    H, J, mJ, nu, ρss, basis, _ =
        build_linearised_model(; Nh=Nh, Nc=Nc, g=g, κh=κh, κc=κc, nh=nh, nc=nc)
    c1, c2 = fcscumulants_recursive(H, J, mJ, 2, ρss, nu)
    return c1, c2, length(basis), H, J, mJ, nu, ρss
end

function build_dense_circuit_qhe_model(
    Nh::Int,
    Nc::Int;
    Ωh=5.0,
    Ωc=1.0,
    EJ=1.75,
    λh=0.20,
    λc=0.25,
    κh=1.0,
    κc=1.0,
    nbarh=0.50,
    nbarc=0.05,
)
    basis_h = FockBasis(Nh)
    basis_c = FockBasis(Nc)
    basis = basis_h ⊗ basis_c

    Id_h = one(basis_h)
    Id_c = one(basis_c)
    ah = destroy(basis_h) ⊗ Id_c
    ac = Id_h ⊗ destroy(basis_c)
    adh = dagger(ah)
    adc = dagger(ac)

    Φ = λh * (adh + ah) + λc * (adc + ac)
    exp_2iΦ = exp(2im * dense(Φ))
    cos_2Φ = (exp_2iΦ + dagger(exp_2iΦ)) / 2

    H = dense(Ωh * adh * ah + Ωc * adc * ac) - EJ * cos_2Φ
    J = [
        sqrt(κh * (nbarh + 1)) * ah,
        sqrt(κc * (nbarc + 1)) * ac,
        sqrt(κh * nbarh) * adh,
        sqrt(κc * nbarc) * adc,
    ]

    mJ = [J[1], J[3]]
    nu = [-1, 1]
    ρss = steadystate.iterative(H, J)

    return H, J, mJ, nu, ρss, basis
end

function highest_joint_fock_population(ρss)
    # In the current tensor-product basis, the last diagonal entry is the
    # joint cutoff-corner state |N_h - 1, N_c - 1><N_h - 1, N_c - 1|.
    return real(ρss.data[end, end])
end
