# Production FCS run: driven-dissipative JC, fixed g/κ drive sweeps

Generated 2026-07-21T13:16:34.925; started 2026-07-20T17:53:26.488.

## Model

Rotating-frame JC Hamiltonian `H = -Δ(a†a + σ₊σ₋) + g(a†σ₋ + aσ₊) - E(a + a†)`
with cavity loss only: master equation `ρ̇ = -i[H,ρ] + κ𝒟[a]ρ`.
Counted jump: cavity emission `κ a ρ a†` (`mJ = [√κ a]`, `ν = [1]`).
Units: `κ = 1.0`. Carmichael conversion: `κ_C = κ/2`.

## Scan

- `g/κ = 14.0` (fixed); drive sweep `x = 2E/g ∈ [0.05, 1.45]`, 89 points per cut
- detuning cuts `Δ/κ ∈ [0.0, 0.55, 0.7]`
- cumulants: `nC = 3` (c₁, c₂, c₃ of the cavity-emission counting statistics)

## Dynamic cutoff

Per-point Fock cutoff from the semiclassical bright-branch estimate
(`jc_semiclassical_bright_n`), tiers `(150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500)`, occupation cap
`occ_max = 0.5`, headroom `pad_sigma·√n + 25.0` with
pad_sigma = Δ=0.0→14.0, Δ=0.55→6.0, Δ=0.7→6.0.
Cutoffs used per cut:

- `Δ/κ = 0.0`: x∈[0.05,1.08]→N=150, x∈[1.1,1.12]→N=175, x∈[1.14,1.14]→N=200, x∈[1.16,1.18]→N=225, x∈[1.2,1.2]→N=250, x∈[1.22,1.24]→N=275, x∈[1.26,1.26]→N=300, x∈[1.28,1.32]→N=350, x∈[1.34,1.38]→N=400, x∈[1.4,1.45]→N=450
- `Δ/κ = 0.55`: x∈[0.05,0.69]→N=150, x∈[0.7,0.71]→N=175, x∈[0.72,0.74]→N=200, x∈[0.75,0.77]→N=225, x∈[0.78,0.8]→N=250, x∈[0.81,0.84]→N=275, x∈[0.85,0.87]→N=300, x∈[0.88,0.94]→N=350, x∈[0.95,1.0]→N=400, x∈[1.02,1.08]→N=450, x∈[1.1,1.45]→N=500
- `Δ/κ = 0.7`: x∈[0.05,0.62]→N=150, x∈[0.63,0.67]→N=175, x∈[0.68,0.71]→N=200, x∈[0.72,0.76]→N=225, x∈[0.77,0.8]→N=250, x∈[0.81,0.85]→N=275, x∈[0.86,0.9]→N=300, x∈[0.91,0.99]→N=350, x∈[1.0,1.08]→N=400, x∈[1.1,1.18]→N=450, x∈[1.2,1.45]→N=500

## Solvers

Steady state (per point, continuation within each cutoff segment):
trace-constrained GMRES, warm-started, shifted crout ILU `τ = 0.1`,
shift factor `1.0e-6`, adaptive rebuild at `> 80`
iterations (itmax 120, fallback 300), Krylov memory
60, rtol 1.0e-10, atol 1.0e-14.

FCS (QuantumFCS.jl): `method = :iterative`, steady-state ILU injected as `Pl`
(right-preconditioned, true-residual stopping), rtol 1.0e-8,
itmax 300, memory 60. Acceptance gate:
`|c₁/(κ⟨n⟩) − 1| ≤ 0.0001`, finite c₂/c₃, no solver
warnings; one retry with the internal preconditioner otherwise.

## Run quality

- 267 points; 0 FCS retries; 0 non-safe cutoff states
- worst `|current_check − 1|` = 2.66e-15; worst tail `p_{N-1}+p_N` = 4.82e-6
- lower-cutoff spot validation (relative drift, production vs one tier down):
  - fcs Δ=0.0, x=1.44: N=450 vs 400: |Δc1|/c1 = 5.71e-7, |Δc2|/c2 = 9.51e-5, |Δc3|/|c3| = 0.00123
  - fcs Δ=0.0, x=1.45: N=450 vs 400: |Δc1|/c1 = 1.59e-6, |Δc2|/c2 = 0.00025, |Δc3|/|c3| = 0.00323
  - fcs Δ=0.55, x=0.7: N=175 vs 150: |Δc1|/c1 = 1.13e-6, |Δc2|/c2 = 1.36e-6, |Δc3|/|c3| = 1.77e-5
  - fcs Δ=0.55, x=1.45: N=500 vs 450: |Δc1|/c1 = 0.00103, |Δc2|/c2 = 0.0149, |Δc3|/|c3| = 0.0114
  - fcs Δ=0.7, x=0.65: N=175 vs 150: |Δc1|/c1 = 4.02e-8, |Δc2|/c2 = 1.82e-9, |Δc3|/|c3| = 3.25e-7
  - fcs Δ=0.7, x=1.45: N=500 vs 450: |Δc1|/c1 = 3.31e-11, |Δc2|/c2 = 2.13e-9, |Δc3|/|c3| = 3.11e-9

## Provenance

- IncompleteLU: v0.2.1 [registry]
- Krylov: v0.10.6 [registry]
- LinearSolve: v3.26.0 [registry]
- QuantumFCS: v1.0.0 [path: /Users/jano/dev/QuantumFCS]
- QuantumToolbox: v0.28.0 [registry]
- Julia 1.12.6, 4 threads, host Marcelos-MacBook-Pro-3.local

## Files

- `/Users/jano/dev/QuantumFCS_paper_writing/FCS.jl-application/data/jc_fcs_production_g14_results.jld2`
- `/Users/jano/dev/QuantumFCS_paper_writing/FCS.jl-application/data/jc_fcs_production_g14_rows.csv`
- `/Users/jano/dev/QuantumFCS_paper_writing/FCS.jl-application/data/jc_fcs_production_g14_metadata.md`
- `/Users/jano/dev/QuantumFCS_paper_writing/QuantumFCSjl_paper/figures/jc_fcs_g14.pdf`
