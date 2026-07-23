# Regenerate the paper-application figures from the checked-in data.
#
# Cheap: no sweeps are rerun, only plotting. This is the paper-application
# counterpart of `make plots` for the benchmark half.
#
# Run:  julia --project=. scripts/plot_paper_figures.jl
#       make paper-figures

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

using CairoMakie
using DataFrames
using JLD2
using LaTeXStrings

include(joinpath(PROJECT_ROOT, "src", "data_io.jl"))
include(joinpath(PROJECT_ROOT, "src", "paper_figures.jl"))

CairoMakie.activate!(type = "png")
figures = joinpath(PROJECT_ROOT, "figures")
mkpath(figures)

function save_both(fig, name)
    png = joinpath(figures, string(name, ".png"))
    pdf = joinpath(figures, string(name, ".pdf"))
    save(png, fig; px_per_unit = 2)
    save(pdf, fig)
    println("  wrote $png")
    println("  wrote $pdf")
end

println("Jaynes-Cummings (paper Fig. 2)")
jc = load_jc_production()
save_both(jc_paper_figure(jc.rows, jc.detuning_cuts, jc.fock; g = jc.g), "jc_fcs_g14")

println("Circuit-QED heat engine (paper Figs. 3 and 4)")
q = load_qhe_paper_sweeps()
save_both(qhe_antibunching_paper_figure(q.antibunching_g, q.antibunching_λc),
          "qhe_antibunching")
save_both(qhe_finite_affinity_paper_figure(q.finite_affinity_g, q.finite_affinity_λc),
          "qhe_finite_affinity_tur")

println("Validation appendix figures")
save_both(qhe_antibunching_validation_figure(q.antibunching_g, q.antibunching_λc),
          "qhe_antibunching_validation")
save_both(qhe_finite_affinity_validation_figure(q.finite_affinity_g, q.finite_affinity_λc),
          "qhe_finite_affinity_validation")

println("Done.")
