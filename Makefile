JULIA ?= julia
JULIA_FLAGS ?= --project=.
WOLFRAMSCRIPT ?= wolframscript
WOLFRAM_FLAGS ?=
MELT_ALLOW_DOWNLOAD ?= 0

LINEARISED_N_VALUES ?= 1,2,3,4,5,6,7,8
DENSE_N_VALUES ?= 1,2,3,4,5,6,7
# Julia timing sample counts use BenchmarkTools and can afford larger samples.
LINEARISED_SAMPLES ?= 100
DENSE_SAMPLES ?= 10
# MELT timing samples are separate because high-dimensional Wolfram runs are slow.
MELT_LINEARISED_SAMPLES ?= 5
MELT_DENSE_SAMPLES ?= 10
EVALS ?= 1
NC ?= 2

# Shared model parameters passed into both Julia and MELT benchmark producers.
LINEARISED_G ?= 0.35
LINEARISED_NH ?= 0.5
LINEARISED_NC ?= 0.05
DENSE_EJ ?= 1.75
DENSE_NBAR_H ?= 0.5
DENSE_NBAR_C ?= 0.05

.PHONY: help instantiate benchmark all machine-info benchmarks julia-benchmarks \
	mathematica-benchmarks melt-benchmarks plots linearised-benchmark \
	dense-benchmark melt-linearised-benchmark melt-dense-benchmark \
	linearised-plot dense-plot smoke clean-generated-plots \
	test paper-figures jc-production qhe-sweeps paper-applications

help:
	@echo "Benchmark targets:"
	@echo "  make plots                 regenerate figures from checked-in CSV data"
	@echo "  make smoke                 cheap plotting-only reproducibility check"
	@echo "  make benchmark             collect metadata, run Julia + MELT benchmarks, regenerate plots"
	@echo "  make instantiate           instantiate the Julia environment"
	@echo "  make machine-info          save local machine metadata to data/machine_info.toml"
	@echo "  make julia-benchmarks      run linearised and dense Julia benchmarks"
	@echo "  make mathematica-benchmarks run linearised and dense MELT scripts"
	@echo "  make linearised-benchmark  run only the linearised Julia benchmark"
	@echo "  make dense-benchmark       run only the dense Julia benchmark"
	@echo "  make melt-linearised-benchmark run only the linearised MELT script"
	@echo "  make melt-dense-benchmark  run only the dense MELT script"
	@echo ""
	@echo "Paper-application targets (Secs. 4 and 5):"
	@echo "  make test                  run the regression and parity test suite (~80 s)"
	@echo "  make paper-figures         regenerate Figs. 2-4 from checked-in data (no sweeps)"
	@echo "  make qhe-sweeps            recompute the heat-engine sweeps (~9 min)"
	@echo "  make jc-production         recompute the full Jaynes-Cummings sweep (~1 hour)"
	@echo "  make paper-applications    qhe-sweeps + jc-production + paper-figures"
	@echo ""
	@echo "  Reduced runs, e.g.:"
	@echo "    make qhe-sweeps QHE_SWEEPS=antibunching_g QHE_POINTS=25"
	@echo "    make jc-production JC_DETUNINGS=0.0 JC_X_MIN=0.4 JC_X_MAX=0.6 JC_X_STEP=0.05 JC_N_OVERRIDE=40"

all: benchmark

benchmark: machine-info julia-benchmarks mathematica-benchmarks plots

smoke: plots

instantiate:
	$(JULIA) $(JULIA_FLAGS) -e 'using Pkg; Pkg.instantiate()'

machine-info:
	$(JULIA) $(JULIA_FLAGS) scripts/collect_machine_info.jl

benchmarks: julia-benchmarks mathematica-benchmarks

julia-benchmarks: linearised-benchmark dense-benchmark

mathematica-benchmarks: melt-benchmarks

melt-benchmarks: melt-linearised-benchmark melt-dense-benchmark

linearised-benchmark:
	LINEARISED_N_VALUES="$(LINEARISED_N_VALUES)" \
	LINEARISED_SAMPLES="$(LINEARISED_SAMPLES)" \
	EVALS="$(EVALS)" \
	NC="$(NC)" \
	LINEARISED_G="$(LINEARISED_G)" \
	LINEARISED_NH="$(LINEARISED_NH)" \
	LINEARISED_NC="$(LINEARISED_NC)" \
	$(JULIA) $(JULIA_FLAGS) scripts/run_linearised_qhe_benchmark.jl

dense-benchmark:
	DENSE_N_VALUES="$(DENSE_N_VALUES)" \
	DENSE_SAMPLES="$(DENSE_SAMPLES)" \
	EVALS="$(EVALS)" \
	NC="$(NC)" \
	DENSE_EJ="$(DENSE_EJ)" \
	DENSE_NBAR_H="$(DENSE_NBAR_H)" \
	DENSE_NBAR_C="$(DENSE_NBAR_C)" \
	$(JULIA) $(JULIA_FLAGS) scripts/run_dense_circuit_qhe_benchmark.jl

melt-linearised-benchmark:
	FCS_PROJECT_ROOT="$(CURDIR)" \
	MELT_ALLOW_DOWNLOAD="$(MELT_ALLOW_DOWNLOAD)" \
	MELT_LINEARISED_SAMPLES="$(MELT_LINEARISED_SAMPLES)" \
	MELT_LINEARISED_G="$(LINEARISED_G)" \
	MELT_LINEARISED_NH="$(LINEARISED_NH)" \
	MELT_LINEARISED_NC="$(LINEARISED_NC)" \
	$(WOLFRAMSCRIPT) $(WOLFRAM_FLAGS) -script scripts/run_linearised_qhe_melt.wl

melt-dense-benchmark:
	FCS_PROJECT_ROOT="$(CURDIR)" \
	MELT_ALLOW_DOWNLOAD="$(MELT_ALLOW_DOWNLOAD)" \
	MELT_DENSE_SAMPLES="$(MELT_DENSE_SAMPLES)" \
	MELT_DENSE_EJ="$(DENSE_EJ)" \
	MELT_DENSE_NBAR_H="$(DENSE_NBAR_H)" \
	MELT_DENSE_NBAR_C="$(DENSE_NBAR_C)" \
	$(WOLFRAMSCRIPT) $(WOLFRAM_FLAGS) -script scripts/run_dense_circuit_qhe_melt.wl

plots: linearised-plot dense-plot

linearised-plot:
	$(JULIA) $(JULIA_FLAGS) scripts/plot_linearised_qhe_comparison.jl

dense-plot:
	$(JULIA) $(JULIA_FLAGS) scripts/plot_dense_circuit_qhe_comparison.jl

# --- Paper applications (manuscript Secs. 4 and 5) --------------------------
# Sweep sizes are controlled by environment variables read inside the scripts:
#   QHE_SWEEPS, QHE_POINTS
#   JC_DETUNINGS, JC_X_MIN, JC_X_MAX, JC_X_STEP, JC_N_OVERRIDE, JC_G, JC_NC

test:
	$(JULIA) $(JULIA_FLAGS) test/runtests.jl

paper-figures:
	$(JULIA) $(JULIA_FLAGS) scripts/plot_paper_figures.jl

qhe-sweeps:
	$(JULIA) $(JULIA_FLAGS) scripts/run_qhe_paper_sweeps.jl

jc-production:
	$(JULIA) $(JULIA_FLAGS) scripts/run_jc_fcs_production_sweep.jl

paper-applications: qhe-sweeps jc-production paper-figures

clean-generated-plots:
	$(RM) figures/benchmark_compare.png figures/benchmark_compare.pdf
	$(RM) figures/benchmark_dense_circuit_qhe_compare.png figures/benchmark_dense_circuit_qhe_compare.pdf
	$(RM) figures/benchmark_linearised_qhe_cumulant_errors.png figures/benchmark_linearised_qhe_cumulant_errors.pdf
	$(RM) figures/benchmark_dense_circuit_qhe_highest_fock_population_scaled.png figures/benchmark_dense_circuit_qhe_highest_fock_population_scaled.pdf

