using LaTeXStrings

function read_two_column_csv(path::AbstractString; header=false)
    isfile(path) || return nothing
    return CSV.read(path, DataFrame; header=header, delim=',')
end

const QUANTUM_FIGURE_SIZE = (252, 168)
const QUANTUM_WIDE_FIGURE_SIZE = (2 * QUANTUM_FIGURE_SIZE[1], 2 * QUANTUM_FIGURE_SIZE[2])
const QUANTUMFCS_COLOR = "#0072B2"
const MELT_COLOR = "#E69F00"
const CUMULANT_1_COLOR = "#009E73"
const CUMULANT_2_COLOR = "#D55E00"
const DIMENSION_LABEL = L"\mathrm{Hilbert}\!-\!\mathrm{space\ dimension}\ d"
const RUNTIME_LABEL = L"\text{Mean runtime (ms)}"
const QUANTUMFCS_LABEL = L"\text{QuantumFCS}"
const MELT_LABEL = L"\text{MELT}"
const LINEARISED_ERROR_LABEL = L"\mathrm{Err}/|c_{\mathrm{ana}}|"
const LINEARISED_DIFFERENCE_ERROR_LABEL = L"|c_{\mathrm{ana}}-c_{\mathrm{num}}|/|c_{\mathrm{ana}}|"
const DENSE_ERROR_LABEL = L"P_d/|c|"
const FIRST_CUMULANT_LABEL = L"\text{First cumulant}"
const SECOND_CUMULANT_LABEL = L"\text{Second cumulant}"

function save_publication_figure(fig, output_file::AbstractString)
    save(output_file, fig; px_per_unit=4)
    root, _ = splitext(output_file)
    pdf_file = root * ".pdf"
    save(pdf_file, fig)
    return (raster=output_file, vector=pdf_file)
end

function benchmark_axis(fig; row=1, col=1, title="", xlabel=DIMENSION_LABEL)
    return Axis(
        fig[row, col];
        title=title,
        titlesize=8,
        yscale=log10,
        xlabel=xlabel,
        ylabel=RUNTIME_LABEL,
        xlabelsize=8,
        ylabelsize=8,
        xticklabelsize=7,
        yticklabelsize=7,
        xgridcolor=(:black, 0.08),
        ygridcolor=(:black, 0.08),
        spinewidth=0.6,
        xtickwidth=0.6,
        ytickwidth=0.6,
    )
end

function diagnostic_axis(fig; row=1, col=1, title="", xlabel=DIMENSION_LABEL, ylabel, yscale=identity)
    return Axis(
        fig[row, col];
        title=title,
        titlesize=8,
        xlabel=xlabel,
        ylabel=ylabel,
        yscale=yscale,
        xlabelsize=8,
        ylabelsize=8,
        xticklabelsize=7,
        yticklabelsize=7,
        xgridcolor=(:black, 0.08),
        ygridcolor=(:black, 0.08),
        spinewidth=0.6,
        xtickwidth=0.6,
        ytickwidth=0.6,
    )
end

function style_publication_figure!(fig)
    colgap!(fig.layout, 4)
    rowgap!(fig.layout, 4)
    return fig
end

function add_publication_legend!(ax; position=:rt)
    axislegend(
        ax;
        position=position,
        framevisible=false,
        labelsize=7,
        patchsize=(10, 6),
        rowgap=2,
    )
    return ax
end

function hide_x_ticks!(ax)
    # Hide both the tick marks and tick labels on shared upper-panel axes.
    ax.xticksvisible = false
    ax.xticklabelsvisible = false
    return ax
end

function add_panel_label!(fig, row, col, label)
    # Put panel labels in the top-left protrusion of each layout cell.
    Label(
        fig[row, col, TopLeft()],
        label;
        fontsize=9,
        font=:bold,
        padding=(0, 5, 5, 0),
        halign=:right,
    )
    return fig
end

function plot_runtime_comparison_on_axis!(ax, julia_file, melt_file; require_julia=false)
    # Runtime CSVs are intentionally kept as two-column, no-header files so the
    # Julia and MELT outputs can share the same plotting path.
    dj = read_two_column_csv(julia_file)
    dm = read_two_column_csv(melt_file)

    if require_julia && (dj === nothing || ncol(dj) < 2)
        error("Missing or invalid Julia benchmark CSV: $julia_file")
    end

    plotted = false
    if dj !== nothing && ncol(dj) >= 2
        lines!(ax, dj[:, 1], dj[:, 2]; color=QUANTUMFCS_COLOR, linewidth=1.4, label=QUANTUMFCS_LABEL)
        scatter!(ax, dj[:, 1], dj[:, 2]; color=QUANTUMFCS_COLOR, marker=:circle, markersize=4)
        plotted = true
    end
    if dm !== nothing && ncol(dm) >= 2
        # Existing MELT exports store runtime in microseconds.
        melt_times_ms = Float64.(dm[:, 2]) ./ 1000
        lines!(
            ax,
            dm[:, 1],
            melt_times_ms;
            color=MELT_COLOR,
            linestyle=:dash,
            linewidth=1.4,
            label=MELT_LABEL,
        )
        scatter!(ax, dm[:, 1], melt_times_ms; color=MELT_COLOR, marker=:diamond, markersize=4)
        plotted = true
    end

    return plotted
end

function plot_linearised_cumulant_errors_on_axis!(
    ax,
    cumulants_file;
    c1_label=L"c_1",
    c2_label=L"c_2",
)
    isfile(cumulants_file) || error("Missing linearised cumulant CSV: $cumulants_file")

    data = CSV.read(cumulants_file, DataFrame)
    dimensions = Float64.(data.hilbert_dimension)
    c1 = Float64.(data.cumulant_1)
    c2 = Float64.(data.cumulant_2)
    analytic_c1 = Float64.(data.analytic_cumulant_1)
    analytic_c2 = Float64.(data.analytic_cumulant_2)
    # Plot relative absolute errors against the analytical low-impedance
    # formulas; this is a convergence check, not another timing benchmark.
    c1_error = abs.(c1 .- analytic_c1) ./ abs.(analytic_c1)
    c2_error = abs.(c2 .- analytic_c2) ./ abs.(analytic_c2)

    lines!(ax, dimensions, c1_error; color=CUMULANT_1_COLOR, linewidth=1.4, label=c1_label)
    scatter!(ax, dimensions, c1_error; color=CUMULANT_1_COLOR, marker=:circle, markersize=4)
    lines!(ax, dimensions, c2_error; color=CUMULANT_2_COLOR, linewidth=1.4, label=c2_label)
    scatter!(ax, dimensions, c2_error; color=CUMULANT_2_COLOR, marker=:diamond, markersize=4)
    return ax
end

function plot_dense_highest_fock_population_scaling_on_axis!(
    ax,
    cumulants_file,
    population_file;
    c1_label=L"c_1",
    c2_label=L"c_2",
)
    isfile(cumulants_file) || error("Missing dense cumulant CSV: $cumulants_file")
    isfile(population_file) || error("Missing dense highest-Fock-population CSV: $population_file")

    cumulants = CSV.read(cumulants_file, DataFrame)
    populations = CSV.read(population_file, DataFrame)
    dimensions = Float64.(cumulants.hilbert_dimension)
    population_dimensions = Float64.(populations.hilbert_dimension)
    dimensions == population_dimensions || error("Dense cumulant and population CSV dimensions differ")

    highest_population = Float64.(populations.highest_joint_fock_population)
    c1 = Float64.(cumulants.cumulant_1)
    c2 = Float64.(cumulants.cumulant_2)

    # Scale the same cutoff-corner population by each cumulant magnitude so the
    # two convergence diagnostics are visible on one plot.
    scaled_by_c1 = highest_population ./ abs.(c1)
    scaled_by_c2 = highest_population ./ abs.(c2)

    lines!(ax, dimensions, scaled_by_c1; color=CUMULANT_1_COLOR, linewidth=1.4, label=c1_label)
    scatter!(ax, dimensions, scaled_by_c1; color=CUMULANT_1_COLOR, marker=:circle, markersize=4)
    lines!(ax, dimensions, scaled_by_c2; color=CUMULANT_2_COLOR, linewidth=1.4, label=c2_label)
    scatter!(ax, dimensions, scaled_by_c2; color=CUMULANT_2_COLOR, marker=:diamond, markersize=4)
    return ax
end

function plot_linearised_comparison(;
    julia_file=raw_data_path("benchmark_linearised_qhe_vs_dimension.csv"),
    melt_file=raw_data_path("benchmark_melt_linearised_vs_dimension.csv"),
    output_file=figure_path("benchmark_compare.png"),
)
    f = Figure(size=QUANTUM_FIGURE_SIZE, fontsize=8)
    ax = benchmark_axis(f)
    plotted = plot_runtime_comparison_on_axis!(ax, julia_file, melt_file)

    if plotted
        add_publication_legend!(ax; position=:lt)
    else
        text!(ax, "No valid CSVs found"; position=(0.5, 0.5), align=(:center, :center), fontsize=8)
    end

    style_publication_figure!(f)
    save_publication_figure(f, output_file)
    return output_file
end

function plot_linearised_cumulant_errors(;
    cumulants_file=raw_data_path("benchmark_linearised_qhe_cumulants_vs_dimension.csv"),
    output_file=figure_path("benchmark_linearised_qhe_cumulant_errors.png"),
)
    f = Figure(size=QUANTUM_FIGURE_SIZE, fontsize=8)
    ax = diagnostic_axis(f; ylabel=LINEARISED_ERROR_LABEL)
    plot_linearised_cumulant_errors_on_axis!(ax, cumulants_file)
    add_publication_legend!(ax; position=:rt)
    style_publication_figure!(f)
    save_publication_figure(f, output_file)
    return output_file
end

function plot_dense_comparison(;
    julia_file=raw_data_path("benchmark_dense_circuit_qhe_vs_dimension.csv"),
    melt_file=raw_data_path("benchmark_melt_dense_circuit_qhe_vs_dimension.csv"),
    output_file=figure_path("benchmark_dense_circuit_qhe_compare.png"),
)
    f = Figure(size=QUANTUM_FIGURE_SIZE, fontsize=8)
    ax = benchmark_axis(f)
    plot_runtime_comparison_on_axis!(ax, julia_file, melt_file; require_julia=true)
    add_publication_legend!(ax; position=:lt)

    style_publication_figure!(f)
    save_publication_figure(f, output_file)
    return output_file
end

function plot_dense_highest_fock_population_scaling(;
    cumulants_file=raw_data_path("benchmark_dense_circuit_qhe_cumulants_vs_dimension.csv"),
    population_file=raw_data_path("benchmark_dense_circuit_qhe_highest_fock_population_vs_dimension.csv"),
    output_file=figure_path("benchmark_dense_circuit_qhe_highest_fock_population_scaled.png"),
)
    f = Figure(size=QUANTUM_FIGURE_SIZE, fontsize=8)
    ax = diagnostic_axis(f; ylabel=DENSE_ERROR_LABEL, yscale=log10)
    plot_dense_highest_fock_population_scaling_on_axis!(ax, cumulants_file, population_file)
    add_publication_legend!(ax; position=:rt)
    style_publication_figure!(f)
    save_publication_figure(f, output_file)
    return output_file
end

function plot_combined_benchmark_summary(;
    linearised_julia_file=raw_data_path("benchmark_linearised_qhe_vs_dimension.csv"),
    linearised_melt_file=raw_data_path("benchmark_melt_linearised_vs_dimension.csv"),
    dense_julia_file=raw_data_path("benchmark_dense_circuit_qhe_vs_dimension.csv"),
    dense_melt_file=raw_data_path("benchmark_melt_dense_circuit_qhe_vs_dimension.csv"),
    linearised_cumulants_file=raw_data_path("benchmark_linearised_qhe_cumulants_vs_dimension.csv"),
    dense_cumulants_file=raw_data_path("benchmark_dense_circuit_qhe_cumulants_vs_dimension.csv"),
    dense_population_file=raw_data_path("benchmark_dense_circuit_qhe_highest_fock_population_vs_dimension.csv"),
    output_file=figure_path("benchmark_combined_summary.png"),
)
    # Arrange the four existing benchmark views in a wide two-column Quantum
    # layout so the runtime and convergence diagnostics can be read together.
    f = Figure(size=QUANTUM_WIDE_FIGURE_SIZE, fontsize=8)

    ax_linearised = benchmark_axis(f; row=1, col=1, xlabel="")
    hide_x_ticks!(ax_linearised)
    add_panel_label!(f, 1, 1, "(a)")
    plotted = plot_runtime_comparison_on_axis!(ax_linearised, linearised_julia_file, linearised_melt_file)
    plotted && add_publication_legend!(ax_linearised; position=:lt)

    ax_dense = benchmark_axis(f; row=1, col=2, xlabel="")
    hide_x_ticks!(ax_dense)
    add_panel_label!(f, 1, 2, "(b)")
    plot_runtime_comparison_on_axis!(ax_dense, dense_julia_file, dense_melt_file; require_julia=true)
    add_publication_legend!(ax_dense; position=:lt)

    ax_linearised_error = diagnostic_axis(
        f;
        row=2,
        col=1,
        ylabel=LINEARISED_DIFFERENCE_ERROR_LABEL,
    )
    add_panel_label!(f, 2, 1, "(c)")
    plot_linearised_cumulant_errors_on_axis!(
        ax_linearised_error,
        linearised_cumulants_file;
        c1_label=FIRST_CUMULANT_LABEL,
        c2_label=SECOND_CUMULANT_LABEL,
    )
    add_publication_legend!(ax_linearised_error; position=:rt)

    ax_dense_error = diagnostic_axis(
        f;
        row=2,
        col=2,
        ylabel=DENSE_ERROR_LABEL,
        yscale=log10,
    )
    add_panel_label!(f, 2, 2, "(d)")
    plot_dense_highest_fock_population_scaling_on_axis!(
        ax_dense_error,
        dense_cumulants_file,
        dense_population_file,
        c1_label=FIRST_CUMULANT_LABEL,
        c2_label=SECOND_CUMULANT_LABEL,
    )
    add_publication_legend!(ax_dense_error; position=:rt)

    style_publication_figure!(f)
    save_publication_figure(f, output_file)
    return output_file
end
