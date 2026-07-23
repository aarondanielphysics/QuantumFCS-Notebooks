# Publication figure functions for the QuantumFCS.jl paper applications.
#
# These are the exact plotting routines that produce the manuscript figures:
#   * jc_fcs_production_plot        -> Fig. 2 (driven-dissipative Jaynes-Cummings)
#   * qhe_two_sweep_regime_plot     -> Figs. 3 and 4 (circuit-QED heat engine)
#   * qhe_two_sweep_validation_plot -> the RWA/cutoff validation appendix figures
#
# Kept separate from src/plots.jl (the benchmark plotting) so the two never
# interfere. Ported from the research repository; the LU-vs-iterative runtime
# comparison plots were dropped along with the study that fed them.
using CairoMakie
using LaTeXStrings

@isdefined(QHE_DEFAULT_SIZE) || const QHE_DEFAULT_SIZE = (980, 760)
@isdefined(QHE_DEFAULT_FONTSIZE) || const QHE_DEFAULT_FONTSIZE = 22
@isdefined(QHE_DEFAULT_LINEWIDTH) || const QHE_DEFAULT_LINEWIDTH = 2.5

latexify_label(label) = label isa LaTeXString ? label : LaTeXString(string(label))

function qhe_column(data, column)
    column isa Symbol && return getproperty(data, column)
    return column
end

function qhe_first_column(data, columns::Symbol...)
    names = propertynames(data)
    for column in columns
        column in names && return getproperty(data, column)
    end
    error("None of the requested columns exist: $(columns)")
end

function hide_linked_xdecorations!(axes; keep_last=true)
    keep_last || return axes
    for ax in axes[1:end-1]
        hidexdecorations!(ax; grid=false)
    end
    return axes
end

function set_grid_visibility!(axes; xgridvisible=true, ygridvisible=true)
    for ax in axes
        ax.xgridvisible = xgridvisible
        ax.ygridvisible = ygridvisible
    end
    return axes
end

"""
    qhe_transport_statistics_plot(data, xcol; kwargs...) -> Figure

Plot the nonlinear QHE transport observables used in the focused antibunching
and TUR-saturation analyses. The required columns are `Jh`, `Jc`, `Omega_h`,
`Omega_c`, `W_out`, `k`, `l`, `Fh`, `Fc`, `Fh_TUR_threshold`,
`Fc_TUR_threshold`, `Qh`, and `Qc`. Hot and cold Fano factors are kept on
separate axes so one channel cannot visually compress the other.
"""
function qhe_transport_statistics_plot(data, xcol;
        xlabel=L"x",
        current_ylabel="Averages",
        hot_fano_ylabel=L"\mathcal{F}_h",
        cold_fano_ylabel=L"\mathcal{F}_c",
        tur_ylabel=L"\mathcal{Q}",
        title="",
        figure_size=(980, 940),
        fontsize=QHE_DEFAULT_FONTSIZE,
        linewidth=QHE_DEFAULT_LINEWIDTH,
        current_labels=(L"J_h/\Omega_h", L"J_c/\Omega_c", L"P/(\Omega_h-2\Omega_c)"),
        hot_fano_labels=(L"\mathcal{F}_h", L"2/(l \mathcal{A})", L"\mathrm{Poisson}"),
        cold_fano_labels=(L"\mathcal{F}_c", L"2/(k \mathcal{A})", L"\mathrm{Poisson}"),
        tur_labels=(L"\mathcal{Q}_h", L"\mathcal{Q}_c", L"\mathrm{TUR\ bound}"),
        current_linestyles=(:solid, :solid, :solid),
        fano_linestyles=(:solid, :dash, :dot),
        tur_linestyles=(:solid, :solid, :dash),
        colors=(Makie.wong_colors()[1], Makie.wong_colors()[2], Makie.wong_colors()[3]),
        threshold_color=:gray35,
        tur_bound_color=:black,
        poisson_color=:gray35,
        fano_yscale=log10,
        legend_position=:rb,
        link_xaxis=true,
        hide_inner_xticks=true,
        xgridvisible=true,
        ygridvisible=true,
        axis_kwargs=(;),
        legend_kwargs=(;))
    x = qhe_column(data, xcol)
    fig = Figure(; size=figure_size, fontsize)

    ax1 = Axis(fig[1, 1];
        xlabel=latexify_label(xlabel),
        ylabel=latexify_label(current_ylabel),
        title,
        axis_kwargs...)
    Omega_h = qhe_first_column(data, :Omega_h, :Ωh)
    Omega_c = qhe_first_column(data, :Omega_c, :Ωc)
    k = qhe_first_column(data, :k, :lh)
    l = qhe_first_column(data, :l, :lc)
    W_out = qhe_first_column(data, :W_out, :P)

    scatterlines!(ax1, x, data.Jh ./ Omega_h;
        label=latexify_label(current_labels[1]), linewidth, color=colors[1],
        linestyle=current_linestyles[1])
    scatterlines!(ax1, x, data.Jc ./ Omega_c;
        label=latexify_label(current_labels[2]), linewidth, color=colors[2],
        linestyle=current_linestyles[2])
    work_scale = abs.(k .* Omega_h .- l .* Omega_c)
    scatterlines!(ax1, x, W_out ./ work_scale;
        label=latexify_label(current_labels[3]), linewidth, color=colors[3],
        linestyle=current_linestyles[3])
    axislegend(ax1; position=legend_position, legend_kwargs...)

    ax2 = Axis(fig[2, 1];
        xlabel=latexify_label(xlabel),
        ylabel=latexify_label(hot_fano_ylabel),
        yscale=fano_yscale,
        axis_kwargs...)
    scatterlines!(ax2, x, data.Fh;
        label=latexify_label(hot_fano_labels[1]), linewidth, color=colors[1],
        linestyle=fano_linestyles[1])
    qhe_reference_lines!(ax2, x, data.Fh_TUR_threshold;
        label=latexify_label(hot_fano_labels[2]), linewidth, color=colors[1],
        linestyle=fano_linestyles[2])
    # hlines!(ax2, [1.0];
    #     label=latexify_label(hot_fano_labels[3]), linewidth, color=poisson_color,
    #     linestyle=fano_linestyles[3])
    axislegend(ax2; position=legend_position, legend_kwargs...)

    ax3 = Axis(fig[3, 1];
        xlabel=latexify_label(xlabel),
        ylabel=latexify_label(cold_fano_ylabel),
        yscale=fano_yscale,
        axis_kwargs...)
    scatterlines!(ax3, x, data.Fc;
        label=latexify_label(cold_fano_labels[1]), linewidth, color=colors[2],
        linestyle=fano_linestyles[1])
    qhe_reference_lines!(ax3, x, data.Fc_TUR_threshold;
        label=latexify_label(cold_fano_labels[2]), linewidth, color=colors[2],
        linestyle=fano_linestyles[2])
    # hlines!(ax3, [1.0];
    #     label=latexify_label(cold_fano_labels[3]), linewidth, color=poisson_color,
    #     linestyle=fano_linestyles[3])
    axislegend(ax3; position=legend_position, legend_kwargs...)

    ax4 = Axis(fig[4, 1];
        xlabel=latexify_label(xlabel),
        ylabel=latexify_label(tur_ylabel),
        axis_kwargs...)
    scatterlines!(ax4, x, data.Qh;
        label=latexify_label(tur_labels[1]), linewidth, color=colors[1],
        linestyle=tur_linestyles[1])
    scatterlines!(ax4, x, data.Qc;
        label=latexify_label(tur_labels[2]), linewidth, color=colors[2],
        linestyle=tur_linestyles[2])
    hlines!(ax4, [2.0];
        label=latexify_label(tur_labels[3]), linewidth, color=tur_bound_color,
        linestyle=tur_linestyles[3])
    axislegend(ax4; position=legend_position, legend_kwargs...)

    axes = (ax1, ax2, ax3, ax4)
    link_xaxis && linkxaxes!(axes...)
    hide_inner_xticks && hide_linked_xdecorations!(collect(axes))
    set_grid_visibility!(axes; xgridvisible, ygridvisible)
    return fig
end

"""
    qhe_model_validation_plot(data, xcol; kwargs...) -> Figure

	Plot the diagnostic quantities used to check RWA and truncation assumptions.
	The required columns are `epsilon_off`, `hot_tail`, `cold_tail`,
	`hot_occupation_fraction`, `cold_occupation_fraction`, and
	`tight_coupling_error`.
	"""
	function qhe_model_validation_plot(data, xcol;
	        xlabel=L"x",
	        epsilon_ylabel=L"\epsilon_\mathrm{off}",
	        tail_ylabel=L"p_{N-1}+p_N",
	        occupation_ylabel=L"\langle n\rangle/N",
	        tight_coupling_ylabel=L"\mathrm{tight\ coupling\ error}",
	        title="",
	        figure_size=(980, 920),
	        fontsize=QHE_DEFAULT_FONTSIZE,
	        linewidth=QHE_DEFAULT_LINEWIDTH,
	        labels=(L"\epsilon_\mathrm{off}", L"\mathrm{hot\ tail}", L"\mathrm{cold\ tail}",
	            L"\mathrm{cutoff}", L"\mathrm{hot\ occupation}",
	            L"\mathrm{cold\ occupation}", L"\mathrm{occupation\ cap}",
	            L"\mathrm{tight\ coupling\ error}"),
	        epsilon_tol=0.05,
	        tail_tol=1e-3,
	        occupation_tol=0.5,
	        colors=(Makie.wong_colors()[1], Makie.wong_colors()[2], Makie.wong_colors()[3]),
	        threshold_color=:black,
	        linestyles=(:solid, :solid, :solid, :dash),
	        tail_yscale=log10,
        tight_coupling_yscale=log10,
        legend_position=:rt,
        link_xaxis=true,
        hide_inner_xticks=true,
        xgridvisible=true,
        ygridvisible=true,
        axis_kwargs=(;),
        legend_kwargs=(;))
    x = qhe_column(data, xcol)
    fig = Figure(; size=figure_size, fontsize)

    ax1 = Axis(fig[1, 1];
        xlabel=latexify_label(xlabel),
        ylabel=latexify_label(epsilon_ylabel),
        title,
        axis_kwargs...)
    scatterlines!(ax1, x, data.epsilon_off;
        label=latexify_label(labels[1]), linewidth, color=colors[1],
        linestyle=linestyles[1])
    hlines!(ax1, [epsilon_tol];
        color=threshold_color, linewidth, linestyle=linestyles[4],
        label=latexify_label(labels[4]))
    axislegend(ax1; position=legend_position, legend_kwargs...)

    ax2 = Axis(fig[2, 1];
        xlabel=latexify_label(xlabel),
        ylabel=latexify_label(tail_ylabel),
        yscale=tail_yscale,
        axis_kwargs...)
    scatterlines!(ax2, x, data.hot_tail;
        label=latexify_label(labels[2]), linewidth, color=colors[2],
        linestyle=linestyles[2])
    scatterlines!(ax2, x, data.cold_tail;
        label=latexify_label(labels[3]), linewidth, color=colors[3],
        linestyle=linestyles[3])
	    hlines!(ax2, [tail_tol];
	        color=threshold_color, linewidth, linestyle=linestyles[4],
	        label=latexify_label(labels[4]))
	    axislegend(ax2; position=legend_position, legend_kwargs...)

	    ax3 = Axis(fig[3, 1];
	        xlabel=latexify_label(xlabel),
	        ylabel=latexify_label(occupation_ylabel),
	        axis_kwargs...)
	    scatterlines!(ax3, x, data.hot_occupation_fraction;
	        label=latexify_label(labels[5]), linewidth, color=colors[2],
	        linestyle=linestyles[2])
	    scatterlines!(ax3, x, data.cold_occupation_fraction;
	        label=latexify_label(labels[6]), linewidth, color=colors[3],
	        linestyle=linestyles[3])
	    hlines!(ax3, [occupation_tol];
	        color=threshold_color, linewidth, linestyle=linestyles[4],
	        label=latexify_label(labels[7]))
	    axislegend(ax3; position=legend_position, legend_kwargs...)

	    ax4 = Axis(fig[4, 1];
	        xlabel=latexify_label(xlabel),
	        ylabel=latexify_label(tight_coupling_ylabel),
	        yscale=tight_coupling_yscale,
	        axis_kwargs...)
	    scatterlines!(ax4, x, data.tight_coupling_error;
	        label=latexify_label(labels[8]), linewidth, color=colors[1],
	        linestyle=linestyles[1])

	    axes = (ax1, ax2, ax3, ax4)
    link_xaxis && linkxaxes!(axes...)
    hide_inner_xticks && hide_linked_xdecorations!(collect(axes))
    set_grid_visibility!(axes; xgridvisible, ygridvisible)
    return fig
end

"""
    qhe_rwa_coherence_plot(data, xcol; kwargs...) -> Figure

Plot the RWA-link coherence diagnostic `C_RWA` returned by `qhe_point`.
"""
function qhe_rwa_coherence_plot(data, xcol;
        xlabel=L"x",
        ylabel=L"\mathcal{C}_\mathrm{RWA}",
        title="",
        figure_size=(900, 420),
        fontsize=QHE_DEFAULT_FONTSIZE,
        linewidth=QHE_DEFAULT_LINEWIDTH,
        label=L"\mathcal{C}_\mathrm{RWA}",
        color=Makie.wong_colors()[4],
        linestyle=:solid,
        legend_position=:rt,
        xgridvisible=true,
        ygridvisible=true,
        axis_kwargs=(;),
        legend_kwargs=(;))
    x = qhe_column(data, xcol)
    fig = Figure(; size=figure_size, fontsize)
    ax = Axis(fig[1, 1];
        xlabel=latexify_label(xlabel),
        ylabel=latexify_label(ylabel),
        title,
        axis_kwargs...)
    scatterlines!(ax, x, data.CRWA; label=latexify_label(label), linewidth, color, linestyle)
    axislegend(ax; position=legend_position, legend_kwargs...)
    set_grid_visibility!((ax,); xgridvisible, ygridvisible)
    return fig
end

"""
    qhe_focused_tur_regime_plot(data, xcol; kwargs...) -> Figure

Plot the observables used to assess the finite-affinity quantum-TUR regime:
scaled heat currents and power, RWA-link coherence, separated hot/cold Fano
factors with their TUR thresholds, and the thermodynamic uncertainty products.
"""
function qhe_focused_tur_regime_plot(data, xcol;
        xlabel=L"x",
        current_ylabel="Averages",
        coherence_ylabel=L"\mathcal{C}_\mathrm{RWA}",
        hot_fano_ylabel=L"\mathcal{F}_h",
        cold_fano_ylabel=L"\mathcal{F}_c",
        tur_ylabel=L"\mathcal{Q}",
        title="",
        figure_size=(980, 1120),
        fontsize=QHE_DEFAULT_FONTSIZE,
        linewidth=QHE_DEFAULT_LINEWIDTH,
        current_labels=(L"J_h/\Omega_h", L"J_c/\Omega_c", L"P/(\Omega_h-2\Omega_c)"),
        coherence_label=L"\mathcal{C}_\mathrm{RWA}",
        hot_fano_labels=(L"F_h", L"2/(l\mathcal{A})", L"\mathrm{Poisson}"),
        cold_fano_labels=(L"F_c", L"2/(k\mathcal{A})", L"\mathrm{Poisson}"),
        tur_labels=(L"\mathcal{Q}_h", L"\mathcal{Q}_c", L"\mathrm{TUR\ bound}"),
        current_linestyles=(:solid, :solid, :solid),
        coherence_linestyle=:solid,
        fano_linestyles=(:solid, :dash, :dot),
        tur_linestyles=(:solid, :solid, :dash),
        colors=(Makie.wong_colors()[1], Makie.wong_colors()[2], Makie.wong_colors()[3],
            Makie.wong_colors()[4]),
        threshold_color=:gray35,
        tur_bound_color=:black,
        poisson_color=:gray35,
        fano_yscale=log10,
        legend_position=:rb,
        link_xaxis=true,
        hide_inner_xticks=true,
        xgridvisible=true,
        ygridvisible=true,
        axis_kwargs=(;),
        legend_kwargs=(;))
    x = qhe_column(data, xcol)
    fig = Figure(; size=figure_size, fontsize)

    Omega_h = qhe_first_column(data, :Omega_h, :Ωh)
    Omega_c = qhe_first_column(data, :Omega_c, :Ωc)
    k = qhe_first_column(data, :k, :lh)
    l = qhe_first_column(data, :l, :lc)
    W_out = qhe_first_column(data, :W_out, :P)

    ax1 = Axis(fig[1, 1];
        xlabel=latexify_label(xlabel),
        ylabel=latexify_label(current_ylabel),
        title,
        axis_kwargs...)
    scatterlines!(ax1, x, data.Jh ./ Omega_h;
        label=latexify_label(current_labels[1]), linewidth, color=colors[1],
        linestyle=current_linestyles[1])
    scatterlines!(ax1, x, data.Jc ./ Omega_c;
        label=latexify_label(current_labels[2]), linewidth, color=colors[2],
        linestyle=current_linestyles[2])
    work_scale = abs.(k .* Omega_h .- l .* Omega_c)
    scatterlines!(ax1, x, W_out ./ work_scale;
        label=latexify_label(current_labels[3]), linewidth, color=colors[3],
        linestyle=current_linestyles[3])
    axislegend(ax1; position=legend_position, legend_kwargs...)

    ax2 = Axis(fig[2, 1];
        xlabel=latexify_label(xlabel),
        ylabel=latexify_label(hot_fano_ylabel),
        yscale=fano_yscale,
        axis_kwargs...)
    scatterlines!(ax2, x, data.Fh;
        label=latexify_label(hot_fano_labels[1]), linewidth, color=colors[1],
        linestyle=fano_linestyles[1])
    qhe_reference_lines!(ax2, x, data.Fh_TUR_threshold;
        label=latexify_label(hot_fano_labels[2]), linewidth, color=colors[1],
        linestyle=fano_linestyles[2])
    # hlines!(ax2, [1.0];
    #     label=latexify_label(hot_fano_labels[3]), linewidth, color=poisson_color,
    #     linestyle=fano_linestyles[3])
    axislegend(ax2; position=legend_position, legend_kwargs...)

    ax3 = Axis(fig[3, 1];
        xlabel=latexify_label(xlabel),
        ylabel=latexify_label(cold_fano_ylabel),
        yscale=fano_yscale,
        axis_kwargs...)
    scatterlines!(ax3, x, data.Fc;
        label=latexify_label(cold_fano_labels[1]), linewidth, color=colors[2],
        linestyle=fano_linestyles[1])
    qhe_reference_lines!(ax3, x, data.Fc_TUR_threshold;
        label=latexify_label(cold_fano_labels[2]), linewidth, color=colors[2],
        linestyle=fano_linestyles[2])
    # hlines!(ax3, [1.0];
    #     label=latexify_label(cold_fano_labels[3]), linewidth, color=poisson_color,
    #     linestyle=fano_linestyles[3])
    axislegend(ax3; position=legend_position, legend_kwargs...)

    ax4 = Axis(fig[4, 1];
        xlabel=latexify_label(xlabel),
        ylabel=latexify_label(tur_ylabel),
        axis_kwargs...)
    scatterlines!(ax4, x, data.Qh;
        label=latexify_label(tur_labels[1]), linewidth, color=colors[1],
        linestyle=tur_linestyles[1])
    scatterlines!(ax4, x, data.Qc;
        label=latexify_label(tur_labels[2]), linewidth, color=colors[2],
        linestyle=tur_linestyles[2])
    hlines!(ax4, [2.0];
        label=latexify_label(tur_labels[3]), linewidth, color=tur_bound_color,
        linestyle=tur_linestyles[3])
    axislegend(ax4; position=legend_position, legend_kwargs...)

    ax5 = Axis(fig[5, 1];
        xlabel=latexify_label(xlabel),
        ylabel=latexify_label(coherence_ylabel),
        axis_kwargs...)
    scatterlines!(ax5, x, data.CRWA;
        label=latexify_label(coherence_label), linewidth, color=colors[4],
        linestyle=coherence_linestyle)
    axislegend(ax5; position=legend_position, legend_kwargs...)

    axes = (ax1, ax2, ax3, ax4, ax5)
    link_xaxis && linkxaxes!(axes...)
    hide_inner_xticks && hide_linked_xdecorations!(collect(axes))
    set_grid_visibility!(axes; xgridvisible, ygridvisible)
    return fig
end

"""
    qhe_tur_paper_plot(data, xcol; kwargs...) -> Figure

Paper-oriented finite-affinity TUR plot. The layout keeps the directly
measured hot and cold Fano factors on separate linked panels, avoiding the
visual compression caused by plotting both channels on one y axis. The TUR
criterion is shown as the channel-specific Fano threshold in each panel, plus
the direct `Q_h`/`Q_c` TUR-product panel.
"""
function qhe_tur_paper_plot(data, xcol;
        xlabel=L"x",
        current_ylabel="Averages",
        hot_fano_ylabel=L"\mathcal{F}_h",
        cold_fano_ylabel=L"\mathcal{F}_c",
        tur_ylabel=L"\mathcal{Q}",
        coherence_ylabel=L"\mathcal{C}_\mathrm{RWA}",
        title="",
        figure_size=(980, 1120),
        fontsize=QHE_DEFAULT_FONTSIZE,
        linewidth=QHE_DEFAULT_LINEWIDTH,
        current_labels=(L"J_h/\Omega_h", L"J_c/\Omega_c", L"P/(\Omega_h-2\Omega_c)"),
        hot_fano_labels=(L"\mathcal{F}_h", L"\mathcal{F}^{\mathrm{TUR}}_h", L"\mathrm{Poisson}"),
        cold_fano_labels=(L"\mathcal{F}_c", L"\mathcal{F}^{\mathrm{TUR}}_c", L"\mathrm{Poisson}"),
        tur_labels=(L"\mathcal{Q}_h", L"\mathcal{Q}_c", L"\mathrm{TUR\ bound}"),
        current_linestyles=(:solid, :solid, :solid),
        fano_linestyles=(:solid, :dash, :dot),
        tur_linestyles=(:solid, :solid, :dash),
        coherence_linestyle=:solid,
        colors=(Makie.wong_colors()[1], Makie.wong_colors()[2], Makie.wong_colors()[3],
            Makie.wong_colors()[4]),
        threshold_color=:gray35,
        poisson_color=:gray35,
        hot_fano_yscale=identity,
        cold_fano_yscale=identity,
        coherence_color=Makie.wong_colors()[4],
        legend_position=:rb,
        link_xaxis=true,
        hide_inner_xticks=true,
        xgridvisible=true,
        ygridvisible=true,
        axis_kwargs=(;),
        legend_kwargs=(;))
    x = qhe_column(data, xcol)
    fig = Figure(; size=figure_size, fontsize)

    Omega_h = qhe_first_column(data, :Omega_h, :Ωh)
    Omega_c = qhe_first_column(data, :Omega_c, :Ωc)
    k = qhe_first_column(data, :k, :lh)
    l = qhe_first_column(data, :l, :lc)
    W_out = qhe_first_column(data, :W_out, :P)

    ax1 = Axis(fig[1, 1];
        xlabel=latexify_label(xlabel),
        ylabel=latexify_label(current_ylabel),
        title,
        axis_kwargs...)
    scatterlines!(ax1, x, data.Jh ./ Omega_h;
        label=latexify_label(current_labels[1]), linewidth, color=colors[1],
        linestyle=current_linestyles[1])
    scatterlines!(ax1, x, data.Jc ./ Omega_c;
        label=latexify_label(current_labels[2]), linewidth, color=colors[2],
        linestyle=current_linestyles[2])
    work_scale = abs.(k .* Omega_h .- l .* Omega_c)
    scatterlines!(ax1, x, W_out ./ work_scale;
        label=latexify_label(current_labels[3]), linewidth, color=colors[3],
        linestyle=current_linestyles[3])
    axislegend(ax1; position=legend_position, legend_kwargs...)

    ax2 = Axis(fig[2, 1];
        xlabel=latexify_label(xlabel),
        ylabel=latexify_label(hot_fano_ylabel),
        yscale=hot_fano_yscale,
        axis_kwargs...)
    scatterlines!(ax2, x, data.Fh;
        label=latexify_label(hot_fano_labels[1]), linewidth, color=colors[1],
        linestyle=fano_linestyles[1])
    qhe_reference_lines!(ax2, x, data.Fh_TUR_threshold;
        label=latexify_label(hot_fano_labels[2]), linewidth, color=threshold_color,
        linestyle=fano_linestyles[2])
    # hlines!(ax2, [1.0];
    #     label=latexify_label(hot_fano_labels[3]), linewidth, color=poisson_color,
    #     linestyle=fano_linestyles[3])
    axislegend(ax2; position=legend_position, legend_kwargs...)

    ax3 = Axis(fig[3, 1];
        xlabel=latexify_label(xlabel),
        ylabel=latexify_label(cold_fano_ylabel),
        yscale=cold_fano_yscale,
        axis_kwargs...)
    scatterlines!(ax3, x, data.Fc;
        label=latexify_label(cold_fano_labels[1]), linewidth, color=colors[2],
        linestyle=fano_linestyles[1])
    qhe_reference_lines!(ax3, x, data.Fc_TUR_threshold;
        label=latexify_label(cold_fano_labels[2]), linewidth, color=threshold_color,
        linestyle=fano_linestyles[2])
    # hlines!(ax3, [1.0];
    #     label=latexify_label(cold_fano_labels[3]), linewidth, color=poisson_color,
    #     linestyle=fano_linestyles[3])
    axislegend(ax3; position=legend_position, legend_kwargs...)

    ax4 = Axis(fig[4, 1];
        xlabel=latexify_label(xlabel),
        ylabel=latexify_label(tur_ylabel),
        axis_kwargs...)
    scatterlines!(ax4, x, data.Qh;
        label=latexify_label(tur_labels[1]), linewidth, color=colors[1],
        linestyle=tur_linestyles[1])
    scatterlines!(ax4, x, data.Qc;
        label=latexify_label(tur_labels[2]), linewidth, color=colors[2],
        linestyle=tur_linestyles[2])
    hlines!(ax4, [2.0];
        label=latexify_label(tur_labels[3]), linewidth, color=threshold_color,
        linestyle=tur_linestyles[3])
    axislegend(ax4; position=legend_position, legend_kwargs...)

    ax5 = Axis(fig[5, 1];
        xlabel=latexify_label(xlabel),
        ylabel=latexify_label(coherence_ylabel),
        axis_kwargs...)
    scatterlines!(ax5, x, data.CRWA;
        linewidth, color=coherence_color, linestyle=coherence_linestyle)

    axes = (ax1, ax2, ax3, ax4, ax5)
    link_xaxis && linkxaxes!(axes...)
    hide_inner_xticks && hide_linked_xdecorations!(collect(axes))
    set_grid_visibility!(axes; xgridvisible, ygridvisible)
    return fig
end

function qhe_apply_axis_limits!(ax; xlimits=nothing, ylimits=nothing)
    xlimits === nothing || xlims!(ax, xlimits...)
    ylimits === nothing || ylims!(ax, ylimits...)
    return ax
end

function qhe_panel_label!(ax, label; fontsize=18, color=:black, offset=(6, -6))
    label_text = string(label)
    isempty(label_text) && return ax
    text!(ax, 0.02, 0.96;
        text=label_text,
        space=:relative,
        align=(:left, :top),
        offset,
        fontsize,
        font=:bold,
        color)
    return ax
end

function qhe_should_show_legend(legend_columns, col)
    legend_columns === :both && return true
    legend_columns === :left && return col == 1
    legend_columns === :right && return col == 2
    return false
end

function qhe_panel_legend_position(position, col, fallback)
    position === nothing && return fallback
    if position isa Tuple && length(position) == 2 && all(p -> p isa Symbol, position)
        return position[col]
    end
    return position
end

function qhe_positive_floor(values; floor=1e-16)
    return ifelse.(values .> floor, values, floor)
end

function qhe_reference_lines!(ax, x, y; label=nothing, linewidth=QHE_DEFAULT_LINEWIDTH,
        color=:black, linestyle=:dash)
    return lines!(ax, x, y; label, linewidth, color, linestyle)
end

function qhe_bad_point_mask(data)
    n = length(data[!, propertynames(data)[1]])
    mask = falses(n)
    for column in (:cutoff_ok, :solver_ok, :fcs_ok)
        if column in propertynames(data)
            mask .|= .!Bool.(data[!, column])
        end
    end
    for column in (:hot_fcs_retry, :cold_fcs_retry)
        if column in propertynames(data)
            mask .|= Bool.(data[!, column])
        end
    end
    return mask
end

function qhe_ring_bad_points!(ax, x, y, mask;
        color=:red, marker=:xcross, markersize=10, strokewidth=1.8)
    any(mask) || return ax
    scatter!(ax, x[mask], y[mask];
        color, marker, markersize, strokewidth)
    return ax
end

"""
    qhe_two_sweep_regime_plot(g_data, lambda_data; kwargs...) -> Figure

Paper-oriented Section 5 figure for one nonlinear-QHE regime. The figure keeps
the physical observables in the main manuscript panels; numerical reliability is
handled by `qhe_two_sweep_validation_plot` for appendix material.
"""
function qhe_two_sweep_regime_plot(g_data, lambda_data;
        g_xcol=:g,
        lambda_xcol=:λc,
        g_xlabel=L"g",
        lambda_xlabel=L"\lambda_c",
        column_labels=nothing,
        panel_labels=("a", "b", "c", "d", "e", "f", "g", "h"),
        current_ylabel="Averages",
        hot_fano_ylabel=L"\mathcal{F}_h",
        cold_fano_ylabel=L"\mathcal{F}_c",
        tur_ylabel=L"\mathcal{Q}",
        coherence_ylabel=L"\mathcal{C}_\mathrm{RWA}",
        figure_size=(1650, 1180),
        fontsize=QHE_DEFAULT_FONTSIZE,
        panel_label_fontsize=18,
        linewidth=QHE_DEFAULT_LINEWIDTH,
        row_heights=(1.0, 1.0, 1.0, 0.75),
        current_labels=(L"\langle J_h\rangle/\Omega_h",
            L"\langle J_c\rangle/\Omega_c",
            L"\langle P\rangle/(\Omega_h-2\Omega_c)"),
        hot_fano_labels=(L"\mathcal{F}_h", L"\mathcal{F}^{\mathrm{TUR}}_h",
            L"\mathrm{Poisson}"),
        cold_fano_labels=(L"\mathcal{F}_c", L"\mathcal{F}^{\mathrm{TUR}}_c",
            L"\mathrm{Poisson}"),
        tur_labels=(L"\mathcal{Q}_h", L"\mathcal{Q}_c", L"\mathrm{TUR\ bound}"),
        current_linestyles=(:solid, :solid, :solid),
        fano_linestyles=(:solid, :dash, :dot),
        tur_linestyles=(:solid, :solid, :dash),
        coherence_linestyle=:solid,
        hot_color=Makie.wong_colors()[1],
        cold_color=Makie.wong_colors()[2],
        power_color=Makie.wong_colors()[3],
        coherence_color=Makie.wong_colors()[4],
        threshold_color=:black,
        poisson_color=:gray35,
        bad_point_color=:red,
        hot_fano_yscale=identity,
        cold_fano_yscale=identity,
        show_poisson=false,
        mark_bad_points=false,
        legend_columns=:left,
        legend_position=:rb,
        current_legend_position=nothing,
        hot_fano_legend_position=nothing,
        cold_fano_legend_position=nothing,
        link_xaxis=true,
        hide_inner_xticks=true,
        repeat_ylabels=false,
        g_xlimits=nothing,
        lambda_xlimits=nothing,
        current_ylimits=nothing,
        hot_fano_ylimits=nothing,
        cold_fano_ylimits=nothing,
        tur_ylimits=nothing,
        coherence_ylimits=nothing,
        link_row_yaxes=false,
        xgridvisible=true,
        ygridvisible=true,
        xlabelsize=nothing,
        ylabelsize=nothing,
        xticklabelsize=nothing,
        yticklabelsize=nothing,
        axis_kwargs=(;),
        legend_kwargs=(; framevisible=false))
    _ = (tur_ylabel, tur_labels, tur_linestyles, threshold_color, tur_ylimits)
    fig = Figure(; size=figure_size, fontsize)
    axis_options = axis_kwargs
    xlabelsize === nothing || (axis_options = merge(axis_options, (; xlabelsize)))
    ylabelsize === nothing || (axis_options = merge(axis_options, (; ylabelsize)))
    xticklabelsize === nothing || (axis_options = merge(axis_options, (; xticklabelsize)))
    yticklabelsize === nothing || (axis_options = merge(axis_options, (; yticklabelsize)))

    datasets = (
        (data=g_data, xcol=g_xcol, xlabel=g_xlabel, xlimits=g_xlimits),
        (data=lambda_data, xcol=lambda_xcol, xlabel=lambda_xlabel, xlimits=lambda_xlimits),
    )
    axes = Matrix{Any}(undef, 4, 2)
    panel_labels = collect(panel_labels)

    for (col, sweep) in enumerate(datasets)
        data = sweep.data
        x = qhe_column(data, sweep.xcol)
        Omega_h = qhe_first_column(data, :Omega_h, :Ωh)
        Omega_c = qhe_first_column(data, :Omega_c, :Ωc)
        k = qhe_first_column(data, :k, :lh)
        l = qhe_first_column(data, :l, :lc)
        W_out = qhe_first_column(data, :W_out, :P)
        bad_mask = mark_bad_points ? qhe_bad_point_mask(data) : nothing
        ylabel_visible = repeat_ylabels || col == 1
        show_legend = qhe_should_show_legend(legend_columns, col)
        current_position = qhe_panel_legend_position(current_legend_position, col,
            legend_position)
        hot_fano_position = qhe_panel_legend_position(hot_fano_legend_position, col,
            legend_position)
        cold_fano_position = qhe_panel_legend_position(cold_fano_legend_position, col,
            legend_position)

        ax1 = Axis(fig[1, col];
            xlabel=latexify_label(sweep.xlabel),
            ylabel=ylabel_visible ? latexify_label(current_ylabel) : "",
            axis_options...)
        scatterlines!(ax1, x, data.Jh ./ Omega_h;
            label=latexify_label(current_labels[1]), linewidth, color=hot_color,
            linestyle=current_linestyles[1])
        scatterlines!(ax1, x, data.Jc ./ Omega_c;
            label=latexify_label(current_labels[2]), linewidth, color=cold_color,
            linestyle=current_linestyles[2])
        work_scale = abs.(k .* Omega_h .- l .* Omega_c)
        scatterlines!(ax1, x, W_out ./ work_scale;
            label=latexify_label(current_labels[3]), linewidth, color=power_color,
            linestyle=current_linestyles[3])
        show_legend && axislegend(ax1; position=current_position, legend_kwargs...)

        ax2 = Axis(fig[2, col];
            xlabel=latexify_label(sweep.xlabel),
            ylabel=ylabel_visible ? latexify_label(hot_fano_ylabel) : "",
            yscale=hot_fano_yscale,
            axis_options...)
        scatterlines!(ax2, x, data.Fh;
            label=latexify_label(hot_fano_labels[1]), linewidth, color=hot_color,
            linestyle=fano_linestyles[1])
        qhe_reference_lines!(ax2, x, data.Fh_TUR_threshold;
            label=latexify_label(hot_fano_labels[2]), linewidth, color=hot_color,
            linestyle=fano_linestyles[2])
        show_poisson && hlines!(ax2, [1.0];
            label=latexify_label(hot_fano_labels[3]), linewidth, color=poisson_color,
            linestyle=fano_linestyles[3])
        show_legend && axislegend(ax2; position=hot_fano_position, legend_kwargs...)

        ax3 = Axis(fig[3, col];
            xlabel=latexify_label(sweep.xlabel),
            ylabel=ylabel_visible ? latexify_label(cold_fano_ylabel) : "",
            yscale=cold_fano_yscale,
            axis_options...)
        scatterlines!(ax3, x, data.Fc;
            label=latexify_label(cold_fano_labels[1]), linewidth, color=cold_color,
            linestyle=fano_linestyles[1])
        qhe_reference_lines!(ax3, x, data.Fc_TUR_threshold;
            label=latexify_label(cold_fano_labels[2]), linewidth, color=cold_color,
            linestyle=fano_linestyles[2])
        show_poisson && hlines!(ax3, [1.0];
            label=latexify_label(cold_fano_labels[3]), linewidth, color=poisson_color,
            linestyle=fano_linestyles[3])
        show_legend && axislegend(ax3; position=cold_fano_position, legend_kwargs...)

        ax4 = Axis(fig[4, col];
            xlabel=latexify_label(sweep.xlabel),
            ylabel=ylabel_visible ? latexify_label(coherence_ylabel) : "",
            axis_options...)
        scatterlines!(ax4, x, data.CRWA;
            linewidth, color=coherence_color, linestyle=coherence_linestyle)

        if mark_bad_points
            qhe_ring_bad_points!(ax1, x, data.Jh ./ Omega_h, bad_mask; color=bad_point_color)
            qhe_ring_bad_points!(ax2, x, data.Fh, bad_mask; color=bad_point_color)
            qhe_ring_bad_points!(ax3, x, data.Fc, bad_mask; color=bad_point_color)
            qhe_ring_bad_points!(ax4, x, data.CRWA, bad_mask; color=bad_point_color)
        end

        axes[:, col] .= (ax1, ax2, ax3, ax4)
        qhe_apply_axis_limits!(ax1; xlimits=sweep.xlimits, ylimits=current_ylimits)
        qhe_apply_axis_limits!(ax2; xlimits=sweep.xlimits, ylimits=hot_fano_ylimits)
        qhe_apply_axis_limits!(ax3; xlimits=sweep.xlimits, ylimits=cold_fano_ylimits)
        qhe_apply_axis_limits!(ax4; xlimits=sweep.xlimits, ylimits=coherence_ylimits)
    end

    if column_labels !== nothing
        labels = collect(column_labels)
        for (col, label) in enumerate(labels)
            Label(fig[0, col], latexify_label(label); fontsize=fontsize)
        end
    end

    for row in 1:4, col in 1:2
        label_index = row + 4 * (col - 1)
        label = label_index <= length(panel_labels) ? "($(panel_labels[label_index]))" : ""
        qhe_panel_label!(axes[row, col], label; fontsize=panel_label_fontsize)
    end

    if link_xaxis
        linkxaxes!(axes[:, 1]...)
        linkxaxes!(axes[:, 2]...)
    end
    if link_row_yaxes
        for row in 1:4
            linkyaxes!(axes[row, :]...)
        end
    end
    if hide_inner_xticks
        hide_linked_xdecorations!(collect(axes[:, 1]))
        hide_linked_xdecorations!(collect(axes[:, 2]))
    end
    set_grid_visibility!(axes; xgridvisible, ygridvisible)

    total_row_height = sum(row_heights)
    for (i, height) in enumerate(row_heights)
        rowsize!(fig.layout, i, Relative(height / total_row_height))
    end

    return fig
end

"""
    qhe_two_sweep_validation_plot(g_data, lambda_data; kwargs...) -> Figure

Appendix-oriented diagnostic figure for the two paper sweeps of one nonlinear-QHE
regime. The rows show the off-resonant/RWA estimate, Hilbert-space tail
probabilities, occupation fractions, and the tight-coupling/current consistency
error.
"""
function qhe_two_sweep_validation_plot(g_data, lambda_data;
        g_xcol=:g,
        lambda_xcol=:λc,
        g_xlabel=L"g",
        lambda_xlabel=L"\lambda_c",
        panel_labels=("a", "b", "c", "d", "e", "f", "g", "h"),
        epsilon_ylabel=L"\epsilon_\mathrm{off}",
        tail_ylabel=L"p_{N-1}+p_N",
        occupation_ylabel=L"\langle n\rangle/N",
        tight_coupling_ylabel=L"\mathrm{tight\ coupling\ error}",
        figure_size=(1650, 1120),
        fontsize=QHE_DEFAULT_FONTSIZE,
        panel_label_fontsize=18,
        linewidth=QHE_DEFAULT_LINEWIDTH,
        row_heights=(1.0, 1.0, 0.8, 1.0),
        labels=(L"\epsilon_\mathrm{off}", L"\mathrm{hot\ tail}",
            L"\mathrm{cold\ tail}", L"\mathrm{cutoff}",
            L"\mathrm{hot\ occupation}", L"\mathrm{cold\ occupation}",
            L"\mathrm{occupation\ cap}", L"\mathrm{tight\ coupling\ error}"),
        epsilon_tol=0.05,
        tail_tol=1e-3,
        occupation_tol=0.5,
        tight_coupling_tol=nothing,
        epsilon_color=Makie.wong_colors()[1],
        hot_tail_color=Makie.wong_colors()[1],
        cold_tail_color=Makie.wong_colors()[2],
        hot_occupation_color=Makie.wong_colors()[1],
        cold_occupation_color=Makie.wong_colors()[2],
        tight_coupling_color=Makie.wong_colors()[3],
        threshold_color=:gray35,
        bad_point_color=:red,
        linestyles=(:solid, :solid, :solid, :dash),
        epsilon_yscale=log10,
        tail_yscale=log10,
        tight_coupling_yscale=log10,
        positive_floor=1e-16,
        legend_columns=:left,
        legend_position=:rt,
        epsilon_legend_position=nothing,
        tail_legend_position=nothing,
        occupation_legend_position=nothing,
        tight_coupling_legend_position=nothing,
        mark_bad_points=true,
        link_xaxis=true,
        hide_inner_xticks=true,
        repeat_ylabels=false,
        g_xlimits=nothing,
        lambda_xlimits=nothing,
        epsilon_ylimits=nothing,
        tail_ylimits=nothing,
        occupation_ylimits=nothing,
        tight_coupling_ylimits=nothing,
        link_row_yaxes=false,
        xgridvisible=true,
        ygridvisible=true,
        xlabelsize=nothing,
        ylabelsize=nothing,
        xticklabelsize=nothing,
        yticklabelsize=nothing,
        axis_kwargs=(;),
        legend_kwargs=(; framevisible=false))
    fig = Figure(; size=figure_size, fontsize)
    axis_options = axis_kwargs
    xlabelsize === nothing || (axis_options = merge(axis_options, (; xlabelsize)))
    ylabelsize === nothing || (axis_options = merge(axis_options, (; ylabelsize)))
    xticklabelsize === nothing || (axis_options = merge(axis_options, (; xticklabelsize)))
    yticklabelsize === nothing || (axis_options = merge(axis_options, (; yticklabelsize)))

    datasets = (
        (data=g_data, xcol=g_xcol, xlabel=g_xlabel, xlimits=g_xlimits),
        (data=lambda_data, xcol=lambda_xcol, xlabel=lambda_xlabel, xlimits=lambda_xlimits),
    )
    axes = Matrix{Any}(undef, 4, 2)
    panel_labels = collect(panel_labels)

    for (col, sweep) in enumerate(datasets)
        data = sweep.data
        x = qhe_column(data, sweep.xcol)
        ylabel_visible = repeat_ylabels || col == 1
        bad_mask = mark_bad_points ? qhe_bad_point_mask(data) : nothing
        show_epsilon_legend = qhe_should_show_legend(legend_columns, col)
        show_tail_legend = qhe_should_show_legend(legend_columns, col)
        show_occupation_legend = qhe_should_show_legend(legend_columns, col)
        show_tight_coupling_legend = tight_coupling_tol !== nothing &&
            qhe_should_show_legend(legend_columns, col)
        epsilon_position = qhe_panel_legend_position(epsilon_legend_position, col,
            legend_position)
        tail_position = qhe_panel_legend_position(tail_legend_position, col,
            legend_position)
        occupation_position = qhe_panel_legend_position(occupation_legend_position, col,
            legend_position)
        tight_coupling_position = qhe_panel_legend_position(tight_coupling_legend_position,
            col, legend_position)

        ax1 = Axis(fig[1, col];
            xlabel=latexify_label(sweep.xlabel),
            ylabel=ylabel_visible ? latexify_label(epsilon_ylabel) : "",
            yscale=epsilon_yscale,
            axis_options...)
        scatterlines!(ax1, x, qhe_positive_floor(data.epsilon_off; floor=positive_floor);
            label=show_epsilon_legend ? latexify_label(labels[1]) : nothing,
            linewidth, color=epsilon_color,
            linestyle=linestyles[1])
        hlines!(ax1, [epsilon_tol];
            color=threshold_color, linewidth, linestyle=linestyles[4],
            label=show_epsilon_legend ? latexify_label(labels[4]) : nothing)
        show_epsilon_legend && axislegend(ax1; position=epsilon_position, legend_kwargs...)

        ax2 = Axis(fig[2, col];
            xlabel=latexify_label(sweep.xlabel),
            ylabel=ylabel_visible ? latexify_label(tail_ylabel) : "",
            yscale=tail_yscale,
            axis_options...)
        scatterlines!(ax2, x, qhe_positive_floor(data.hot_tail; floor=positive_floor);
            label=show_tail_legend ? latexify_label(labels[2]) : nothing,
            linewidth, color=hot_tail_color,
            linestyle=linestyles[2])
        scatterlines!(ax2, x, qhe_positive_floor(data.cold_tail; floor=positive_floor);
            label=show_tail_legend ? latexify_label(labels[3]) : nothing,
            linewidth, color=cold_tail_color,
            linestyle=linestyles[3])
        hlines!(ax2, [tail_tol];
            color=threshold_color, linewidth, linestyle=linestyles[4],
            label=show_tail_legend ? latexify_label(labels[4]) : nothing)
        show_tail_legend && axislegend(ax2; position=tail_position, legend_kwargs...)

        ax3 = Axis(fig[3, col];
            xlabel=latexify_label(sweep.xlabel),
            ylabel=ylabel_visible ? latexify_label(occupation_ylabel) : "",
            axis_options...)
        scatterlines!(ax3, x, data.hot_occupation_fraction;
            label=show_occupation_legend ? latexify_label(labels[5]) : nothing,
            linewidth, color=hot_occupation_color,
            linestyle=linestyles[2])
        scatterlines!(ax3, x, data.cold_occupation_fraction;
            label=show_occupation_legend ? latexify_label(labels[6]) : nothing,
            linewidth, color=cold_occupation_color,
            linestyle=linestyles[3])
        hlines!(ax3, [occupation_tol];
            color=threshold_color, linewidth, linestyle=linestyles[4],
            label=show_occupation_legend ? latexify_label(labels[7]) : nothing)
        show_occupation_legend && axislegend(ax3; position=occupation_position, legend_kwargs...)

        ax4 = Axis(fig[4, col];
            xlabel=latexify_label(sweep.xlabel),
            ylabel=ylabel_visible ? latexify_label(tight_coupling_ylabel) : "",
            yscale=tight_coupling_yscale,
            axis_options...)
        scatterlines!(ax4, x, qhe_positive_floor(data.tight_coupling_error; floor=positive_floor);
            label=show_tight_coupling_legend ? latexify_label(labels[8]) : nothing,
            linewidth, color=tight_coupling_color,
            linestyle=linestyles[1])
        if tight_coupling_tol !== nothing
            hlines!(ax4, [tight_coupling_tol];
                color=threshold_color, linewidth, linestyle=linestyles[4],
                label=show_tight_coupling_legend ? latexify_label(labels[4]) : nothing)
        end
        show_tight_coupling_legend &&
            axislegend(ax4; position=tight_coupling_position, legend_kwargs...)

        if mark_bad_points
            qhe_ring_bad_points!(ax1, x,
                qhe_positive_floor(data.epsilon_off; floor=positive_floor),
                bad_mask; color=bad_point_color)
            qhe_ring_bad_points!(ax2, x,
                max.(qhe_positive_floor(data.hot_tail; floor=positive_floor),
                    qhe_positive_floor(data.cold_tail; floor=positive_floor)),
                bad_mask; color=bad_point_color)
            qhe_ring_bad_points!(ax3, x,
                max.(data.hot_occupation_fraction, data.cold_occupation_fraction),
                bad_mask; color=bad_point_color)
            qhe_ring_bad_points!(ax4, x,
                qhe_positive_floor(data.tight_coupling_error; floor=positive_floor),
                bad_mask; color=bad_point_color)
        end

        axes[:, col] .= (ax1, ax2, ax3, ax4)
        qhe_apply_axis_limits!(ax1; xlimits=sweep.xlimits, ylimits=epsilon_ylimits)
        qhe_apply_axis_limits!(ax2; xlimits=sweep.xlimits, ylimits=tail_ylimits)
        qhe_apply_axis_limits!(ax3; xlimits=sweep.xlimits, ylimits=occupation_ylimits)
        qhe_apply_axis_limits!(ax4; xlimits=sweep.xlimits, ylimits=tight_coupling_ylimits)
    end

    for row in 1:4, col in 1:2
        label_index = row + 4 * (col - 1)
        label = label_index <= length(panel_labels) ? "($(panel_labels[label_index]))" : ""
        qhe_panel_label!(axes[row, col], label; fontsize=panel_label_fontsize)
    end

    if link_xaxis
        linkxaxes!(axes[:, 1]...)
        linkxaxes!(axes[:, 2]...)
    end
    if link_row_yaxes
        for row in 1:4
            linkyaxes!(axes[row, :]...)
        end
    end
    if hide_inner_xticks
        hide_linked_xdecorations!(collect(axes[:, 1]))
        hide_linked_xdecorations!(collect(axes[:, 2]))
    end
    set_grid_visibility!(axes; xgridvisible, ygridvisible)

    total_row_height = sum(row_heights)
    for (i, height) in enumerate(row_heights)
        rowsize!(fig.layout, i, Relative(height / total_row_height))
    end

    return fig
end


"""
    qhe_heatmap_plot(x, y, z; kwargs...) -> Figure

Plot a single QHE scan heatmap, for example `Q_h(g, lambda_c)` or
`Q_h(g, n_c)`.
"""
function qhe_heatmap_plot(x, y, z;
        xlabel=L"x",
        ylabel=L"y",
        colorbar_label=L"\mathcal{Q}_h",
        title="",
        figure_size=(900, 620),
        fontsize=QHE_DEFAULT_FONTSIZE,
        colormap=:viridis,
        lowclip=nothing,
        highclip=nothing,
        colorrange=nothing,
        axis_kwargs=(;),
        colorbar_kwargs=(;))
    fig = Figure(; size=figure_size, fontsize)
    ax = Axis(fig[1, 1];
        xlabel=latexify_label(xlabel),
        ylabel=latexify_label(ylabel),
        title,
        axis_kwargs...)
    heatmap_kwargs = (; colormap, lowclip, highclip)
    colorrange === nothing || (heatmap_kwargs = merge(heatmap_kwargs, (; colorrange)))
    hm = heatmap!(ax, x, y, z; heatmap_kwargs...)
    Colorbar(fig[1, 2], hm; label=latexify_label(colorbar_label), colorbar_kwargs...)
    return fig
end

"""
    qhe_tur_coherence_heatmaps(x, y, Q, CRWA; kwargs...) -> Figure

Plot a finite-affinity blockade/coherence map with the TUR product and RWA
coherence on linked axes.
"""
function qhe_tur_coherence_heatmaps(x, y, Q, CRWA;
        xlabel=L"g",
        ylabel=L"\lambda_c",
        q_colorbar_label=L"\mathcal{Q}_h",
        coherence_colorbar_label=L"\mathcal{C}_\mathrm{RWA}",
        q_title="",
        coherence_title="",
        figure_size=(920, 820),
        fontsize=QHE_DEFAULT_FONTSIZE,
        q_colormap=:viridis,
        coherence_colormap=:magma,
        q_colorrange=nothing,
        coherence_colorrange=nothing,
        q_bound=2.0,
        show_q_bound=true,
        marker_x=Float64[],
        marker_y=Float64[],
        marker_color=:white,
        marker_linewidth=2.0,
        marker_linestyle=:dash,
        xgridvisible=true,
        ygridvisible=true,
        axis_kwargs=(;),
        colorbar_kwargs=(;))
    fig = Figure(; size=figure_size, fontsize)

    ax_q = Axis(fig[1, 1];
        xlabel=latexify_label(xlabel),
        ylabel=latexify_label(ylabel),
        title=q_title,
        axis_kwargs...)
    q_kwargs = (; colormap=q_colormap)
    q_colorrange === nothing || (q_kwargs = merge(q_kwargs, (; colorrange=q_colorrange)))
    hm_q = heatmap!(ax_q, x, y, Q; q_kwargs...)
    show_q_bound && contour!(ax_q, x, y, Q; levels=[q_bound], color=:white,
        linewidth=marker_linewidth, linestyle=:solid)
    isempty(marker_x) || vlines!(ax_q, marker_x; color=marker_color,
        linewidth=marker_linewidth, linestyle=marker_linestyle)
    isempty(marker_y) || hlines!(ax_q, marker_y; color=marker_color,
        linewidth=marker_linewidth, linestyle=marker_linestyle)
    Colorbar(fig[1, 2], hm_q; label=latexify_label(q_colorbar_label), colorbar_kwargs...)

    ax_c = Axis(fig[2, 1];
        xlabel=latexify_label(xlabel),
        ylabel=latexify_label(ylabel),
        title=coherence_title,
        axis_kwargs...)
    c_kwargs = (; colormap=coherence_colormap)
    coherence_colorrange === nothing || (c_kwargs = merge(c_kwargs, (; colorrange=coherence_colorrange)))
    hm_c = heatmap!(ax_c, x, y, CRWA; c_kwargs...)
    isempty(marker_x) || vlines!(ax_c, marker_x; color=marker_color,
        linewidth=marker_linewidth, linestyle=marker_linestyle)
    isempty(marker_y) || hlines!(ax_c, marker_y; color=marker_color,
        linewidth=marker_linewidth, linestyle=marker_linestyle)
    Colorbar(fig[2, 2], hm_c; label=latexify_label(coherence_colorbar_label), colorbar_kwargs...)

    linkxaxes!(ax_q, ax_c)
    linkyaxes!(ax_q, ax_c)
    hide_linked_xdecorations!([ax_q, ax_c])
    set_grid_visibility!((ax_q, ax_c); xgridvisible, ygridvisible)
    return fig
end

"""
    qhe_laguerre_prefactors_plot(lambda_grid, prefactor; kwargs...) -> Figure

Plot Laguerre blockade prefactors for hot and cold process orders. `prefactor`
must be callable as `prefactor(n, k, lambda)`.
"""
function qhe_laguerre_prefactors_plot(lambda_grid, prefactor;
        hot_order=1,
        cold_order=2,
        hot_levels=0:3,
        cold_levels=0:3,
        lambda_h=nothing,
        lambda_c=nothing,
        xlabel=L"\lambda",
        ylabel=L"A_n^k(\lambda)",
        title="",
        figure_size=(900, 560),
        fontsize=QHE_DEFAULT_FONTSIZE,
        linewidth=QHE_DEFAULT_LINEWIDTH,
        hot_linestyle=:solid,
        cold_linestyle=:dash,
        hot_color=:crimson,
        cold_color=:royalblue,
        marker_linewidth=2.0,
        marker_linestyle=:dash,
        legend_position=:rt,
        nbanks=2,
        xgridvisible=true,
        ygridvisible=true,
        axis_kwargs=(;),
        legend_kwargs=(;))
    fig = Figure(; size=figure_size, fontsize)
    ax = Axis(fig[1, 1];
        xlabel=latexify_label(xlabel),
        ylabel=latexify_label(ylabel),
        title,
        axis_kwargs...)

    palette = Makie.wong_colors()
    for (i, n) in enumerate(hot_levels)
        scatterlines!(ax, lambda_grid, prefactor.(n, hot_order, lambda_grid);
            label=latexify_label("\$A_{$n}^{$hot_order}\$"),
            linewidth,
            color=palette[mod1(i, length(palette))],
            linestyle=hot_linestyle)
    end
    for (i, n) in enumerate(cold_levels)
        scatterlines!(ax, lambda_grid, prefactor.(n, cold_order, lambda_grid);
            label=latexify_label("\$A_{$n}^{$cold_order}\$"),
            linewidth,
            color=palette[mod1(i + length(hot_levels), length(palette))],
            linestyle=cold_linestyle)
    end

    lambda_h === nothing || vlines!(ax, [lambda_h];
        color=hot_color, linewidth=marker_linewidth, linestyle=marker_linestyle)
    lambda_c === nothing || vlines!(ax, [lambda_c];
        color=cold_color, linewidth=marker_linewidth, linestyle=marker_linestyle)

    axislegend(ax; position=legend_position, nbanks, legend_kwargs...)
    set_grid_visibility!((ax,); xgridvisible, ygridvisible)
    return fig
end

function plot_laguerre_prefactors(; λh=0.47, λc=0.89, λgrid=LinRange(0.0, 1.5, 500),
        prefactor=laguerre_prefactor, kwargs...)
    return qhe_laguerre_prefactors_plot(λgrid, prefactor; lambda_h=λh, lambda_c=λc, kwargs...)
end

@isdefined(JC_DEFAULT_TRUNC_TOL) || const JC_DEFAULT_TRUNC_TOL = 1e-3
@isdefined(JC_DEFAULT_WIDE_TAIL_FACTOR) || const JC_DEFAULT_WIDE_TAIL_FACTOR = 10.0
@isdefined(JC_PLOT_FONTSIZE) || const JC_PLOT_FONTSIZE = 25
@isdefined(JC_PLOT_LINEWIDTH) || const JC_PLOT_LINEWIDTH = 4
@isdefined(JC_PANEL_FONTSIZE) || const JC_PANEL_FONTSIZE = 28

jc_positive_floor(values; floor=1e-18) = max.(values, floor)

function jc_panel_label!(ax, label; fontsize=JC_PANEL_FONTSIZE)
    text!(ax, 0.02, 0.96;
        text=label,
        space=:relative,
        align=(:left, :top),
        offset=(6, -6),
        fontsize=fontsize,
        font=:bold)
    return ax
end

function jc_style_axes!(axes)
    for ax in axes
        ax.xgridvisible = true
        ax.ygridvisible = true
        ax.xgridcolor = (:gray70, 0.35)
        ax.ygridcolor = (:gray70, 0.35)
        ax.xgridwidth = 1
        ax.ygridwidth = 1
    end
    return axes
end

function jc_status_style(status, color)
    status === :safe && return (color=color, alpha=1.0, marker=:circle, strokewidth=0.0)
    status === :ambiguous && return (color=color, alpha=0.45, marker=:utriangle, strokewidth=1.5)
    return (color=:gray55, alpha=0.55, marker=:xcross, strokewidth=0.0)
end

function jc_figures_dir()
    # This repository's figures/ directory. The research original wrote straight
    # into the manuscript tree; here figures are produced as repo artefacts.
    return joinpath(normpath(@__DIR__, ".."), "figures")
end

function save_jc_figure(fig, filename; figures_dir=jc_figures_dir())
    mkpath(figures_dir)
    path = joinpath(figures_dir, filename)
    save(path, fig)
    return path
end

"""
    jc_fcs_production_plot(rows, detuning_cuts, metadata; kwargs...) -> Figure

Publication figure for the fixed-g/κ production FCS drive sweeps
(`run_jc_fcs_production_sweep`): a 3×2 grid versus the swept drive `x = 2E/g`,
one colour per detuning cut, with physics in the left column and numerical
reliability in the right column.

Left (physics): (a) count rate `c₁/κ` (log), (b) Fano factor `c₂/c₁` (log,
Poisson line at 1), (c) skewness `c₃/c₂^{3/2}` (symlog, so the low-drive Poisson
divergence and the negative bright-branch values are both visible).

Right (reliability): (d) the dynamic cutoff `N` used at each point, one step
curve per cut; (e) truncation tail `p_{N-1}+p_N` (log, `trunc_tol` line);
(f) occupation `n/N` with the scheduler cap `occ_max`. Points whose cutoff was
clamped to the tier ceiling (`cutoff_budget_exceeded`) are ringed in red on (d)
and (f).
"""
function jc_fcs_production_plot(rows, detuning_cuts, metadata;
        κ=1.0,
        figure_size=nothing,
        fontsize=JC_PLOT_FONTSIZE,
        linewidth=JC_PLOT_LINEWIDTH,
        trunc_tol=JC_DEFAULT_TRUNC_TOL,
        occ_max=0.7,
        c1_floor=1e-6,
        fock_distributions=nothing,
        g=nothing,
        pn_floor=1e-4,
        pn_linestyles=(:dot, :dash, :solid),
        pn_xscale=sqrt,
        pn_xlims=nothing,
        pn_xticks=[0, 1, 10, 50, 100, 300, 400, 500],
        pn_bright_band=(50, Inf),
        show_cutoff_limited=false,
        # title=jc_metadata_string(metadata) * ", production FCS")
        title= " ")
    # When `fock_distributions` is supplied, prepend a Fock-state-distribution
    # column (Pₙ) as column 1 and shift the existing physics/reliability panels
    # to columns 2 and 3 (a 3×3 figure). Otherwise the original 3×2 layout is
    # reproduced unchanged.
    has_fock = fock_distributions !== nothing
    has_fock && g === nothing &&
        error("jc_fcs_production_plot: `g` is required when `fock_distributions` " *
              "is supplied (n_scale = (g/κ)²).")
    figure_size = figure_size === nothing ?
        (has_fock ? (2600, 1400) : (1500, 1300)) : figure_size
    col0 = has_fock ? 1 : 0
    fig = Figure(size=figure_size; fontsize)
    isempty(title) || Label(fig[0, 1:(2 + col0)], title; fontsize=16)

    # Physics column then numerical-reliability column (shifted right by `col0`).
    ax_c1 = Axis(fig[1, 1 + col0], ylabel=L"\langle a^\dagger a \rangle", yscale=log10)
    ax_f = Axis(fig[2, 1 + col0], ylabel=L"\mathcal{F}", yscale=log10)
    ax_skew = Axis(fig[3, 1 + col0], xlabel=L"2\mathcal{E}/g", ylabel=L"\mathcal{G}",
        yscale=Makie.pseudolog10)
    ax_N = Axis(fig[1, 2 + col0], ylabel=L"\mathrm{cutoff}\ N")
    ax_tail = Axis(fig[2, 2 + col0], ylabel=L"p_{N-1}+p_N", yscale=log10)
    ax_occ = Axis(fig[3, 2 + col0], xlabel=L"2\mathcal{E}/g", ylabel=L"\langle a^\dagger a \rangle/N")

    left_axes = (ax_c1, ax_f, ax_skew)
    right_axes = (ax_N, ax_tail, ax_occ)

    cut_colors = Makie.wong_colors()[1:length(detuning_cuts)]
    cut_markers = (:circle, :rect, :diamond, :hexagon, :star5)
    for (color, marker, Δ_val) in zip(cut_colors, Iterators.cycle(cut_markers), detuning_cuts)
        cut = sort(filter(r -> r.Δ == Δ_val, rows), by=r -> r.x)
        isempty(cut) && continue
        label = LaTeXString("\\Delta/\\kappa=$(Δ_val)")
        x = [r.x for r in cut]
        common = (; color, marker, linewidth, markersize=9, strokewidth=0.0)

        # Left: physics
        scatterlines!(ax_c1, x, jc_positive_floor([r.c1 / κ for r in cut]; floor=c1_floor);
            label, common...)
        scatterlines!(ax_f, x, jc_positive_floor([r.Fano for r in cut]; floor=1e-6);
            common...)
        scatterlines!(ax_skew, x,
            [r.c2 > 0 ? r.c3 / r.c2^(3 / 2) : NaN for r in cut]; common...)
        ax_skew.yticks = ([-1e+2, -1e+1, 0.0, 1e1, 1e+2, 1e+3], ["-10²", "-10¹", "0", "10¹", "10²", "10³"])
        # Right: reliability
        scatterlines!(ax_N, x, [Float64(r.N) for r in cut]; common...)
        scatterlines!(ax_tail, x, jc_positive_floor([r.cavity_tail for r in cut]); common...)
        scatterlines!(ax_occ, x, [r.occupation_fraction for r in cut]; common...)

        # Highlight (2× marker) the drive points shown in the Pₙ column on the
        # physics panels, so each Pₙ curve can be located on the FCS scans.
        if has_fock
            hx = sort!([fr.x for fr in fock_distributions if fr.Δ == Δ_val])
            picks = [cut[argmin(abs.(x .- hxi))] for hxi in hx]
            xp = [r.x for r in picks]
            hl = (; color, marker, markersize=26, strokewidth=0.0)
            scatter!(ax_c1, xp, jc_positive_floor([r.c1 / κ for r in picks]; floor=c1_floor); hl...)
            scatter!(ax_f, xp, jc_positive_floor([r.Fano for r in picks]; floor=1e-6); hl...)
            scatter!(ax_skew, xp,
                [r.c2 > 0 ? r.c3 / r.c2^(3 / 2) : NaN for r in picks]; hl...)
        end
    end

    hlines!(ax_f, [1.0]; color=:gray35, linestyle=:dash, linewidth=2.5,
        label=L"\mathrm{Poisson}")
    hlines!(ax_skew, [0.0]; color=:gray35, linestyle=:dash, linewidth=2.5)
    hlines!(ax_tail, [trunc_tol]; color=:black, linestyle=:dash, linewidth=2.5,
        label=L"\varepsilon_{\mathrm{trunc}}")
    hlines!(ax_occ, [occ_max]; color=:black, linestyle=:dash, linewidth=2.5
        # label=L"n/N=%$(occ_max)"
        )

    # Ring the cutoff-limited points (estimate exceeded the top tier) on the
    # reliability panels where they matter (opt-in via `show_cutoff_limited`).
    if show_cutoff_limited && !isempty(rows) && :cutoff_budget_exceeded in propertynames(first(rows))
        clamped = filter(r -> r.cutoff_budget_exceeded, rows)
        if !isempty(clamped)
            scatter!(ax_N, [r.x for r in clamped], [Float64(r.N) for r in clamped];
                color=(:red, 0.0), marker=:circle, markersize=18,
                strokecolor=:red, strokewidth=2.5, label="cutoff-limited")
            scatter!(ax_occ, [r.x for r in clamped], [r.occupation_fraction for r in clamped];
                color=(:red, 0.0), marker=:circle, markersize=18,
                strokecolor=:red, strokewidth=2.5)
        end
    end

    # ax_N.yticks = collect(JC_DEFAULT_FCS_PROD_TIERS)
    ax_N.yticks = [150, 200, 250, 300, 350, 400, 450, 500]
    ax_tail.yticks = ([1e-18, 1e-14, 1e-10, 1e-6, 1e-4], ["10⁻¹⁸", "10⁻¹⁴", "10⁻¹⁰", "10⁻⁶", "10⁻⁴"])
    ax_occ.yticks = [0.1, 0.3, 0.5, 0.7, 0.9]
    ylims!(ax_occ, 0, min(1.0, occ_max + 0.25))

    # Panel letters run down each column then to the next: col 1 = (a,b,c),
    # col 2 = (d,e,f), col 3 = (g,h,i). With the Pₙ column the physics panels are
    # col 2 (d,e,f) and reliability col 3 (g,h,i); the Pₙ column (below) is (a,b,c).
    existing_labels = has_fock ?
        ("(d)", "(e)", "(f)", "(g)", "(h)", "(i)") :
        ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
    for (ax, lbl) in zip((ax_c1, ax_f, ax_skew, ax_N, ax_tail, ax_occ), existing_labels)
        jc_panel_label!(ax, lbl)
    end
    linkxaxes!(ax_c1, ax_f, ax_skew, ax_N, ax_tail, ax_occ)
    for ax in (ax_c1, ax_f, ax_N, ax_tail)
        hidexdecorations!(ax; grid=false)
    end
    jc_style_axes!((left_axes..., right_axes...))
    # Legends placed in each panel's empty corner, clear of the top-left labels.
    axislegend(ax_c1; position=:rc, framevisible=false, labelsize = 26)      # cut legend (c₁ high at right)
    # axislegend(ax_f; position=:rt, framevisible=false, labelsize = 22)       # Poisson line
    axislegend(ax_tail; position= (0.97, 0.75), framevisible=false, labelsize = 26)    # ε_trunc line
    # axislegend(ax_occ; position=:rb, framevisible=false, labelsize = 24)     # n/N cap line
    (show_cutoff_limited && !isempty(rows) && :cutoff_budget_exceeded in propertynames(first(rows)) &&
        any(r -> r.cutoff_budget_exceeded, rows)) &&
        axislegend(ax_N; position=:rb, framevisible=false, labelsize = 24)   # cutoff-limited flag

    # New Pₙ column (column 1): cavity Fock-state distributions probed by the FCS
    # panels. One panel per detuning cut, coloured by cut (same mapping as the FCS
    # panels); the representative drive values within a panel — below / at / after
    # the transition — are drawn dotted / dashed / solid. A dashed vertical line
    # marks n_scale = (g/κ)². The three Pₙ panels share the n-axis (only the
    # bottom shows n ticks); the top panel carries a dim | n_scale | bright axis.
    if has_fock
        n_scale = (g / κ)^2
        nmax = maximum(r.N for r in fock_distributions)
        # xautolimitmargin=0 on the low side: a `sqrt` (or other nonlinear) scale
        # would otherwise pad the lower limit below 0, where its inverse is undefined.
        pn_axes = [Axis(fig[r, 1]; yscale=log10, xscale=pn_xscale, ylabel=L"P_n",
                        xautolimitmargin=(0.0, 0.0))
                   for r in 1:length(detuning_cuts)]
        pn_axes[end].xlabel = L"n"
        for (k, Δ_val) in enumerate(detuning_cuts)
            ax = pn_axes[k]
            color = cut_colors[k]
            fcut = sort(filter(r -> r.Δ == Δ_val, fock_distributions), by=r -> r.x)
            isempty(fcut) && continue
            # Light shaded band marking the bright region (drawn first, behind curves).
            # An infinite upper edge extends the band to the plot's right limit.
            band_hi = isfinite(pn_bright_band[2]) ? pn_bright_band[2] :
                (pn_xlims === nothing ? nmax : pn_xlims[2])
            vspan!(ax, pn_bright_band[1], band_hi; color=(:gray, 0.12))
            # Curves ordered by drive: below / at / after the transition, drawn
            # dotted / dashed / solid (the style list cycles if more are supplied).
            for (j, fr) in enumerate(fcut)
                ns = collect(0:(length(fr.Pn) - 1))
                lines!(ax, ns, jc_positive_floor(fr.Pn; floor=pn_floor);
                    color=color, linewidth=linewidth,
                    linestyle=pn_linestyles[mod1(j, length(pn_linestyles))],
                    label=L"2\mathcal{E}/g=%$(fr.x)")
            end
            vlines!(ax, [n_scale]; color=:gray35, linestyle=:dash, linewidth=2.5)
            text!(ax, 0.5, 0.9; text=LaTeXString("\\Delta/\\kappa=$(Δ_val)"),
                space=:relative, align=(:right, :top), offset=(-6, -6),
                color=color, fontsize=30, font=:bold)
            # Bottom-axis ticks: numeric ticks plus a labelled n_scale tick.
            pn_tickpos = Float64.(vcat(pn_xticks, n_scale))
            pn_ticklab = vcat([LaTeXString(string(t)) for t in pn_xticks], [L"n_\text{scale}"])
            pn_perm = sortperm(pn_tickpos)
            ax.xticks = (pn_tickpos[pn_perm], pn_ticklab[pn_perm])
            # Per-panel legend of the drive values (2ℰ/g) shown, with their styles.
            axislegend(ax; position=:rt, framevisible=false, labelsize=22)
            ylims!(ax, pn_floor, nothing)
        end
        pn_xl = pn_xlims === nothing ? (0, nmax) : pn_xlims
        linkxaxes!(pn_axes...)
        xlims!(pn_axes[1], pn_xl...)
        for ax in pn_axes[1:(end - 1)]
            hidexdecorations!(ax; grid=false)
        end
        jc_style_axes!(pn_axes)
        # Top secondary axis on the top Pₙ panel: region ticks centred in the dim
        # (n<n_scale) and bright (n>n_scale) regions, with n_scale itself marked.
        ax_top = Axis(fig[1, 1]; xaxisposition=:top, xscale=pn_xscale,
            backgroundcolor=:transparent, xautolimitmargin=(0.0, 0.0))
        linkxaxes!(pn_axes[1], ax_top)
        xlims!(ax_top, pn_xl...)
        hideydecorations!(ax_top)
        ax_top.xgridvisible = false
        ax_top.ygridvisible = false
        # Top-axis region labels: "dim" pushed into the low-n end, "bright" over
        # the shaded bright region (n_scale itself is now marked on the bottom axis).
        xit = Makie.inverse_transform(pn_xscale)
        dim_center = xit(pn_xscale(n_scale) / 4)
        ax_top.xticks = ([dim_center, n_scale],
            [L"\text{dim (blockade)}", L"\text{bright}"])
        for (ax, lbl) in zip(pn_axes, ("(a)", "(b)", "(c)"))
            jc_panel_label!(ax, lbl)
        end
    end

    rowgap!(fig.layout, 8)
    colgap!(fig.layout, 24)
    return fig
end
