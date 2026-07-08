function benchmark_project_root()
    return normpath(joinpath(@__DIR__, ".."))
end

function benchmark_dir()
    return benchmark_project_root()
end

function raw_data_dir()
    return joinpath(benchmark_dir(), "data")
end

function figures_dir()
    return joinpath(benchmark_dir(), "figures")
end

function ensure_benchmark_dirs()
    mkpath(raw_data_dir())
    mkpath(figures_dir())
    return (raw_data=raw_data_dir(), figures=figures_dir())
end

function raw_data_path(filename::AbstractString)
    ensure_benchmark_dirs()
    return joinpath(raw_data_dir(), filename)
end

function figure_path(filename::AbstractString)
    ensure_benchmark_dirs()
    return joinpath(figures_dir(), filename)
end
