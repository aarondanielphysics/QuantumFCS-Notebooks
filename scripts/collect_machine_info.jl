const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

using Dates
using InteractiveUtils
using LinearAlgebra
using Sockets
using TOML

include(joinpath(PROJECT_ROOT, "src", "paths.jl"))

ensure_benchmark_dirs()

function maybe_read_cmd(cmd)
    try
        return strip(read(cmd, String))
    catch
        return ""
    end
end

function os_release()
    path = "/etc/os-release"
    if isfile(path)
        data = Dict{String,String}()
        for line in eachline(path)
            isempty(line) && continue
            startswith(line, "#") && continue
            parts = split(line, "="; limit=2)
            length(parts) == 2 || continue
            data[parts[1]] = strip(parts[2], ['"', '\''])
        end
        return data
    end
    return Dict{String,String}()
end

function cpu_summary()
    info = Sys.cpu_info()
    first_cpu = isempty(info) ? nothing : first(info)
    return Dict{String,Any}(
        "model" => first_cpu === nothing ? "" : first_cpu.model,
        "reported_cores" => length(info),
        "julia_threads" => Threads.nthreads(),
        "cpu_threads" => Sys.CPU_THREADS,
    )
end

function package_versions()
    project_path = joinpath(PROJECT_ROOT, "Project.toml")
    manifest_path = joinpath(PROJECT_ROOT, "Manifest.toml")
    isfile(project_path) || return Dict{String,String}()

    project = TOML.parsefile(project_path)
    manifest = isfile(manifest_path) ? TOML.parsefile(manifest_path) : Dict{String,Any}()
    project_deps = get(project, "deps", Dict{String,Any}())
    manifest_deps = get(manifest, "deps", Dict{String,Any}())

    packages = Dict{String,String}()
    for name in sort(collect(keys(project_deps)))
        entry = get(manifest_deps, name, nothing)
        if entry isa Vector && !isempty(entry)
            entry = first(entry)
        end
        packages[name] = entry isa Dict ? string(get(entry, "version", "stdlib")) : "stdlib"
    end
    return packages
end

function versioninfo_string()
    io = IOBuffer()
    versioninfo(io; verbose=false)
    return String(take!(io))
end

git = Dict{String,Any}(
    "commit" => maybe_read_cmd(`git -C $PROJECT_ROOT rev-parse HEAD`),
    "branch" => maybe_read_cmd(`git -C $PROJECT_ROOT branch --show-current`),
    "status_short" => maybe_read_cmd(`git -C $PROJECT_ROOT status --short`),
)

blas_config = try
    string(BLAS.get_config())
catch
    ""
end

metadata = Dict{String,Any}(
    "captured_at" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
    "project_root" => PROJECT_ROOT,
    "hostname" => gethostname(),
    "julia" => Dict{String,Any}(
        "version" => string(VERSION),
        "bindir" => Sys.BINDIR,
        "executable" => joinpath(Sys.BINDIR, Base.julia_exename()),
        "machine" => Sys.MACHINE,
        "word_size" => Sys.WORD_SIZE,
        "versioninfo" => versioninfo_string(),
    ),
    "os" => Dict{String,Any}(
        "kernel" => string(Sys.KERNEL),
        "arch" => string(Sys.ARCH),
        "is_linux" => Sys.islinux(),
        "is_macos" => Sys.isapple(),
        "is_windows" => Sys.iswindows(),
        "release" => os_release(),
    ),
    "hardware" => Dict{String,Any}(
        "cpu" => cpu_summary(),
        "total_memory_bytes" => Sys.total_memory(),
        "free_memory_bytes" => Sys.free_memory(),
    ),
    "linear_algebra" => Dict{String,Any}(
        "blas_threads" => BLAS.get_num_threads(),
        "blas_config" => blas_config,
    ),
    "environment" => Dict{String,Any}(
        "JULIA_NUM_THREADS" => get(ENV, "JULIA_NUM_THREADS", ""),
        "JULIA_EXCLUSIVE" => get(ENV, "JULIA_EXCLUSIVE", ""),
        "JULIA_CPU_TARGET" => get(ENV, "JULIA_CPU_TARGET", ""),
    ),
    "git" => git,
    "direct_package_versions" => package_versions(),
)

out = raw_data_path("machine_info.toml")
open(out, "w") do io
    TOML.print(io, metadata)
end

println("Saved: ", out)
