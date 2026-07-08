function parse_int_env(name::AbstractString, default::Integer)
    value = get(ENV, name, "")
    isempty(strip(value)) && return Int(default)
    return parse(Int, strip(value))
end

function parse_float_env(name::AbstractString, default::Real)
    value = get(ENV, name, "")
    isempty(strip(value)) && return Float64(default)
    return parse(Float64, strip(value))
end

function parse_int_list_env(name::AbstractString, default_values)
    value = get(ENV, name, "")
    isempty(strip(value)) && return collect(Int, default_values)
    return parse.(Int, split(replace(value, ";" => ","), ","; keepempty=false))
end

function write_two_column_csv(path::AbstractString, x, y)
    writedlm(path, hcat(x, y), ',')
    return path
end

function write_columns_csv(path::AbstractString, header, columns...)
    lengths = length.(columns)
    all(==(first(lengths)), lengths) || error("CSV columns must have the same length")

    open(path, "w") do io
        println(io, join(header, ","))
        for i in 1:first(lengths)
            println(io, join((column[i] for column in columns), ","))
        end
    end
    return path
end
