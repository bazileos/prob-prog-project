include("./sr_model.jl")

function run(file_path::String, num_iter::Integer)
    xs::Vector{Float64} = []
    ys::Vector{Float64} = []

    f = open(file_path, "r")
    for line in readlines(f)
        append!(xs, parse(Float64, split(line, ";")[1]))
        append!(ys, parse(Float64, split(line, ";")[2]))
    end
    close(f)

    solutions = find_and_print_expressions(xs, ys, num_iter)
end