include("./inference.jl")

function run(file_path::String, nr_particles::Integer)
    f = open(file_path, "r")
    o ::Vector{Int} = zeros(0)
    for line in readlines(f)
        append!(o, parse(Int, split(line, "\t")[2]))
    end
    close(f)
    result = unfold_particle_filter(nr_particles, o);
    println(result)
end
