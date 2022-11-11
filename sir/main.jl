include("./inference.jl")
file_name = ARGS[1]
nr_particles = ARGS[2]
f = open(file_name, "r")
o ::Vector{Int} = zeros(0)
for line in readlines(f)
    append!(o, parse(Int, split(line, "\t")[2]))
end
close(f)
nr_par = parse(Int, nr_particles)
result = unfold_particle_filter(nr_par, o);
println(result)
