using Gen
include("./inference.jl")

# Read in data file
f = open("./reasoning_about_geo_structures/data.txt", "r")
t ::Vector{Float32} = zeros(0)
head :: Vector{Float32} = zeros(0)
o ::Vector{Float32} = zeros(0)

lines = readlines(f)
append!(t, parse.(Float64, split(lines[1])))
append!(head, parse.(Float64, split(lines[2])))
append!(o, parse.(Float64, split(lines[3])))
iterations_to_perform = parse(Int, lines[4])
close(f)
println(t)
println(head)
println(o)
println(iterations_to_perform)

Kv, Sskv, Sske, nclay = perform_geo_inference(o, 1000)
println("Kv: $Kv, Sskv: $Sskv, Sske: $Sske, nclay: $nclay")
