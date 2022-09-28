include("./inference.jl")
using Plots

f = open("data.txt", "r")
o ::Vector{Int} = zeros(0)
for line in readlines(f)
    append!(o, parse(Int, split(line, "\t")[2]))
end
close(f)

x = 2:60
plot(x, o)