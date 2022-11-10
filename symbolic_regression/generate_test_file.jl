test_fn(xs) = xs.^2 .+ xs .+ 4

xs::Vector{Float64} = 0.1:0.1:10
ys::Vector{Float64} = test_fn(xs)

touch("./symbolic_regression/data.txt")
file = open("./symbolic_regression/data.txt", "w")
n = length(xs)
for i = 1:n
    write(file, "$(xs[i])\t$(ys[i])\n")
end
close(file)