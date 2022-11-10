include("./sr_model.jl")

test_fn(xs) = xs.^2 .+ xs .+ 4

xs::Vector{Float64} = 0.1:0.1:10
ys::Vector{Float64} = test_fn(xs)

test_model(xs, ys, 500, test_fn)