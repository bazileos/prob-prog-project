include("./sr.jl")

function test_fn(xs)
    return xs.^2 .+ xs .+ 4
end

xs::Vector{Float64} = 0.1:0.1:10
ys::Vector{Float64} = test_function(xs)

test_model(xs, ys, 500, test_fn)