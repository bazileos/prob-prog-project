include("./model.jl")
using Gen

trace = simulate(symbolic_regression_model, ())
choices = get_choices(trace)

print(choices)