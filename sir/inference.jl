include("./model.jl")
using Gen

function sir_inference(o::Vector{Int}, num_iters::Int)

    constraints = choicemap()
    for (t, o_t) in enumerate(o)
        constraints["o-$t"] = o_t
    end
    
    # Run the model, constrained by `constraints`,
    # to get an initial execution trace
    (trace, _) = generate(sir_model, (), constraints)
    
    # Iteratively update the slope then the intercept,
    # using Gen's metropolis_hastings operator.
    for iter = 1:num_iters
        (trace, _) = metropolis_hastings(trace, select(:tau))
        (trace, _) = metropolis_hastings(trace, select(:R0))
        (trace, _) = metropolis_hastings(trace, select(:rho0))
        (trace, _) = metropolis_hastings(trace, select(:rho1))
        (trace, _) = metropolis_hastings(trace, select(:rho2))
        (trace, _) = metropolis_hastings(trace, select(:switch_to_rho1))
        (trace, _) = metropolis_hastings(trace, select(:switch_to_rho2))
    end
    
    choices = get_choices(trace)
    return (choices[:tau], choices[:R0], choices[:rho0], choices[:rho1], 
        choices[:rho2], choices[:switch_to_rho1], choices[:switch_to_rho2])
end