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

function unfold_particle_filter(num_particles::Int, os::Vector{Int})
    init_obs = Gen.choicemap((:chain => 1 => :obs, os[1]))
    state = Gen.initialize_particle_filter(unfold_model, (1,), init_obs, num_particles)
    old_obs = init_obs
    for t=2:length(os)-1
        nr_acc = 0
        all_Nan = true
        # For each particle do a random MH walk for one of the initial parameters on all encountered observations up until now
        for i=1:num_particles            
            initial_choices = select(:tau, :R0, :rho0, :rho1, :rho1, :rho2, :switch_to_rho1, :switch_to_rho2)
            state.traces[i], _  = mh(state.traces[i], initial_choices, check=true, observations=old_obs)
        end
        mxval, mxindx = findmax(state.log_weights)
        println("MAX value weight found : ", mxval, " ", ", iteration: ", t, ", obs:", os[t])

        # Resample particles if the effective sample size is below half the number of particles
        maybe_resample!(state, ess_threshold=num_particles)
        # Observation of this timestep 
        obs = Gen.choicemap((:chain => t => :obs, os[t]))
        
        # t::Int, prev_state::State, Population::Int, tau::Int, R0::Float64, 
        #j switch_to_rho1::Int, switch_to_rho2::Int, rho0::Float64, rho1::Float64, rho2::Float64)
        # All observations encountered up until now
        old_obs = Base.merge(init_obs, obs)
        arg_kernel = (t, state, 600, select(:tau, :R0, :switch_to_rho1, :switch_to_rho2, :rho0, :rho1, :rho2))
        # Particle filter step with current observation step
        Gen.particle_filter_step!(state, (t,), (kernel(), arg_kernel), obs)
        # TODO: Remove samples that have a certain weight of NaN? Or at least do something with them/ set their weight to zero?        for i=1:num_particles
    end
    (_, log_normalized_weights) = Gen.normalize_weights(state.log_weights)
    weights = exp.(log_normalized_weights)
    mxval, mxindx = findmax(state.log_weights)
    print("MAP estimate weight value: ")
    println(mxval)
    return state.traces[mxindx]
end
    
    

