include("./model.jl")
using Gen
using Statistics

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

function visualize_traces_R0_tau(traces, num_particles, t)
    R0 = Vector{Float64}(undef, num_particles)
    tau = Vector{Int}(undef, num_particles)
    switch_to_rho1 = Vector{Int}(undef, num_particles)
    switch_to_rho2 = Vector{Int}(undef, num_particles)
    S2I = Vector{Int}(undef, num_particles)
    for trace_ind = 1:num_particles
        temp_choices = Gen.get_choices(traces[trace_ind])
        R0[trace_ind] = temp_choices[:R0]
        tau[trace_ind] = temp_choices[:tau]
        switch_to_rho1[trace_ind] = temp_choices[:switch_to_rho1]
        switch_to_rho2[trace_ind] = temp_choices[:switch_to_rho2]
        S2I[trace_ind] = temp_choices[:chain => t => :S2I]
    end
    mx_min, mxindx = findmin(S2I)
    println("Timestep:", t, ", Variance R0:", var(R0), ", tau: ", var(tau), ", S21:", var(S2I), ", ", mx_min)
end

function particle_resimulation(trace)
    initial_choice = select(:tau, :R0, :rho0, :rho1, :rho1, :rho2, :switch_to_rho1, :switch_to_rho2)
    trace, a = metropolis_hastings(trace, initial_choice)
    return trace
end

@gen function state_proposal(trace, t_begin, t_now)
    t_begin = max(1, t_begin)

    # for each timestep in the interval we want to update the state parameters
    # most simple way is to just resample them and see if the sample is more likely
    # beta_t, S2I, I2R
    beta_t = 0.0
    S2I = 0
    I2R = 0

    for t = t_begin : t_now
        beta_t = normal(trace[:chain => t => :beta_t], 0.1)
        S2I = uniform_discrete(trace[:chain => t => :S2I] - 1, trace[:chain => t => :S2I] + 1)
        I2R = uniform_discrete(trace[:chain => t => :I2R] - 1, trace[:chain => t => :I2R] + 1)
    end
    return beta_t, S2I, I2R
end

function particle_rejuvenation(trace, t_begin, t_now)
    return Gen.metropolis_hastings(trace, state_proposal, (t_begin, t_now))
end

function unfold_particle_filter(num_particles::Int, os::Vector{Int})
    init_obs = Gen.choicemap((:chain => 1 => :obs, os[1]))
    state = Gen.initialize_particle_filter(unfold_model, (1,), init_obs, num_particles)
    old_obs = init_obs
    best_trace = ""
    for t=2:length(os)-1
        nr_acc = 0
        # For each particle do a random MH walk for one of the initial parameters on all encountered observations up until now
        for i=1:num_particles     

            # Resimulate the whole path of the trace to update the global initial parameters
            state.traces[i] = particle_resimulation(state.traces[i])
            
            state.traces[i], a = particle_rejuvenation(state.traces[i], t-5, t-1)
            if a > 0
                nr_acc += 1
            end
        end
    
        # visualize_traces_R0_tau(state.traces, num_particles, t-1)


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
        Gen.particle_filter_step!(state, (t,), (kernel, arg_kernel), obs)
        mxval, mxindx = findmax(state.log_weights)
        if mxval != 0
            cho_res = state.traces[mxindx]
            best_trace = "tau: $(cho_res[:tau]), R0: $(cho_res[:R0]), rho0: $(cho_res[:rho0]), rho1: $(cho_res[:rho1]), rho2: $(cho_res[:rho2]), switch_to_rho1: $(cho_res[:switch_to_rho1]), switch_to_rho2: $(cho_res[:switch_to_rho2])"
        end
        println("MAX value weight found : ", mxval, ", iteration: ", t, ", obs:", os[t], ", acc:", nr_acc)

    end
    # (_, log_normalized_weights) = Gen.normalize_weights(state.log_weights)
    # weights = exp.(log_normalized_weights)
    # mxval, mxindx = findmax(old_weights)
    # print("MAP estimate weight value: ")
    # println(mxval)
    return best_trace
end
    
    

