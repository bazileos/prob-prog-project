using Gen

@dist lognormal(x, y) = exp(normal(log(x), y))

@gen function sir_model()
    T = 60 # duration of simulation (in days)
    Population = 600 # population size
    S_0 = 599 # susceptible part of population
    I_0 = 1 # number of infected people at time step 1
    beta_0 = 1.0
    
    tau ~ uniform_discrete(2, 9)
    R0 ~ lognormal(0.0, 1.0)
    rho0 ~ beta(2, 4)
    rho1 ~ beta(4, 4)
    rho2 ~ beta(8, 4)
    switch_to_rho1 ~ uniform_discrete(15, 40)
    switch_to_rho2 ~ uniform_discrete(30, 60)

    beta_t = beta_0
    I_t = I_0
    S_t = S_0

    for t = 2:T
        beta_t = @trace(lognormal(beta_t, 0.1), "beta_$t")
        R_t = R0 * beta_t
        
        individual_rate = R_t/tau

        """
        compute the probability that any pair (susceptible, infected)
        pair of individuals results in an infection at this time step.
        """
        p = individual_rate / Population
        combined_p = 1 - (1-p)^I_t

        S2I = @trace(binom(S_t, combined_p), "S2I_$t") # susceptible to infected
        I2R = @trace(binom(I_t, 1/tau), "I2R_$t") # infected to recovered
        
        S_t = S_t - S2I
        I_t = I_t + S2I - I2R
        
        rho_to_use = t >= switch_to_rho2 ? rho2 : (t > switch_to_rho1 ? rho1 : rho0)
        
        o_t = @trace(binom(S2I, rho_to_use), "o_$t")
    end
end