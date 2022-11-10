using PyCall
using Gen

pushfirst!(pyimport("sys")."path", "./building_safety/")
@pyinclude("./building_safety/building_safety.py")
np = pyimport("numpy")

@dist lognormal(x, y) = exp(normal(x, y))

a = Vector{Float64}(undef,6)
b = Vector{Float64}(undef,6)

@gen function generative_model(env)
    @param muA::Float64
    @param muB::Float64

    sigmaA = muA * 0.5
    sigmaB = muB * 0.2
    logMuA = log(muA^2/sqrt(muA^2+sigmaA^2))
    logSigmaA = sqrt(log(1+sigmaA^2/muA^2))

    num_comp = env.model.numCols

    for t = 1:env.numSteps
        # Sample a and b for the components' damage increment distributions
        for i = 1:num_comp
            a[i] = @trace(lognormal(logMuA, logSigmaA), :A => t => i)
            b[i] = @trace(normal(muB, sigmaB), :B => t => i)
        end

        py"""
        $env.shapes += $a * ($env.ages + 1) ** $b - $a * $env.ages ** $b

        # create the continuous RV which is the expected value for the model
        damages = np.random.gamma(shape=$env.shapes, scale=1/$env.u, size=(5, $env.model.numCols))
        modalDispls = np.mean(np.abs(np.array([$env.model.eigen_analysis(dams) for dams in damages])), axis=0)
        obsMus = np.abs($env.model.eigen_analysis(np.random.gamma(shape=$env.shapes, scale=1/$env.u)))
        noises = obsMus * $env.noise

        obs_mu = modalDispls.reshape((-1, 1))
        obs_sigma = noises.reshape((-1, 1))
        """

        for i = 1:num_comp
            @trace(normal(py"obs_mu"[i], py"obs_sigma"[i]), :obs => t => i)
        end

        env.ages = env.ages .+ 1

        #println("Inference for decision step $t is completed")
    end
    #println("Episode ran successfully")
end


function train_model(env, data::ChoiceMap, muA_init, muB_init, max_iter, num_samples)
    init_param!(generative_model, :muA, muA_init)
    init_param!(generative_model, :muB, muB_init)
    update = ParamUpdate(FixedStepGradientDescent(0.01), generative_model)
    for iter=1:max_iter
        traces, weights = do_inference(env, data, num_samples)
        for (trace, weight) in zip(traces, weights)
            accumulate_param_gradients!(trace, nothing, weight)
        end

        apply!(update)
    end
end

function do_inference(env, data, num_samples)
    (traces, log_weights, _) = importance_sampling(generative_model, (env,), data, num_samples)
    weights = exp.(log_weights)
    (traces, weights,)
end

function parse_file(file_path::String)
    muA_init = 0
    muB_init = 0
    noise = 0
    num_iter = 0
    fixed_obs = [Vector{Float64}(undef,6) for _ in 1:20]

    f = open(file_path, "r")
    i = 1
    for line in readlines(f)
        if i == 1
            muA_init = parse(Float64, line)
        elseif i == 2
            muB_init = parse(Float64, line)
        elseif i == 3
            noise = parse(Float64, line)
        else
            row = split(line, ";")
            obs_row::Vector{Float64} = [
                parse(Float64, row[1]),
                parse(Float64, row[2]),
                parse(Float64, row[3]),
                parse(Float64, row[4]),
                parse(Float64, row[5]),
                parse(Float64, row[6])
            ]
            fixed_obs[i-3] = obs_row
        end
        i += 1
    end
    close(f)
    return (muA_init, muB_init, noise, fixed_obs,)
end

function main(file_path::String, num_iter::Integer)
    (muA_init, muB_init, noise, fixed_obs,) = parse_file(file_path)

    py"""
    env = frameEnv($noise, numSteps = 20)
    _, totalReward, done = env.reset(), 0, False
    """
    env = py"env"

    observations = choicemap()
    n = length(fixed_obs)
    num_comp = env.model.numCols
    for t = 1:n
        for i = 1:num_comp
            observations[:obs => t => i] = fixed_obs[t][i]
        end
    end

    max_train_iter = 10
    num_samples = trunc(Int, num_iter / 5)
    @time train_model(env, observations, muA_init, muB_init, max_train_iter, num_samples)

    muA_final = get_param(generative_model, :muA)
    muB_final = get_param(generative_model, :muB)

    println("muA: $muA_final")
    println("muB: $muB_final")
end