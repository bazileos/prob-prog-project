from bayesinf import frameEnv
import numpy as np
muA = 0.1
muB = 1.8

noise = 0.1 # coefficient of variation, meaning that sigma = omegaNoise * mu
total_steps = 20
env = frameEnv(noise, muA, muB)

observation, totalReward, done, nr_steps = env.reset(), 0, False, 0

while nr_steps < total_steps:

    # Actions were chosen by the DRL agent
    # To display only the inference part, they are chosen at random
    action = np.zeros(6)

    # Can not ensure that no assertion error will occur
    # In the complete script the code would just continue with the next episode
    try:
        state_, reward, done, _ = env.step(action)
    except AssertionError:
        print("Assertion Error during sampling")

    totalReward += reward
    state = state_

    print(f"Inference for decision step {env.decisionStep} is completed")

print("Episode ran successfully")