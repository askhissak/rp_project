import numpy as np
from food_truck import FoodTruck
from time import sleep
import math
import operator

# Set up the environment
env = FoodTruck()

# Parameters
gamma = 0.99
k = 0.01 #0.9
eta = - math.log(gamma)/k

def true_hyperbolic(t):
    return 1 / (1 + k * t)

def approx_hyperbolic(reward, total_reward):
    mod_total_reward = reward*math.exp(eta*(total_reward/reward - 1))
    return mod_total_reward

if __name__ == "__main__":
    # Reset the environment
    state = env.reset()

    # Compute state values and the policy
    value_est, policy = np.zeros((env.w, env.h)), np.zeros((env.w, env.h))

    env.clear_text()
    env.draw_values(value_est)
    env.draw_actions(policy)
    env.render()
    sleep(5)

    # policy_stable = True
    delta = 0.1
    n_iter = 0
    while n_iter < 100:
        delta = 0
        t = 0
        for i in range(env.w):
            for j in range(env.h):
                old_action = policy[i, j]
                value_temp = value_est[i, j]
                neighbor_values = np.zeros(4)
                # gamma = true_hyperbolic(t)
                # total_reward_left = 0 # approx hyperbolic
                # total_reward_down = 0 # approx hyperbolic
                # total_reward_right = 0 # approx hyperbolic
                # total_reward_up = 0 # approx hyperbolic
                for k in range(len(env.transitions[i, j, env.LEFT])):
                    if env.transitions[i, j, env.LEFT][k][2] is not True:
                        if env.transitions[i, j, env.LEFT][k][0] != None:
                            reward = env.transitions[i, j, env.LEFT][k][1]
                            # if reward == 0: reward = 0.1 # approx hyperbolic
                            # neighbor_values[0] = neighbor_values[0] + env.transitions[i, j, env.LEFT][k][3]*( # approx hyperbolic
                            #     approx_hyperbolic(reward, total_reward_left) + gamma*value_est[env.transitions[i, j, env.LEFT][k][0]])
                            # neighbor_values[0] = neighbor_values[0] + env.transitions[i, j, env.LEFT][k][3]*( # true hyperbolic
                            #     reward + true_hyperbolic(t)*value_est[env.transitions[i, j, env.LEFT][k][0]])
                            neighbor_values[0] = neighbor_values[0] + env.transitions[i, j, env.LEFT][k][3]*( # exponential
                                reward + gamma*value_est[env.transitions[i, j, env.LEFT][k][0]])
                            # total_reward_left += reward # approx hyperbolic
                for k in range(len(env.transitions[i, j, env.DOWN])):
                    if env.transitions[i, j, env.DOWN][k][2] is not True:
                        if env.transitions[i, j, env.DOWN][k][0] != None:
                            reward = env.transitions[i, j, env.DOWN][k][1]
                            # if reward == 0: reward = 0.1 # approx hyperbolic
                            # neighbor_values[1] = neighbor_values[1] + env.transitions[i, j, env.DOWN][k][3]*( # approx hyperbolic
                            #     approx_hyperbolic(reward, total_reward_down) + gamma*value_est[env.transitions[i, j, env.DOWN][k][0]])
                            # neighbor_values[1] = neighbor_values[1] + env.transitions[i, j, env.DOWN][k][3]*( # true hyperbolic
                            #     reward + true_hyperbolic(t)*value_est[env.transitions[i, j, env.DOWN][k][0]])
                            neighbor_values[1] = neighbor_values[1] + env.transitions[i, j, env.DOWN][k][3]*( # exponential
                                reward + gamma*value_est[env.transitions[i, j, env.DOWN][k][0]])
                            # total_reward_down += reward # approx hyperbolic
                for k in range(len(env.transitions[i, j, env.RIGHT])):
                    if env.transitions[i, j, env.RIGHT][k][2] is not True:
                        if env.transitions[i, j, env.RIGHT][k][0] != None:
                            reward = env.transitions[i, j, env.RIGHT][k][1]
                            # if reward == 0: reward = 0.1 # approx hyperbolic
                            # neighbor_values[2] = neighbor_values[2] + env.transitions[i, j, env.RIGHT][k][3]*( # approx hyperbolic
                            #     approx_hyperbolic(reward, total_reward_right) + gamma*value_est[env.transitions[i, j, env.RIGHT][k][0]])
                            # neighbor_values[2] = neighbor_values[2] + env.transitions[i, j, env.RIGHT][k][3]*( # true hyperbolic
                            #     reward + true_hyperbolic(t)*value_est[env.transitions[i, j, env.RIGHT][k][0]])
                            neighbor_values[2] = neighbor_values[2] + env.transitions[i, j, env.RIGHT][k][3]*( # exponential
                                reward + gamma*value_est[env.transitions[i, j, env.RIGHT][k][0]])
                            # total_reward_right += reward # approx hyperbolic
                for k in range(len(env.transitions[i, j, env.UP])):
                    if env.transitions[i, j, env.UP][k][2] is not True:
                        if env.transitions[i, j, env.UP][k][0] != None:
                            reward = env.transitions[i, j, env.UP][k][1]
                            # if reward == 0: reward = 0.1 # approx hyperbolic
                            # neighbor_values[3] = neighbor_values[3] + env.transitions[i, j, env.UP][k][3]*( # approx hyperbolic
                            #     approx_hyperbolic(reward, total_reward_up) + gamma*value_est[env.transitions[i, j, env.UP][k][0]])
                            # neighbor_values[3] = neighbor_values[3] + env.transitions[i, j, env.UP][k][3]*( # true hyperbolic
                            #     reward + true_hyperbolic(t)*value_est[env.transitions[i, j, env.UP][k][0]])
                            neighbor_values[3] = neighbor_values[3] + env.transitions[i, j, env.UP][k][3]*( # exponential
                                reward + gamma*value_est[env.transitions[i, j, env.UP][k][0]])
                            # total_reward_up += reward # approx hyperbolic
                # print(i, j)
                # print(neighbor_values)
                index, value_est[i, j] = max(enumerate(neighbor_values), key=operator.itemgetter(1))
                # print(value_est[i, j])
                # print(index)
                policy[i, j] = index
                delta = max(delta, abs(value_est[i, j] - value_temp))
                t += 1
        #         if old_action != policy[i,j]:
        #             policy_stable = False
        # if policy_stable:
        #     break
        env.clear_text()
        env.draw_values(value_est)
        env.draw_actions(policy)
        env.render()
        sleep(1)
        n_iter += 1
        # print(n_iter)
        #delta = np.copy(max(delta, abs(value_temp-value_est)))

    # Show the values and the policy
    env.clear_text()
    env.draw_values(value_est)
    env.draw_actions(policy)
    env.render()
    sleep(1)

    # Save the state values and the policy
    fnames = "values.npy", "policy.npy"
    np.save(fnames[0], value_est)
    np.save(fnames[1], policy)
    print("Saved state values and policy to", *fnames)

    # Run a single episode
    # TODO: Run multiple episodes and compute the discounted returns (Task 4)
    num_ep = 1
    disc_returns = list(range(num_ep))
    for i in range(num_ep):
        done = False
        j = 0
        ini_g = 0
        while not done:
            # Select a random action
            # TODO: Use the policy to take the optimal action (Task 2)
            action = policy[state]

            # Step the environment
            state, reward, done, _ = env.step(action)
            ini_g += (gamma**j)*reward

            # # Render and sleep
            env.render()
            sleep(0.5)
            j += 1
        disc_returns[i] = ini_g
        state = env.reset()
    print(np.mean(disc_returns))
    print(np.std(disc_returns))
