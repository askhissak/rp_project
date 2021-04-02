import numpy as np
from food_truck import FoodTruck
from time import sleep
import math
import operator
# from matplotlib import pyplot as plt

# #Set new plot size
# plt.rcParams["figure.figsize"] = [9.6,7.2]

# Set up the environment
env = FoodTruck()

# Parameters
hyperbolic = True
gamma = 0.9
k = 0.1 #0.9
eta = - math.log(gamma)/k

def true_hyperbolic(t):
    return 1 / (1 + k * t)

def approx_hyperbolic(reward, next_reward):
    mod_reward = reward*math.exp(eta*(next_reward/reward - 1))
    return mod_reward

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
    delta = 3
    n_iter = 0
    ep_lengths, epl_avg = [], []
    while delta > 1:
        delta = 0
        t = 0
        for i in range(env.w):
            for j in range(env.h):
                old_action = policy[i, j]
                old_value = value_est[i, j]
                neighbor_values = np.zeros(4)
                # gamma = true_hyperbolic(t)

                total_reward = 0 # approx hyperbolic
                for action in range(env.NO_ACTIONS):
                    print(action)
                    for k in range(len(env.transitions[i, j, action])):
                        next_state, reward, done, prob = env.transitions[i, j, action][k]
                        # if env.transitions[i, j, env.LEFT][k][0] == None and env.transitions[i, j, env.LEFT][k][2] and env.visited == 0:
                        #     state_left, reward_left, done_left, prob_left = (i,j), env.delayed_rewards[i,j], False, 1
                        #     env.visited = 1
                        # elif env.transitions[i, j, env.LEFT][k][0] == None and env.transitions[i, j, env.LEFT][k][2] and env.visited == 1:
                        #     env.visited = 0
                        if done is not True:
                            if next_state != None:
                                if hyperbolic:
                                    # if reward_left == 0: reward_left = 0.1 # approx hyperbolic
                                    neighbor_values[0] = neighbor_values[0] + prob * \
                                        (approx_hyperbolic(reward, total_reward) +
                                        gamma * value_est[next_state])  # approx hyperbolic
                                    total_reward += reward # approx hyperbolic

                                # neighbor_values[0] = neighbor_values[0] + prob_left * (
                                #     reward_left + true_hyperbolic(t) * value_est[state_left])  # true hyperbolic
                                else:
                                    neighbor_values[0] = neighbor_values[0] + prob * \
                                        (reward + gamma *
                                         value_est[next_state])  # exponential

                # total_reward_left = 0 # approx hyperbolic
                # total_reward_down = 0 # approx hyperbolic
                # total_reward_right = 0 # approx hyperbolic
                # total_reward_up = 0 # approx hyperbolic
                # for k in range(len(env.transitions[i, j, env.LEFT])):
                #     state_left, reward_left, done_left, prob_left = env.transitions[i, j, env.LEFT][k]
                #     # if env.transitions[i, j, env.LEFT][k][0] == None and env.transitions[i, j, env.LEFT][k][2] and env.visited == 0:
                #     #     state_left, reward_left, done_left, prob_left = (i,j), env.delayed_rewards[i,j], False, 1
                #     #     env.visited = 1
                #     # elif env.transitions[i, j, env.LEFT][k][0] == None and env.transitions[i, j, env.LEFT][k][2] and env.visited == 1:
                #     #     env.visited = 0
                #     if done_left is not True:
                #         if state_left != None:
                #             # if reward_left == 0: reward_left = 0.1 # approx hyperbolic
                #             neighbor_values[0] = neighbor_values[0] + prob_left * \
                #                 (approx_hyperbolic(reward_left, total_reward_left) +
                #                  gamma * value_est[state_left])  # approx hyperbolic

                #             # neighbor_values[0] = neighbor_values[0] + prob_left * (
                #             #     reward_left + true_hyperbolic(t) * value_est[state_left])  # true hyperbolic

                #             # neighbor_values[0] = neighbor_values[0] + prob_left * \
                #             #     (reward_left + gamma *
                #             #      value_est[state_left])  # exponential
                #             total_reward_left += reward_left # approx hyperbolic
                # for k in range(len(env.transitions[i, j, env.DOWN])):
                #     state_down, reward_down, done_down, prob_down = env.transitions[i, j, env.DOWN][k]
                #     if env.transitions[i, j, env.DOWN][k][0] == None and env.transitions[i, j, env.DOWN][k][2] and env.visited == 0:
                #         state_down, reward_down, done_down, prob_down = (i,j), env.delayed_rewards[i,j], False, 1
                #         env.visited = 1
                #     elif env.transitions[i, j, env.DOWN][k][0] == None and env.transitions[i, j, env.DOWN][k][2] and env.visited == 1:
                #         env.visited = 0
                #     if done_down is not True:
                #         if state_down != None:
                #             # if reward_down == 0: reward_down = 0.1 # approx hyperbolic
                #             neighbor_values[1] = neighbor_values[1] + prob_down * \
                #                 (approx_hyperbolic(reward_down, total_reward_down) +
                #                  gamma * value_est[state_down])  # approx hyperbolic

                #             # neighbor_values[1] = neighbor_values[1] + prob_down * (
                #             #     reward_down + true_hyperbolic(t) * value_est[state_down])  # true hyperbolic

                #             # neighbor_values[1] = neighbor_values[1] + prob_down * \
                #             #     (reward_down + gamma *
                #             #      value_est[state_down])  # exponential
                #             total_reward_down += reward_down # approx hyperbolic
                # for k in range(len(env.transitions[i, j, env.RIGHT])):
                #     state_right, reward_right, done_right, prob_right = env.transitions[i, j, env.RIGHT][k]
                #     if env.transitions[i, j, env.RIGHT][k][0] == None and env.transitions[i, j, env.RIGHT][k][2] and env.visited == 0:
                #         state_right, reward_right, done_right, prob_right = (i,j), env.delayed_rewards[i,j], False, 1
                #         env.visited = 1
                #     elif env.transitions[i, j, env.RIGHT][k][0] == None and env.transitions[i, j, env.RIGHT][k][2] and env.visited == 1:
                #         env.visited = 0
                #     if done_right is not True:
                #         if state_right != None:
                #             # if reward_right == 0: reward_right = 0.1 # approx hyperbolic
                #             neighbor_values[2] = neighbor_values[2] + prob_right * \
                #                 (approx_hyperbolic(reward_right, total_reward_right) +
                #                  gamma * value_est[state_right])  # approx hyperbolic

                #             # neighbor_values[2] = neighbor_values[2] + prob_right * (
                #             #     reward_right + true_hyperbolic(t) * value_est[state_right])  # true hyperbolic

                #             # neighbor_values[2] = neighbor_values[2] + prob_right * \
                #             #     (reward_right + gamma * 
                #             #     value_est[state_right])  # exponential
                #             total_reward_right += reward_right # approx hyperbolic
                # for k in range(len(env.transitions[i, j, env.UP])):
                #     state_up, reward_up, done_up, prob_up = env.transitions[i, j, env.UP][k]
                #     if env.transitions[i, j, env.UP][k][0] == None and env.transitions[i, j, env.UP][k][2] and env.visited == 0:
                #         state_up, reward_up, done_up, prob_up = (i,j), env.delayed_rewards[i,j], False, 1
                #         env.visited = 1
                #     elif env.transitions[i, j, env.UP][k][0] == None and env.transitions[i, j, env.UP][k][2] and env.visited == 1:
                #         env.visited = 0
                #     if done_up is not True:
                #         if state_up != None:
                #             # if reward_up == 0: reward_up = 0.1 # approx hyperbolic
                #             neighbor_values[3] = neighbor_values[3] + prob_up * \
                #                 (approx_hyperbolic(reward_up, total_reward_up) +
                #                  gamma * value_est[state_up])  # approx hyperbolic

                #             # neighbor_values[3] = neighbor_values[3] + prob_up * (
                #             #     reward_up + true_hyperbolic(t) * value_est[state_up])  # true hyperbolic

                #             # neighbor_values[3] = neighbor_values[3] + prob_up * \
                #             #     (reward_up + gamma *
                #             #      value_est[state_up])  # exponential
                #             total_reward_up += reward_up # approx hyperbolic
                # print(i, j)
                # print(neighbor_values)
                new_action, value_est[i, j] = max(enumerate(neighbor_values), key=operator.itemgetter(1))
                # print(value_est[i, j])
                # print(index)
                policy[i, j] = new_action
                print("Delta", delta)
                print("State", i,j)
                print("Old value", old_value)
                print("New value", value_est[i,j])
                delta = max(delta, abs(value_est[i, j] - old_value))
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

        # ep_lengths.append(t)
        # epl_avg.append(np.mean(ep_lengths[max(0, n_iter-2):]))
        # if n_iter % 2 == 0:
        #     print("Episode {}, average timesteps: {:.2f}".format(n_iter, np.mean(ep_lengths[max(0, n_iter-2):])))

        n_iter += 1
        # print(n_iter)
        #delta = np.copy(max(delta, abs(old_value-value_est)))
        

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
            if state == None and done and env.visited == 0:
                state, reward, done, _ = state, env.delayed_rewards[state], False, 1
                env.visited = 1
            elif state == None and done and env.visited == 1:
                env.visited = 0
            ini_g += (gamma**j)*reward

            # # Render and sleep
            env.render()
            sleep(0.5)
            j += 1
        disc_returns[i] = ini_g
        state = env.reset()
    print(np.mean(disc_returns))
    print(np.std(disc_returns))

    # Draw plots
    # plt.plot(ep_lengths)
    # plt.plot(epl_avg)
    # plt.legend(["Episode length", "500 episode average"])
    # plt.title("Episode lengths")
    # plt.show()
