import sys
import argparse
import gym
import numpy as np
from time import sleep
import operator
from matplotlib import pyplot as plt
import math

# Parameters
hyperbolic = True
gamma = 0.99
k = 1000
eta = - math.log(gamma)/k

# Draw plots
def plot_returns(ep_lengths, epl_avg):
    plt.plot(ep_lengths)
    plt.plot(epl_avg)
    plt.legend(["Episode length", "500 episode average"])
    plt.title("Episode lengths")
    plt.show()

# Command line arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None,
                        help="Model to be tested")
    parser.add_argument("--env", type=str, default="FrozenLake8x8-v0",
                        help="Environment to use")
    return parser.parse_args(args)

def true_hyperbolic(t):
    return 1 / (1 + k * t)

def approx_hyperbolic(reward, total_reward):
    mod_reward = reward*math.exp(eta*(total_reward/reward - 1))
    return mod_reward

def update_values(env, state, value_est, neighbor_values, gamma):
    total_reward = 0
    for action in range(env.nA):
        transitions = np.array(env.P[state][action])
        (x,y) = np.shape(transitions)

        for i in range(x):
            next_state = int(transitions[i][1])
            prob = transitions[i][0]
            reward = transitions[i][2]
            if reward == 0: reward = 0.1  # cost
            if reward == 1: reward = 10  # final reward
            if hyperbolic:
                # if reward == 1: reward = 10 # approximate hyperbolic
                neighbor_values[action] += prob*(approx_hyperbolic(reward, total_reward) + gamma * value_est[next_state]) # approx hyperbolic
                total_reward += reward
            else:
                neighbor_values[action] += prob*(reward + gamma * value_est[next_state]) # exponential & true hyperbolic

    return neighbor_values

# The main function
def main(args):    
    # Create a Gym environment
    env = gym.make(args.env)

    # Print environment parameters 
    print("Environment:", args.env)
    print("Observation space dimensions:", env.nS)
    print("Action space dimensions:", env.nA)
    
    # Parameters
    delta = 0.1

    # Initialize state values and policy
    value_est = np.zeros(env.nS)
    policy = np.zeros((env.nS))

    # Reset the environment
    state = env.reset()

    while delta > 0.000001:
        delta = 0
        for state in range(env.nS):
            value_temp = value_est[state]
            neighbor_values = np.zeros(env.nA)
            neighbor_values = update_values(env, state, value_est, neighbor_values, gamma) # exponential & approx hyperbolic
            # update_values(env, state, value_est, neighbor_values, true_hyperbolic(t), k) # true hyperbolic
            index, value_est[state] = max(enumerate(neighbor_values), key=operator.itemgetter(1))
            policy[state] = index
            delta = max(delta,abs(value_est[state] - value_temp))

    print("Values: ", value_est)
    print("Policy: ", policy)

    # Save the state values and the policy
    fnames = "values.npy", "policy.npy"
    np.save(fnames[0], value_est)
    np.save(fnames[1], policy)
    print("Saved state values and policy to", *fnames)

    env.render()

    y_positions = np.array(list(range(0, 8)))
    x_positions = np.array(list(range(0, 8)))
    grid = np.reshape(value_est, (8,8))
    cmap = plt.cm.binary
    cmap.set_under((1,0,0,1))
    cmap.set_over((0,1,0,1))
    cmap.set_bad((0,0,1,1))
    # colored = cmap(grid)    
    for y_, y in enumerate(y_positions):
        for x_, x in enumerate(x_positions):
            color = "white"
            label = round(grid[y,x], 2)
            if x == 0 and y == 0:
                color = "black"
                label = "Start"
                grid[y,x] = 10
            elif x == 3 and y == 2:
                label = "Hole"
                grid[y,x] = np.nan
            elif x == 5 and y == 3:
                label = "Hole"
                grid[y,x] = np.nan
            elif x == 3 and y == 4:
                label = "Hole"
                grid[y,x] = np.nan
            elif x == 1 and y == 5:
                label = "Hole"
                grid[y,x] = np.nan
            elif x == 2 and y == 5:
                label = "Hole"
                grid[y,x] = np.nan
            elif x == 6 and y == 5:
                label = "Hole"
                grid[y,x] = np.nan
            elif x == 1 and y == 6:
                label = "Hole"
                grid[y,x] = np.nan
            elif x == 4 and y == 6:
                label = "Hole"
                grid[y,x] = np.nan
            elif x == 6 and y == 6:
                label = "Hole"
                grid[y,x] = np.nan
            elif x == 3 and y == 7:
                label = "Hole"
                grid[y,x] = np.nan
            elif x == 7 and y == 7:
                label = "Goal"
                grid[y,x] = 25
            plt.text(x, y, label, color=color, ha='center', va='center')
    plt.imshow(grid, cmap=cmap)
    plt.show()
    
    e=0
    for i_episode in range(100):
        c = env.reset()
        for t in range(10000):
            c, reward, done, info = env.step(int(policy[c]))
            if done:
                if reward == 1:
                    e +=1
                break
    print(" agent succeeded to reach goal {} out of 100 Episodes using this policy ".format(e+1))

    # s = env.reset()
    # for t in range(50):
    #     s, reward, done, info = env.step(int(policy[s]))
    #     env.render()
    #     if done:
    #         break

    # env.close()

# Entry point
if __name__ == "__main__":
    args = parse_args()
    main(args)
