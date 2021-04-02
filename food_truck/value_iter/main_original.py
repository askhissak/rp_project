import numpy as np
from food_truck_original import FoodTruck
from time import sleep
import math
import operator
from matplotlib import pyplot as plt

# Set up the environment
env = FoodTruck()

# Parameters
hyperbolic = True
gamma = 0.99
k = 100 #0.9
eta = - math.log(gamma)/k

def true_hyperbolic(t):
    return 1 / (1 + k * t)

def approx_hyperbolic(reward, total_reward):
    mod_reward = reward*math.exp(eta*(total_reward/reward - 1))
    return mod_reward

if __name__ == "__main__":
    # Reset the environment
    state = env.reset()

    # Compute state values and the policy
    value_est, policy = np.zeros(env.n_states_extended), np.zeros(env.n_states_extended)

    # policy_stable = True
    delta = 1
    while delta > 0.000001:
        delta = 0
        for state in range(env.n_states_extended):
            env.state = state
            old_value = value_est[state]
            neighbor_values = np.zeros(4)
            total_reward = 0 # hyperbolic
            for action in range(env.n_actions):
                next_state, reward, done, _ = env.step(action)
                if not done:
                    if hyperbolic:
                        neighbor_values[action] += approx_hyperbolic(reward, total_reward) + gamma * value_est[next_state]  # hyperbolic
                        total_reward += reward # approx hyperbolic
                    else:
                        neighbor_values[action] += reward + gamma * value_est[next_state]  # exponential

            # print(neighbor_values)
            new_action, value_est[state] = max(enumerate(neighbor_values), key=operator.itemgetter(1))
            policy[state] = new_action
            delta = max(delta, abs(value_est[state] - old_value))

    print(value_est)
    print(policy)

    y_positions = np.array(list(range(7, -1,-1)))
    x_positions = np.array(list(range(0, 6)))
    grid = np.reshape(value_est[:48], (8,6))
    cmap = plt.cm.binary
    # cmap.set_under((1,0,0,1))
    # cmap.set_over((0,1,0,1))
    cmap.set_bad((0,0,1,1))
    # colored = cmap(grid)    
    for y_, y in enumerate(y_positions):
        for x_, x in enumerate(x_positions):
            color = "white"
            label = round(grid[y,x], 2)
            # if x == 0 and y == 0:
            #     color = "black"
            #     label = "Start"
            #     grid[y,x] = -800
            # elif label == 0.35:
            #     label = "Hole"
            #     grid[y,x] = np.nan
            # if x == 7 and y == 7:
            #     label = "Goal"
            #     grid[y,x] = 5000
            plt.text(x, y, label, color=color, ha='center', va='center')
    plt.imshow(grid, cmap=cmap)
    plt.show()

    # Run test episodes
    num_ep = 10
    disc_returns = list(range(num_ep))
    test_state = env.reset()
    for i in range(num_ep):
        done = False
        j = 0
        ini_g = 0
        while not done:
            # Select a random action
            action = policy[test_state]

            # Step the environment
            test_state, reward, done, _ = env.step(int(action))
            ini_g += (gamma**j)*reward

            j += 1
        disc_returns[i] = ini_g
        test_state = env.reset()
    print(np.mean(disc_returns))
    print(np.std(disc_returns))
