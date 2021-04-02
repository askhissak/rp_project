import gym
import numpy as np
from matplotlib import pyplot as plt
import sys
import argparse
import math

# Parameters
hyperbolic = True
gamma = 0.99
k = 100
eta = - math.log(gamma)/k

# Plot the heatmap
def plot_heatmap(values, x_grid, v_grid):
    # TODO: Plot the heatmap here using Seaborn or Matplotlib
    plt.title("Heatmap after training")
    plt.xlabel("Position")
    plt.xticks(np.arange(len(x_grid))+0.5,np.around(x_grid,2),rotation=90)
    plt.ylabel("Velocity")
    plt.yticks(np.arange(len(v_grid))+0.5,np.around(v_grid,2))
    plt.imshow(values)
    plt.colorbar()
    plt.show()

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
    parser.add_argument("--env", type=str, default="MountainCar-v0",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=10000,
                        help="Number of episodes to train for")
    parser.add_argument("--test_episodes", type=int, default=10,
                        help="Number of episodes to test for")
    parser.add_argument("--render_train", action='store_true',
                        help="Render each frame during training. Will be slower.")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    return parser.parse_args(args)

def true_hyperbolic(t):
    return 1 / (1 + k * t)

def approx_hyperbolic(reward, total_reward):
    mod_total_reward = reward*math.exp(eta*(total_reward/reward - 1))
    return mod_total_reward

# The main function
def main(args):
    # Create a Gym environment
    env = gym.make(args.env)

    # Parameters
    alpha = 0.1
    # epsilon = 0.2
    target_eps = 0.1
    a = round(args.train_episodes*target_eps/(1-target_eps))    

    # Diagnostics
    ep_lengths, epl_avg = [], []

    # Action space
    num_of_actions = env.action_space.n

    # State space - discretize continous state space
    discr = 36
    x_min, x_max = -1.2, 0.6
    v_min, v_max = -0.07, 0.07

    x_grid = np.linspace(x_min, x_max, discr-1) # state[0] table
    v_grid = np.linspace(v_min, v_max, discr-1) # state[1] table
    q_grid = np.zeros((discr, discr, num_of_actions)) # Q grid

    # Print environment parameters 
    print("Environment:", args.env)
    print("Observation space dimensions:", discr)
    print("Action space dimensions:", num_of_actions)

    ep_lengths, epl_avg = [], []
    for ep in range(args.train_episodes+args.test_episodes):
        test = ep >= args.train_episodes # check for train vs test
        state, done, steps = env.reset(), False, 0 # initialize environment - sample state
        epsilon = a/(a+ep)  # T1: GLIE 
        total_reward = 0

        while not done:
            #Discretize state values
            x_ind = np.digitize(state[0],x_grid)
            v_ind = np.digitize(state[1],v_grid)

            # Sample actions and choose one
            random_action = int(np.random.rand()*2)
            best_action = np.argmax(q_grid[x_ind,v_ind])
            action = np.random.choice([random_action,best_action],p=[epsilon,1-epsilon])

            # Step through the environment
            new_state, reward, done, _ = env.step(action)

            #Discretize new state values
            x_new_ind = np.digitize(new_state[0],x_grid)
            v_new_ind = np.digitize(new_state[1],v_grid)

            # if reward == -1: reward = 0.1
            # if reward == 0: reward = 100

            # Calculate Q values
            if not test:
                if not done: 
                    if hyperbolic:
                        q_grid[x_ind,v_ind,action] += alpha*(approx_hyperbolic(reward, total_reward) + gamma*(np.max(q_grid[x_new_ind,v_new_ind])) - q_grid[x_ind,v_ind,action])
                    else:
                        q_grid[x_ind,v_ind,action] += alpha*(reward + gamma*(np.max(q_grid[x_new_ind,v_new_ind])) - q_grid[x_ind,v_ind,action])
                else:
                    q_grid[x_ind,v_ind,action] += alpha*(reward + gamma*0 - q_grid[x_ind,v_ind,action])
            else:
                env.render()
            total_reward += reward
            state = new_state
            steps += 1

        #Heatmap after a single episode
        # if ep == 0:
        #     plt.title("Heatmap after a single episode")
        #     plt.xlabel("Angle Theta")
        #     plt.xticks(np.arange(len(x_grid))+0.5,np.around(x_grid,2),rotation=90)
        #     plt.ylabel("Position x")
        #     plt.yticks(np.arange(len(v_grid))+0.5,np.around(v_grid,2))     
        #     plt.imshow(np.max(q_grid,axis=2))
        #     plt.colorbar()       
        #     plt.show()

        # Print some logs
        ep_lengths.append(steps)
        epl_avg.append(np.mean(ep_lengths[max(0, ep-500):]))
        if ep % 200 == 0:
            print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep-200):])))

    # Save the Q-value array
    np.save("q_values.npy", q_grid)

    # Calculate the value function V(s)
    values = np.max(q_grid, axis=2)
    np.save("value_func.npy", values)  

    # Plot results
    plot_heatmap(values, x_grid, v_grid)
    plot_returns(ep_lengths, epl_avg)

# Entry point
if __name__ == "__main__":
    args = parse_args()
    main(args)