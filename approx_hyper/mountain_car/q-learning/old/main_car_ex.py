import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd

# k = 0.0001 # true hyperbolic
k = 0.1 # approx hyperbolic
gamma = 0.995

def trueHyperbolic(t):
    return 1/(1+k*t)

def approxHyperbolic(t):
    
    return 0

def exponential(t):
    return gamma**t

def plotCompare():
    T = np.array(range(200))
    true_hyp = []
    expon = []
    # approx_hyp = []
    for t_ in T:
        true_hyp.append(trueHyperbolic(t_))
        # expon.append(exponential(t_))
        # coef_hyp.append(hyperbolic_coefs(t_))

    plt.plot(T, true_hyp, label="true hyperbolic")
    # plt.plot(T, expon, label="exponential")
    # plt.plot(T, coef_hyp, label="approx")

    plt.xlabel('Time delay')
    plt.ylabel('Discount rate')
    plt.title('Hyperbolic vs Exponential')
    plt.legend()
    plt.show() 
    return 

# Define Q-learning function
def QLearning(env, learning, discount, epsilon, min_eps, episodes):
    # Determine size of discretized state space
    num_states = (env.observation_space.high - env.observation_space.low)*\
                    np.array([10, 100])
    num_states = np.round(num_states, 0).astype(int) + 1
    
    # Initialize Q table
    Q = np.random.uniform(low = -1, high = 1, 
                        size = (num_states[0], num_states[1], 
                                env.action_space.n))
    
    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []
    
    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps)/episodes
    
    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        state = env.reset()
        
        # Discretize state
        state_adj = (state - env.observation_space.low)*np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)
        t = 0
        while done != True:   
            # Render environment for last five episodes
            if i >= (episodes - 20):
                env.render()
                
            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]]) 
            else:
                action = np.random.randint(0, env.action_space.n)
                
            # Get next state and reward
            state2, reward, done, info = env.step(action) 
            
            # Discretize state2
            state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)
            
            reward_transform = reward*np.exp(-(np.log(discount)/k)*(1/reward-1))

            #Allow for terminal states
            if done and state2[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward
                
            # Adjust Q value for current state
            # reward*np.exp(-(np.log(discount)/k)*(1/reward-1))
            else:
                delta = learning*(reward_transform + 
                                discount*np.max(Q[state2_adj[0], state2_adj[1]]) - 
                                Q[state_adj[0], state_adj[1],action])
                Q[state_adj[0], state_adj[1], action] += delta
                                    
            # Update variables
            tot_reward += reward
            state_adj = state2_adj

            t += 1
        
        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= reduction
        
        # Track rewards
        reward_list.append(tot_reward)
        
        if (i+1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []
            
        if (i+1) % 100 == 0:    
            print('Episode {} Average Reward: {}'.format(i+1, ave_reward))
            
    env.close()
    
    return ave_reward_list

# Policy training function
def train(print_things=True, train_run_id=0, train_episodes=5000):
    # Create a Gym environment
    env = gym.make('MountainCar-v0')
    env.reset()

    # Run Q-learning algorithm
    rewards = QLearning(env, 0.2, 0.9, 0.8, 0, train_episodes)

    # plotCompare()

    # Plot Rewards
    plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig('rewards.jpg')     
    plt.close() 

# Function to test a trained policy
# def test(env_name, episodes, params, render):
#     # Create a Gym environment
#     env = gym.make(env_name)

#     # Get dimensionalities of actions and observations
#     action_space_dim = env.action_space.shape[-1]
#     observation_space_dim = env.observation_space.shape[-1]

#     # Instantiate agent and its policy
#     policy = Policy(observation_space_dim, action_space_dim)
#     policy.load_state_dict(params)
#     agent = Agent(policy)

#     test_reward, test_len = 0, 0
#     for ep in range(episodes):
#         done = False
#         observation = env.reset()
#         while not done:
#             # Similar to the training loop above -
#             # get the action, act on the environment, save total reward
#             # (evaluation=True makes the agent always return what it thinks to be
#             # the best action - there is no exploration at this point)
#             action, _ = agent.get_action(observation, evaluation=True)
#             observation, reward, done, info = env.step(action.detach().cpu().numpy())

#             if render:
#                 env.render()
#             test_reward += reward
#             test_len += 1
#     print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--train_episodes", type=int, default=5000, help="Number of episodes to train for")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    args = parser.parse_args()

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        try:
            train(train_episodes=args.train_episodes)
        # Handle Ctrl+C - save model and go to tests
        except KeyboardInterrupt:
            print("Interrupted!")
    else:
        state_dict = torch.load(args.test)
        print("Testing...")
        # test(args.env, 100, state_dict, args.render_test)

