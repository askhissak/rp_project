import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from agent import ReplayBuffer, NormalizedActions, ValueNetwork, SoftQNetwork, PolicyNetwork#, soft_q_update

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

# # Policy training function
# def train(env_name, print_things=True, train_run_id=0, train_episodes=5000):
#     # Create a Gym environment
#     env = gym.make(env_name)

#     # Get dimensionalities of actions and observations
#     action_space_dim = env.action_space.shape[-1]
#     observation_space_dim = env.observation_space.shape[-1]

#     # Instantiate agent and its policy
#     policy = Policy(observation_space_dim, action_space_dim)
#     agent = Agent(policy)

#     # Arrays to keep track of rewards
#     reward_history, timestep_history = [], []
#     average_reward_history = []

#     # Run actual training
#     for episode_number in range(train_episodes):
#         reward_sum, timesteps = 0, 0
#         done = False
#         # Reset the environment and observe the initial state
#         observation = env.reset()

#         # Loop until the episode is over
#         while not done:
#             # Get action from the agent
#             action, action_probabilities, state_values = agent.get_action(observation)
#             previous_observation = observation

#             # Perform the action on the environment, get new state and reward
#             observation, reward, done, info = env.step(action.detach().numpy())
            
#             _, _, next_state_values = agent.get_action(observation)

#             # Store action's outcome (so that the agent can improve its policy)
#             # T3
#             # agent.store_outcome(previous_observation, action_probabilities, action, reward, state_values)

#             #T4
#             agent.store_outcome(previous_observation, action_probabilities, action, reward, state_values, next_state_values)

#             if timesteps != 0 and timesteps%10 == 0:
#                 agent.episode_finished(episode_number, done)

#             # Store total episode reward
#             reward_sum += reward
#             timesteps += 1

#         if print_things:
#             print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
#                   .format(episode_number, reward_sum, timesteps))

#         # Bookkeeping (mainly for generating plots)
#         reward_history.append(reward_sum)
#         timestep_history.append(timesteps)
#         if episode_number > 100:
#             avg = np.mean(reward_history[-100:])
#         else:
#             avg = np.mean(reward_history)
#         average_reward_history.append(avg)

#         # Let the agent do its magic (update the policy)
#         # agent.episode_finished(episode_number, done) # T3

#     # Training is finished - plot rewards
#     if print_things:
#         plt.plot(reward_history)
#         plt.plot(average_reward_history)
#         plt.legend(["Reward", "100-episode average"])
#         plt.title("Reward history")
#         plt.show()
#         print("Training finished.")
#     data = pd.DataFrame({"episode": np.arange(len(reward_history)),
#                          "train_run_id": [train_run_id]*len(reward_history),
#                          # TODO: Change algorithm name for plots, if you want
#                          "algorithm": ["PG"]*len(reward_history),
#                          "reward": reward_history})
#     torch.save(agent.policy.state_dict(), "model_%s_%d.mdl" % (env_name, train_run_id))
#     return data


# # Function to test a trained policy
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


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
#     parser.add_argument("--env", type=str, default="ContinuousCartPole-v0", help="Environment to use")
#     parser.add_argument("--train_episodes", type=int, default=5000, help="Number of episodes to train for")
#     parser.add_argument("--render_test", action='store_true', help="Render test")
#     args = parser.parse_args()

#     # If no model was passed, train a policy from scratch.
#     # Otherwise load the policy from the file and go directly to testing.
#     if args.test is None:
#         try:
#             train(args.env, train_episodes=args.train_episodes)
#         # Handle Ctrl+C - save model and go to tests
#         except KeyboardInterrupt:
#             print("Interrupted!")
#     else:
#         state_dict = torch.load(args.test)
#         print("Testing...")
#         test(args.env, 100, state_dict, args.render_test)

env = NormalizedActions(gym.make("Pendulum-v0"))

action_dim = env.action_space.shape[0]
state_dim  = env.observation_space.shape[0]
hidden_dim = 256

value_net        = ValueNetwork(state_dim, hidden_dim).to(device)
target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)

soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)
    

value_criterion  = nn.MSELoss()
soft_q_criterion = nn.MSELoss()

value_lr  = 3e-4
soft_q_lr = 3e-4
policy_lr = 3e-4

value_optimizer  = optim.Adam(value_net.parameters(), lr=value_lr)
soft_q_optimizer = optim.Adam(soft_q_net.parameters(), lr=soft_q_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)


replay_buffer_size = 1000000
replay_buffer = ReplayBuffer(replay_buffer_size)

max_frames  = 40000
max_steps   = 500
frame_idx   = 0
rewards     = []
batch_size  = 128

max_frames  = 40000

while frame_idx < max_frames:
    state = env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        action = policy_net.get_action(state)
        next_state, reward, done, _ = env.step(action)
        
        replay_buffer.push(state, action, reward, next_state, done)
        if len(replay_buffer) > batch_size:
            soft_q_update(batch_size)
        
        state = next_state
        episode_reward += reward
        frame_idx += 1
        
        if frame_idx % 1000 == 0:
            plot(frame_idx, rewards)
        
        if done:
            break
        
    rewards.append(episode_reward)

