import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        self.fc3 = torch.nn.Linear(self.hidden, 1)
        self.sigma = torch.nn.Parameter(torch.tensor([10.])) # T2b
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        mu = self.fc2_mean(x)
        sigma = F.softplus(self.sigma) # T2

        # DONE: Instantiate and return a normal distribution
        # with mean mu and std of sigma (T1)
        dist = Normal(mu, sigma)

        # DONE: Add a layer for state value calculation (T3)
        x = self.fc3(x)
        return dist, x


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.values = []
        self.next_values = []

    def episode_finished(self, episode_number, done):
        action_probs = torch.stack(self.action_probs, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        values = torch.stack(self.values, dim=1).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards, self.values = [], [], [], []
        next_values = torch.stack(self.next_values, dim=1).to(self.train_device).squeeze(-1)
        self.next_values = []
        
        # DONE: Compute discounted rewards (use the discount_rewards function)
        # T3
        discounted_rewards = discount_rewards(rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards) # T1c
        discounted_rewards /= torch.std(discounted_rewards) # T1c
        # T4
        # if not done:
        #     td0_rewards = rewards + self.gamma*next_values
        # else:
        #     td0_rewards = rewards

        # DONE: Compute critic loss and advantages (T3)
        advantage = discounted_rewards - values # T3
        # advantage = td0_rewards - values # T4
        critic_loss  = advantage.pow(2).mean()

        # DONE: Compute the optimization term (T1, T3)
        actor_loss = -(action_probs * advantage.detach()).mean()
        loss = actor_loss + 0.1*critic_loss

        # DONE: Compute the gradients of loss w.r.t. network parameters (T1)
        loss.backward()

        # DONE: Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(np.ndarray([observation])).float().to(self.train_device)

        # DONE: Pass state x through the policy network (T1)
        dist, value = self.policy.forward(x)

        # DONE: Return mean if evaluation, else sample from the distribution
        # returned by the policy (T1)
        if evaluation:
            action =  torch.mean(dist)
        else:
            action = dist.sample()

        # DONE: Calculate the log probability of the action (T1)
        act_log_prob = dist.log_prob(action)
    
        # DONE: Return state value prediction, and/or save it somewhere (T3)

        return action, act_log_prob, value

    # T3
    def store_outcome(self, observation, action_prob, action_taken, reward, state_values):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.values.append(state_values)

    # T4
    # def store_outcome(self, observation, action_prob, action_taken, reward, state_values, next_state_values):
    #     self.states.append(observation)
    #     self.action_probs.append(action_prob)
    #     self.rewards.append(torch.Tensor([reward]))
    #     self.values.append(state_values)
    #     self.next_values.append(next_state_values)