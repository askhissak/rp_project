import gym
from gym import spaces
from gym.utils import seeding
import torch
from itertools import product
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle, Circle
import numpy as np

WALL = -1
GRID = 1
DONUT_NORTH_FIRST = 2
DONUT_SOUTH_FIRST = 3
VEGAN_FIRST = 4
NOODLE_FIRST = 5

MAPS = {"original": torch.tensor([  [DONUT_SOUTH_FIRST, WALL, WALL, GRID, WALL, WALL],
                                    [GRID, GRID, GRID, GRID, WALL, WALL],
                                    [WALL, WALL, WALL, GRID, WALL, NOODLE_FIRST],
                                    [WALL, WALL, WALL, GRID, GRID, GRID],
                                    [WALL, WALL, WALL, GRID, WALL,GRID],
                                    [WALL, WALL, DONUT_NORTH_FIRST, GRID, WALL, GRID],
                                    [WALL, WALL, WALL, GRID, GRID, GRID],
                                    [WALL, WALL, WALL, WALL, VEGAN_FIRST, WALL]])}

REWARDS = { "original": ((10,-10), (10,-10), (-10,20), (0,0)), 
            "version_1": ((11,-10), (11,-10), (-10,21), (0,0)),
            "version_1_effort": ((11-1,-10-1), (11-1,-10-1), (-10-1,20-1), (0-1,0-1)),
            "version_2_effort": ((11-0.2,-10-0.2), (11-0.2,-10-0.2), (-10-0.2,20-0.2), (0-0.2,0-0.2))}


class FoodTruck(gym.Env):
    NO_ACTIONS = 4
    LEFT, DOWN, RIGHT, UP = range(NO_ACTIONS)

    # Required
    def __init__(self, reward_type = "version_1", time_cost = -0.01, hit_wall_cost=-2):
        self.grid_map = MAPS["original"]
        self.time_cost = time_cost
        self.hit_wall_cost = hit_wall_cost
        self.rest_rewards = REWARDS[reward_type]
        self.seed()

        self.h = self.grid_map.shape[0]
        self.w = self.grid_map.shape[1]       
        self.init_x = 3
        self.init_y = 1 
        self.state = 0, 0

        self.rewards = np.zeros((self.w, self.h))
        self.terminate = np.zeros((self.w,self.h))
        self.delayed_rewards = np.zeros((self.w, self.h))
        # self.states = self.grid_map.reshape(-1)
        # self.wrong_action_prob = 0.0005
        self.visited = np.zeros(4)
        self.transitions = None

        #Render
        self.fig, self.ax = None, None
        # self.maximized = False
        self.episode_finished = False

        self.grid1_x = 3, 6
        self.grid2_x = 3, 6
        self.grid3_x = 0, 4
        self.grid4_x = 3, 4
        self.grid5_x = 5, 6
        self.grid1_y = 6, 7
        self.grid2_y = 3, 4
        self.grid3_y = 1, 2
        self.grid4_y = 0, 7
        self.grid5_y = 3, 7

        self.rest1_x = 0, 1
        self.rest1_y = 0, 1
        self.rest2_x = 5, 6
        self.rest2_y = 2, 3
        self.rest3_x = 2, 3
        self.rest3_y = 5, 6
        self.rest4_x = 4, 5
        self.rest4_y = 7, 8

        # self.n_states = self.h * self.w
        # self.action_space = spaces.Discrete(self.n_actions)
        # self.observation_space = spaces.Discrete(self.n_states)

        # self.R = torch.zeros((self.n_states,self.n_actions,self.n_states)) + self.time_cost #state reward
        # self.P = torch.zeros((self.n_states, self.n_actions, self.n_states)) #transition probability
        self.reset()

        self.terminate[self.rest1_x[0],self.rest1_y[0]] = 1
        self.terminate[self.rest2_x[0],self.rest2_y[0]] = 1
        self.terminate[self.rest3_x[0],self.rest3_y[0]] = 1
        self.terminate[self.rest4_x[0],self.rest4_y[0]] = 1

        self.rewards[self.rest1_x[0],self.rest1_y[0]] = self.rest_rewards[1][0]
        self.rewards[self.rest2_x[0],self.rest2_y[0]] = self.rest_rewards[3][0]
        self.rewards[self.rest3_x[0],self.rest3_y[0]] = self.rest_rewards[0][0]
        self.rewards[self.rest4_x[0],self.rest4_y[0]] = self.rest_rewards[2][0]

        self.delayed_rewards[self.rest1_x[0],self.rest1_y[0]] = self.rest_rewards[1][1]
        self.delayed_rewards[self.rest2_x[0],self.rest2_y[0]] = self.rest_rewards[3][1]
        self.delayed_rewards[self.rest3_x[0],self.rest3_y[0]] = self.rest_rewards[0][1]
        self.delayed_rewards[self.rest4_x[0],self.rest4_y[0]] = self.rest_rewards[2][1]

        for s_col, s_row in product(range(self.h),range(self.w)):
            if self.grid_map[s_col,s_row] == WALL:
                    self.terminate[s_row, s_col] = 2
                    self.rewards[s_row, s_col] = self.hit_wall_cost

        self._update_transitions()

        # self.P[-1, :, -1] = 1.0
        # self.R[-1,:,:] = 0


    # Required
    def step(self, action):
        """ Moves the simulation one step forward.

        Args:
            action: The action taken by the agent (int)

        Returns:
            Tuple (new_state, reward, done, info)
            new_state: new state of the environment
            reward: reward for the transition
            done: whether the environment is finished or not
            info: empty dictionary """
        if self.episode_finished:
            print("Episode is finished! Reset the environment first!")
            return self.state, 0, True, {}
        info = {}

        # Get possible next states for this action, along with their probabilities
        action = int(action)
        transitions = self.transitions[self.state[0], self.state[1], action]

        # Sample next state from the transitions.
        r = np.random.rand()
        for state, reward, done, p in transitions:
            if r < p:
                self.state = state
                break
            else:
                r -= p

        self.episode_finished = done
        return self.state, reward, done, info

        # assert self.action_space.contains(action)
        # reward = self.R[self.state,action,0]
        # done = False
        # if self.extended_states[self.state] == TERMINAL:
        #     done = True
        #     self.state = torch.argmax(self.P[self.state, action, :]).item()

        # else:
        #     self.state = torch.argmax(self.P[self.state, action, :]).item()

        # return self.state, reward, done, {}

    # Required
    def reset(self):
        if self.fig:
            plt.close(self.fig)

        self.state = self.init_x, self.init_y

        self.fig, self.ax = plt.subplots(1,figsize=(9,10))
        xt = np.arange(0, 1, 1/self.w)
        yt = np.arange(0, 1, 1/self.h)
        self.ax.set_xticks(xt)
        self.ax.set_yticks(yt)

        return self.state

    # Required
    def render(self):
        """Draw the environment on screen."""
        self.ax.patches.clear()
        self.ax.set_yticklabels([])
        self.ax.set_xticklabels([])

        # Fill with gray
        bg = Rectangle((0, 0), 1, 1, facecolor='#c1c1c0')
        self.ax.add_patch(bg)

        grid1 = Rectangle((self.grid1_x[0]/self.w, self.grid1_y[0]/self.h),
                           (self.grid1_x[1]-self.grid1_x[0])/self.w,
                           (self.grid1_y[1]-self.grid1_y[0])/self.h, facecolor="#ffffff")
        self.ax.add_patch(grid1)

        grid2 = Rectangle((self.grid2_x[0]/self.w, self.grid2_y[0]/self.h),
                           (self.grid2_x[1]-self.grid2_x[0])/self.w,
                           (self.grid2_y[1]-self.grid2_y[0])/self.h, facecolor="#ffffff")
        self.ax.add_patch(grid2)

        grid3 = Rectangle((self.grid3_x[0]/self.w, self.grid3_y[0]/self.h),
                           (self.grid3_x[1]-self.grid3_x[0])/self.w,
                           (self.grid3_y[1]-self.grid3_y[0])/self.h, facecolor="#ffffff")
        self.ax.add_patch(grid3)

        grid4 = Rectangle((self.grid4_x[0]/self.w, self.grid4_y[0]/self.h),
                           (self.grid4_x[1]-self.grid4_x[0])/self.w,
                           (self.grid4_y[1]-self.grid4_y[0])/self.h, facecolor="#ffffff")
        self.ax.add_patch(grid4)

        grid5 = Rectangle((self.grid5_x[0]/self.w, self.grid5_y[0]/self.h),
                           (self.grid5_x[1]-self.grid5_x[0])/self.w,
                           (self.grid5_y[1]-self.grid5_y[0])/self.h, facecolor="#ffffff")
        self.ax.add_patch(grid5)

        rest1 = Rectangle((self.rest1_x[0]/self.w, self.rest1_y[0]/self.h),
                            1/self.w, 1/self.h, facecolor="#ff9999")
        self.ax.add_patch(rest1)

        rest2 = Rectangle((self.rest2_x[0]/self.w, self.rest2_y[0]/self.h),
                            1/self.w, 1/self.h, facecolor="#ff9999")
        self.ax.add_patch(rest2)

        rest3 = Rectangle((self.rest3_x[0]/self.w, self.rest3_y[0]/self.h),
                            1/self.w, 1/self.h, facecolor="#ff9999")
        self.ax.add_patch(rest3)

        rest4 = Rectangle((self.rest4_x[0]/self.w, self.rest4_y[0]/self.h),
                            1/self.w, 1/self.h, facecolor="#99ff99")
        self.ax.add_patch(rest4)

        if self.state is not None:
            truck_x = 0.5/self.w + self.state[0]/self.w 
            truck_y = 0.5/self.h + self.state[1]/self.h
            truck = Circle(xy=(truck_x, truck_y), radius=0.03, fill=True, facecolor="#0066ff")
            self.ax.add_patch(truck)
        plt.grid(True, color="#e8e8e8", lw=2)

        # # Maximize when using Tk
        # if "Tk" in mpl.get_backend() and not self.maximized:
        #     manager = plt.get_current_fig_manager()
        #     manager.resize(*manager.window.maxsize())

        self.fig.canvas.draw()
        # self.maximized = True
        plt.pause(0.01)


    # Additional for rendering
    def draw_values(self, values):
        self._draw_floats(values, v_offset=0.5, label="V")
        self._draw_floats(self.rewards, v_offset=0.8, label="r")

    def _draw_floats(self, values, v_offset=0.8, label="V"):
        """Draw an array of float values on the grid.
           Doesn't automatically render the environment - a separate call
           to render is needed afterwards.

           Args:
               values: a width*height array of floating point numbers"""
        for i, row in enumerate(values):
            rx = (i+0.5)/self.w
            for j, value in enumerate(row):
                ry = (j+v_offset)/self.h
                self.ax.text(rx, ry, "{}={:.2f}".format(label, value), ha='center', va='center')

    def draw_actions(self, policy):
        """Draw all the actions on the grid.
           Doesn't automatically render the environment - a separate call
           to render is needed afterwards.

           Args:
               policy: a width*height array of floating point numbers"""

        pol_str = policy.astype(int).astype(str)
        pol_str[pol_str == str(self.RIGHT)] = "Right"
        pol_str[pol_str == str(self.LEFT)] = "Left"
        pol_str[pol_str == str(self.UP)] = "Up"
        pol_str[pol_str == str(self.DOWN)] = "Down"
        for i, row in enumerate(pol_str):
            rx = (i+0.5)/self.w
            for j, value in enumerate(row):
                ry = (j+0.2)/self.h
                self.ax.text(rx, ry, "a: {}".format(value), ha='center', va='center')

    def clear_text(self):
        """Removes all text from the environment before it's rendered."""
        self.ax.texts.clear()

    # Random seed
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # def update(self):
        
    #     for s_row,s_col, a in product(range(self.h),range(self.w), range(self.n_actions)):
    #         s_type = self.grid_map[s_row,s_col]
    #         p_index = self.coord_to_state(s_col, s_row)
            
    #         #Wall states are inaccessible anyays
    #         if s_type == WALL:
    #             continue

    #         #First entry to restaurants, whatever you do, will go to the second-time state of the same restaurant
    #         #Trick for handling the "delayed reward" in the example
    #         elif s_type == DONUT_NORTH_FIRST:
    #             self.P[p_index,:,p_index] = 1.0
    #             self.R[p_index,:,:] = self.rewards[0][1]

    #         elif s_type == DONUT_SOUTH_FIRST:
    #             self.P[p_index,:,p_index] = 1.0
    #             self.R[p_index,:,:] = self.rewards[1][1]

    #         elif s_type == VEGAN_FIRST:
    #             self.P[p_index,:,p_index] = 1.0
    #             self.R[p_index,:,:] = self.rewards[2][1]

    #         elif s_type == NOODLE_FIRST:
    #             self.P[p_index,:,p_index] = 1.0
    #             self.R[p_index,:,:] = self.rewards[3][1]

    #         #If it is an accessible grid
    #         elif s_type == GRID:
    #             if (s_col - 1) >= 0 and self.grid_map[s_row,s_col-1] != WALL:
    #                 self.P[p_index,self.LEFT,p_index-1] = 1.0
    #             else:
    #                 self.P[p_index,self.LEFT,p_index] = 1.0
    #                 self.R[p_index,self.LEFT,:] = self.hit_wall_cost
    #                 self.render_rewards[s_col, s_row] = 0.0

    #             if (s_col + 1) < self.w and self.grid_map[s_row,s_col+1] != WALL:
    #                 self.P[p_index,self.RIGHT,p_index+1] = 1.0
    #             else:
    #                 self.P[p_index,self.RIGHT,p_index] = 1.0
    #                 self.R[p_index,self.RIGHT,:] = self.hit_wall_cost
    #                 self.render_rewards[s_col, s_row] = 0.0

    #             if (s_row - 1) >= 0 and self.grid_map[s_row-1,s_col] != WALL:
    #                 self.P[p_index,self.UP,p_index-self.w] = 1.0
    #             else:
    #                 self.P[p_index,self.UP,p_index] = 1.0
    #                 self.R[p_index,self.UP,:] = self.hit_wall_cost
    #                 self.render_rewards[s_col, s_row] = 0.0

    #             if (s_row + 1) < self.h and self.grid_map[s_row+1,s_col] != WALL:
    #                 self.P[p_index,self.DOWN,p_index+self.w] = 1.0
    #             else:
    #                 self.P[p_index,self.DOWN,p_index] = 1.0
    #                 self.R[p_index,self.DOWN,:] = self.hit_wall_cost
    #                 self.render_rewards[s_col, s_row] = 0.0

            # elif self.terminate(p_index) == 1:
            #     return [(None, 0, True, 1)]

    def coord_to_state(self, x, y):
        return x + y*self.w

    def state_to_coord(self, state):
        return state % self.w, state // self.w

    #def true_hyperbolic(self, k, t):
    #    return 1 / (1 + k * t)

    def _update_transitions(self):
        """Updates the state transition model after rewards etc. were changed."""
        self.transitions = np.empty((self.w, self.h, self.NO_ACTIONS), dtype=list)
        for x, y, a in product(range(self.w), range(self.h), range(self.NO_ACTIONS)):
            self.transitions[x, y, a] = self._get_possible_transitions((x, y), a)

    def _get_neighbouring_state(self, state, relative_pos):
        """Returns the next state to be reached when action is taken in state.
           Assumes everything to be deterministic.

           Args:
               state: current state
               relative_pos: action to be taken/evaluated

            Returns:
                The next state (as numpy.array)"""
        if relative_pos == self.LEFT:
            if state[0] > 0:
                return state[0]-1, state[1]
            else:
                return state
        elif relative_pos == self.RIGHT:
            if state[0] < self.w-1:
                return state[0]+1, state[1]
            else:
                return state
        elif relative_pos == self.DOWN:
            if state[1] > 0:
                return state[0], state[1]-1
            else:
                return state
        elif relative_pos == self.UP:
            if state[1] < self.h-1:
                return state[0], state[1]+1
            else:
                return state
        else:
            raise ValueError("Invalid action: %s" % relative_pos)

    def _get_possible_transitions(self, state, action):
        """ Returns an array of possible future states when
            given action is taken in given state.

            Args:
                state - current state
                action -  action to be taken/evaluated
            Returns:
                 an array of (state, reward, done, prob) uples:
                [(state1, reward1, done1, prob1), (state2, reward2, done2, prob2)...].
                State is None if the episode terminates."""
        
        if self.terminate[state]==2:
            return [(None, 0, False, 1)]

        if state[0] == self.rest1_x and state[1] == self.rest1_y and self.visited[0]==0:
            self.visited[0] = 1
            return [(state, self.delayed_rewards[state], False, 1)]
        elif state[0] == self.rest2_x and state[1] == self.rest2_y and self.visited[1]==0:
            self.visited[1] = 1
            return [(state, self.delayed_rewards[state], False, 1)]
        elif state[0] == self.rest3_x and state[1] == self.rest3_y and self.visited[2]==0:
            self.visited[2] = 1
            return [(state, self.delayed_rewards[state], False, 1)]
        elif state[0] == self.rest4_x and state[1] == self.rest4_y and self.visited[3]==0:
            self.visited[3] = 1
            return [(state, self.delayed_rewards[state], False, 1)]    

        if self.terminate[state]:
            return [(None, 0, True, 1)]

        transitions = []

        next_state = self._get_neighbouring_state(state, action)
        next_reward = self.rewards[next_state]
        next_terminate = self.terminate[next_state]
        transitions.append((next_state, next_reward, next_terminate, 1))
        return transitions