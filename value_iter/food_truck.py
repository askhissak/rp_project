import gym
from gym import spaces
from gym.utils import seeding
import torch
from itertools import product
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle, Circle
import numpy as np



# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3

WALL = -1
GRID = 1
DONUT_NORTH_FIRST = 2
DONUT_NORTH_SECOND = 22

DONUT_SOUTH_FIRST = 3
DONUT_SOUTH_SECOND = 33

VEGAN_FIRST = 4
VEGAN_SECOND = 44

NOODLE_FIRST = 5
NOODLE_SECOND = 55

TERMINAL = -2

MAPS = {"original": torch.tensor([  [WALL, WALL, WALL, WALL, VEGAN_FIRST, WALL],
                                [WALL, WALL, WALL, GRID, GRID, GRID],
                                [WALL, WALL, DONUT_NORTH_FIRST, GRID, WALL, GRID],
                                [WALL, WALL, WALL, GRID, WALL,GRID],
                                [WALL, WALL, WALL, GRID, GRID, GRID],
                                [WALL, WALL, WALL, GRID, WALL, NOODLE_FIRST],
                                [GRID, GRID, GRID, GRID, WALL, WALL],
                                [DONUT_SOUTH_FIRST, WALL, WALL, GRID, WALL, WALL]])}

REWARDS = { "original": ((10,-10), (10,-10), (-10,20), (0,0)), 
            "version_1": ((11,-10), (11,-10), (-10,21), (0,0)),
            "version_1_effort": ((11-1,-10-1), (11-1,-10-1), (-10-1,20-1), (0-1,0-1)),
            "version_2_effort": ((11-0.2,-10-0.2), (11-0.2,-10-0.2), (-10-0.2,20-0.2), (0-0.2,0-0.2))}


class FoodTruck(gym.Env):
    LEFT, DOWN, RIGHT, UP = 0,1,2,3

    # Required
    def __init__(self, map_matrix = MAPS["original"], reward_type = "version_1", time_cost = -0.01, hit_wall_cost=-2):
        self.h = map_matrix.shape[0]
        self.w = map_matrix.shape[1]
        self.grid_map = map_matrix
        self.states = self.grid_map.reshape(-1)
        self.rewards = REWARDS[reward_type]
        self.time_cost = time_cost
        self.state = 3, 1  # Start at beginning of the chain
        self.seed()
        self.hit_wall_cost = hit_wall_cost

        #Render
        self.rewards = np.zeros((self.w, self.h))
        self.terminate = np.zeros((self.w, self.h))
        self.state = 0, 0
        self.fig, self.ax = None, None
        self.maximized = False
        self.episode_finished = False
        self.transitions = None
        self.reset()

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

        #States with feature >1 are restaurants.
        # self.n_vegan = torch.sum(self.states == 4).item()
        # self.n_donut_south = torch.sum(self.states == 3).item()
        # self.n_donut_north = torch.sum(self.states == 2).item()
        # self.n_noodle = torch.sum(self.states == 5).item()

        #Add the pre-terminal states per restaurant. and a terminal state
        self.extended_states = torch.cat((self.states,torch.tensor([22,33,44,55,-2])))

        self.n_states_extended = self.extended_states.shape[0]
        self.n_states = self.h * self.w
        self.n_actions = 4
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_states_extended)
        self.R = torch.zeros((self.n_states_extended,self.n_actions,self.n_states_extended)) + self.time_cost

        self.P = torch.zeros((self.n_states_extended, self.n_actions, self.n_states_extended))

        for s_row,s_col, a in product(range(self.h),range(self.w), range(self.n_actions)):
            s_type = self.grid_map[s_row,s_col]
            p_index = (s_row * self.w) + s_col
            #Wall states are inaccessible anyays
            if s_type == WALL:
                continue

            #First entry to restaurants, whatever you do, will go to the second-time state of the same restaurant
            #Trick for handling the "delayed reward" in the example
            elif s_type == DONUT_NORTH_FIRST:
                self.P[p_index,:,-5] = 1.0
                self.R[p_index,:,:] = self.rewards[0][0]
            elif s_type == DONUT_SOUTH_FIRST:
                self.P[p_index,:,-4] = 1.0
                self.R[p_index,:,:] = self.rewards[1][0]

            elif s_type == VEGAN_FIRST:
                self.P[p_index,:,-3] = 1.0
                self.R[p_index,:,:] = self.rewards[2][0]

            elif s_type == NOODLE_FIRST:
                self.P[p_index,:,-2] = 1.0
                self.R[p_index,:,:] = self.rewards[3][0]

            #If it is an accessible grid
            elif s_type == GRID:
                if (s_col - 1) >= 0 and self.grid_map[s_row,s_col-1] != WALL:
                    self.P[p_index,self.LEFT,p_index-1] = 1.0
                else:
                    self.P[p_index,self.LEFT,p_index] = 1.0
                    self.R[p_index,self.LEFT,:] = self.hit_wall_cost


                if (s_col + 1) < self.w and self.grid_map[s_row,s_col+1] != WALL:
                    self.P[p_index,self.RIGHT,p_index+1] = 1.0
                else:
                    self.P[p_index,self.RIGHT,p_index] = 1.0
                    self.R[p_index,self.RIGHT,:] = self.hit_wall_cost


                if (s_row - 1) >= 0 and self.grid_map[s_row-1,s_col] != WALL:
                    self.P[p_index,self.UP,p_index-self.w] = 1.0
                else:
                    self.P[p_index,self.UP,p_index] = 1.0
                    self.R[p_index,self.UP,:] = self.hit_wall_cost


                if (s_row + 1) < self.h and self.grid_map[s_row+1,s_col] != WALL:
                    self.P[p_index,self.DOWN,p_index+self.w] = 1.0
                else:
                    self.P[p_index,self.DOWN,p_index] = 1.0
                    self.R[p_index,self.DOWN,:] = self.hit_wall_cost


        #Another hack. Terminal states. No escape.

        self.P[-5, :, -1] = 1.0
        self.R[-5,:,:] = self.rewards[0][1]

        self.P[-4, :, -1] = 1.0
        self.R[-4,:,:] = self.rewards[1][1]

        self.P[-3, :, -1] = 1.0
        self.R[-3,:,:] = self.rewards[2][1]

        self.P[-2, :, -1] = 1.0
        self.R[-2,:,:] = self.rewards[3][1]

        self.P[-1, :, -1] = 1.0
        self.R[-1,:,:] = 0



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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

        self.state = 3, 1

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