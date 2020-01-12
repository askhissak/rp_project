import gym
from gym import spaces
from gym.utils import seeding
import torch
from itertools import product


# Actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Rewards for different destinations and order
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

# Map
MAPS = {"original": torch.tensor([[WALL, WALL, WALL, WALL, VEGAN_FIRST, WALL],
                                  [WALL, WALL, WALL, GRID, GRID, GRID],
                                  [WALL, WALL, DONUT_NORTH_FIRST,GRID,WALL,GRID],
                                  [WALL,WALL,WALL,GRID,WALL,GRID],
                                  [WALL, WALL, WALL, GRID, GRID, GRID],
                                  [WALL, WALL, WALL, GRID, WALL, NOODLE_FIRST],
                                  [GRID, GRID, GRID, GRID, WALL, WALL],
                                  [DONUT_SOUTH_FIRST, WALL, WALL, GRID, WALL, WALL]])}

#
REWARDS = {"original": ((10,-10), (10,-10), (-10,20), (0,0)), 
          "version_1": ((11,-10), (11,-10), (-10,20), (0,0))}


class FoodTruck(gym.Env):

    def __init__(self):#, map_matrix = MAPS["original"], rewards=REWARDS["original"], time_cost = -0.01):
        self.grid_map = MAPS["original"] # map
        self.states = self.grid_map.reshape(-1) # map reshaped into vector
        self.rewards = REWARDS["original"] # rewards
        self.time_cost = -0.01 # cost of a time unit (second or millisecond)
        self.state = 39  # Start at beginning of the chain
        self.seed() # Sets the seed for this env's random number generator(s)

        #States with feature >1 are restaurants.
        # self.n_vegan = torch.sum(self.states == 4).item() # number of vegan restaraunts
        # self.n_donut_south = torch.sum(self.states == 3).item() 
        # self.n_donut_north = torch.sum(self.states == 2).item()
        # self.n_noodle = torch.sum(self.states == 5).item()

        self.action_space = [LEFT,DOWN,RIGHT,UP] # action space
        self.observation_space = torch.cat((self.states,torch.tensor([22,33,44,55,-2]))) # observation space based on number of extended states plus terminal state

        self.n_states = self.grid_map.shape[0] * self.grid_map.shape[1] # number of states
        self.n_actions = len(self.action_space) # number of actions
        self.n_states_extended = self.observation_space.shape[0] #number of extended states (states + pre-terminal states)

        self.R = torch.zeros((self.n_states_extended,self.n_actions,self.n_states_extended)) + self.time_cost #with time cost
        self.P = torch.zeros((self.n_states_extended,self.n_actions,self.n_states_extended)) #without time cost

        # for s_row,s_col, a in product(range(self.n_row),range(self.n_col), range(self.n_actions)):
        #     s_type = self.grid_map[s_row,s_col]
        #     p_index = (s_row * self.n_col) + s_col
        #     #Wall states are inaccessible anyays
        #     if s_type == WALL:
        #         continue

        #     #First entry to restaurants, whatever you do, will go to the second-time state of the same restaurant
        #     #Trick for handling the "delayed reward" in the example
        #     elif s_type == DONUT_NORTH_FIRST:
        #         self.P[p_index,:,-5] = 1.0
        #         self.R[p_index,:,:] = self.rewards[0][0]
        #     elif s_type == DONUT_SOUTH_FIRST:
        #         self.P[p_index,:,-4] = 1.0
        #         self.R[p_index,:,:] = self.rewards[1][0]

        #     elif s_type == VEGAN_FIRST:
        #         self.P[p_index,:,-3] = 1.0
        #         self.R[p_index,:,:] = self.rewards[2][0]

        #     elif s_type == NOODLE_FIRST:
        #         self.P[p_index,:,-2] = 1.0
        #         self.R[p_index,:,:] = self.rewards[3][0]

        #     #If it is an accessible grid
        #     elif s_type == GRID:
        #         if (s_col - 1) >= 0 and self.grid_map[s_row,s_col-1] != WALL:
        #             self.P[p_index,LEFT,p_index-1] = 1.0
        #         else:
        #             self.P[p_index,LEFT,p_index] = 1.0

        #         if (s_col + 1) < self.n_col and self.grid_map[s_row,s_col+1] != WALL:
        #             self.P[p_index,RIGHT,p_index+1] = 1.0
        #         else:
        #             self.P[p_index,RIGHT,p_index] = 1.0

        #         if (s_row - 1) >= 0 and self.grid_map[s_row-1,s_col] != WALL:
        #             self.P[p_index,UP,p_index-self.n_col] = 1.0
        #         else:
        #             self.P[p_index,UP,p_index] = 1.0

        #         if (s_row + 1) < self.n_row and self.grid_map[s_row+1,s_col] != WALL:
        #             self.P[p_index,DOWN,p_index+self.n_col] = 1.0
        #         else:
        #             self.P[p_index,DOWN,p_index] = 1.0

        # #Another hack. Terminal states. No escape.

        # self.P[-5, :, -1] = 1.0
        # self.R[-5,:,:] = self.rewards[0][1]

        # self.P[-4, :, -1] = 1.0
        # self.R[-4,:,:] = self.rewards[1][1]

        # self.P[-3, :, -1] = 1.0
        # self.R[-3,:,:] = self.rewards[2][1]

        # self.P[-2, :, -1] = 1.0
        # self.R[-2,:,:] = self.rewards[3][1]

        # self.P[-1, :, -1] = 1.0
        # self.R[-1,:,:] = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        reward = self.R[self.state,0,0] #calculate reward for a step

        if self.extended_states[self.state] == TERMINAL:
            print("Episode is finished!")
            done = True
            self.state = torch.argmax(self.P[self.state, action, :]).item()

        else:
            done = False
            self.state = torch.argmax(self.P[self.state, action, :]).item()

        return self.state, reward, done, {}

    def reset(self):
        self.state = 39
        return self.state

    # def render(self):
    #     """Draw the environment on screen."""
    #     self.ax.patches.clear()
    #     self.ax.set_yticklabels([])
    #     self.ax.set_xticklabels([])
    #     # Fill with blue
    #     bg = Rectangle((0, 0), 1, 1, facecolor="#75daff")
    #     self.ax.add_patch(bg)
    #     rocks1 = Rectangle((self.rocks1_x[0]/self.w, self.rocks1_y[0]/self.h),
    #                        (self.rocks1_x[1]-self.rocks1_x[0])/self.w,
    #                        (self.rocks1_y[1]-self.rocks1_y[0])/self.h, facecolor="#c1c1c0")
    #     self.ax.add_patch(rocks1)
    #     rocks2 = Rectangle((self.rocks2_x[0]/self.w, self.rocks2_y[0]/self.h),
    #                        (self.rocks2_x[1]-self.rocks2_x[0])/self.w,
    #                        (self.rocks2_y[1]-self.rocks2_y[0])/self.h, facecolor="#c1c1c0")
    #     self.ax.add_patch(rocks2)
    #     wind = Rectangle((self.wind_x[0]/self.w, self.wind_y[0]/self.h),
    #                        (self.wind_x[1]-self.wind_x[0])/self.w,
    #                        (self.wind_y[1]-self.wind_y[0])/self.h, facecolor="#0F97CA")
    #     self.ax.add_patch(wind)
    #     harbour = Rectangle((self.harbour_x/self.w, self.harbour_y/self.h),
    #                         1/self.w, 1/self.h, facecolor="#7AE266")
    #     self.ax.add_patch(harbour)

    #     if self.state is not None:
    #         boat_x = np.array([0.1, 0.9, 0.7, 0.3])/self.w + self.state[0]/self.w
    #         boat_y = np.array([0.6, 0.6, 0.3, 0.3])/self.h + self.state[1]/self.h
    #         boat = Polygon(xy=list(zip(boat_x, boat_y)), fill=True,
    #                        edgecolor="#ac9280", facecolor="#ecc8af")
    #         self.ax.add_patch(boat)
    #     plt.grid(True, color="#e8e8e8", lw=2)

    #     # Maximize when using Tk
    #     if "Tk" in mpl.get_backend() and not self.maximized:
    #         manager = plt.get_current_fig_manager()
    #         manager.resize(*manager.window.maxsize())

    #     self.fig.canvas.draw()
    #     self.maximized = True
    #     plt.pause(0.01)

    # def close(self):
        # if self.viewer:
        #     self.viewer.close()
        #     self.viewer = None