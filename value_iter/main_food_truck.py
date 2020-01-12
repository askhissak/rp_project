import numpy as np
from food_truck import FoodTruck
from time import sleep

import operator

# Set up the environment
env = FoodTruck()
value_est = np.zeros((env.w, env.h))
env.draw_values(value_est)

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

    #Value iteration
    gamma = 0.9
    policy_stable = True
    delta = 0.1
    while delta > 0.0001:
        delta = 0
        for i in range(env.w):
            for j in range(env.h):
                old_action = policy[i,j]
                value_temp = value_est[i,j]
                neighbor_values = np.zeros(4)
                for k in range(len(env.transitions[i,j,env.LEFT])):
                    if env.transitions[i,j,env.LEFT][k][2] is not True:
                        neighbor_values[0] = neighbor_values[0] + env.transitions[i,j,env.LEFT][k][3]*(env.transitions[i,j,env.LEFT][k][1] + gamma*value_est[env.transitions[i,j,env.LEFT][k][0]])
                for k in range(len(env.transitions[i,j,env.DOWN])):
                    if env.transitions[i,j,env.DOWN][k][2] is not True:    
                        neighbor_values[1] = neighbor_values[1] + env.transitions[i,j,env.DOWN][k][3]*(env.transitions[i,j,env.DOWN][k][1] + gamma*value_est[env.transitions[i,j,env.DOWN][k][0]])
                for k in range(len(env.transitions[i,j,env.RIGHT])):
                    if env.transitions[i,j,env.RIGHT][k][2] is not True:    
                        neighbor_values[2] = neighbor_values[2] + env.transitions[i,j,env.RIGHT][k][3]*(env.transitions[i,j,env.RIGHT][k][1] + gamma*value_est[env.transitions[i,j,env.RIGHT][k][0]])
                for k in range(len(env.transitions[i,j,env.UP])):
                    if env.transitions[i,j,env.UP][k][2] is not True:    
                        neighbor_values[3] = neighbor_values[3] + env.transitions[i,j,env.UP][k][3]*(env.transitions[i,j,env.UP][k][1] + gamma*value_est[env.transitions[i,j,env.UP][k][0]])
                # print(neighbor_values)
                index, value_est[i,j] = max(enumerate(neighbor_values), key=operator.itemgetter(1))
                policy[i,j] = index
                delta = max(delta,abs(value_est[i,j] - value_temp))
        #         if old_action != policy[i,j]:
        #             policy_stable = False
        # if policy_stable:
        #     break    
        env.clear_text()
        env.draw_values(value_est)
        env.draw_actions(policy)
        env.render()
        sleep(1)
        #delta = np.copy(max(delta, abs(value_temp-value_est)))

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
    num_ep = 1000
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
            ini_g += (gamma**j)*reward

            # # Render and sleep
            # env.render()
            # sleep(0.5)
            j += 1
        disc_returns[i] = ini_g
        state = env.reset()
    print(np.mean(disc_returns))
    print(np.std(disc_returns))