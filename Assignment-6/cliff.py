import random
import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_gridworlds
env = gym.make('Cliff-v0')  # substitute environment's name

num_episodes = 500

runs = 50

alpha = 0.5
gamma = 1
epsilon = 0.1

rewards_table_Q = np.zeros(num_episodes)
rewards_table_SARSA = np.zeros(num_episodes)

for i_run in range(1,runs):
    q_table = np.zeros([env.observation_space[0].n,env.observation_space[1].n, env.action_space.n])

    for i_episode in range(1,num_episodes):
        state = env.reset()
        action = None
        sum_of_rewards = 0
        done = False
        t=0
        while not done:
            #print(state)
            l_state = list(state)
            if random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[l_state[0],l_state[1]])
            #print(action)

            next_state, reward, done, info = env.step(action)

            #print(next_state)
            l_next_state = list(next_state)
            old_value = q_table[l_state[0],l_state[1], action]
            next_max = np.max(q_table[l_next_state[0],l_next_state[1]])

            #Q-learning
            q_table[l_state[0],l_state[1], action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            state = next_state
            sum_of_rewards = sum_of_rewards + reward
            if done:
                print("Q-Episode finished after {} timesteps".format(t+1))
                break
            t = t+1
        rewards_table_Q[i_episode] += sum_of_rewards

    #SARSA

    q_table_SARSA = np.zeros([env.observation_space[0].n,env.observation_space[1].n, env.action_space.n])
    for i_episode in range(num_episodes):
        state = env.reset()
        l_state = list(state)
        action = None
        sum_of_rewards = 0
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table_SARSA[l_state[0],l_state[1]])
        done = False
        t = 0
        while not done:
            #print(state)

            next_state, reward, done, info = env.step(action)

            l_next_state = list(next_state)

            if random.uniform(0,1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table_SARSA[l_next_state[0],l_next_state[1]])

            old_value = q_table_SARSA[l_state[0],l_state[1], action]
            next_value = q_table_SARSA[l_next_state[0],l_next_state[1], next_action]

            q_table_SARSA[l_state[0],l_state[1], action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_value)
            state = next_state
            l_state = list(state)
            action = next_action
            sum_of_rewards = sum_of_rewards + reward
            if done:
                print("S-Episode finished after {} timesteps".format(t+1))
                break
            t=t+1
        rewards_table_SARSA[i_episode] += sum_of_rewards
env.close()

rewards_table_Q /= runs
rewards_table_SARSA /= runs

plt.plot(rewards_table_Q, label='Q-Learning')
plt.plot(rewards_table_SARSA, label='SARSA')
plt.xlabel('Episodes')
plt.ylabel('Sum of rewards during the episode')
plt.ylim([-100, 0])
plt.legend()

plt.savefig('figure.png')
plt.show()
plt.close()
