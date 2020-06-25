import os
os.chdir('E:\\Master_E\\Workspace\\bidding_learning') 
import sys
#sys.path.append('../src/')
#import gym
import numpy as np
import matplotlib.pyplot as plt

from src.agent_ddpg import agent_ddpg
from src.utils import OUNoise, Memory, GaussianNoise
from src.environment_bid_market import EnvironmentBidMarket



'''
 High-Level Interface that calls learning algorithm and Energy-Market Environment
 subject to user-specified inputs

Environment Parameters

capacities:   np.array 1x(number of Agents)
costs:        np.array 1x(number of Agents)

Attention these np.arrays have to correspond to the number of Agents

Demand:       np.array 1x2
Chooses demand from arang between [min,max-1]
For fixed Demand, write: the preferred [Number, Number +1] (e.g. Demand = 99 -> [99,100])

Agents:       scalar
Number of learning agents

Rewards:      scalar
Type of Reward function. Usually 0.

Split: binary
Allow Split Bids

past_action: binary
Allow agents to learn from all agents past actions

lr_actor:    float
Learning Rate Actor


lr_critic:   float
Learning Rate Critic

Discrete:    binary
Enables Discrete Spaces (Not yet functional)
'''


POWER_CAPACITIES = [20]
PRODUCTION_COSTS = [0]
DEMAND = [70,71]
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3
NUMBER_OF_AGENTS = 1

env = EnvironmentBidMarket(capacities = POWER_CAPACITIES, costs = PRODUCTION_COSTS, demand = DEMAND, agents = NUMBER_OF_AGENTS, 
                           fringe_player = 1, rewards = 0, split = 0, past_action= 0, 
                           lr_actor = LEARNING_RATE_ACTOR, lr_critic = LEARNING_RATE_CRITIC, discrete = [0, 10, 10])

agents = env.create_agents(env)
rewards = []
avg_rewards = []
# 2 different noise models

# Ohrenstein Ullenbck Noise
# This is a popular noise in machine learning. 
# It starts with one distribution and then converges to another.
# Frequently, this is used to explore more in the beginning than in the end of the algorithm.
noise = OUNoise(env.action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000)

# Gaussian Noise 
# The standard normal distributed noise with variance sigma scaled to the action spaces size
#(default: (mean = 0, sigma = 0.1) * action_space_distance)
#noise = GaussianNoise(env.action_space, mu= 0, sigma = 0.1, regulation_coef= 0.01, decay_rate = 0.1)



# Learning continues for a number of episodes, 
# divided into batches consisting of rounds
# Each episode resets the environment, it consits of rounds
# After a number of rounds equal to the batch size, the neural networks are updated
total_episodes = 50
rounds_per_episode = 500
batch_size = 128

# Start Learning

for episode in range(total_episodes):
    state = env.reset()
    noise.reset()
    episode_reward = 0
 
    for step in range(rounds_per_episode):
        actions = []
        for n in range(len(agents)):
            #Neural Network Chooses Action and Adds Noise
            action_temp = agents[n].get_action(state)
            action_temp = noise.get_action(action_temp, episode) 
            actions.append(action_temp[:])
    
        actions = np.asarray(actions)
        # Environment delivers output
        new_state, reward, done, _ = env.step(actions)   
    
        # Add new experience to memory
        for n in range(len(agents)):
            agents[n].memory.push(state, actions[n], np.array([reward[n]]), new_state, done)
   
        #Update Neural Network
        if len(agents[0].memory) > batch_size:
            for n in range(len(agents)):
                agents[n].update(batch_size)
            
    
        state = new_state
        episode_reward += reward

        if done:
            sys.stdout.write("***episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            env.render()
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))


plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
