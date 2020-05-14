# -*- coding: utf-8 -*- 
"""

Created on Wed Mar  4 10:05:17 2020

@author: Viktor


"""


import sys
#import gym
import numpy as np
import matplotlib.pyplot as plt
from DDPG_main import DDPGagent_main
from utils_main import OUNoise, Memory, GaussianNoise
from BiddingMarket_energy_Environment import BiddingMarket_energy_Environment

# length of lists(CAP, costs) has to correspond to the number of Agents
# if you wanna have a fixed Demand, write: the preferred [Number, Number +1] (e.g. Demand = 99 -> [99,100])

capacitys = [1]
costs = [0]

env = BiddingMarket_energy_Environment(CAP = capacitys, costs = costs, Demand =[5,6], Agents = 1, 
                                       Fringe = 1, Rewards = 0, Split = 0, past_action= 1,
                                       lr_actor = 1e-4, lr_critic = 1e-3, Discrete = 0)

agents = env.create_agents(env)

# 2 different noise models

# Ohrenstein Ullenbck Noise
# This is a popular noise in machine learning. 
# It starts with one distribution and then converges to another.
# Frequently, this is used to explore more in the beginning than in the end of the algorithm.
# noise = OUNoise(env.action_space, max_sigma=0.3, discrete = env.Discrete, discrete_split = env.Split)

# Gaussian Noise 
# The standard normal distributed noise with variance sigma scaled to the action spaces size
#(default: (mean = 0, sigma = 0.1) * actio_space_distance)
noise = GaussianNoise(env.action_space, mu= 0, sigma = 0.01, regulation_coef= 0.1, decay_rate = 0)


batch_size = 128
rewards = []
avg_rewards = []



for episode in range(50):
    state = env.reset()
    noise.reset()
    episode_reward = 0
    
    for step in range(500):
        actions = []
        for n in range(len(agents)):
            action_temp = agents[n].get_action(state)
            action_temp = noise.get_action(action_temp, step+episode) ## episode statt step!!
            actions.append(action_temp[:])
        
        actions = np.asarray(actions)
        new_state, reward, done, _ = env.step(actions)   
        
        for n in range(len(agents)):
            agents[n].memory.push(state, actions[n], np.array([reward[n]]), new_state, done)
       
        
        if len(agents[0].memory) > batch_size:
            for n in range(len(agents)):
                agents[n].update(batch_size)
                
        
        state = new_state
        episode_reward = reward

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
#plt.savefig('lrAC6-4_regcoef_01_decay0_rescalee-5_agents2_R0_Fr0_Sp0_pa1_03_noclipnotipp.pdf')
plt.show()



