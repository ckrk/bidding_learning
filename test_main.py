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
from utils_main import OUNoise, Memory
from BiddingMarket_energy_Environment import BiddingMarket_energy_Environment



env = BiddingMarket_energy_Environment(CAP = np.array([500,500,500]), costs = np.array([20,20,20]), 
                                       Fringe = 0, Rewards = 1, Split = 0, past_action= 0, Agents = 2)

agents = env.create_agents(env)
noise = OUNoise(env.action_space)
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
            action_temp = noise.get_action(action_temp, step)
            actions.append(action_temp[:])
        
        actions = np.asarray(actions)
        new_state, reward, done, _ = env.step(actions)   
        
        for n in range(len(agents)):
            agents[n].memory.push(state, actions[n], np.array([reward[n]]), new_state, done)
       
        
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


