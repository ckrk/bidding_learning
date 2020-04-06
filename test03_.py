# -*- coding: utf-8 -*- 
"""

Created on Wed Mar  4 10:05:17 2020

@author: Viktor
works only for EnMarketEnv07_ and DDPG03_
supposed for the 3 Player Games



"""

###### Versuch mit 2 agents ########################

import sys
#import gym
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from DDPG03_ import DDPGagent03
from utils_ import OUNoise, Memory
from EnMarketEnv07_ import EnMarketEnv07 



env = EnMarketEnv07(CAP = np.array([500,500,500]), costs = np.array([20,20,20]), Fringe=0, Rewards = 4)



agent0 = DDPGagent03(env)
agent1 = DDPGagent03(env)
agent2 = DDPGagent03(env)
noise = OUNoise(env.action_space)
batch_size = 128
rewards = []
avg_rewards = []
last_action = np.array([0,0,0])


for episode in range(50):
    state = env.reset()
    noise.reset()
    episode_reward = 0
    
    for step in range(500):
        
        action0 = agent0.get_action(state)
        action0 = noise.get_action(action0, step)
        action1 = agent1.get_action(state)
        action1 = noise.get_action(action1, step)
        action2 = agent2.get_action(state)
        action2 = noise.get_action(action2, step)
        
        action = np.concatenate([action0, action1, action2])
        #action = np.concatenate([action0, action1]) #if trying with fringe Player !!!
        new_state, reward, done, _ = env.step(action, last_action)   
        
        agent0.memory.push(state, np.array([action[0]]), np.array([reward[0]]), new_state, done)
        agent1.memory.push(state, np.array([action[1]]), np.array([reward[1]]), new_state, done)
        agent2.memory.push(state, np.array([action[2]]), np.array([reward[2]]), new_state, done)


        
        if len(agent1.memory) > batch_size:            
            agent0.update(batch_size)
            agent1.update(batch_size)
            agent2.update(batch_size) 
        
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


