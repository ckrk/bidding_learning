# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:47:21 2020

@author: Viktor
"""

import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DDPG_ import DDPGagent
#from utils_ import *
from utils_ import OUNoise, Memory
from EnMarketEnv01_ import EnMarketEnv ##alway NameError

#env = NormalizedEnv(gym.make("Pendulum-v0")) ## NotImplementedError

#env = gym.make("Pendulum-v0")   # funktioniert 

#env = PendulumEnv()   ## arbeitet ewig ohne Ergebniss

#env = DummyVecEnv([lambda: StockTradingEnv(df)])  ## funtioniert nicht (müsste tensorflow installieren) nicht notwendig

#env = StockTradingEnv(df) ### würde funtionieren; Code läuft aber nicht (evtl. wegen falschen dimensionen)

#env.observation_space.shape[:]
#env.seed(10000191)

#seed(1011)

env = EnMarketEnv(CAP = 300, costs = 30)



agent = DDPGagent(env)
noise = OUNoise(env.action_space)
batch_size = 128
rewards = []
avg_rewards = []

for episode in range(50):
    state = env.reset()
    noise.reset()
    episode_reward = 0
    
    for step in range(500):
        action = agent.get_action(state)
        action = noise.get_action(action, step)         # t=0 ist default?!?
        #action = np.array([agent.get_action(state)])
        new_state, reward, done, _ = env.step(action)   ## hier entsteht NotImplementedError (evtl. muss im Wrapper die step function auch def) ;;; und Error bei StockEnv (evtl. weil numpy format und kein panda?)
        agent.memory.push(state, action, reward, new_state, done)
        
        if len(agent.memory) > batch_size:            # nicht ganz klar
            agent.update(batch_size)        
        
        state = new_state
        episode_reward += reward

        if done:
            sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
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

#plt.savefig('Plot.png')



############

#agent = DDPGagent(env)
#state = env.reset()
#action = agent.get_action(state)
#action
#action = noise.get_action(action, step)    
#action

#agent.memory.buffer
