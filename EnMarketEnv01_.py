# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 08:28:40 2020

@author: Viktor
"""

#### First Try EnMarketEnv #####

###### leere dimension zu action shape hinzufügen

import random
import gym
from gym import spaces
import numpy as np
import pandas as pd
import random

import torch
import torch.autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from collections import deque


#C = 30
#CAP = 300
#env = EnMarketEnv(CAP = 300, costs = 30)

#env.observation_space.shape[:]
#env.action_space.shape[:]

class EnMarketEnv(gym.Env):
    
    """Energy Market environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}   ### ?

    def __init__(self, CAP, costs):              ##### mit df ?
        super(EnMarketEnv, self).__init__()
        
        self.CAP = CAP
        self.costs = costs
        #self.n = 4
        
        # Continous action space for bids
        ##self.action_space = spaces.Box(low=0, high=10000,shape=(1, 1), dtype=np.float16)
        self.action_space = spaces.Box(low=np.array([-10]), high=np.array([10000]), dtype=np.float16)
        
        # Discrete Demand opportunities
        ##self.observation_space = spaces.Discrete(self.n)## eig. nur 1, oder dim von allen möglichen demands
        ##self.observation_space = spaces.Tuple((spaces.Discrete(4), self.observation_space))
        self.observation_space = spaces.Box(low=np.array([50]), high=np.array([450]), dtype=np.float16)
        
        
    def _next_observation(self):
        
        Q = np.array([50, 150, 300, 450])
        #Q = np.array([100])
      
        obs = np.asarray([random.choice(Q)])
        
        
        return obs

    def step(self, action):
        # what is happening with chosen action
        # market clearing
        self.current_step += 1
        

        
        q = self._next_observation()
        
        if q > self.CAP:
            q = self.CAP
        
        self.total_soldq += q

        
        z = action #* 10000  #* 250000 #- self.costs
        
        
        #if z < 1:
         #   z = z + (1-z)
            
        self.bid = z
        
        
        
        self.sum_z += z
        self.avg_z = self.sum_z/self.current_step
        
        #reward = q*z**2 + (q*0.2)*z - 50
        reward = (q*z)#/ self.total_soldq
        
        self.safe(z, self.current_step)
        #if reward <= q:
         #   reward = reward * (-100)
     
        
        self.profit = q*z
        self.total_profit += self.profit #* -1
        self.avg_profit = self.total_profit/self.current_step       
        
        #done = self.total_profit >= 10000
        done = self.current_step == 128#256#128 
        #done = self.total_soldq >= CAP*12
        
        obs = self._next_observation()

       

        return obs, reward, done, {}
    
    def safe(self, action, current_step):
        
        Aktionen = (action, current_step)
        self.AllAktionen.append(Aktionen)
        
    
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.total_soldq = 0
        self.total_profit = 0
        self.profit = 0
        self.avg_z = 0
        self.sum_z = 0
        self.avg_profit = 0
        self.AllAktionen = deque(maxlen=500)
        
        return self._next_observation()
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        # ???
        print(f'Step: {self.current_step}')
        #print(f'Profit: {self.profit}')
        print(f'Average Profit: {self.avg_profit}')
        print(f'Total Sold Q: {self.total_soldq}')
        print(f'Total Profit: {self.total_profit}')
        print(f'AllAktionen: {self.AllAktionen}')
        #print(f'OverView: {self.overview}')
        print(f'Average Bid: {self.avg_z}')
        #print(f'Bid/action: {self.bid}')
        
  


###### Test Place #######
        

        
