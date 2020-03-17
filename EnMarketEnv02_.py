# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:05:34 2020

@author: Viktor
"""

#### Environment für 2 Player#####



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
#env = EnMarketEnv02(CAP = np.array([100,100]), costs = 30)

#env.observation_space.shape[:]
#env.action_space.shape[0]-1

class EnMarketEnv02(gym.Env):
    
    """Energy Market environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}   ### ?

    def __init__(self, CAP, costs):              ##### mit df ?
        super(EnMarketEnv02, self).__init__()
        
        self.CAP = CAP
        self.costs = costs
        
        
        # Continous action space for bids
        self.action_space = spaces.Box(low=np.array([-10]), high=np.array([10000]), dtype=np.float16)
        #self.action_space = spaces.Box(low=np.array([-10, -10]), high=np.array([10000, 10000]), dtype=np.float16)

        # Discrete Demand opportunities
        self.observation_space = spaces.Box(low=np.array([50]), high=np.array([450]), dtype=np.float16)
        
        
    def _next_observation(self):
        
        #Q = np.array([50, 150, 300, 450])
        Q = np.array([200])
      
        obs = np.asarray([random.choice(Q)])
        
        
        return obs

    def step(self, action):
        # market clearing
        self.current_step += 1
        
        z = action
        

        q = self._next_observation()
        
        Sup0 = np.array([0, action[0], CAP[0]])
        Sup1 = np.array([1, action[1], CAP[1]])
        Sup2 = np.array([2, action[2], CAP[2]])
        
        # if Sup0 is lowest bidder
        
        if Sup0[1] < Sup1[1] and Sup0[1] < Sup2[1]:
            minSup = Sup0
            
            if Sup1[1] < Sup2[1]:
                medSup = Sup1
                maxSup = Sup2
            else:
                medSup = Sup2
                maxSup = Sup1
                
        # if Sup1 is lowest bidder

        if Sup1[1] < Sup0[1] and Sup1[1] < Sup2[1]:
            minSup = Sup1
            
            if Sup0[1] < Sup2[1]:
                medSup = Sup0
                maxSup = Sup2
            else:
                medSup = Sup2
                maxSup = Sup0
                
        # if Sup2 is lowest bidder
                
        if Sup2[1] < Sup0[1] and Sup2[1] < Sup1[1]:
            minSup = Sup2
            
            if Sup0[1] < Sup1[1]:
                medSup = Sup0
                maxSup = Sup1
            else:
                medSup = Sup1
                maxSup = Sup0
                
        # first reward
        reward_min = minSup[1] * minSup[2]
        reward_min = np.append(minSup[0], reward_min)
        
        q = q - minSup[2]
        
        # second reward
        if q > medSup[2]:
            reward_med = medSup[1] * medSup[2]
            reward_med = np.append(medSup[0], reward_med)

        else:
            reward_med = medSup[1] * q
            reward_med = np.append(medSup[0], reward_med)

            
        q = q - medSup[2]
        
        # third reward
        if q > maxSup[2]:
            reward_max = maxSup[1] * maxSup[2]
            reward_max = np.append(maxSup[0], reward_max)

        else:
            reward_max = maxSup[1] * q
            reward_max = np.append(maxSup[0], reward_max)

        
       # find your partner again
       
       if reward_min[0] == 0:
           reward0 = reward_min[1]
           
           if reward_med[0] == 1:
               reward1 = reward_med[1]
               reward2 = reward_max[1]
           else:
               reward2 = reward_med[1]
               reward1 = reward_max[1]
               
        if reward_min[0] == 1:
            reward1 = reward_min[1]
           
            if reward_med[0] == 0:
                reward0 = reward_med[1]
                reward2 = reward_max[1]
            else:
                reward2 = reward_med[1]
                reward0 = reward_max[1]
           
        if reward_min[0] == 2:
            reward2 = reward_min[1]
           
            if reward_med[0] == 0:
                reward0 = reward_med[1]
                reward1 = reward_max[1]
            else:
                reward1 = reward_med[1]
                reward0 = reward_max[1]
       
    

        reward = np.append(reward0, reward1)
        reward = np.append(reward, reward2)
      
        
        ## Render Commands 
        self.safe(z, self.current_step)
        
        self.total_soldq += q
        self.sum_z += z
        self.avg_z = self.sum_z/self.current_step
        self.profit = q*z
        self.total_profit += self.profit #* -1
        self.avg_profit = self.total_profit/self.current_step       
        
        #### DONE
        done = self.current_step == 128#256#128 
        
        ##### Next Obs
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
#m =200
#n = np.array([100, 200, 200])
#r = (m*0.5)*n[0] + (m*0.5)*n[1]
#max(n)
#n[1] *2
#m = np.array([100, 200])

#np.append(n,m)