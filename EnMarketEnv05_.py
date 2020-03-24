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
from market_clearing import market_clearing


#C = 30
#CAP = 300
#env = EnMarketEnv02(CAP = np.array([100,100]), costs = 30)

#env.observation_space.shape[:]
#env.action_space.shape[0]-1

class EnMarketEnv05(gym.Env):
    
    """Energy Market environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}   ### ?

    def __init__(self, CAP, costs):              ##### mit df ?
        super(EnMarketEnv05, self).__init__()
        
        self.CAP = CAP
        self.costs = costs
        
        
        # Continous action space for bids
        self.action_space = spaces.Box(low=np.array([-10]), high=np.array([10000]), dtype=np.float16)
        #self.action_space = spaces.Box(low=np.array([-10, -10]), high=np.array([10000, 10000]), dtype=np.float16)

        # Discrete Demand opportunities
        self.observation_space = spaces.Box(low=np.array([50]), high=np.array([4500]), dtype=np.float16)
        
        
    def _next_observation(self):
        
        #Q = np.array([500, 1000, 1500])
        Q = np.array([400])
      
        obs = np.asarray([random.choice(Q)])
        
        
        return obs

    def step(self, action):
        # market clearing
        self.current_step += 1
        
        z = action
        

        q = self._next_observation()
        
        Sup0 = np.array([0, self.CAP[0], action[0]])
        Sup1 = np.array([1, self.CAP[1], action[1]])
        Sup2 = np.array([2, self.CAP[2], action[2]])
        
        All = np.stack((Sup0, Sup1, Sup2))
        # if Sup0 is lowest bidder
        
        market = market_clearing(0, q, All)
        
        allorderd = market[1]
        
        minSup = allorderd[0]
        medSup = allorderd[1]
        maxSup = allorderd[2]
        
        p = market[0]
        
                
        ### if bids are equal ###
        
        ### if minSUp and medSup are equal
        
        if minSup[2] == medSup[2]:
            if minSup[1] >= q*0.5:
                if medSup[1] >= q*0.5:
                    reward_min = p * (q*0.5)
                    reward_med = p * (q*0.5)
                    
                    qmax = 0
                    reward_max = p * qmax
                    
                else:
                    qx = (q*0.5) - medSup[1]
                    qmin = np.clip(minSup[1], 0, ((q*0.5) + qx))
                    reward_min = p * qmin
                    
                    qmed = np.clip(medSup[1], 0, (q*0.5))
                    reward_med = p * qmed
                    q = q - qmin - qmed
                    q = np.clip(q, 0, maxSup[1])
                    qmax = np.clip(maxSup[1], 0, q)
                    reward_max = p * qmax
            else:
                if medSup[1] >= q*0.5:
                    qx = (q*0.5) - minSup[1]
                    qmed= np.clip(medSup[1], 0, ((q*0.5) + qx))
                    reward_med = p * qmed
                    qmin = np.clip(minSup[1], 0, (q*0.5))
                    reward_min = p * qmin
                    
                    q = q - qmin - qmed
                    q = np.clip(q, 0, maxSup[1])
                    qmax = np.clip(maxSup[1], 0, q)
                    reward_max = p * qmax
                
                reward_min = p * minSup[1]
                reward_med = p * medSup[1]
                
                q = q - minSup[1] - medSup[1]
                q = np.clip(q, 0, maxSup[1])
                qmax = np.clip(maxSup[1], 0, q)
                reward_max = p * qmax
        
        ### if medSup and maxSup are equal
        
        if medSup[2] == maxSup[2]:
            qmin = np.clip(minSup[1], 0, q)
            reward_min = p * qmin
            q = q - qmin
            q = np.clip(q, 0 , (medSup[1] + maxSup[1]))
            
            if medSup[1] >= q*0.5:
                if maxSup[1] >= q*0.5:
                    reward_med = p * (q*0.5)
                    reward_max = p * (q*0.5)
                    
                else:
                    qx = (q*0.5) - maxSup[1]
                    qmed = np.clip(medSup[1], 0, ((q*0.5) + qx))
                    reward_med = p * qmed
                    qmax = np.clip(maxSup[1], 0, (q*0.5))
                    reward_max = p * qmax
            else:
                if maxSup[1] >= q*0.5:
                    qx = (q*0.5) - medSup[1]
                    qmax= np.clip(maxSup[1], 0, ((q*0.5) + qx))
                    reward_max = p * qmax
                    qmed = np.clip(medSup[1], 0, (q*0.5))
                    reward_med = p * qmed
                    
                
                reward_med = p * minSup[1]
                reward_max = p * medSup[1]
                
        
        ### if minSup and maxSup are equal, also medSup must be the same
        
        ### if all are equal ###
        if minSup[2] == medSup[2] and medSup[2] == maxSup[2]:
            if minSup[1] >= (q*(1/3)):
                if medSup[1] >= (q*(1/3)):
                    if maxSup[1] >= (q*(1/3)):   ## if all SUp- Cap are over a third
                        reward_min = p * (q*(1/3))
                        reward_med = p * (q*(1/3))
                        reward_max = p * (q*(1/3))
                    else:   ## if min and med Sup- Cap are over a third
                        qx = 0.5* ((q*(1/3)) - maxSup[1])
                        qmin = np.clip(minSup[1], 0, (q*(1/3) + qx))
                        qmed = np.clip(medSup[1], 0, (q*(1/3) + qx))
                        reward_min = p * qmin
                        reward_med = p * qmed
                        qmax = np.clip(maxSup[1], 0, (q*(1/3)))
                        reward_max = p * qmax
                else:  
                    if maxSup[1] >= (q*(1/3)): ## if min and max Sup- Cap are over a third
                        qx = 0.5* ((q*(1/3)) - medSup[1])
                        qmin = np.clip(minSup[1], 0, (q*(1/3) + qx))
                        qmax = np.clip(maxSup[1], 0, (q*(1/3) + qx))
                        reward_min = p * qmin
                        reward_max = p * qmax
                        qmed = np.clip(medSup[1], 0, (q*(1/3)))
                        reward_med = p * qmed
                    ## if only min Sup- Cap is over a third
                    qx = ((q*(2/3)) - medSup[1] - maxSup[1])
                    qmin = np.clip(minSup[1], 0, (q*(1/3) + qx))
                    reward_min = p * qmin
                    qmed = (medSup[1], 0, (q*(1/3)))
                    qmax = (maxSup[1], 0, (q*(1/3)))
                    reward_med = p * qmed
                    reward_max = p * qmax
            else:
                if medSup[1] >= (q*(1/3)):
                    if maxSup[1] >= (q*(1/3)): ## if med and max Sup- Cap are over a third
                        qx = 0.5* ((q*(1/3)) - minSup[1])
                        qmin = np.clip(minSup[1], 0, (q*(1/3) + qx))
                        qmed = np.clip(medSup[1], 0, (q*(1/3) + qx))
                        reward_med = p * qmin
                        reward_max = p * qmax
                        qmin = np.clip(minSup[1], 0, (q*(1/3)))
                        reward_min = p * qmin
                    else: ## if only med Sup- Cap is over a third
                        qx = ((q*(2/3)) - minSup[1] - maxSup[1])
                        qmed = np.clip(medSup[1], 0, (q*(1/3) + qx))
                        reward_med = p * qmin
                        qmin = (minSup[1], 0, (q*(1/3)))
                        qmax = (maxSup[1], 0, (q*(1/3)))
                        reward_min = p * qmin
                        reward_max = p * qmax
                if maxSup[1] >= (q*(1/3)): ## if only max Sup- Cap is over a third
                    qx = ((q*(2/3)) - minSup[1] - medSup[1])
                    qmax = np.clip(maxSup[1], 0, (q*(1/3) + qx))
                    reward_max = p * qmax
                    qmin = (minSup[1], 0, (q*(1/3)))
                    qmed = (medSup[1], 0, (q*(1/3)))
                    reward_min = p * qmin
                    reward_med = p * qmed
            ## if all are under a third
            qmin = np.clip(minSup[1], 0, (q*(1/3)))
            qmed = np.clip(medSup[1], 0, (q*(1/3)))
            qmax = np.clip(maxSup[1], 0, (q*(1/3)))
            
            reward_min = p * qmin
            reward_med = p * qmed
            reward_max = p * qmax
                    
                        
                        
                        
            
        
        ### if not equality ###
        
        if minSup[2] != medSup[2] and medSup[2] != maxSup[2]:
            
         
            # rewards
            qmin = np.clip(minSup[1], 0, q)
            reward_min = p * qmin
            
            q = q - qmin
            qx = np.clip(q, 0, (medSup[1] + maxSup[1]))
            qmed = np.clip(medSup[1], 0, qx)
            reward_med = p * qmed
           
            q = q - qmed
            q = np.clip(q, 0, maxSup[1])
            qmax = np.clip(maxSup[1], 0, q)
            reward_max = p * qmax
        
        
        ### get information back ###
        #reward_min = reward_min - (qmin * minSup[3])
        #reward_med = reward_med - (qmed * medSup[3])
        #reward_max = reward_max - (qmax * maxSup[3])
        
        reward_min = np.append(minSup[0], reward_min)
        reward_med = np.append(medSup[0], reward_med)
        reward_max = np.append(maxSup[0], reward_max)
            
        ### find your partner again ###
            
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
        #self.safe(z, self.current_step)
        
        #self.total_soldq += q
        self.sum_z += z
        self.avg_z = self.sum_z/self.current_step
        #self.profit = q*z
        #self.total_profit += self.profit #* -1
        #self.avg_profit = self.total_profit/self.current_step       
        
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
        
  


