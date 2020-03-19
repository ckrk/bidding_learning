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

class EnMarketEnv03(gym.Env):
    
    """Energy Market environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}   ### ?

    def __init__(self, CAP, costs):              ##### mit df ?
        super(EnMarketEnv03, self).__init__()
        
        self.CAP = CAP
        self.costs = costs
        
        
        # Continous action space for bids
        self.action_space = spaces.Box(low=np.array([-10]), high=np.array([10000]), dtype=np.float16)
        #self.action_space = spaces.Box(low=np.array([-10, -10]), high=np.array([10000, 10000]), dtype=np.float16)

        # Discrete Demand opportunities
        self.observation_space = spaces.Box(low=np.array([50]), high=np.array([4500]), dtype=np.float16)
        
        
    def _next_observation(self):
        
        Q = np.array([500, 1000, 1500])
        #Q = np.array([200])
      
        obs = np.asarray([random.choice(Q)])
        
        
        return obs

    def step(self, action):
        # market clearing
        self.current_step += 1
        
        z = action
        

        q = self._next_observation()
        
        Sup0 = np.array([0, action[0], self.CAP[0]])
        Sup1 = np.array([1, action[1], self.CAP[1]])
        Sup2 = np.array([2, action[2], self.CAP[2]])
        
        #Sup0 = np.array([0, 300, 100])
        #Sup1 = np.array([1, 400, 100])
        #Sup2 = np.array([2, 500, 100])
        #q = 1000
        # if Sup0 is lowest bidder
        
        if Sup0[1] <= Sup1[1] and Sup0[1] <= Sup2[1]:
            minSup = Sup0
            
            if Sup1[1] <= Sup2[1]:
                medSup = Sup1
                maxSup = Sup2
            else:
                medSup = Sup2
                maxSup = Sup1
                
        # if Sup1 is lowest bidder

        if Sup1[1] <= Sup0[1] and Sup1[1] <= Sup2[1]:
            minSup = Sup1
            
            if Sup0[1] <= Sup2[1]:
                medSup = Sup0
                maxSup = Sup2
            else:
                medSup = Sup2
                maxSup = Sup0
                
        # if Sup2 is lowest bidder
                
        if Sup2[1] <= Sup0[1] and Sup2[1] <= Sup1[1]:
            minSup = Sup2
            
            if Sup0[1] <= Sup1[1]:
                medSup = Sup0
                maxSup = Sup1
            else:
                medSup = Sup1
                maxSup = Sup0
                
        ### if bids are equal ###
        
        ### if minSUp and medSup are equal
        
        if minSup[1] == medSup[1]:
            if minSup[2] >= q*0.5:
                if medSup[2] >= q*0.5:
                    reward_min = minSup[1] * (q*0.5)
                    reward_med = medSup[1] * (q*0.5)
                    
                    qmax = 0
                    reward_max = maxSup[1] * qmax
                    
                else:
                    qx = (q*0.5) - medSup[2]
                    qmin = np.clip(minSup[1], 0, ((q*0.5) + qx))
                    reward_min = minSup[1] * qmin
                    
                    qmed = np.clip(medSup[2], 0, (q*0.5))
                    reward_med = medSup[1] * qmed
                    q = q - qmin - qmed
                    q = np.clip(q, 0, maxSup[2])
                    qmax = np.clip(maxSup[2], 0, q)
                    reward_max = maxSup[1] * qmax
            else:
                if medSup[2] >= q*0.5:
                    qx = (q*0.5) - minSup[2]
                    qmed= np.clip(medSup[1], 0, ((q*0.5) + qx))
                    reward_med = medSup[1] * qmed
                    qmin = np.clip(minSup[2], 0, (q*0.5))
                    reward_min = minSup[1] * qmin
                    
                    q = q - qmin - qmed
                    q = np.clip(q, 0, maxSup[2])
                    qmax = np.clip(maxSup[2], 0, q)
                    reward_max = maxSup[1] * qmax
                
                reward_min = minSup[1] * minSup[2]
                reward_med = medSup[1] * medSup[2]
                
                q = q - minSup[2] - medSup[2]
                q = np.clip(q, 0, maxSup[2])
                qmax = np.clip(maxSup[2], 0, q)
                reward_max = maxSup[1] * qmax
        
        ### if medSup and maxSup are equal
        
        if medSup[1] == maxSup[1]:
            qmin = np.clip(minSup[2], 0, q)
            reward_min = minSup[1] * qmin
            q = q - qmin
            q = np.clip(q, 0 , (medSup[2] + maxSup[2]))
            
            if medSup[2] >= q*0.5:
                if maxSup[2] >= q*0.5:
                    reward_med = minSup[1] * (q*0.5)
                    reward_max = medSup[1] * (q*0.5)
                    
                else:
                    qx = (q*0.5) - maxSup[2]
                    qmed = np.clip(medSup[1], 0, ((q*0.5) + qx))
                    reward_med = medSup[1] * qmed
                    qmax = np.clip(maxSup[2], 0, (q*0.5))
                    reward_max = maxSup[1] * qmax
            else:
                if maxSup[2] >= q*0.5:
                    qx = (q*0.5) - medSup[2]
                    qmax= np.clip(maxSup[1], 0, ((q*0.5) + qx))
                    reward_max = maxSup[1] * qmax
                    qmed = np.clip(medSup[2], 0, (q*0.5))
                    reward_med = medSup[1] * qmed
                    
                
                reward_med = minSup[1] * minSup[2]
                reward_max = medSup[1] * medSup[2]
                
        
        ### if minSup and maxSup are equal, also medSup must be the same
        
        ### if all are equal ###
        if minSup[1] == medSup[1] and medSup[1] == maxSup[1]:
            if minSup[2] >= (q*(1/3)):
                if medSup[2] >= (q*(1/3)):
                    if maxSup[2] >= (q*(1/3)):   ## if all SUp- Cap are over a third
                        reward_min = minSup[1] * (q*(1/3))
                        reward_med = medSup[1] * (q*(1/3))
                        reward_max = maxSup[1] * (q*(1/3))
                    else:   ## if min and med Sup- Cap are over a third
                        qx = 0.5* ((q*(1/3)) - maxSup[2])
                        qmin = np.clip(minSup[1], 0, (q*(1/3) + qx))
                        qmed = np.clip(medSup[1], 0, (q*(1/3) + qx))
                        reward_min = minSup[1] * qmin
                        reward_med = medSup[1] * qmed
                        qmax = np.clip(maxSup[2], 0, (q*(1/3)))
                        reward_max = maxSup[1] * qmax
                else:  
                    if maxSup[2] >= (q*(1/3)): ## if min and max Sup- Cap are over a third
                        qx = 0.5* ((q*(1/3)) - medSup[2])
                        qmin = np.clip(minSup[1], 0, (q*(1/3) + qx))
                        qmax = np.clip(maxSup[1], 0, (q*(1/3) + qx))
                        reward_min = minSup[1] * qmin
                        reward_max = maxSup[1] * qmax
                        qmed = np.clip(medSup[2], 0, (q*(1/3)))
                        reward_med = medSup[1] * qmed
                    ## if only min Sup- Cap is over a third
                    qx = ((q*(2/3)) - medSup[2] - maxSup[2])
                    qmin = np.clip(minSup[1], 0, (q*(1/3) + qx))
                    reward_min = minSup[1] * qmin
                    qmed = (medSup[2], 0, (q*(1/3)))
                    qmax = (maxSup[2], 0, (q*(1/3)))
                    reward_med = medSup[1] * qmed
                    reward_max = maxSup[1] * qmax
            else:
                if medSup[2] >= (q*(1/3)):
                    if maxSup[2] >= (q*(1/3)): ## if med and max Sup- Cap are over a third
                        qx = 0.5* ((q*(1/3)) - minSup[2])
                        qmin = np.clip(minSup[1], 0, (q*(1/3) + qx))
                        qmed = np.clip(medSup[1], 0, (q*(1/3) + qx))
                        reward_med = medSup[1] * qmin
                        reward_max = maxSup[1] * qmax
                        qmin = np.clip(minSup[2], 0, (q*(1/3)))
                        reward_min = minSup[1] * qmin
                    else: ## if only med Sup- Cap is over a third
                        qx = ((q*(2/3)) - minSup[2] - maxSup[2])
                        qmed = np.clip(medSup[1], 0, (q*(1/3) + qx))
                        reward_med = medSup[1] * qmin
                        qmin = (minSup[2], 0, (q*(1/3)))
                        qmax = (maxSup[2], 0, (q*(1/3)))
                        reward_min = minSup[1] * qmin
                        reward_max = maxSup[1] * qmax
                if maxSup[2] >= (q*(1/3)): ## if only max Sup- Cap is over a third
                    qx = ((q*(2/3)) - minSup[2] - medSup[2])
                    qmax = np.clip(maxSup[1], 0, (q*(1/3) + qx))
                    reward_max = maxSup[1] * qmax
                    qmin = (minSup[2], 0, (q*(1/3)))
                    qmed = (medSup[2], 0, (q*(1/3)))
                    reward_min = minSup[1] * qmin
                    reward_med = medSup[1] * qmed
            ## if all are under a third
            qmin = np.clip(minSup[2], 0, (q*(1/3)))
            qmed = np.clip(medSup[2], 0, (q*(1/3)))
            qmax = np.clip(maxSup[2], 0, (q*(1/3)))
            
            reward_min = minSup[1] * qmin
            reward_med = medSup[1] * qmed
            reward_max = maxSup[1] * qmax
                    
                        
                        
                        
            
        
        ### if not equality ###
        
        if minSup[1] != medSup[1] and medSup[1] != maxSup[1]:
            
         
            # rewards
            qmin = np.clip(minSup[2], 0, q)
            reward_min = minSup[1] * qmin
            
            q = q - qmin
            qx = np.clip(q, 0, (medSup[2] + maxSup[2]))
            qmed = np.clip(medSup[2], 0, qx)
            reward_med = medSup[1] * qmed
           
            q = q - qmed
            q = np.clip(q, 0, maxSup[2])
            qmax = np.clip(maxSup[2], 0, q)
            reward_max = maxSup[1] * qmax
        
        
        ### get information back ###
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
        
  


###### Test Place #######
#m =200
#n = np.array([100, 200, 200])
#r = (m*0.5)*n[0] + (m*0.5)*n[1]
#max(n)
#n[1] *2
#m = np.array([100, 200])

#np.append(n,m)