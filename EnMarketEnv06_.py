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


from collections import deque
from market_clearing import market_clearing


#C = 30
#CAP = 300
#env = EnMarketEnv02(CAP = np.array([100,100]), costs = 30)

#env.observation_space.shape[:]
#env.action_space.shape[0]-1

class EnMarketEnv06(gym.Env):
    
    """Energy Market environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}   ### ?

    def __init__(self, CAP, costs):              ##### mit df ?
        super(EnMarketEnv06, self).__init__()
        
        self.CAP = CAP
        self.costs = costs
        
        
        # Continous action space for bids
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([10000]), dtype=np.float16)
        #self.action_space = spaces.Box(low=np.array([-10, -10]), high=np.array([10000, 10000]), dtype=np.float16)

        # Discrete Demand opportunities
        self.observation_space = spaces.Box(low=np.array([50]), high=np.array([4500]), dtype=np.float16)
        
        
    def _next_observation(self):
        
        #Q = np.array([500, 1000, 1500])
        Q = np.array([1000])
      
        obs = np.asarray([random.choice(Q)])
        
        
        return obs

    def step(self, action):
        # market clearing
        self.current_step += 1
        
        z = action
        

        q = self._next_observation()
        Demand = q
        
        Sup0 = np.array([0, self.CAP[0], action[0]])
        Sup1 = np.array([1, self.CAP[1], action[1]])
        Sup2 = np.array([2, self.CAP[2], action[2]])
        
        All = np.stack((Sup0, Sup1, Sup2))
        
        market = market_clearing(0, q, All)
        
        allorderd = market[1]
        
        #minSup = np.append(allorderd[0], self.costs[0])
        #medSup = np.append(allorderd[1], self.costs[1])
        #maxSup = np.append(allorderd[2], self.costs[2])
        
        minSup = allorderd[0]
        medSup = allorderd[1]
        maxSup = allorderd[2]
    
        
        p = market[0]
        
    
        
        # rewards
        qmin = np.clip(minSup[1], 0, q)
        reward_min = p * qmin
        
        q = q - qmin
        qmed = np.clip(medSup[1], 0, q)
        reward_med = p * qmed
           
        q = q - qmed
        qmax = np.clip(maxSup[1], 0, q)
        reward_max = p * qmax
        
        
        
        #reward_min = reward_min - (qmin * minSup[3])
        #reward_med = reward_med - (qmed * medSup[3])
        #reward_max = reward_max - (qmax * maxSup[3])
        
        
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
        
        self.sum_q += Demand
        self.avg_q = self.sum_q/self.current_step
        self.sum_z += z
        self.avg_z = self.sum_z/self.current_step
        self.current_q = Demand
        self.last_rewards = reward
        self.last_bids = z
        self.sum_rewards += reward
        self.avg_rewards = self.sum_rewards/self.current_step
        
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
        self.avg_z = 0
        self.sum_z = 0
        self.sum_q = 0
        self.sum_rewards = 0
        self.AllAktionen = deque(maxlen=500)
        
        return self._next_observation()
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Step: {self.current_step}')
        #print(f'AllAktionen: {self.AllAktionen}')
        print(f'Last Demand of this Episode: {self.current_q}')
        print(f'Last Bid of this Episode: {self.last_bids}')
        print(f'Last Reward of this Episode: {self.last_rewards}')
        print(f'Average Demand: {self.avg_q}')
        print(f'Average Bid: {self.avg_z}')
        print(f'Average Reward: {self.avg_rewards}')
        
        
  


