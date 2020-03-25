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

class EnMarketEnv07(gym.Env):
    
    """
    Energy Market environment for OpenAI gym
    market_clearing included
    
<<<<<<< HEAD
    Changes: observation_space with additional Dimensons: shape(1,7): 1demand, 3Capcitys, 3actions
=======
    Changes: observation_space with additional Dimensons: shape(7,1): 1 demand, 3 Capcitys, 3 actions
>>>>>>> development
    
    only works with test03 and DDPG03_
    
    """
    metadata = {'render.modes': ['human']}   ### ?

    def __init__(self, CAP, costs):              ##### mit df ?
        super(EnMarketEnv07, self).__init__()
        
        self.CAP = CAP
        self.costs = costs
        
        
        # Continous action space for bids
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([10000]), dtype=np.float16)
        #self.action_space = spaces.Box(low=np.array([-10, -10]), high=np.array([10000, 10000]), dtype=np.float16)

        # Discrete Demand opportunities
        #self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([5000, 5000]), dtype=np.float16)
        self.observation_space = spaces.Box(low=0, high=10000, shape=(7,1), dtype=np.float16)

<<<<<<< HEAD
=======
        self.reward_range = (0, 1)
>>>>>>> development
        
    def _next_observation(self, last_action):
        
        if self.current_step == 0:
            last_action = self.start_action
        else:
            last_action = last_action
        
        #Q = np.array([500, 1000, 1500])
<<<<<<< HEAD
    
        Q = np.array([800])
        Q = random.choice(Q)
      
        obs = np.array([Q, self.CAP[0], self.CAP[1], self.CAP[2], last_action[0], last_action[1], last_action[2]])
=======
        #Q = np.array([800])
        #Q = random.choice(Q)
        Q = np.random.randint(900, 1100, 1)
      
        obs = np.array([Q[0], self.CAP[0], self.CAP[1], self.CAP[2], last_action[0], last_action[1], last_action[2]])
>>>>>>> development
        
        
        return obs

    def step(self, action, last_action):
        # market clearing
        
        
        z = action
        
        if self.current_step == 0:
            last_action = self.start_action
            obs = self._next_observation(last_action)
        else:
            obs = self._next_observation(last_action)
        
        self.current_step += 1
        
        Demand = obs[0]
        q = obs[0]
        
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
       
    
<<<<<<< HEAD

=======
        
        #reward0 = np.clip(reward0, 0, Demand)
        #reward1 = np.clip(reward1, 0, Demand)
        #reward2 = np.clip(reward2, 0, Demand)
        #E:\Master_E\Workspace\Bidding-Learning\EnMarketEnv07_.py:181: 
        #RuntimeWarning: invalid value encountered in double_scalars
        
        #reward0 = reward0 / (p * Sup0[1])
        #reward1 = reward1 / (p * Sup1[1])
        #reward2 = reward2 / (p * Sup2[1])
        
>>>>>>> development
        reward = np.append(reward0, reward1)
        reward = np.append(reward, reward2)

        
      
        
        ## Render Commands 
        self.safe(z, self.current_step)
        
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
        
        obs = self._next_observation(action)
        last_action = action

       

        return obs, reward, done, last_action, {}
    
    def safe(self, action, current_step):
        
        Aktionen = (action, current_step)
        self.AllAktionen.append(Aktionen)
        #last_action = action
        
       # return last_action
        
    
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.avg_z = 0
        self.sum_z = 0
        self.sum_q = 0
        self.sum_rewards = 0
        self.AllAktionen = deque(maxlen=500)
        self.start_action = np.array([0, 0, 0])
        
        return self._next_observation(self.start_action)
    
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
        
        
  


