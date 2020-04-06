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
    

    Changes: observation_space with additional Dimensons: shape(7,1): 1 demand, 3 Capcitys, 3 actions (maybe add costs)
    Rewards = 0: Default, Reward is (price-costs)*acceptedCAP 
    Rewards = 1: Reward is (price-costs)*acceptedCAP - (price*unsoldCAP)
    Rewards = 2: Reward is ((price-costs)*acceptedCAP)/(cost*maxCAP)
    Rewards = 3: Reward is ((price-costs)*acceptedCAP - (price*unsoldCAP))/(cost*maxCAP)
    Rewards = 4: Reward is ((price-costs)*acceptedCAP - (price*unsoldCAP))/((ownBid-cost)*maxCAP)

    
    only works with test03 and DDPG03_
    
    """
    metadata = {'render.modes': ['human']}   ### ?

    def __init__(self, CAP, costs, Fringe=0, Rewards=0):              ##### mit df ?
        super(EnMarketEnv07, self).__init__()
        
        self.CAP = CAP
        self.costs = costs
        self.Fringe = Fringe
        self.Rewards = Rewards
        
        # Continous action space for bids
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([10000]), dtype=np.float16)

        # Discrete Demand opportunities
        self.observation_space = spaces.Box(low=0, high=10000, shape=(7,1), dtype=np.float16)
        


        #Fringe or Strategic Player
        #        # Test move to init
        #Readout fringe players from other.csv (m)
        #Readout fringe players from other.csv (m)
        read_out = np.genfromtxt("others.csv",delimiter=";",autostrip=True,comments="#",skip_header=1,usecols=(0,1))
        #Readout fringe switched to conform with format; finge[0]=quantity fringe[1]=bid
        self.fringe = np.fliplr(read_out)
        self.fringe = np.pad(self.fringe,((0,0),(1,0)),mode='constant')


        self.reward_range = (0, 1)

        
    def _next_observation(self, last_action):
        
        """
        Get State:
            includes the current Demand  -> Q
            - the Capacitys of all Players -> self.CAP[], (Maybe will change to: sold Capcitys from the Round before)
            - Memory of the Bids from the round before of all Players -> last_action[]
            (Consideration: of including Memory from more played rounds)
        
        Output:
            State as np.array of shape [7,1] 
    
        """
        
        if self.current_step == 0:
            last_action = self.start_action           
        else:
            last_action = last_action
            
        
        #Q = np.array([500, 1000, 1500])
        #Q = np.array([800])
        #Q = random.choice(Q)
        
        Q = np.random.randint(900, 1100, 1)
      
        obs = np.array([Q[0], self.CAP[0], self.CAP[1], self.CAP[2], last_action[0], last_action[1], last_action[2]])
        
        
        return obs

    def step(self, action, last_action):
        
        self.current_step += 1
        
        # get current state, which includes Memory of the bids from the round before        
        obs = self._next_observation(last_action)
            
        Demand = obs[0]
        q = obs[0]
                  
        #Decision on Strategic or Fringe Player 0
        if self.Fringe == 1:
            Sup0 = self.fringe
        else:
            Sup0 = np.array([int(0), self.CAP[0], action[0], self.costs[0], self.CAP[0]])            
                 
        #Strategic Players
        Sup1 = np.array([int(1), self.CAP[1], action[1], self.costs[1], self.CAP[1]])
        Sup2 = np.array([int(2), self.CAP[2], action[2], self.costs[2], self.CAP[2]])
                
        All = np.stack((Sup0, Sup1, Sup2))
        
        # Returns all Players orderd by lowest bid and assigns to them their quantities they can sell
        # (Output: [0]= price, [1]= Orderd Player lists, [2]= quantities to sell in original order)
        market = market_clearing(q, All)
        
        p = market[0]
        sold_quantities = market[2]

     
        #### rewards
        
        reward0 = (p - Sup0[3]) * sold_quantities[0]                        
        reward1 = (p - Sup1[3]) * sold_quantities[1]
        reward2 = (p - Sup2[3]) * sold_quantities[2]
        
        
        if self.Rewards == 1:
            reward0 = reward0 - (Sup0[3] * (Sup0[4] - sold_quantities[0]))
            reward1 = reward1 - (Sup1[3] * (Sup1[4] - sold_quantities[1]))
            reward2 = reward2 - (Sup2[3] * (Sup2[4] - sold_quantities[2]))        
        
        if self.Rewards == 2:
            reward0 = reward0 / (Sup0[3] * Sup0[4])
            reward1 = reward1 / (Sup1[3] * Sup1[4])
            reward2 = reward2 / (Sup2[3] * Sup2[4])
            
        if self.Rewards == 3: 
            reward0 = reward0 - (Sup0[3] * (Sup0[4] - sold_quantities[0]))
            reward1 = reward1 - (Sup1[3] * (Sup1[4] - sold_quantities[1]))
            reward2 = reward2 - (Sup2[3] * (Sup2[4] - sold_quantities[2]))
            
            reward0 = reward0 / (Sup0[3] * Sup0[4])
            reward1 = reward1 / (Sup1[3] * Sup1[4])
            reward2 = reward2 / (Sup2[3] * Sup2[4])            
        
        if self.Rewards == 4:  
            reward0 = reward0 - (Sup0[3] * (Sup0[4] - sold_quantities[0]))
            reward1 = reward1 - (Sup1[3] * (Sup1[4] - sold_quantities[1]))
            reward2 = reward2 - (Sup2[3] * (Sup2[4] - sold_quantities[2]))
            
            expWin0 = (Sup0[2] - Sup0[3]) * sold_quantities[0]
            expWin1 = (Sup1[2] - Sup1[3]) * sold_quantities[1]
            expWin2 = (Sup2[2] - Sup2[3]) * sold_quantities[2]
            
            # is this a correct way to avoid producig nan
            if expWin0 == 0:
                expWin0 = 0.0000000000001
            if expWin1 == 0:
                expWin1 = 0.0000000000001
            if expWin2 == 0:
                expWin2 = 0.0000000000001
            
            reward0 = reward0 / expWin0
            reward1 = reward1 / expWin1
            reward2 = reward2 / expWin2
        
     
        
        # Idea of setting reward in relation to their Max reward
        #E:\Master_E\Workspace\Bidding-Learning\EnMarketEnv07_.py:181: 
        #RuntimeWarning: invalid value encountered in double_scalars
        #reward0 = reward0/(Sup0[3] * Sup0[1])
        #reward1 = reward1/(Sup1[3] * Sup1[1])
        #reward2 = reward2/(Sup2[3] * Sup2[1])


        reward = np.append(reward0, reward1)
        reward = np.append(reward, reward2)

        
      
        
        ## Render Commands 
        self.safe(action, self.current_step)
        
        self.sum_q += Demand
        self.avg_q = self.sum_q/self.current_step
        self.sum_action += action
        self.avg_action = self.sum_action/self.current_step
        self.current_q = Demand
        self.last_rewards = reward
        self.last_bids = action
        self.sum_rewards += reward
        self.avg_rewards = self.sum_rewards/self.current_step
        
        #### DONE
        done = self.current_step == 128#256#128 
        
        ##### Next Obs
        
        obs = self._next_observation(action)
        last_action = action        

       

        return obs, reward, done, {}
    
    def safe(self, action, current_step):
        
        Aktionen = (action, current_step)
        self.AllAktionen.append(Aktionen)
        #last_action = action
        
       # return last_action
        
    
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.avg_action = 0
        self.sum_action = 0
        self.sum_q = 0
        self.sum_rewards = 0
        self.AllAktionen = deque(maxlen=500)
        self.start_action = np.array([0, 0, 0])
        
        return self._next_observation(self.start_action)
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Step: {self.current_step}')
        print(f'AllAktionen: {self.AllAktionen}')
        print(f'Last Demand of this Episode: {self.current_q}')
        print(f'Last Bid of this Episode: {self.last_bids}')
        print(f'Last Reward of this Episode: {self.last_rewards}')
        print(f'Average Demand: {self.avg_q}')
        print(f'Average Bid: {self.avg_action}')
        print(f'Average Reward: {self.avg_rewards}')
        
        
  


