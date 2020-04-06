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

class EnMarketEnv_2Player01(gym.Env):
    
    """
    Energy Market environment for OpenAI gym
    market_clearing included
    

    Changes: observation_space with additional Dimensons: shape(7,1): 1 demand, 3 Capcitys, 3 actions (maybe add costs)
    Rewards = 0: Default, Reward is (price-costs)*acceptedCAP 
    Rewards = 1: Default, Reward is (price-costs)*acceptedCAP - (price*unsoldCAP)
    Rewards = 2: Default, Reward is ((price-costs)*acceptedCAP)/(cost*maxCAP)
    Rewards = 3: Default, Reward is ((price-costs)*acceptedCAP - (price*unsoldCAP))/(cost*maxCAP)

    
    only works with test04 and DDPG03_
    
    """
    metadata = {'render.modes': ['human']}   ### ?

    def __init__(self, CAP, costs, Fringe=0, Rewards=0):              ##### mit df ?
        super(EnMarketEnv_2Player01, self).__init__()
        
        self.CAP = CAP
        self.costs = costs
        self.Fringe = Fringe
        self.Rewards = Rewards
        
        # Continous action space for bids
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([10000]), dtype=np.float16)

        # Discrete Demand opportunities
        #self.observation_space = spaces.Box(low=0, high=10000, shape=(7,1), dtype=np.float16)
        self.observation_space = spaces.Box(low=0, high=10000, shape=(5,1), dtype=np.float16)


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
            - Memory of the Rewards from the round before of all Players -> last_rewards[]
            (Consideration: of including Memory from more played rounds)
        
        Output:
            State as np.array of shape [7,1] 
    
        """
        
        #if self.current_step == 0:
         #   last_action = self.start_action
          #  last_rewards = self.start_rewards
        #else:
         #   last_action = last_action
          #  last_rewards = last_rewards
        
        #Q = np.array([500, 1000, 1500])
        Q = np.array([70])
        #Q = random.choice(Q)
        
        #Q = np.random.randint(900, 1100, 1)
      
        obs = np.array([Q[0], self.CAP[0], self.CAP[1], last_action[0], last_action[1]])
        #obs = np.array([Q[0], self.CAP[0], self.CAP[1], last_action[0], last_action[1], 
                        #last_rewards[0], last_rewards[1]])
    
        #obs = np.array([Q[0], soldCAP[0], soldCAP[1], last_action[0], last_action[1], 
                        #last_rewards[0], last_rewards[1], self.costs[0],self.costs[1]])
        
        return obs

    def step(self, action, last_action):
        
        
        obs = self._next_observation(last_action)
        
        self.current_step += 1
        
        Demand = obs[0]
        q = obs[0]
        
        

        
        #Decision on Strategic or Fringe Player 0
        if self.Fringe == 1:
            Sup0 = self.fringe
        else:
            Sup0 = np.array([0, self.CAP[0], action[0], self.costs, self.CAP[0]])
            
       
        #Strategic Players
        Sup1 = np.array([1, self.CAP[1], action[1], self.costs, self.CAP[1]])
        #Sup2 = np.array([2, self.CAP[2], action[2], self.costs, self.CAP[2]])
        

        All = np.stack((Sup0, Sup1))
        
        market = market_clearing(q, All)
        
        #Naming the results of the Market Clearing
        p = market[0]
        allorderd = market[1]

        #sold_quantities = market[2]

        minSup = allorderd[0]
        medSup = allorderd[1]
        #maxSup = allorderd[2]
        
        #if minSup[2] == medSup[2]:
         #   minSup[1] = q*0.5
          #  medSup[1] = q*0.5
    
    
        
        # rewards
        qmin = np.clip(minSup[1], 0, q)
        reward_min = (p - minSup[3]) * qmin 
                
        q = q - qmin
        qmed = np.clip(medSup[1], 0, q)
        reward_med = (p - medSup[3]) * qmed 
           
        #q = q - qmed
        #qmax = np.clip(maxSup[1], 0, q)
        #reward_max = (p - maxSup[3]) * qmax 
        
        
        if self.Rewards == 1:
            reward_min = reward_min - (minSup[3] * (minSup[4] - minSup[1]))
            reward_med = reward_med - (medSup[3] * (medSup[4] - medSup[1]))
            #reward_max = reward_max - (maxSup[3] * (maxSup[4] - maxSup[1]))
        
        if self.Rewards == 2:
            reward_min = reward_min / (minSup[3] * minSup[4])
            reward_med = reward_med / (medSup[3] * medSup[4])
            #reward_max = reward_max / (maxSup[3] * maxSup[4])
            
        if self.Rewards == 3:           
            reward_min = reward_min - (minSup[3] * (minSup[4] - minSup[1]))
            reward_med = reward_med - (medSup[3] * (medSup[4] - medSup[1]))
            #reward_max = reward_max - (maxSup[3] * (maxSup[4] - maxSup[1]))
            
            reward_min = reward_min / (minSup[3] * minSup[4])
            reward_med = reward_med / (medSup[3] * medSup[4])
            #reward_max = reward_max / (maxSup[3] * maxSup[4])
        
        if self.Rewards == 4: 
            #4or4a with X
            #reward_min = reward_min - (minSup[3] * (minSup[4] - minSup[1]))
            #reward_med = reward_med - (medSup[3] * (medSup[4] - medSup[1]))
            #reward_max = reward_max - (maxSup[3] * (maxSup[4] - maxSup[1]))
            
            #4
            minWin = np.clip((minSup[2]-minSup[3]), 0.0001, minSup[2])
            medWin = np.clip((medSup[2]-medSup[3]), 0.0001, medSup[2])
            #maxWin = np.clip((maxSup[2]-maxSup[3]), 0.0001, maxSup[2])
            #4a (not useful)
            #minWin = np.clip(minSup[2], 0.0001, minSup[2])
            #medWin = np.clip(medSup[2], 0.0001, medSup[2])
            #maxWin = np.clip(maxSup[2], 0.0001, maxSup[2])
            
            reward_min = reward_min / (minWin * minSup[4])
            reward_med = reward_med / (medWin* medSup[4])
            #reward_max = reward_max / (maxWin * maxSup[4])
        
        
        
        
        #reward_min = reward_min - (qmin * minSup[3])
        #reward_med = reward_med - (qmed * medSup[3])
        #reward_max = reward_max - (qmax * maxSup[3])
        
        
        ### get information back ###
        reward_min = np.append(minSup[0], reward_min)
        reward_med = np.append(medSup[0], reward_med)
        #reward_max = np.append(maxSup[0], reward_max)
            
        ### find your partner again ###
            
        if reward_min[0] == 0:
            reward0 = reward_min[1]
            reward1 = reward_med[1]
           
        else:
            reward0 = reward_med[1]
            reward1 = reward_min[1]  
           
       
        
        
        
        #Maybe the Order could already be redone in the market clearing, 
        #than we woudn't need the "find your partner again" - part anymore
        #reward0 = sold_quantities[0]*p 
        #reward1 = sold_quantities[1]*p
        #reward2 = sold_quantities[2]*p

        reward = np.append(reward0, reward1)
        #reward = np.append(reward, reward2)

        
      
        
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
        done = self.current_step == 128  
        
        ##### Next Obs
        
        last_action = action
        
        
        obs = self._next_observation(action)
        
        

       

        return obs, reward, done, {}
    
    def safe(self, action, current_step):
        
        Aktionen = (action, current_step)
        self.AllAktionen.append(Aktionen)
    
        
    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.avg_action = 0
        self.sum_action = 0
        self.sum_q = 0
        self.sum_rewards = 0
        self.AllAktionen = deque(maxlen=500)
        self.start_action = np.array([0,0,0])
        
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
        
        
  


