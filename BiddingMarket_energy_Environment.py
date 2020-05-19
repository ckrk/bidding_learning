import numpy as np

import gym
from gym import spaces
from collections import deque

from market_clearing import market_clearing, converter
from DDPG_main import DDPGagent_main

class BiddingMarket_energy_Environment(gym.Env):
    
    """
    Energy Market environment for OpenAI gym
    
    Sets_up an envrionment from several static parameters
    Once set_up it receives actions from players, 
    then outputs rewards and determines next state of environment
    """
    metadata = {'render.modes': ['human']}   ### ?

    def __init__(self, CAP, costs, Demand =[5, 6], Agents = 1, Fringe=1, Rewards=0, Split=0, past_action = 1, lr_actor = 1e-4, lr_critic = 1e-3, Discrete = 0):              
        super(BiddingMarket_energy_Environment, self).__init__()
        
        # basic game parameters
        self.CAP = CAP
        self.costs = costs
        self.Demand = Demand
        self.Agents = Agents
        # additional opptions
        self.Fringe = Fringe
        self.Rewards = Rewards
        self.Split = Split
        self.past_action = past_action
        self.Discrete = Discrete
        # learning rate parameters for (DDPG)Agents (for the Neuronal Networks)
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.price_cap = 10000
        
        # Continous action space for bids
        self.action_space = spaces.Box(low=np.array([-100]), high=np.array([10000]), dtype=np.float16)
        
        # fit observation_space size to choosen environment settings
        observation_space_size = 1 + self.Agents*2
        if self.Fringe == 1:
            observation_space_size = observation_space_size + 60 #self.fringe.shape[0]
        
        if self.Split == 1:
            self.action_space = spaces.Box(low=np.array([0,0,0]), high=np.array([10000,10000,1]), dtype=np.float16)
            observation_space_size = 1+ self.Agents*2 + self.Agents
            if self.Fringe == 1:
                observation_space_size = observation_space_size + 60 #self.fringe.shape[0]
                
        if past_action == 0:
            observation_space_size = 1 + self.Agents
        
        # Set observation space continious   
        self.observation_space = spaces.Box(low=0, high=10000, shape=(observation_space_size,1), dtype=np.float16)
        
        # Discrete Action space
        if self.Discrete == 1:
            self.action_space = spaces.Discrete(9)
            
        # Reward Range
        self.reward_range = (0, 1000000)
    
    def create_agents(self, env):
        
        agents_list = []
        
        for n in range(self.Agents):
            agents_list.append(DDPGagent_main(env, actor_learning_rate=self.lr_actor, critic_learning_rate=self.lr_critic,
                                              discrete = self.Discrete, discrete_split = self.Split))
            
        return agents_list
    
    def set_up_suppliers(self, action, nmb_agents):
        """
        Sets Up all the Agents to act as Suppliers on Energy Market
        Supplier: Agent Number (int), their own Capacity, their Action, their cost, again their Capacity
        output is on big 2 dimensional np.array containing all Suppliers (optional: + fringe Players)
    
        """
        
        suppliers = [0]*nmb_agents
        
        for n in range(nmb_agents):
            a1 = action[n,0]
            suppliers[n] = [int(n), self.CAP[n], a1, self.costs[n]]
            
            if self.Split == 1:
                a1,a2,a3 = action[n]
                suppliers[n] = [int(n), self.CAP[n], a1, a2, a3, self.costs[n]]
                
        suppliers = np.asarray(suppliers)
        
        if self.Fringe == 1:
            suppliers = np.concatenate([suppliers, self.fringe])
        
        return suppliers
        
    def _next_observation(self, nmb_agents):
        
        """
        Set Up State
        State includes: Demand, Capacitys of all Players, sort by from lowest to highest last Actions of all Players (Optional)
    
        """
        #Q = np.array([500, 1000, 1500])
        #Q = np.random.choice(Q)
        #Q = np.array([Q])
        
        Q = np.random.randint(self.Demand[0], self.Demand[1], 1)
        obs = np.append(Q, self.CAP)
        
        if self.past_action == 1:
            #obs = np.insert(obs, nmb_agents+1, self.last_action)
            obs = np.concatenate([obs, self.last_action])
            if self.Fringe == 1:
                obs = np.concatenate([obs, self.fringe[:,2]])   ## last actions fringe

        return  obs


    def step(self, action):
        
        self.current_step += 1
        
        # get current state        
        obs = self._next_observation(self.Agents)
        Demand = obs[0]
        q = obs[0]
        
        if self.Discrete == 1:
            action = action #* 100
        
        # set up all the agents as suppliers in the market
        all_suppliers = self.set_up_suppliers(action, self.Agents)
        
        # market_clearing: orders all suppliers from lowest to highest bid, 
        # last bid of cumsum offerd capacitys determines the price; also the real sold quantities are derived
        # if using splits, convert them in the right shape for market_clearing-function 
        # and after that combine sold quantities of the same supplier again
        if self.Split == 0:
            market = market_clearing(q, all_suppliers)
            self.last_action= action
        else:
            all_suppliers_split = converter(all_suppliers, self.Agents)
            market = market_clearing(q, all_suppliers_split)
            self.last_action = action[:,0:2]
        
        # save last actions for next state (= next obeservation) and sort them by lowest bids
        self.last_action = np.sort(self.last_action, axis = None)
        
        #market price and sold quantities determined through market clearing
        market_price = market[0]
        sold_quantities = market[2]
        
        # caluclate rewards
        reward = self.reward_function(all_suppliers, sold_quantities, market_price, self.Agents, self.Rewards, action)
        

        # Render Commands 
        self.safe(action, self.current_step)
        self.sold_quantities = sold_quantities
        self.last_market_price = market_price
        self.Suppliers = all_suppliers 
        
        self.last_q = Demand
        self.sum_q += Demand
        self.avg_q = self.sum_q/self.current_step
        
        self.last_bids = action
        self.sum_action += action
        self.avg_action = self.sum_action/self.current_step
        
        self.last_rewards = reward
        self.sum_rewards += reward
        self.avg_rewards = self.sum_rewards/self.current_step
        
        
        #### DONE and next_state
        done = self.current_step == 128 
        obs = self._next_observation(self.Agents)
        

        return obs, reward, done, {}
    
    
    def safe(self, action, current_step):
        # to save all actions during one round
        Aktionen = (action, current_step)
        self.AllAktionen.append(Aktionen)
        
    def reward_function(self, suppliers, sold_quantities, p, nmb_agents, Penalty, action):
        '''
        Different Options of calculating the Reward
        Rewards = 0: Default, Reward is (price-costs)*acceptedCAP 
        Rewards = 1: Reward is (price-costs)*acceptedCAP - (price*unsoldCAP)
        Rewards = 2: Reward is ((price-costs)*acceptedCAP)/(cost*maxCAP)
        Rewards = 3 (= combination of 1 and 2): Reward is ((price-costs)*acceptedCAP - (price*unsoldCAP))/(cost*maxCAP)
        #Reward 4 only works without Splits#
        Rewards = 4: Reward is (Reward 1)/((ownBid-cost)*maxCAP)
        
        '''
        # rescaling the rewards to avoid hard weight Updates of the Criticer 
        rescale = 0.00001
        maxreward = 10
        if self.Fringe == 1:
            rescale = 0.01
            maxreward = 10000
            
        # Position of costs is diffrent between suppliers with and without Split
        cost_position = 3
        if self.Split == 1:
            cost_position = 5
            
        reward = [0]*nmb_agents
        
        for n in range(nmb_agents):
            reward[n] = (p - suppliers[n,cost_position]) * sold_quantities[n] * rescale # "clipping/rescaling rewards"
        reward = np.asarray(reward)
        

        if Penalty == 1:
            for n in range(nmb_agents):
                reward[n] = reward[n] - (suppliers[n,cost_position]*(suppliers[n,1] - sold_quantities[n]))       
        
        if Penalty == 2:
            for n in range(nmb_agents):
                reward[n] = reward[n] / (suppliers[n,cost_position] * suppliers[n,1])       
            
        if Penalty == 3:
            for n in range(nmb_agents):
                reward[n] = reward[n] - (suppliers[n,cost_position]*(suppliers[n,1] - sold_quantities[n]))
                reward[n] = reward[n] / (suppliers[n,cost_position] * suppliers[n,1]) 
          
        if Penalty == 4:
            for n in range(nmb_agents):
                reward[n] = reward[n] - (suppliers[n,cost_position]*(suppliers[n,1] - sold_quantities[n]))
                #expWin = Suppliers[n,2]  * sold_quantities[n] # Alternative!!
                expWin = (suppliers[n,2] - suppliers[n,cost_position]) * sold_quantities[n] #auskommentiern wenn mit alternative
                expWin = np.clip(expWin, 0.0000001, 10000000)
                reward[n] = reward [n] /expWin
                if self.Split == 1:
                    break
                print('ERROR: only works without Split')
        
        # TIPP (especially for games vs Fringe Player needed)
        for n in range(nmb_agents):
            if action[n] <= 0:
                reward[n] = 0 
        
        # unsure yet, if clipping is needed
        #reward = np.clip(reward,-10, maxreward) ## limit und scaling bei "MITfringe" deutlich höher (dafür rescaling niedriger)        

        return reward
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.avg_action = 0
        self.sum_action = 0
        self.sum_q = 0
        self.sum_rewards = 0
        self.avg_rewards = 0
        self.AllAktionen = deque(maxlen=500)
        
        self.last_action = np.zeros(self.Agents)
        
        if self.Split == 1:
            self.last_action = np.zeros(self.Agents*2)
       
        if self.Fringe == 1:
            #Fringe Player
            #Readout fringe players from other.csv (m)

            #read_out = np.genfromtxt("others.csv",delimiter=";",autostrip=True,comments="#",skip_header=1,usecols=(0,1))
            
            #Readout fringe players from simple_fringe.csv (m)
            read_out = np.genfromtxt("simple_fringe.csv",delimiter=";",autostrip=True,comments="#",skip_header=1,usecols=(0,1))
            
            
            #Readout fringe switched to conform with format; finge[0]=quantity fringe[1]=bid
            self.fringe = np.fliplr(read_out)
            self.last_action = np.zeros(self.Agents)
            
            if self.Split == 1:
                self.fringe = np.pad(self.fringe,((0,0),(1,3)),mode='constant', constant_values=(self.Agents, 1))
                self.last_action = np.zeros(self.Agents*2)
            else:
                self.fringe = np.pad(self.fringe,((0,0),(1,1)),mode='constant', constant_values=(self.Agents, 1))
        
        # Errors
        if len(self.CAP) != self.Agents or len(self.costs) != self.Agents or len(self.CAP) != len(self.costs):
            return print('******************************\n ERROR: length of CAP and costs has to correspond to the number of Agents \n******************************')


        
        return self._next_observation(self.Agents)
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Step: {self.current_step}')
        print(f'AllAktionen: {self.AllAktionen}')
        print(f'Last Demand of this Episode: {self.last_q}')
        print(f'Last Bid of this Episode:\n {self.last_bids}')
        print(f'Average Bid:\n {self.avg_action}')
        print(f'Last Reward of this Episode: {self.last_rewards}')
        print(f'Average Reward: {self.avg_rewards}')
        #print(f'Last_action: {self.last_action}')
        #print(f'Suppliers: {self.Suppliers}')
        print(f'Average Demand: {self.avg_q}')
        print(f'sold Qs:{self.sold_quantities}')
        print(f'Last Market Price: {self.last_market_price}')
        
        
        
