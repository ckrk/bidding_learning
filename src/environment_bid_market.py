import numpy as np
import os

import gym
from gym import spaces
from collections import deque
import logging

from src.market_clearing import market_clearing, converter
from src.agent_ddpg import agent_ddpg


class EnvironmentBidMarket(gym.Env):
    
    """
    Energy Market environment for OpenAI gym
    
    Sets_up an envrionment from several static parameters
    Once set_up it receives actions from players, 
    then outputs rewards and determines next state of environment
    Discrete Box explanation: [starting point, how many steps, how big are the steps(if 0, Discrete action is disabled)]
    """
    metadata = {'render.modes': ['human']}   ### ?

    def __init__(self, capacities, costs, demand =[500, 501], agents = 1, fringe_player=1, rewards=0, split=0, past_action = 1, lr_actor = 1e-6, lr_critic = 1e-4, normalization = 'none', discrete = [0, 10, 0], reward_scaling = 1, action_limits = [-100,100], price_cap = 40):              
        super(EnvironmentBidMarket, self).__init__()
        
        # basic game parameters
        self.capacities = capacities
        self.costs = costs
        self.demand = demand
        self.agents = agents
        self.action_limits = action_limits # displayed action limit
        self.price_cap = price_cap # bids above price cap enter tie-break
        # additional opptions
        self.fringe_player = fringe_player
        self.rewards = rewards
        self.reward_scaling = reward_scaling
        self.split = split
        self.past_action = past_action
        self.discrete = discrete
        # learning rate parameters for (DDPG)Agents (for the Neuronal Networks)
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.normalization = normalization
        
        # Continous action space for bids
        self.action_space = spaces.Box(low=np.array([self.action_limits[0]]), high=np.array([self.action_limits[1]]), dtype=np.float16)
        
        # fit observation_space size to choosen environment settings
        observation_space_size = 1 + self.agents*2
        
        if self.split == 1:
            self.action_space = spaces.Box(low=np.array([self.action_limits[0],self.action_limits[0],0]), high=np.array([self.action_limits[1],self.action_limits[1],1]), dtype=np.float16)
            observation_space_size = 1+ self.agents*2 + self.agents
        
        if self.fringe_player == 1:
            observation_space_size = observation_space_size + 60 #self.fringe.shape[0]
            
        # Discrete Action space
        if self.discrete[2] != 0:
            # including splits ?
            self.action_space = spaces.Box(low=0, high=1, shape=(self.discrete[1],1), dtype=np.float16)
            
            
        if past_action == 0:
            observation_space_size = 1 + self.agents
        
        # Set observation space continious   
        self.observation_space = spaces.Box(low=self.action_limits[0], high=self.action_limits[1], shape=(observation_space_size,1), dtype=np.float16)
        
            
        # Reward Range
        # This cant be fixed!
        self.reward_range = (0, 1000000)
    
    def create_agents(self, env):
        
        agents_list = []
        
        for n in range(self.agents):
            agents_list.append(agent_ddpg(env, actor_learning_rate=self.lr_actor, critic_learning_rate=self.lr_critic,
                                              discrete = self.discrete, norm = self.normalization))
            
        return agents_list
    
    def discretization_of_actions(self, action, nmb_agents):
        
        discret_action_space = np.arange(self.discrete[0], self.discrete[1]*self.discrete[2], self.discrete[1])
        
        discretised_action =[]
        
        for n in range(nmb_agents):
            discretised_action.append([discret_action_space[np.argmax(action[n])]])
            
        discretised_action = np.asarray(discretised_action)
        
        return discretised_action
    
    def set_up_suppliers(self, action, nmb_agents):
        """
        Sets Up all the Agents to act as Suppliers on Energy Market
        Supplier: Agent Number (int), their own Capacity, their Action, their cost, again their Capacity
        output is on big 2 dimensional np.array containing all Suppliers (optional: + fringe Players)
    
        """
        
        suppliers = [0]*nmb_agents
        
        for n in range(nmb_agents):
            a1 = action[n,0]
            suppliers[n] = [int(n), self.capacities[n], a1, self.costs[n], self.capacities[n]]
            
            if self.split == 1:
                a1,a2,a3 = action[n]
                suppliers[n] = [int(n), self.capacities[n], a1, a2, a3, self.costs[n], self.capacities[n]]
                
        suppliers = np.asarray(suppliers)
        
        if self.fringe_player == 1:
            suppliers = np.concatenate([suppliers, self.fringe])
        
        return suppliers
        
    def _next_observation(self, nmb_agents):
        
        """
        Set Up State
        State includes: Demand, Capacitys of all Players, sort by from lowest to highest last Actions of all Players (Optional)
    
        """
        
        #demand = np.random.randint(self.demand[0], self.demand[1], 1)
        demand = np.random.uniform(self.demand[0], self.demand[1], 1)
        obs = np.append(demand, self.capacities)
        
        if self.past_action == 1:
            #obs = np.insert(obs, nmb_agents+1, self.last_action)
            obs = np.concatenate([obs, self.last_action])
            if self.fringe_player == 1:
                obs = np.concatenate([obs, self.fringe[:,2]])   ## last actions fringe
        
        #obs = np.asarray(obs)

        return  obs


    def step(self, action):
        
        self.current_step += 1
        
        # get current state        
        obs = self._next_observation(self.agents)
        demand = obs[0]
        
        # for discrete action space, action must first be discretised
        if self.discrete[2] != 0:
            true_action = action
            action = self.discretization_of_actions(true_action, self.agents) 
        # test scaling sigmoid
        #action = action *1000
        
        # set up all the agents as suppliers in the market
        all_suppliers = self.set_up_suppliers(action, self.agents)

        # market_clearing: orders all suppliers from lowest to highest bid, 
        # last bid of cumsum offerd capacitys determines the price; also the real sold quantities are derived
        # if using splits, convert them in the right shape for market_clearing-function 
        # and after that combine sold quantities of the same supplier again
        if self.split == 0:
            market = market_clearing(demand, all_suppliers, self.price_cap)
            self.last_action= action
        else:
            all_suppliers_split = converter(all_suppliers, self.agents)
            market = market_clearing(demand, all_suppliers_split, self.price_cap)
            self.last_action = action[:,0:2]
            
        
        # save last actions for next state (= next obeservation) and sort them by lowest bids
        self.last_action = np.sort(self.last_action, axis = None)
        
        #market price and sold quantities determined through market clearing
        market_price = market[0]
        sold_quantities = market[2]

        # caluclate rewards
        reward = self.reward_function(all_suppliers, sold_quantities, market_price, self.agents, self.rewards, action)
        

        # Render Commands 
        self.safe(action, self.current_step)
        self.sold_quantities = sold_quantities
        self.last_market_price = market_price
        self.Suppliers = all_suppliers 
        
        self.last_q = demand
        self.sum_q += demand
        self.avg_q = self.sum_q/self.current_step
        
        self.last_bids = action
        self.sum_action += action
        self.avg_action = self.sum_action/self.current_step
        
        self.last_rewards = reward
        self.sum_rewards += reward
        self.avg_rewards = self.sum_rewards/self.current_step
        
        
        #### DONE and next_state
        done = self.current_step >= 1 #128 #!!! 
        obs = self._next_observation(self.agents)
        

        return obs, reward, done, {}
    
    
    def safe(self, action, current_step):
        # to save all actions during one round
        Aktionen = (action, current_step)
        self.AllAktionen.append(Aktionen)
        
    def reward_function(self, suppliers, sold_quantities, p, nmb_agents, penalty, action):
        '''
        Different Options of calculating the Reward
        Rewards = 0: Default, Reward is (price-costs)*acceptedCAP 
        Rewards = 1: Reward is (price-costs)*acceptedCAP - (price*unsoldCAP)
        Rewards = 2: Reward is ((price-costs)*acceptedCAP)/(cost*maxCAP)
        Rewards = 3: (= combination of 1 and 2): Reward is ((price-costs)*acceptedCAP - (price*unsoldCAP))/(cost*maxCAP)
        Rewards = 4: Reward is Reward - abs(Reward - maxReward)
        
        '''
        # rescaling the rewards to avoid hard weight Updates of the Criticer 
        rescale = self.reward_scaling #0.01#0.00025#0.0001
        #maxreward = self.capacities[0] *50 *rescale
        p = np.clip(p,-100, self.price_cap)
        
        if self.fringe_player == 1:
            rescale = 0.01 #0.01
            #maxreward = self.actio_limit

        # Position of costs is diffrent between suppliers with and without Split
        cost_position = 3
        if self.split == 1:
            cost_position = 5
            
        reward = [0]*nmb_agents
        
        for n in range(nmb_agents):
            reward[n] = (p - suppliers[n,cost_position]) * sold_quantities[n] * rescale # "clipping/rescaling rewards"
    
        reward = np.asarray(reward)

        if penalty == 1:
            for n in range(nmb_agents):
                reward[n] = reward[n] - (suppliers[n,cost_position]*(suppliers[n,1] - sold_quantities[n]))       
        
        if penalty == 2:
            for n in range(nmb_agents):
                reward[n] = reward[n] / (suppliers[n,cost_position] * suppliers[n,1])       
            
        if penalty == 3:
            for n in range(nmb_agents):
                reward[n] = reward[n] - (suppliers[n,cost_position]*(suppliers[n,1] - sold_quantities[n]))
                reward[n] = reward[n] / (suppliers[n,cost_position] * suppliers[n,1]) 
          
        if penalty == 4:
            '''
            !!! Should be unitzied if capacaties and cost structers differs between the Agents !!!
            '''
            for n in range(nmb_agents):
                reward[n] = reward[n] - abs(reward[n] - max(reward))
        
        # Tipp (especially for games vs Fringe Player needed); Split would need an own implementation (if both actions are =0)
        
        if self.split == 0 and self.fringe_player == 1:
        #if self.split == 0:
            for n in range(nmb_agents):
                if action[n] <= 0:
                    reward[n] = 0
                    #reward[n] = np.clip(reward[n], reward[n], 0) 
        
        # unsure yet, if clipping is needed
        #reward = np.clip(reward,-100, maxreward) ## limit und scaling bei "MITfringe" deutlich höher (dafür rescaling niedriger)        

        
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
        
        self.last_action = np.zeros(self.agents)
        
        if self.split == 1:
            self.last_action = np.zeros(self.agents*2)
       
        if self.fringe_player == 1:
            #Fringe Player
            #Readout fringe players from others.csv (m)

            #path = os.path.join(os.path.dirname(__file__), '../data/fringe-players/others.csv')            
            #read_out = np.genfromtxt(path,delimiter=";",autostrip=True,comments="#",skip_header=1,usecols=(0,1))
            
            #Readout fringe players from test_fringe02.csv (m)
            
            path = os.path.join(os.path.dirname(__file__), '../data/fringe-players/others.csv')            
            read_out = np.genfromtxt(path,delimiter=";",autostrip=True,comments="#",skip_header=1,usecols=(0,1))
            
            
            #Readout fringe switched to conform with format; finge[0]=quantity fringe[1]=bid
            self.fringe = np.fliplr(read_out)
            self.last_action = np.zeros(self.agents)
            
            if self.split == 1:
                self.fringe = np.pad(self.fringe,((0,0),(1,4)),mode='constant', constant_values=(self.agents, 1))
                self.last_action = np.zeros(self.agents*2)
            else:
                self.fringe = np.pad(self.fringe,((0,0),(1,2)),mode='constant', constant_values=(self.agents, 1))
        
        # Errors
        if len(self.capacities) != self.agents or len(self.costs) != self.agents or len(self.capacities) != len(self.costs):
            return print('******************************\n ERROR: length of CAP and costs has to correspond to the number of Agents \n******************************')


        
        return self._next_observation(self.agents)
    
    def render(self, mode='human', close=False):
        # Calls an output of several important parameters during the learning
        # This defines the content of the output
        #print(f'Step: {self.current_step}')
        print(f'AllAktionen: {self.AllAktionen}')
        #print(f'Last Demand of this Episode: {self.last_q}')
        #print(f'Last Bid of this Episode:\n {self.last_bids}')
        #print(f'Last Reward of this Episode: {self.last_rewards}')
        print(f'last sold Qs:{self.sold_quantities}')
        #print(f'Last Market Price: {self.last_market_price}')
        #print(f'Average Bid:\n {self.avg_action}')
        #print(f'Average Reward: {self.avg_rewards}')
        #print(f'Last_action: {self.last_action}')
        #print(f'Suppliers: {self.Suppliers}')
        #print(f'Average Demand: {self.avg_q}')
        
    def variable_render(self):
        return self.sold_quantities, self.last_market_price
        

    def logger(self, episode, test_round):        
        ####Logger
        logging.basicConfig(filename = 'lr6_4_1-vs-1_costs20cap50demand70_rescaling01_woTippwoPastAction_60gauNoise2RC100DC1_pricecap40realcap100_wTieBreak40_batchx4_00.log', level= logging.INFO, format='%(levelname)s:%(asctime)s:%(message)s')
        
        logging.info(f'Test Round: {test_round}')
        logging.info(f'Episode: {episode}')
        logging.info(f'AllAktionen: {self.AllAktionen}')
        #logging.info(f'Average Reward: {self.avg_rewards}')
        #logging.info(f'Last Market Price: {self.last_market_price}')
        
        
        
