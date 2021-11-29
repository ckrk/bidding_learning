import numpy as np
import os

import gym
from gym import spaces
from collections import deque

from src.market_clearing import market_clearing
from src.agent_ddpg import agent_ddpg
from src.demand_models import demand_normal


class EnvironmentBidMarket(gym.Env):
    
    """
    Energy Market environment for OpenAI gym
    
    Sets_up an envrionment from several static parameters
    Once set_up, it receives actions from players, 
    then outputs rewards and determines next state of environment
    """
    #metadata = {'render.modes': ['human']}

    def __init__(self, capacities, costs, demand =[500, 501], agents = 1, fringe_player=1, past_action = 1, lr_actor = 1e-6, lr_critic = 1e-4, normalization = 'none', reward_scaling = 1, action_limits = [-1,1], rounds_per_episode = 1):              
        super(EnvironmentBidMarket, self).__init__()
        
        # basic game parameters considering the agents
        self.capacities = capacities
        self.costs = costs
        self.agents = agents
        self.action_limits = action_limits
        self.rounds_per_episode = rounds_per_episode
        
        # Determine type of demand model
        if  type(demand) == float or type(demand) == list:           
            self.demand = demand
        elif type(demand) == tuple:
            self.demand = demand
            self.means = demand[0]
            self.variances = demand[1]
            # Create normal demand model
            self.demand_model = demand_normal(self.means,self.variances)
        
        
        # additional options
        self.fringe_player = fringe_player
        self.reward_scaling = reward_scaling # rescaling the rewards to avoid hard weight Updates of the Criticer 
        self.past_action = past_action
        
        # learning rate parameters for (DDPG)Agents (for the Neuronal Networks)
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.normalization = normalization
        
        # Continous action space for bids
        self.action_space = spaces.Box(low=np.array([self.action_limits[0]]), high=np.array([self.action_limits[1]]), dtype=np.float16)
        
        # fit observation_space size to choosen environment settings
        observation_space_size = 1 + self.agents*2
        
        # vs a fringe player   
        if self.fringe_player == 1:
            #Readout fringe players from fringe_player_data_00.csv (m)
            path = os.path.join(os.path.dirname(__file__), '../data/fringe_players/fringe_player_data_00.csv')            
            read_out = np.genfromtxt(path,delimiter=";",autostrip=True,comments="#",skip_header=1,usecols=(0,1))
            self.fringe = np.fliplr(read_out)
            self.fringe = np.pad(self.fringe,((0,0),(1,2)),mode='constant', constant_values=(self.agents, 1))
            
            #observation space size including past actions from fringe as well
            observation_space_size = observation_space_size + self.fringe.shape[0]
            
        # if without past_action, observation space decreases by one per agent 
        if past_action == 0:
            observation_space_size = 1 + self.agents
        
        
        # Set observation space continious   
        self.observation_space = spaces.Box(low=self.action_limits[0], high=self.action_limits[1], shape=(observation_space_size,1), dtype=np.float16)
        
    
    def create_agents(self, env):
        
        agents_list = []
        for n in range(self.agents):
            agents_list.append(agent_ddpg(env, actor_learning_rate=self.lr_actor, critic_learning_rate=self.lr_critic, norm = self.normalization))
            
        return agents_list
    
    def set_up_suppliers(self, action, nmb_agents):
        """
        Sets Up all the Agents to act as Suppliers on Energy Market
        Supplier: Agent Number (int), their own Capacity, their Action, their cost, again their Capacity
        output is on big 2 dimensional np.array containing all Suppliers (optional: + fringe Players)
    
        """
        
        suppliers = [0]*nmb_agents
        print(suppliers)
        for n in range(nmb_agents):
            a1 = action[n,0]
            suppliers[n] = [int(n), self.capacities[n], a1, self.costs[n], self.capacities[n]]
                
        suppliers = np.asarray(suppliers)
        print(suppliers)
        
        if self.fringe_player == 1:
            self.fringe[:,2] = self.fringe[:,2]/np.max(self.fringe[:,2]) # to ensure that fringe player bids are also in a range frrom [-1,1]
            suppliers = np.concatenate([suppliers, self.fringe])
        
        return suppliers
        
    def _next_observation(self):
        
        """
        Set Up State
        State includes: Demand, Capacitys of all Players, sort by from lowest to highest last Actions of all Players (Optional)
    
        """
        if type(self.demand) == list:
            demand = np.random.uniform(self.demand[0], self.demand[1], 1)
        if type(self.demand) == tuple:
            demand = self.episodes_demand_timeseries[self.current_step]
        obs = np.append(demand, self.capacities)
        
        if self.past_action == 1:
            obs = np.concatenate([obs, self.last_action])
            if self.fringe_player == 1:
                obs = np.concatenate([obs, self.fringe])   ## last actions fringe
        
        return  obs


    def step(self, action):
        
        self.current_step += 1
        
        # get current state        
        obs = self._next_observation()
        demand = obs[0]
        
        # set up all the agents as suppliers in the market
        all_suppliers = self.set_up_suppliers(action, self.agents)

        # market_clearing: orders all suppliers from lowest to highest bid, 
        # last bid of cumsum offerd capacitys determines the price; also the real sold quantities are derived
        # if using splits, convert them in the right shape for market_clearing-function 
        # and after that combine sold quantities of the same supplier again
        market_price, _ , sold_quantities = market_clearing(demand, all_suppliers)
        self.last_action= action
        
        # save last actions for next state (= next obeservation) and sort them by lowest bids
        self.last_action = np.sort(self.last_action, axis = None)


        # calculate rewards
        reward = self.reward_function(all_suppliers, sold_quantities, market_price, self.agents, action)
        

        # Intersting Variables and Render Commands 
        self.safe(action, self.current_step)
        self.sold_quantities = sold_quantities
        self.market_price = market_price
        self.Suppliers = all_suppliers 
        
        self.last_demand = demand 
        self.sum_demand += demand
        self.avg_demand = self.sum_demand/self.current_step
        
        self.sum_action += action
        self.avg_action = self.sum_action/self.current_step
        
        self.last_rewards = reward
        self.sum_rewards += reward
        self.avg_rewards = self.sum_rewards/self.current_step
        
        
        #### DONE and next_state
        self.render()
        done = self.current_step >= self.rounds_per_episode
        obs = self._next_observation()
        

        return obs, reward, done, {}
    
    
    def safe(self, action, current_step):
        # to save all actions during one round
        Aktionen = (action, current_step)
        self.AllAktionen.append(Aktionen)
        
    def reward_function(self, suppliers, sold_quantities, market_price, nmb_agents, action):

  
        reward = [0]*nmb_agents
        for n in range(nmb_agents):
            reward[n] = ( market_price - suppliers[n,3]) * sold_quantities[n] * self.reward_scaling 
    
        reward = np.asarray(reward)
        
        # Tipp (especially for games vs Fringe Player needed); Split would need an own implementation (if both actions are =0)
        if self.fringe_player == 1:
            for n in range(nmb_agents):
                if action[n] <= 0:
                    reward[n] = 0
                    #reward[n] = np.clip(reward[n], reward[n], 0) 
        
        # unsure yet, if reward clipping is needed
        #maxreward = self.capacities[0] *50 *rescale
        #reward = np.clip(reward,-1000, maxreward)        

        return reward
    
    def reset(self, episode):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.current_episode = episode
        self.avg_action = 0
        self.sum_action = 0
        self.sum_demand = 0
        self.sum_rewards = 0
        self.avg_rewards = 0
        self.AllAktionen = deque(maxlen=500)
        
        if type(self.demand) == tuple:
            self.episodes_demand_timeseries = self.demand_model.generate(self.rounds_per_episode+1) #+1 for round zero
        
        if episode == 0: 
            self.last_action = np.zeros(self.agents)

        
        # Errors
        if len(self.capacities) != self.agents or len(self.costs) != self.agents or len(self.capacities) != len(self.costs):
            return print('******************************\n ERROR: length of CAP and costs has to correspond to the number of Agents \n******************************')

        
        return self._next_observation()
    
    def render(self, mode='demand', close=False):
        # Calls an output of several important parameters during the learning
        # This defines the content of the output
        #print(f'AllAktionen: {self.AllAktionen}')
        print('Episode',self.current_episode,'Step',self.current_step)
        print(f'Last Demand: {self.last_demand}')
        #print(f'Last Reward of this Episode: {self.last_rewards}')
        #print(f'last sold Qs:{self.sold_quantities}')
        #print(f'Last Market Price: {self.market_price}')
        #print(f'Average Reward: {self.avg_rewards}')
        print(f'Average Demand: {self.avg_demand}')
        print('Cumulative Demand',self.sum_demand)
        
        
        
        
