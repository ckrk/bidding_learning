import sys
#sys.path.append('./bin/')
import gym
import numpy as np
import matplotlib.pyplot as plt
from PyPDF2 import PdfFileMerger
import os
import pickle
import datetime
import time

from src.agent_ddpg import agent_ddpg 
from src.utils import  UniformNoise, OUNoise, Memory, GaussianNoise
from src.environment_bid_market import EnvironmentBidMarket

'''
High-Level Interface that calls learning algorithm and Energy-Market Environment 
subject to user-specified inputs

Environment Parameters

capacities:   np.array 1x(number of Agents)
costs:        np.array 1x(number of Agents)

Attention these np.arrays have to correspond to the number of Agents

Demand:       np.array 1x2
Chooses demand from arang between [min,max-1]
For fixed Demand, write: the preferred [Number, Number +1] (e.g. Demand = 99 -> [99,100])

Agents:       scalar
Number of learning agents

Rewards:      scalar
Type of Reward function. Usually 0.

Split: binary
Allow Split Bids

past_action: binary
Allow agents to learn from all agents past actions

lr_actor:    float
Learning Rate Actor

lr_critic:   float
Learning Rate Critic

Discrete:    binary
Enables Discrete Spaces (Not yet functional)
'''

pdfs =[]
time_stamp = datetime.datetime.now()
meta_data_time = time_stamp.strftime('%d-%m-%y %H:%M')

# Agent Parameters
POWER_CAPACITIES = [50/100,50/100] #50
PRODUCTION_COSTS = [20/100,20/100] #20
DEMAND = [70/100,70/100] #70
PRICE_CAP = 100/100
NUMBER_OF_AGENTS = 2
PAST_ACTION = 0

# Neural Network Parameters
ACTION_LIMITS = [-1,1] #[-10/100,100/100]#[-100/100,100/100]
REWARD_SCALING = 1 #0.01 #
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3
NORMALIZATION_METHOD = 'none' # options are BN = Batch Normalization, LN = Layer Normalization, none

# Noise Parameters
NOISE ='GaussianNOISE '  #'GaussianNOISE + UniformNoise'
DECAY_RATE = 0.001 #0.0004 strong; 0.0008 medium; 0.001 soft; # if 0: Not used, if:1: only simple Noise without decay used
REGULATION_COEFFICENT = 10 # if 1: Not used, if:0: only simple Noise used

TOTAL_TEST_RUNS = 1 # How many runs should be executed
EPISODES_PER_TEST_RUN = 10000 # How many episodes should one run contain
ROUNDS_PER_EPISODE = 500 # How many rounds are allowed per episode (by now, only 1 round is always played due 'done'-command)
BATCH_SIZE = 128 # *0.5 # *2


Results = {}

Results['meta-data'] = {
        'date_run':meta_data_time,
        'power_capacities':POWER_CAPACITIES,
        'production_costs':PRODUCTION_COSTS,
        'demand':DEMAND,
        'noise': NOISE,
        'regulation_coef':REGULATION_COEFFICENT,
        'decay_rate' :DECAY_RATE,
        'lr_critic':LEARNING_RATE_CRITIC,
        'lr_actor':LEARNING_RATE_ACTOR,
        'Normalization': NORMALIZATION_METHOD,
        'reward_scaling':REWARD_SCALING,
        'total_test_rounds':TOTAL_TEST_RUNS,
        'episodes_per_test_run':EPISODES_PER_TEST_RUN,
        'batches':BATCH_SIZE,
        'rounds':ROUNDS_PER_EPISODE,
        'agents':NUMBER_OF_AGENTS,
        'action_limits': ACTION_LIMITS,
        'past_action': PAST_ACTION,
        'price_cap':PRICE_CAP}


for test_run in  range(TOTAL_TEST_RUNS):
    
    print('Test Run: {}'.format(test_run))
    
    Results[test_run] ={'episode_results':[], 'runtime':0}
    t_0 = time.time()
    
    env = EnvironmentBidMarket(capacities = POWER_CAPACITIES, costs = PRODUCTION_COSTS, demand = DEMAND, agents = NUMBER_OF_AGENTS, 
                               fringe_player = 0, rewards = 0, split = 0, past_action= PAST_ACTION,
                               lr_actor = LEARNING_RATE_ACTOR, lr_critic = LEARNING_RATE_CRITIC, normalization = NORMALIZATION_METHOD, discrete = [0, 10,0], 
                               reward_scaling = REWARD_SCALING, action_limits = ACTION_LIMITS, price_cap = PRICE_CAP)
    
    agents = env.create_agents(env)
    noise = GaussianNoise(env.action_space, mu= 0, sigma = 0.1, regulation_coef= REGULATION_COEFFICENT, decay_rate = DECAY_RATE) # Gaussian Noise (only instead of Ornstein Uhlenbeck Noise)
    #noise = OUNoise(env.action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000) # Ornstein Uhlenbeck Noise ( only instead of Gaussian Noise)
    random_noise = UniformNoise(env.action_space, env.price_cap, initial_exploration = 0.99, final_exploration = 0.05, decay_rate = 0.999)


    rewards_temp = []
    avg_rewards = []
    bids_temp = []
    med_bids_temp =[]
    
    state = env.reset() # position important when using past action
    
    for episode in range(EPISODES_PER_TEST_RUN):
        Results[test_run][episode] = {'rewards':[], 'quantities':[], 'actions':[]}
        #state = env.reset()
        noise.reset()
        episode_reward = 0

        for step in range(ROUNDS_PER_EPISODE):
            actions = []
            for n in range(len(agents)):
                action_temp = agents[n].get_action(state)
                action_temp = noise.get_action(action_temp, episode)
                #action_temp = random_noise.get_action(action_temp, episode) # if adsitionaly a random noise is wanted
                actions.append(action_temp[:])
        
            actions = np.asarray(actions)
            new_state, reward, done, _ = env.step(actions)   
        
            for n in range(len(agents)):
                agents[n].memory.push(state, actions[n], np.array([reward[n]]), new_state, done)
       
        
            if len(agents[0].memory) > BATCH_SIZE:
                for n in range(len(agents)):
                    agents[n].update(BATCH_SIZE)
                
        
            state = new_state
            episode_reward = reward
            
            # Statistics !!! not working for split option(skip ploting actions/bids, than it will work)
            rewards_temp.append(episode_reward)
            bids_temp.append(actions.squeeze(1))
            rewards = np.asarray(rewards_temp)
            bids = np.asarray(bids_temp)
            

            if done:
                # use some rendering to get insights during running (turned off to save time)
                #sys.stdout.write("***TestRound: {}, episode: {}, reward: {}, average _reward: {} \n".format(test_run, episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
                #env.render()
                break

        med_bids_temp.append(np.median(bids[-10:], axis=0))
        avg_rewards.append(np.median(rewards[-10:], axis=0))
        
        sold_qunatities, market_price = env.variable_render()

        Results[test_run][episode]['rewards'] = episode_reward
        Results[test_run][episode]['avg_reward'] = np.array(episode_reward).mean(axis=0)
        Results[test_run][episode]['actions'] = actions
        Results[test_run][episode]['sold_quantities'] = sold_qunatities
        Results[test_run][episode]['market_price'] = market_price
        
    med_bids = np.asarray(med_bids_temp) *100
    avg_rewards = np.asarray(avg_rewards) *10 *100


    plt.plot([0]*EPISODES_PER_TEST_RUN, color='grey', label = 'Bid Limits', lw =1)
    plt.plot([100]*EPISODES_PER_TEST_RUN, color='grey', lw =1)
    plt.plot([52]*EPISODES_PER_TEST_RUN, color='tab:orange', label = 'Nash Equilibrium', lw =1)
    plt.plot(med_bids[1:,0], 'tab:red', label = 'Bids Agent1', lw =1, linestyle = '--')
    plt.plot(med_bids[1:,1], 'tab:blue', label = 'Bids Agent2', lw =1, linestyle = '--')
    #plt.plot(med_bids[1:,2], 'tab:green', label = 'Bids Agent3', lw =1, linestyle = '--')# 3rd agents
    #plt.plot(avg_rewards[1:,0], 'tab:red', label = 'Rewards Agent1', lw =1) # displaying rewards
    #plt.plot(avg_rewards[1:,1], 'tab:blue', label = 'Rewards Agent2', lw =1) # displaying rewards
    #plt.plot(avg_rewards[1:,2], 'tab:green', label = 'Rewards Agent3', lw =1) # 3rd agents


    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(loc=4, prop={'size': 7})
    plt.title('none lr4-3 woPast Action: Run {}'.format(test_run))
    plt.savefig('temp{}.pdf'.format(test_run))
    #plt.show() # if you want to see plots immediately 
    plt.close()
    
    pdfs.append('temp{}.pdf'.format(test_run))
    
    t_end = time.time()
    time_total = t_end - t_0
    Results[test_run]['runtime'] = time_total


### Merg PDFs and pickle
with open('none_results_lr4-3_woPA_00.pkl', 'wb') as pickle_file:
    pickle.dump(Results, pickle_file)


merger = PdfFileMerger()

for pdf in pdfs:
    merger.append(pdf)

merger.write("none_plots_lr4-3_woPA_00.pdf")
merger.close()