import os, sys
path = os.path.dirname(os.path.realpath('__file__'))
os.chdir(path)
sys.path.append(os.path.dirname(path))

import numpy as np
import matplotlib.pyplot as plt
from PyPDF2 import PdfFileMerger

import pickle
import datetime
import time

from src.utils import UniformNoise, OUNoise, GaussianNoise, plot_run_outcome
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
# if plots should be saved
pdfs = []
# save date/time
time_stamp = datetime.datetime.now()
meta_data_time = time_stamp.strftime('%d-%m-%y %H:%M')

# Agent Parameters
POWER_CAPACITIES = [50 / 100, 50 / 100]  # 50
PRODUCTION_COSTS = [20 / 100, 20 / 100]  # 20
DEMAND = [70 / 100, 70 / 100]  # 70
ACTION_LIMITS = [-1, 1]  # [-10/100,100/100]#[-100/100,100/100]
NUMBER_OF_AGENTS = 2
PAST_ACTION = 0
FRINGE = 1

# Neural Network Parameters
# rescaling the rewards to avoid hard weight Updates of the Criticer 
REWARD_SCALING = 1  # 0.01 #
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3
NORMALIZATION_METHOD = 'none'  # options are BN = Batch Normalization, LN = Layer Normalization, none

# Noise Parameters
NOISE = 'GaussianNoise'  # Options are: 'GaussianNoise',OUNoise','UniformNoise'
DECAY_RATE = 0.001  # 0.0004 strong; 0.0008 medium; 0.001 soft; # if 0: Not used, if:1: only simple Noise without decay used
REGULATION_COEFFICENT = 10  # if 1: Not used, if:0: only simple Noise used

TOTAL_TEST_RUNS = 1  # How many runs should be executed
EPISODES_PER_TEST_RUN = 5000  # How many episodes should one run contain
ROUNDS_PER_EPISODE = 1  # How many rounds are allowed per episode (right now number of rounds has no impact -due 'done' is executed if step >= round- and choosing 1 is easier to interpret; )
BATCH_SIZE = 128  # *0.5 # *2

# Dictionary to save data and Parameter settings
Results = {}
Results['meta-data'] = {
        'date_run':meta_data_time,
        'power_capacities':POWER_CAPACITIES,
        'production_costs':PRODUCTION_COSTS,
        'demand':DEMAND,
        'noise': NOISE,
        'regulation_coef':REGULATION_COEFFICENT,
        'decay_rate':DECAY_RATE,
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
        'fringe:player': FRINGE}

for test_run in  range(TOTAL_TEST_RUNS):
    print('Test Run: {}'.format(test_run))
    
    # save runtime
    Results[test_run] = {'runtime':0}
    t_0 = time.time()
    
    # set up environment
    env = EnvironmentBidMarket(capacities=POWER_CAPACITIES, costs=PRODUCTION_COSTS, demand=DEMAND, agents=NUMBER_OF_AGENTS,
                               fringe_player=FRINGE, past_action=PAST_ACTION, lr_actor=LEARNING_RATE_ACTOR, lr_critic=LEARNING_RATE_CRITIC,
                               normalization=NORMALIZATION_METHOD, reward_scaling=REWARD_SCALING, action_limits=ACTION_LIMITS, rounds_per_episode=ROUNDS_PER_EPISODE)
    # set up agents
    agents = env.create_agents(env)
    
    # set up noise
    if NOISE == 'GaussianNoise':
        noise = GaussianNoise(env.action_space, mu=0, sigma=0.1, regulation_coef=REGULATION_COEFFICENT, decay_rate=DECAY_RATE)  # Gaussian Noise (only instead of Ornstein Uhlenbeck Noise)
    elif NOISE == 'OUNoise':
        noise = OUNoise(env.action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000)  # Ornstein Uhlenbeck Noise ( only instead of Gaussian Noise)
    elif NOISE == 'UniformNoise':
        noise = UniformNoise(env.action_space, initial_exploration=0.99, final_exploration=0.05, decay_rate=0.999)
    
    for episode in range(EPISODES_PER_TEST_RUN):
        Results[test_run][episode] = {'rewards':[], 'actions':[], 'market_price':[] , 'sold_quantities':[],
                                      'round':[], 'state':[], 'new_state':[]}
        # reset noise and state (past_actions resets only at the bginning of a new run)
        state = env.reset(episode)
        noise.reset()  # only important for OUNoise
        
        for step in range(ROUNDS_PER_EPISODE):
            actions = []
            for n in range(len(agents)):
                action_temp = agents[n].get_action(state)
                action_temp = noise.get_action(action_temp, episode)
                actions.append(action_temp[:])
            
            actions = np.asarray(actions)
            
            # get reward an new state from environment
            new_state, reward, done, _ = env.step(actions)   
            
            # save data in memory
            for n in range(len(agents)):
                agents[n].memory.push(state, actions[n], np.array([reward[n]]), new_state, done)
            
            # update
            if len(agents[0].memory) > BATCH_SIZE:
                for n in range(len(agents)):
                    agents[n].update(BATCH_SIZE)
                
            
            # save data in dictionary
            Results[test_run][episode]['rewards'].append(reward)
            Results[test_run][episode]['actions'].append(actions)
            Results[test_run][episode]['state'].append(state)
            Results[test_run][episode]['new_state'].append(new_state)
            Results[test_run][episode]['sold_quantities'].append(env.sold_quantities)
            Results[test_run][episode]['market_price'].append(env.market_price)
            
            # new_state becomes state
            state = new_state
            
            if done:
                # use some rendering to get insights during running (turned off to save time). 
                # can be adjusted directly in the environment
                # sys.stdout.write("***TestRound: {}, episode: {}, reward: {}, average _reward: {} \n".format(test_run, episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
                # env.render()
                break
        
    
    # save running tim in dictionary
    t_end = time.time()
    time_total = t_end - t_0
    Results[test_run]['runtime'] = time_total
    
    # Plot
    # takes as input the data from the saved dictionary, the number of agnets, the maximum action limit, a threshold for an Nash Equilibrium (if: 'none', no threshold gets displayed)
    # the total number of episodes per run, the actual run, which curves shoul be plotted (options: 'actions', rewards' or 'both'),
    # a title for the plot, rescale parameters if needed (usage: rescale[param for actions, param for rewards, param for bid limit])
    # and a window "moving_window" for which a moving median gets computed (recommended for presntation reasons) 
    plot_run_outcome(Results, NUMBER_OF_AGENTS, ACTION_LIMITS[1], 52,
                       EPISODES_PER_TEST_RUN, test_run, curves='actions',
                       title='Norm:{} Agents:{} PA: {} Fringe: {}, Run:{}'.format(NORMALIZATION_METHOD, NUMBER_OF_AGENTS, PAST_ACTION, FRINGE, test_run),
                       rescale=[100, 1000, 100], moving_window=9)
    
    # save plots (uncommment ALL below)
    # plt.savefig('temp{}.pdf'.format(test_run))
    # plt.close()
    # pdfs.append('temp{}.pdf'.format(test_run))

'''
### Merge PDFs and pickle
with open('{}_results_lra{}_lrc{}_Agents{}_PA{}_00.pkl'.format(NORMALIZATION_METHOD,NUMBER_OF_AGENTS,LEARNING_RATE_ACTOR,LEARNING_RATE_CRITIC, PAST_ACTION), 'wb') as pickle_file:
    pickle.dump(Results, pickle_file)


merger = PdfFileMerger()

for pdf in pdfs:
    merger.append(pdf)

merger.write("{}_plots_lra{}_lrc{}_Agents{}_PA{}_00.pdf".format(NORMALIZATION_METHOD,NUMBER_OF_AGENTS,LEARNING_RATE_ACTOR,LEARNING_RATE_CRITIC, PAST_ACTION))
merger.close()

for file in sample(pdfs):
    os.remove(pdfs[file])
'''
