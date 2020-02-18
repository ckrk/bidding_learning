# -*- coding: utf-8 -*-
"""
Initializes global parameters
Calls all relevant subfunctions and runs the whole package
Following the tutorial https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
"""

import gym

from utils import mini_batch_train
from ddpg import DDPGAgent



# Start Test
#env = gym.make("Pendulum-v0") #Standard Gym Enviroment


#Call StockTradingEnv
import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions 
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
import matplotlib.pyplot as plt

env = gym.make('stocks-v0', frame_bound=(50, 100), window_size=10)



max_episodes = 100
max_steps = 500
batch_size = 32

gamma = 0.99
tau = 1e-2
buffer_maxlen = 100000
critic_lr = 1e-3
actor_lr = 1e-3

agent = DDPGAgent(env, gamma, tau, buffer_maxlen, critic_lr, actor_lr)
episode_rewards = mini_batch_train(env, agent, max_episodes, max_steps, batch_size)

plt.cla()
env.render_all()
plt.show()
