import os, sys
path = os.path.dirname(os.path.realpath('__file__'))
path_rangle_challenge = os.path.dirname(path)
path_parent_folder = os.path.dirname(os.path.dirname(path))
os.chdir(path)
sys.path.append(path_rangle_challenge)
sys.path.append(os.path.join(path_parent_folder,'netzerotc'))

import operator 

#import logging

import pandas as pd
import numpy as np
import gym

# needed to create environment with gym.make
import rangl 

from src.agent_ddpg import agent_ddpg
from src.noise_models import UniformNoise, OUNoise, GaussianNoise 

from pathlib import Path

# create environment
env = gym.make("rangl:nztc-open-loop-v0")

# Generate a random action and check it has the right length
# [increment in offshore wind capacity GW, increment in blue hydrogen energy TWh, increment in green hydrogen energy TWh]
action = env.action_space.sample()
assert len(action) == 3

# Check the to_observation method
assert len(env.observation_space.sample()) == len(env.state.to_observation())

# Batch size, gives the size of the sample that is srawn for updating the agent
BATCH_SIZE = 120

# Random Guassian Noise gets added to the actions for exploratation 
REGULATION_COEFFICENT = 1 # only moves the variance (if =1: sigma stays the same)
DECAY_RATE = 1 # basically no decay rate gets apllied
noise = GaussianNoise(env.action_space, mu=0, sigma=1, regulation_coef=REGULATION_COEFFICENT, decay_rate=DECAY_RATE)

# Reset the environment
env.reset()
done = False

# Create agent-ddpg
"increase max memory size ?"
agent = agent_ddpg(env, hidden_size=[400, 300], actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-3, max_memory_size=50000, norm = 'none')


# Training
for step in range(1000):
    env.reset()
    done = False 
    
    print("step:",step)
    
    while not done:
        
        action = agent.get_action(np.asarray([env.state.to_observation()[0]]))
        #action = agent.get_action(env.state.to_observation())
        action = noise.get_action(action)
        print(action)
        
        # Specify the action. Check the effect of any fixed policy by specifying the action here:
        observation, reward, done, _ = env.step(action)
        
        #Saves the played round as tuple in the memory
        agent.memory.push(tuple(map(operator.sub, observation, (1,))), action, reward, observation, done)
        
        if len(agent.memory) > BATCH_SIZE:
            agent.update(BATCH_SIZE)
    
    #if len(agent.memory) > BATCH_SIZE:
        #agent.update(BATCH_SIZE)
        
        
# Testing/Evaluation
# Reset the environment
env.reset()
done = False

while not done:
    
    action = agent.get_action(np.asarray([env.state.to_observation()[0]]))
    
    # Specify the action. Check the effect of any fixed policy by specifying the action here:
    observation, reward, done, _ = env.step(action)
    
    agent.memory.push(tuple(map(operator.sub, observation, (1,))), action, reward, observation, done)
    
   #if len(agent.memory) > BATCH_SIZE: # necessary??
        #agent.update(BATCH_SIZE)


# Plot the episode
# Ploting works only if the environment wasn`t reseted before plottting !!! (So better plot only after a Test/Evaluation run)
env.plot("fixed_policy_DirectDeployment.png")

#assert Path("fixed_policy_DirectDeployment.png").is_file()


