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

from pathlib import Path

# create environment
env = gym.make("rangl:nztc-open-loop-v0")

# Generate a random action and check it has the right length
# [increment in offshore wind capacity GW, increment in blue hydrogen energy TWh, increment in green hydrogen energy TWh]
action = env.action_space.sample()
assert len(action) == 3

# Check the to_observation method
assert len(env.observation_space.sample()) == len(env.state.to_observation())

# Create agent-ddpg
agent = agent_ddpg(env, hidden_size=[400, 300], actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-3, max_memory_size=50000, norm = 'none')

# Batch size, gives the size of the sample that is srawn for updating the agent
BATCH_SIZE = 20

# Reset the environment
env.reset()
done = False

# Training
for step in range(100):
    env.reset()
    while not done:
        
        action = agent.get_action(np.asarray([env.state.to_observation()[0]]))
        
        # Specify the action. Check the effect of any fixed policy by specifying the action here:
        observation, reward, done, _ = env.step(action)
        
        agent.memory.push(tuple(map(operator.sub, observation, (1,))), action, reward, observation, done)
        
        if len(agent.memory) > BATCH_SIZE:
            agent.update(BATCH_SIZE)
        
        
# Testing/Evaluation
# Reset the environment
env.reset()
done = False

while not done:
    
    action = agent.get_action(np.asarray([env.state.to_observation()[0]]))
    
    # Specify the action. Check the effect of any fixed policy by specifying the action here:
    observation, reward, done, _ = env.step(action)
    
    agent.memory.push(tuple(map(operator.sub, observation, (1,))), action, reward, observation, done)
    
    if len(agent.memory) > BATCH_SIZE:
        agent.update(BATCH_SIZE)


# Plot the episode
# Ploting works only if the environment wasn`t reseted before plottting !!! (So better plot only after a Test/Evaluation run)
env.plot("fixed_policy_DirectDeployment.png")

#assert Path("fixed_policy_DirectDeployment.png").is_file()


