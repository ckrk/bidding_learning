import os, sys
path = os.path.dirname(os.path.realpath('__file__'))
path_rangle_challenge = os.path.dirname(path)
path_parent_folder = os.path.dirname(os.path.dirname(path))
os.chdir(path)
sys.path.append(path_rangle_challenge)
sys.path.append(os.path.join(path_parent_folder,'netzerotc'))

import operator 
from comet_ml import Experiment

#import pandas as pd
import numpy as np
import gym

# needed to create environment with gym.make
import rangl 

from src.agent_ddpg import agent_ddpg
from src.noise_models import UniformNoise, OUNoise, GaussianNoise 

#from pathlib import Path

# create environment
env = gym.make("rangl:nztc-open-loop-v0")


# Hyper parameters for DDPG
BATCH_SIZE = 128
ACTOR_LR = 1e-5
CRITIC_LR = 1e-3
GAMMA = 0.99

# Hyper paramters Noise
REGULATION_COEFFICENT = 1 # only moves the variance (if =1: sigma stays the same)
DECAY_RATE = 1 # basically no decay rate gets apllied
NOISE_VARIANCE=4

# Random Guassian Noise gets added to the actions for exploratation 
noise = GaussianNoise(env.action_space, mu=0, sigma=NOISE_VARIANCE, regulation_coef=REGULATION_COEFFICENT, decay_rate=DECAY_RATE)


# Reset the environment
env.reset()
done = False

# Create agent-ddpg
"increase max memory size ?"

agent = agent_ddpg(env, hidden_size=[400, 300], actor_learning_rate=ACTOR_LR, critic_learning_rate=CRITIC_LR, gamma=GAMMA, tau=1e-3, max_memory_size=50000, norm = 'none')

# comet logging
experiment = Experiment(project_name="rangl-challenge-2022",
                        api_key="4fWyWzYNLrJ4X4md1JWg8TBWw")
experiment.log_parameter('Batch Size', BATCH_SIZE)
experiment.log_parameter('Noise Variance', NOISE_VARIANCE)
experiment.log_parameter('Learning Rate (Actor)', ACTOR_LR)
experiment.log_parameter('Learning Rate (Critic)',CRITIC_LR)
experiment.log_parameter('Gamma', GAMMA)

# Training
for step in range(1000):
    env.reset()
    done = False 
    
    print("step:",step)
    
    while not done:
        
        action = agent.get_action(np.asarray([env.state.to_observation()[0]]))
        #action = agent.get_action(env.state.to_observation())
        action = noise.get_action(action)
        #print(action)
        
        # Specify the action. Check the effect of any fixed policy by specifying the action here:
        observation, reward, done, _ = env.step(action)
        experiment.log_metric('Reward (Training Phase)',reward,step=step)
        
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
    print(action)
    # Specify the action. Check the effect of any fixed policy by specifying the action here:
    observation, reward, done, _ = env.step(action)
    experiment.log_metric('Reward (Testing-Phase)',reward)
    
    #agent.memory.push(tuple(map(operator.sub, observation, (1,))), action, reward, observation, done)
    
   #if len(agent.memory) > BATCH_SIZE: # necessary??
        #agent.update(BATCH_SIZE)


# Plot the episode
# Ploting works only if the environment wasn`t reseted before plottting !!! (So better plot only after a Test/Evaluation run)
env.plot("fixed_policy_DirectDeployment.png")
experiment.log_figure(figure_name='Environment Output', figure=env.plot("fixed_policy_DirectDeployment.png"))

experiment.end()
