import os, sys
path = os.path.dirname(os.path.realpath('__file__'))
path_rangle_challenge = os.path.dirname(path)
path_parent_folder = os.path.dirname(os.path.dirname(path))
os.chdir(path)
sys.path.append(path_rangle_challenge)
sys.path.append(os.path.join(path_parent_folder,'netzerotc'))

import operator 
from comet_ml import Experiment
import datetime

# pandas as pd
import numpy as np
import gym
import torch

# needed to create environment with gym.make
import rangl 

from src.agent_ddpg import agent_ddpg
from src.noise_models import UniformNoise, OUNoise, GaussianNoise
import matplotlib.pyplot as plt


#from pathlib import Path

# create environment
env = gym.make("rangl:nztc-open-loop-v0")


# Hyper Parameters for DDPG
BATCH_SIZE = 128
ACTOR_LR =1e-4
CRITIC_LR = 1e-3
GAMMA = 0.99

# Hyper parameters for the noise 
REGULATION_COEFFICENT = 1 # only moves the variance (if =1: sigma stays the same)
DECAY_RATE = 1 # basically no decay rate gets apllied
NOISE_VARIANCE = 4

# Random Guassian Noise gets added to the actions for exploratation
noise = GaussianNoise(env.action_space, mu=0, sigma=NOISE_VARIANCE, regulation_coef=REGULATION_COEFFICENT, decay_rate=DECAY_RATE)
# Noise adjustement for "1 Round Game"
#noise.action_dim = noise.action_dim * 20
#noise.low = np.asarray(noise.low.tolist()*20)
#noise.high = np.asarray(noise.high.tolist()*20)

# Reset the environment
env.reset()
done = False

# Create agent-ddpg
"increase max memory size ?"
agent = agent_ddpg(env, hidden_size=[400, 300], actor_learning_rate=ACTOR_LR, critic_learning_rate=CRITIC_LR, gamma=GAMMA, tau=1e-3, max_memory_size=50000, norm = 'none')

#Create individual state
ind_state = (1,)

# comet logging
experiment = Experiment(project_name="rangl-challenge-2022",
                        api_key="4fWyWzYNLrJ4X4md1JWg8TBWw")
experiment.log_parameter('Batch Size', BATCH_SIZE)
experiment.log_parameter('Noise Variance', NOISE_VARIANCE)
experiment.log_parameter('Learning Rate (Actor)', ACTOR_LR)
experiment.log_parameter('Learning Rate (Critic)',CRITIC_LR)
experiment.log_parameter('Gamma', GAMMA)

# saving outcome as list
all_rewards=[]
all_actions=[]
# Training
for step in range(10):
    
    cum_reward = 0
    actions_per_episode = []

    env.reset()
    done = False 
    
    print("step:",step)
    
    
    while not done:
        action = agent.get_action_rangl(env.state.to_observation())
        action = noise.get_action(action)
        # Specify the action. Check the effect of any fixed policy by specifying the action here:
        observation, reward, done, _ = env.step(action)

        cum_reward += reward
        all_rewards.append(reward)
        all_actions.append(action)
        actions_per_episode = np.append(actions_per_episode, action)


    #Saves the played round as tuple in the memory
    agent.memory.push(ind_state, actions_per_episode, cum_reward/20, ind_state, done)
    
    experiment.log_metric('Reward (Testing-Phase)', cum_reward)
    
    if len(agent.memory) > BATCH_SIZE:
        agent.update(BATCH_SIZE)



"""
# plotting reward progress
#moving median of n_th step (takes median from n rows and outputs an array of the same length as the input)
med_rewards=[]
n=19
for i in range(1000*20):
    temp_rewards =np.mean(all_rewards[i:n], axis=0)
    med_rewards.append(temp_rewards)
    n += 1

recompiled_rewards = np.asarray(med_rewards)
plt.plot(all_rewards)
plt.plot(recompiled_rewards)
"""     
# Testing/Evaluation
# Reset the environment
#env.seed(111)
env.reset()
done = False


cum_rewards=[]
for i in range(10):
    env.reset()
    done = False
    rewards_list=[]

    print(i)
    while not done:
        
        action = agent.get_action_rangl(env.state.to_observation())
        # Specify the action. Check the effect of any fixed policy by specifying the action here:
        observation, reward, done, _ = env.step(action)
        rewards_list.append(reward)

        
    cum_rewards.append(sum(rewards_list))
    
    
sum(cum_rewards)/1000

torch.save(agent.actor.state_dict(), 'trained_agent.pt')

experiment.log_metric('Reward (Testing-Phase)', sum(cum_rewards))
# Plot the episode
# Ploting works only if the environment wasn`t reseted before plottting !!! (So better plot only after a Test/Evaluation run)
env.plot("fixed_policy_DirectDeployment_avgReward_1Action_Noise4_gamma99_.png")
experiment.log_figure(figure_name='Environment Output', figure=env.plot("fixed_policy_DirectDeployment.png"))

time_stamp = datetime.datetime.now()
meta_data_time = time_stamp.strftime('%d-%m-%y %H:%M')
torch.save(agent.actor.state_dict(), 'Actor'+meta_data_time+'.pt')
torch.save(agent.critic.state_dict(), 'Critic'+meta_data_time+'.pt')
experiment.log_other('NN Filenames:','Actor'+meta_data_time+'.pt,'+'Critic'+meta_data_time+'.pt')

experiment.end()
