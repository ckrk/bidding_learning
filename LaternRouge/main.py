import os, sys
path = os.path.dirname(os.path.realpath('__file__'))
path_rangle_challenge = os.path.dirname(path)
path_parent_folder = os.path.dirname(os.path.dirname(path))
os.chdir(path)
sys.path.append(path_rangle_challenge)
sys.path.append(os.path.join(path_parent_folder,'netzerotc'))


import numpy as np
import gym
import torch

from agent_ddpg import agent_ddpg
from utils import GaussianNoise


# create environment
env = gym.make("rangl:nztc-open-loop-v0")


# Hyper Parameters for DDPG
TRAINING_STEPS = 1000
BATCH_SIZE = 256
ACTOR_LR =1e-4
CRITIC_LR = 1e-3
GAMMA = 0

# Hyper parameters for the noise 
REGULATION_COEFFICENT = 1 # only moves the variance (if =1: sigma stays the same)
DECAY_RATE = 1 # basically no decay rate gets apllied
NOISE_VARIANCE = 4

# Random Guassian Noise gets added to the actions for exploratation
noise = GaussianNoise(env.action_space, mu=0, sigma=NOISE_VARIANCE, regulation_coef=REGULATION_COEFFICENT, decay_rate=DECAY_RATE)

# Reset the environment
env.reset()
done = False

# Create agent-ddpg
agent = agent_ddpg(env, hidden_size=[400, 300], actor_learning_rate=ACTOR_LR, critic_learning_rate=CRITIC_LR, gamma=GAMMA, tau=1e-3, max_memory_size=50000, norm = 'none')

#Create individual state in order to train the agent based on a single action step for all 20 time-steps
ind_state = (1,)


# Training
for step in range(TRAINING_STEPS):
    
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
        actions_per_episode = np.append(actions_per_episode, action)


    #Saves the played round as tuple in the memory
    agent.memory.push(ind_state, actions_per_episode, cum_reward/20, ind_state, done)
    
    
    if len(agent.memory) > BATCH_SIZE:
        agent.update(BATCH_SIZE)

#env.plot()

# uncomment to save the trained actor
# torch.save(agent.actor.state_dict(), 'trained_agent.pt')