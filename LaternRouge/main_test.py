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

#Hyper parameter to set up the agent (necessary even if only the stored output vector of the activation function according to the actor network gets executed)
TEST_STEPS = 1000
BATCH_SIZE = 256
ACTOR_LR =1e-4
CRITIC_LR = 1e-3
GAMMA = 0

# create environment
env = gym.make("rangl:nztc-open-loop-v0")

# Create agent-ddpg
agent = agent_ddpg(env, hidden_size=[400, 300], actor_learning_rate=ACTOR_LR, critic_learning_rate=CRITIC_LR, gamma=GAMMA, tau=1e-3, max_memory_size=50000, norm = 'none')
agent.actor.load_state_dict(torch.load('results\\trained_agent_v0.pt'))

env.reset()
done = False


cum_rewards=[]
for i in range(TEST_STEPS):
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


#env.plot()