import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

from actor_critic import Actor, Critic
from utils import Memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  
class agent_ddpg:
    def __init__(self, env, hidden_size=[400, 300], actor_learning_rate=1e-4, critic_learning_rate=5e-3, gamma=0.99, tau=1e-3, max_memory_size=50000, norm = 'none'):
        
        # Gym Environment
        
        self.env = env
        #BiddingMarket_energy_Environment Params
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0] *20 ############# dirty implemntation!!!
        
        self.high_action_limit = self.env.action_space.high
        self.high_action_limit = np.asarray(self.high_action_limit.tolist()*20, dtype=np.float32) ############# dirty implemntation!!!
        
        # DDPG specific Params
        self.gamma = gamma
        self.tau = tau
        self.norm = norm
        self.hidden_size= hidden_size
        self.output_size = 1 #only for critic
        
        # Networks
        self.actor = Actor(self.num_states, self.hidden_size, self.num_actions, self.high_action_limit, norm = self.norm).to(device)
        self.actor_target = Actor(self.num_states, self.hidden_size, self.num_actions, self.high_action_limit, norm = self.norm).to(device)
        self.critic = Critic(self.num_states, self.hidden_size, self.output_size, self.num_actions, norm = self.norm).to(device)
        self.critic_target = Critic(self.num_states, self.hidden_size, self.output_size, self.num_actions, norm = self.norm).to(device)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.memory = Memory(max_memory_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        

    def get_action(self, state):
        #state = np.asarray([state[0]])
        state = Variable(torch.from_numpy(state).float().unsqueeze(0).to(device))
        self.actor.eval()
        action = self.actor.forward(state)
        self.actor.train() 
        action = action.detach().numpy()[0,:]
        #action = action.detach().cpu().numpy()[0,:]  
        return action
    
    def get_action_rangl(self, state):
        
        ind_state = np.asarray([1])
        ind_state = Variable(torch.from_numpy(ind_state).float().unsqueeze(0).to(device))
        self.actor.eval()
        action = self.actor.forward(ind_state)
        self.actor.train() 
        action = action.detach().numpy()[0,:]
        #action = action.detach().cpu().numpy()[0,:]
        action = action[((state[0]+1)*3):((state[0]+1)*3+3)]
        return action
    
    def update(self, batch_size):
        
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states, device=device)
        actions = torch.FloatTensor(np.asarray(actions), device=device)
        rewards = torch.FloatTensor(rewards, device=device) #######################
        next_states = torch.FloatTensor(next_states, device=device)
        
        # reshape rewards
        rewards = rewards.view((len(rewards),1))
        
        # Critic loss       
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        
        # update networks
        self.critic_optimizer.zero_grad()
        # Critic loss
        critic_loss = self.critic_criterion(Qvals, Qprime)
        critic_loss.backward() 
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            
            