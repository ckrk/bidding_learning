import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from src.actor_criticTEST import Actor, Critic
from src.utils import Memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
class agent_ddpg:
    def __init__(self, env, hidden_size=[400, 300], actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-3, max_memory_size=50000, discrete = [0, 10, 0], norm = 'none'):
        
        #BiddingMarket_energy_Environment Params
        self.discrete = discrete
        self.num_states = env.observation_space.shape[0]
        #self.num_actions = env.action_space.shape[0]
        self.num_actions = env.action_space.shape[0]

        self.gamma = gamma
        self.tau = tau
        self.norm = norm
        
        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions, discrete = self.discrete, norm = self.norm).to(device)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions, norm = self.norm).to(device)
        self.critic = Critic(self.num_states, hidden_size, self.num_actions, self.num_actions, norm = self.norm).to(device)
        self.critic_target = Critic(self.num_states, hidden_size, self.num_actions, self.num_actions, norm = self.norm).to(device)

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
        state = Variable(torch.from_numpy(state).float().unsqueeze(0).to(device))
        self.actor.eval() #new
        action = self.actor.forward(state)
        self.actor.train() #new
        #action = action.detach().numpy()[0,:]  
        action = action.detach().cpu().numpy()[0,:]  
        return action
    
    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.cuda.FloatTensor(states, device=device)
        actions = torch.cuda.FloatTensor(actions, device=device)
        rewards = torch.cuda.FloatTensor(rewards, device=device)
        next_states = torch.cuda.FloatTensor(next_states, device=device)
        
        #states = torch.FloatTensor(states, device=device)
        #actions = torch.FloatTensor(actions, device=device)
        #rewards = torch.FloatTensor(rewards, device=device)
        #next_states = torch.FloatTensor(next_states, device=device)
        
        #states = torch.from_numpy(states).float().to(device)
        #actions = torch.from_numpy(actions).float().to(device)
        #rewards = torch.from_numpy(rewards).float().to(device)
        #next_states = torch.from_numpy(next_states).float().to(device)
        
        
        # Critic loss       
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime) 

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            
            