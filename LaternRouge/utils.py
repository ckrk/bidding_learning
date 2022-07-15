import numpy as np
import random

from collections import deque


class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)




class GaussianNoise(object):
    def __init__(self, action_space, mu = 0.0, sigma = 0.1, regulation_coef = 1, decay_rate = 0):
        
        self.action_dim      = action_space.shape[0]
        self.low             = action_space.low
        self.high            = action_space.high
            
        self.distance        = abs(self.low - self.high)
        
        self.decay_rate = decay_rate 
        self.regulation_coef = regulation_coef
        self.mu              = mu
        self.sigma           = sigma
        
        self.reset()
        
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    

    def get_action(self, action, step = 0):
         
        noise_list = np.random.normal(self.mu, self.sigma, self.action_dim)* ((1 - self.decay_rate)**step) * self.regulation_coef 
        
        if (((noise_list)**2)**0.5 < 0.01).any():
            noise_list = np.random.normal(0,0.01,self.action_dim) 
        
        noisy_action = np.clip(action + noise_list, self.low, self.high)

        return noisy_action 


    
        
