# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:44:09 2020

@author: Viktor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        
        x = torch.cat([state, action], 1)   ##### fixed dim ändern
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        #x = torch.tanh(self.linear3(x))
        #x = F.relu(self.linear3(x))
        #x = nn.LeakyReLU(self.linear3(x), negative_slope=0.1)# .negativ_slope nur für leakyReLU relevant
        x = F.leaky_relu(self.linear3(x), 0.1)
        #x = F.softmax(self.linear3(x), dim=0)
        
        return x