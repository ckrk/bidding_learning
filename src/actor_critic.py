import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable

import numpy as np


WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


def fan_in_uniform_init(tensor, fan_in=None):
    """
    Utility function for initializing weigths for actor and critic
    """
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, action_dim=1, norm = 'none'):
        super(Critic, self).__init__()
        
        self.norm = norm
        self.action_dim = action_dim

        #Standard Settings
        self.linear1 = nn.Linear(input_size, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0]+self.action_dim, hidden_size[1]) 
        self.linear3 = nn.Linear(hidden_size[1], output_size)
        
        # Layer Normalization
        if self.norm == 'LN':
            self.ln1 = nn.LayerNorm(hidden_size[0])
            self.ln2 = nn.LayerNorm(hidden_size[1])
         # Batch Normalization
        if self.norm == 'BN':
            self.norm0 = nn.BatchNorm1d(input_size)
            self.bn1 = nn.BatchNorm1d(hidden_size[0])
            self.ReLU = nn.ReLU()
        
        # Weight Init
        #'''
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)
        
        # Final weigths distribution is predfeined
        nn.init.uniform_(self.linear3.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.linear3.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)
        #'''


    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        #x = torch.cat([state, action], 1) # not as suggested by the original paper
        x = state # as suggested by the original paper
        
        if self.norm == 'LN': 
            # Layer Normalization
            # Layer 1
            x = self.linear1(x)
            x = self.ln1(x)
            x = F.relu(x)

            # Layer 2
            x = torch.cat((x, action), 1) # as suggested by the original paper
            x = self.linear2(x)
            x = self.ln2(x)  
            x = F.relu(x)
          
        if self.norm == 'BN':
            # Batch Normalization
            # Layer 1
            x = F.relu(self.linear1(x))
            
            # Layer 2
            x = self.bn1(x) # as suggested by the original paper 
            x = torch.cat((x, action), 1) # as suggested by the original paper
            x = F.relu(self.linear2(x))
        
        if self.norm == 'none': 
            #Standard Settings
            # Layer 1
            x = F.relu(self.linear1(x))
            
            #Layer 2
            x = torch.cat((x, action), 1) 
            x = F.relu(self.linear2(x))
        
        # Layer 3
        x = self.linear3(x) 

        return x

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, high_action_limit, learning_rate = 3e-4, norm = 'none'):
        super(Actor, self).__init__()
        
        self.norm = norm
        self.high_action_limit=high_action_limit
        
        #Standard Settings
        self.linear1 = nn.Linear(input_size, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], output_size)
        
        # Layer Normalization
        if self.norm == 'LN':
            self.ln1 = nn.LayerNorm(hidden_size[0])
            self.ln2 = nn.LayerNorm(hidden_size[1])
            
        # Batch Normalization   
        if self.norm == 'BN':
            self.norm0 = nn.BatchNorm1d(input_size) 
            self.bn1 = nn.BatchNorm1d(hidden_size[0]) 
            self.bn2 = nn.BatchNorm1d(hidden_size[1]) 

        
        #'''
        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)
        
        # Final weigths distribution is predfeined
        nn.init.uniform_(self.linear3.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.linear3.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)
        #'''

        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = state
        
        if self.norm == 'LN':
            # Layer Normalization
            # Layer 1
            x = self.linear1(x)
            x = self.ln1(x) 
            x = F.relu(x)

            # Layer 2
            x = self.linear2(x)
            x = self.ln2(x) 
            x = F.relu(x)
        
        if self.norm == 'BN':
            # Batch Normalization
            # Layer 1
            x = self.norm0(x)
            x = F.relu(self.linear1(x))
            
            # Layer 2
            x = self.bn1(x)
            x = F.relu(self.linear2(x))
            
            x = self.bn2(x)
         
        if self.norm == 'none':
            #Standard Settings
            # Layer 1
            x = F.relu(self.linear1(x))
            # Layer 2
            x = F.relu(self.linear2(x))

        # Layer 3
        ## Output Layer Activation Functions for Continuous Tasks
        #x = F.leaky_relu(self.linear3(x), 0.1) # relu with small negative slope#
        x = torch.tanh(self.linear3(x)) # from -1 to 1 (eventually as alternative to rescaling)
        
        # Apply Rescaling for tanh hyperbolicus to output domain
        x = ((1+ x)*torch.from_numpy(self.high_action_limit))/2
        
        return x