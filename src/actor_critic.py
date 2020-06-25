import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable

#torch.manual_seed(100)

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
        
        x = torch.cat([state, action], 1)   
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4, discrete = [0, 10, 0]):
        super(Actor, self).__init__()
        self.discrete = discrete 
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        ## Output Layer Activation Functions for Continuous Tasks
        
        if self.discrete[2] == 0:
            #x = F.leaky_relu(self.linear3(x), 0.1) # relu with small negative slope#
            #x = F.relu(self.linear3(x)) # without negative values
            #x = torch.sigmoid(self.linear3(x))
            #x = torch.tanh(self.linear3(x)) # from -1 to 1 (eventually as alternative to rescaling)
            x = F.hardtanh(self.linear3(x),min_val=10., max_val=40.) # from -1 to 1 (eventually as alternative to rescaling)

        ## Output Layer Activation Functions for Discrete Tasks
        else:
            x = F.gumbel_softmax(self.linear3(x), tau=1, hard=False, dim=1) # from 0 to 1, sums up to 1
            #x = F.softmax(self.linear3(x), dim=1) # from 0 to 1, sums up to 1 

        
        return x