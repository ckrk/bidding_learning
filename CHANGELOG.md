# Change log

## v1.0.1 - Dez. 11, 2020

Adapted DDPG parameters according to the orginal paper:
"Continuous control with deep reinforcement learning" (Lillicrap et al., 2016)

**Added:**
- Changed hidden layer size from 256x256 to 400x300
- Actions are included as input of the critic network until the 2nd layer (=hidden layer); before they were included until the 1st
- Parameter tau got changed from 1e-2 to 1e-3
- Changed activation function for output layer of actor from leaky ReLU to tanh
- Included Fanin function to actor and critic networks, to initialize final weights (-3e-3, 3e-3) and bias (-3e-4, 3e-4) from a uniform distribution
- Added possible normalization methods for the actor and critic network ('none','BN','LN')
	- 'BN' = Batch Normalization, gets added to each layer and input of the actor and to all layers of the critic befor actions gets included
	- 'LN' = Layer Normlization, gets added to all layers of the actor and critic
	- 'none' = No normlaization methode gets performed


## v1.0.0 - Mar. 2020

**Previous Setup:**
- Hidden Layer size 256X256
- tau = 1e-2
- No normalization methods for actor and critic
- Actions were included until the first layer of the critic
- Leaky ReLU used as activation function for output layer of actor
- No final weigths initialized for final weights of actor and critic networks
- No availability for cuda