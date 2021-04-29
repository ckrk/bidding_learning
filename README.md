# Bidding-Learning 
**Version 1.0.2** - [Change log](CHANGELOG.md)

**Version 1.0.1** - [Change log](CHANGELOG.md)

Implementations of the Deep Q-Learning Algorithms for Auctions.


## What should it do?

- Generally, the algorithm sets up a reinforcement learning algorithm in an user defined environment,
that represents energy market auctions. The user defines number of players, their production costs & capacities as well as learning related hyper parameters of specifities of the bidding rules. Then the algorithm trains the players and outputs statistics about their behavior during training.

## Pay attention that the algorithm involves randomness

The algorithm is seeded and delivers reproducible results with the same seed.
Nontheless, the algorithm intrinsically uses randomness for exploration and different runs will differ randomly.
Sometimes you will get strange results. Hence, try to rerun the algorithm 2-3 times in varying settings if you feel you get non-sensical values to be sure it repeatedly fails. Sometimes you just get unlucky.

## How to run?

- Clone to local repository
- run ./bin/main.py in standard settings (or with appropriate parameters)

- it is recommended to use a package manager for library installation (PIP/Conda)
- Windows users might consider using Anaconda Navigator for package managment
- install the python packages listed in Requirements

## Requirements
- python 3.7
- pytorch=1.6.0
- gym=0.18.0
- numpy=1.19.1
- numpy-groupies=0.9.13 (relatively non-standard package that allows to do things similar to pandas groupby in numpy)
- seaborn=0.11.1 (plotting library based on matplotlib)

## Citing

If you use our algorithm in your work, please cite the accompanying [paper]:

```bibtex
@misc{graf2021computational,
      title={{Computational Performance of Deep Reinforcement Learning to find Nash Equilibria}}, 
      author={Christoph Graf and Viktor Zobernig and Johannes Schmidt and Claude Kl\"ockl},
      year={2021},
      eprint={2104.12895},
      archivePrefix={arXiv},
      primaryClass={cs.GT}
}
```

### How to customize a run of the algorithm?

#### Environment Parameters

The following parameters can be defined by the user by specifying them as inputs to the Environment in environment_bid_market.py. This is usually done via main.py but can be done directly.

EnvironmentBidMarket(capacities = capacities, costs = costs, demand =[5,6], agents = 1,                                       fringe = 1, past_action= 1, lr_actor = 1e-4, lr_critic = 1e-3, normalization = 'none', reward_scaling = 1, action_limits = [-1,1], rounds_per_episode = 1)

- capacities: np.array [cap1,...,capN]             (requires elements matching number of agents) ... Generation capacity an agent can sell 
- costs: np.array [costs1,...,costsN]       (requires elements matching number of agents) ... Generation capacity an agent can sell 
- demand: np.array [minDemand,maxDemand-1]  (2elements) ... Range of demand. Fixed demand D is written as [D,D+1]
- agents: integer ... Number of learning players
- fringe: binary  ... Strategic-Fringe on/off (i.e. a non-learning player submitting constant bids defined by a csv-file)
- past_action: binary ... include the agents last actions as observations for learning on/off
- lr_actor: float < 1 .... learning rate actor network, weighs relevance of new and old experiences
- lr_critic: float < 1 .... learning rate critic network, weighs relevance of new and old experiences
- normalization: defines a normalization method for the neural networks. Options are: 'BN'... batch normalization, 'LN'... layer normalization, 'none'... no normalization (default)
- reward_scaling: a parameter to rescale the reward. In some cases high rewards can lead to too strong update steps of Adam-backprobagation.
- action_limits: should alway be between [-1,1] (default) due the tanh-activation function of the actor network. If another action space is desired an action wrapper is suggested (but not provided yet).
- rounds_per_episode: Number of rounds per episode. Is only necessary to reset the environment when using past_actions, so that past_actions are only get reset at the start of a new test run. Default is 1.

The output mode is hardcoded in the function render belonging to EnvironmentBidMarket

#### Test Parameters

The noise model and its variants is hard-coded in main.py.
There is:
- OU-Noise
- Gaussian Noise (Standard): sigma defines variance
- Uniform Noise: follows an epsioln-greedy strategy

#### Network Architecture

The architecture of the actor and critic netowrks are hardcoded in actor_crtic.py

## Dependency Structure:

  - main.py                                                            (High-level interface thaht accepts user input)
      - agent_ddpg.py                                                            (Learning Algorithm,        3rd Party Code)
          - actor_critic.py                                      (Provides Neural Networks, Actor and Critic, 3rd Party Code)
      - environment_bid_market.py   (Energy Market Envrionment, receives bids from agents and determines outcome)
          - market_clearing.py                                         (Numpy based,Clears bids and demand, outputs reward)
          - fringe_player_data_00.csv                                         (Fixed Bidding Patterns for fringe player)
      - utils.py                                              (Provides Noise Models, Learning Memory, 3rd Party Code)
