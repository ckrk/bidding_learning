# Bidding-Learning 
**Version 1.0.1** - [Change log](CHANGELOG.md)

Implementations of the Deep Q-Learning Algorithms for Auctions.

## What should it do?

- Generally, the algorithm sets up a reinforcement learning algorithm in an user defined environment,
that represents energy market auctions. The user defines number of players, their production costs & capacities as well as learning related hyper parameters of specifities of the bidding rules. Then the algorithm trains the players and outputs statistics about their behavior during training.

- Specifically, if run in the standard settings the algorithm learns in a simple predefined environment that is a minimal working example. It is chosen to be easy to understand and for comparability while debbuging.

## Pay attention that the algorithm involves randomness

This is bad for reproducibility but the algorithm intrinsically uses random steps.
Sometimes you will get strange results. Hence, try to rerun the algorithm 2-3 times in the same settings if you feel you get non-sensical values to be sure it repeatedly fails. Sometimes you just get unlucky.
We get strange results in a significant number of times, if the final graphic just outputs a flat line, try a simple restart.

## What is the Minimum Working Example? What do the standard settings implement?

The Minimal Working Example implements an easy market situation. Specifically:
- There is only a single learning player, while the remaining market is represented as non-learning "fringe player".
- The learning player always bids his whole capacity and is aware of the bids in the last round.
- The fringe player always submits the bids specified in a coresponding csv file.
- Please check "data/fringe-players/simple_fringe02.csv" to understand the behavior of the fringe player. It shows the bids that are submitted each round by the fringe player in our standard setting.
- Essentially, the first unit of energy is sold for free. Every extra unit of energy is sold for an extra 100 price.
- The demand is predefined to equal to 500.
- The strategic player has 0 costs and 1 unit capaciy.
- However, we rescale the parameters to 1/100, in order to scale the rewards into a range that exhibits good learning. This seems necessary, but we do not understand entirely why.
- The market price without the players participation is 400. If the player bids all capacity at 0, this reduces the price to 300. We would expect that the player can gain by becoming the price setting player and offering between 301-399.
- Tie breaking may be relevant. Currently the in case of tie the player with lower number gets everything. Proper tie breaking is involved to program.

## Requirements

- torch
- gym
- numpy-groupies (relatively non-standard package that allows to do things similar to pandas groupby in numpy)
- matplotlib

## How to run?

- Clone to local repository
- run ./src/main.py in standard settings (or with appropriate parameters)

- it is recommended to use a package manager for library installation (PIP/Conda)
- Windows users might consider using Anaconda Navigator for package managment

### How to customize a run of the algorithm?

#### Environment Parameters

The following parameters can be defined by the user by specifying them as inputs to the Environment in environment_bid_market.py. This is usually done via main.py but can be done directly.

EnvironmentBidMarket(CAP = capacitys, costs = costs, Demand =[5,6], Agents = 1,                                       Fringe = 1, Rewards = 0, Split = 0, past_action= 1, lr_actor = 1e-4, lr_critic = 1e-3, Discrete = 0)

- CAP: np.array [cap1,...,capN]             (requires elements matching number of agents) ... Generation capacity an agent can sell 
- costs: np.array [costs1,...,costsN]       (requires elements matching number of agents) ... Generation capacity an agent can sell 
- Demand: np.array [minDemand,maxDemand-1]  (2elements) ... Range of demand. Fixed demand D is written as [D,D+1]
- Agents: integer ... Number of learning players
- Fringe: binary  ... Strategic-Fringe on/off (i.e. a non-learning player submitting constant bids defined by a csv-file)
- Rewards: integer ... different reward functions, set 0 for (price-costs) * sold_quantity
- Split: binary ... Allow offering capacity at 2 different price on/off
- past_action: binary ... include the agents last actions as observations for learning on/off
- lr_actor: float < 1 .... learning rate actor network, weighs relevance of new and old experiences
- lr_critic: float < 1 .... learning rate critic network, weighs relevance of new and old experiences
- Discrete: binary ... discrete state space on/off (not ready yet)

The output mode is hardcoded in the function render belonging to EnvironmentBidMarket

#### Fringe Player

The fringe player reads his bids from a csv-file. The name of the file is hardcoded in the reset function from environment_bid_market.py. All csv's are stored in ./data/fringe-players.
Currently, we provide following standard test csv:
- simple_fringe02.csv (Standard choice, 100 price steps, quantity steps 100)
- simple_fringe03.csv (100 price steps, quantity steps 1)
- others.csv (non-trivial, Test Case by Christoph Graf, for comparision with optimization solver, 60 bids)
- simple_fringe01.csv (easy file, price_bids increase by 1000, quantity_bids increase by 1, 60 bids)

Attention, only csv with 60 bids are compatible!

#### Test Parameters

The noise model and its variants is hard-coded in main.py.
There is:
- OU-Noise
- Gaussian Noise (Standard): sigma defines variance

#### Network Architecture

The architecture of the actor and critic netowrks are hardcoded in actor_crtic.py

## Dependency Structure:

  - main.py                                                            (High-level interface thaht accepts user input)
      - agent_ddpg.py                                                            (Learning Algorithm,        3rd Party Code)
          - actor_critic.py                                      (Provides Neural Networks, Actor and Critic, 3rd Party Code)
      - environment_bid_market.py   (Energy Market Envrionment, receives bids from agents and determines outcome)
          - market_clearing.py                                         (Numpy based,Clears bids and demand, outputs reward)
          - other.csv, simple_fringe.csv                                         (Fixed Bidding Patterns for fringe player)
      - utils.py                                              (Provides Noise Models, Learning Memory, 3rd Party Code)
