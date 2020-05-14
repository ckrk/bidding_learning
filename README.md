# Bidding-Learning
Implementations of the Deep Q-Learning Algorithms for Auctions

## Requirements

- PyTorch
- Gym
- numpy-groupies (relatively non-standard package that allows to do things similar to pandas groupby in numpy)

- it is recommended to use a package manager (PIP/Conda)
- Windows users might consider using Anaconda Navigator for package managment

## How to run?

- Clone to local repository
- run test_main.py in standard settings (or with appropriate parameters)

### How to customize a run of the algorithm?

#### Environment Parameters

The following parameters can be defined by the user by specifying them as inputs to the Environment in BiddingMarket_energy_Environment.py. This is usually done via test_main.py but can be done directly.

BiddingMarket_energy_Environment(CAP = capacitys, costs = costs, Demand =[5,6], Agents = 1,                                       Fringe = 1, Rewards = 0, Split = 0, past_action= 1, lr_actor = 1e-4, lr_critic = 1e-3, Discrete = 0)

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

The output mode is hardcoded in the function render belonging to BiddingMarket_energy_Environment

#### Test Parameters

The noise model and its variants is hard-coded in test_main.py.
There is:
- OU-Noise
- Gaussian Noise (Standard): sigma defines variance

## Dependency Structure:

  - test_main.py                                                            (High-level interface thaht accepts user input)
      - DDPG_main.py                                                            (Learning Algorithm,        3rd Party Code)
          - model_main.py                                      (Provides Neural Networks, Actor and Critic, 3rd Party Code)
      - BiddingMarket_energy_Environment.py   (Energy Market Envrionment, receives bids from agents and determines outcome)
          - market_clearing.py                                         (Numpy based,Clears bids and demand, outputs reward)
          - other.csv, simple_fringe.csv                                         (Fixed Bidding Patterns for fringe player)
      - utils_main.py                                              (Provides Noise Models, Learning Memory, 3rd Party Code)
      - utils_2split.py                                (Provides bid-reshaping, if capacity can be split into several bids)
