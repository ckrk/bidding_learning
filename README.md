# Bidding-Learning 
**Version 1.0.4 Forked to rangL competition branch**
- Readout netzerotc from sister directory

**Version 1.0.3 Forked to Single Player branch**
- Incorporate Single Player
- Added and Normal Distributed Demand
- refactored methods from utils.py into demand_models.py and noise_model.py

**Version 1.0.2** - [Change log](CHANGELOG.md)

**Version 1.0.1** - [Change log](CHANGELOG.md)

Implementations of the Deep Q-Learning Algorithms for Auctions.


## What should it do?

- Run the Netzerotc rangl 2022 challenge

## Pay attention that the algorithm involves randomness

The algorithm is seeded and delivers reproducible results with the same seed.
Nontheless, the algorithm intrinsically uses randomness for exploration and different runs will differ randomly.
Sometimes you will get strange results. Hence, try to rerun the algorithm 2-3 times in varying settings if you feel you get non-sensical values to be sure it repeatedly fails. Sometimes you just get unlucky.

## How to run?

- Clone to local repository
- Clone netzerotc to local repository
- ./single-player-rangle2022 and ./netzerotc need to be in the same parent folder
- run ./single-player-rangle2022/bin/main.py in standard settings (or with appropriate parameters)

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
