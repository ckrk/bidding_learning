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

## Dependency Structure:

  - test_main.py                                                            (High-level interface thaht accepts user input)
      - DDPG_main.py                                                            (Learning Algorithm,        3rd Party Code)
          - model_main.py                                      (Provides Neural Networks, Actor and Critic, 3rd Party Code)
      - BiddingMarket_energy_Environment.py   (Energy Market Envrionment, receives bids from agents and determines outcome)
          - market_clearing.py                                         (Numpy based,Clears bids and demand, outputs reward)
          - other.csv, simple_fringe.csv                                         (Fixed Bidding Patterns for fringe player)
      - utils_main.py                                              (Provides Noise Models, Learning Memory, 3rd Party Code)
      - utils_2split.py                                (Provides bid-reshaping, if capacity can be split into several bids)
