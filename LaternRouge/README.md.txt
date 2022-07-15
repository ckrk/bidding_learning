## Submission LanterneRougeVienna-Boku-Ait

The submission was written mainly for competing in the open-loop. However, it was also submitted to closed-loop. 

The submission was joint work between Claude Klöckl and Viktor Zobernig. 
It is based on the following preprint: https://arxiv.org/pdf/2104.12895.pdf.

We are competing on behalf of the University of Life Sciences and Natural Resources (BOKU) and the Austrian Institute of Technology (AIT).

# 1) What mix of offshore wind, blue and green hydrogen that you found to be optimal?

The agent predominantly relies on building blue hydrogen.

Blue hydrogen is built over 4 time-steps, 
where the maximum possible amount is built.

In contrast, the agent entirely avoids the construction of wind energy and green hydro except for the last time step, where all three technologies are built to the fully allowed extent.

Overall this leads to a ratio of 1:1:4 for wind power, green hydrogen and blue hydrogen respectively.

    
# 2) Why your optimal pathway performs better than the standard IEV models Breeze, Gale and Storm? 

We believe, that our agents strategy is geared towards optimizing the reward function and not necessarily to achieve CO2 reduction. 

This leads to the agent acting "selfish" by maximizing his profits instead of curbing CO2 emissions. 

In howfar, this behavior is better than the proposed scenarios is arguable.

As a matter of fact, the agent almost entirely forfeits the goal of CO2 reduction. Nontheless, it scored comparably well. 

With the exception of the significantly better algorithm of AIM-MATE the algorithm' scores proved to be quite competetive and is quite close to (or even better than some) competing algorithms. 

Moreover, we tried during training, but did not succeed, to steer the algorithm to higher windpower construction by biased noise. 

This indicates, that the rewards of earnings and job creation where relatively big as compared to the cost of investment in green hydrogen, wind power and C02. 

Moreover, the algorithm exhibits a strategy to delay construction in order to build when investment costs are low. 

Apparently, the competition environment did encourage such a behavior, i.e. not pursuing the goal of CO2 abatement,  to some extent. 

We believe, that the relative success of our algorithm should not serve as a role model, but rather as a cautionary tale against two low CO2 prices, otherwise they may not be sufficient to incentivice C02 abatement.
    
# 3) In what ways your approach to constructing an agent improves upon naive RL training?

The algorithm is a standard DDPG design with a twist. 

A main motivation for DDPG, was that it copes well with continous action-spaces such as the one of this challenge.

We reinterpreted the game to a single round game with 60 possible actions. 

In contrast, to mainstream DDPG the algorithm entirely forfeits temporal difference learning, due to the relatively low number of 20 turns per episode.

Our strategy was to sidestep any discounted reward schemes.
Instead, we were aiming to directly optimize the cumulative reward of the 20 turns.

Our reasoning being, that due to the lack of feedback in the open-loop phase, there would be no sense in introducing disortions by discounting future rewards. 
Consequently, we trained the algorithm to directly maximize the cumulative undiscounted reward.
