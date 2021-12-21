import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque



class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)



def plot_run_outcome(data, number_of_agents, bid_limit, NE, episodes, run, curves = 'both', 
                       title = 'Bidding Game',rescale=[1,1,1], moving_window = 9):
    '''
    Plots actions or rewards or both (curves ='actons','rewards'or'both')
    Reads out nessecary data from dictionary (structured like in 'main')
    possible to display bid limit and Nash Equilibrium (NE) threshold (NE can also be 'none')
    actions, rewards and bid_limit can be recaled for reprenstatiom (rescale[param for actions, param, for rewards, param for bid_limit])
    also takes the moving medain of a predefined window (default = 10) for smoothing outputted lines
    '''
    
    med_actions, med_rewards = moving_median_rewards_actions(data,run,episodes, moving_window)
    
    # rescale data
    med_actions = med_actions*rescale[0]
    med_rewards = med_rewards*rescale[1]
    bid_limit = bid_limit*rescale[2] 
    
    plt.plot([bid_limit]*episodes, color='grey',label = 'Bid Limit' , lw =1)
    
    if NE != 'none':
        plt.plot([NE]*episodes, color='C0', label = 'Nash Equilibrium', lw =1)
    
    for i in range(number_of_agents):
        if curves == 'actions':
            plt.plot(med_actions[1:,i], 'C{}'.format(i+1), label = 'Bids Agent{}'.format(i), lw =1, linestyle = '--') # displaying actions
            plt.ylabel('Action')
        elif curves == 'rewards':
            plt.plot(med_rewards[1:,i], 'C{}'.format(i+1), label = 'Rewards Agent{}'.fromat(i), lw =1) # displaying rewards
            plt.ylabel('Reward')
        else:
            plt.plot(med_actions[1:,i], 'C{}'.format(i+1), label = 'Bids Agent{}'.format(i), lw =1, linestyle = '--')
            plt.plot(med_rewards[1:,i], 'C{}'.format(i+1), label = 'Rewards Agent{}'.format(i), lw =1) # displaying rewards
            plt.ylabel('Reward/Action')


    
    plt.xlabel('Episode')
    plt.legend(loc=4, prop={'size': 7})
    plt.title('{}'.format(title))
    plt.plot()
    


def moving_median_rewards_actions(data, run, episodes=15000, n=9): 
    '''
    reads actions and rewards for all episodes in a specific run from dictionary 
    and further calculates the moving median for a specified period "n"
    '''
    # get data from dictionary
    actions =[data[run][i]['actions'] for i in range(episodes)]
    rewards =[data[run][i]['rewards'] for i in range(episodes)]
    actions= np.squeeze(np.asarray(actions))
    rewards= np.squeeze(np.asarray(rewards))
    
    med_rewards = []
    med_actions = []
    #moving median of n_th step (takes median from n rows and outputs an array of the same length as the input)
    for i in range(episodes):
        temp_actions =np.median(actions[i:n], axis=0)
        temp_rewards =np.median(rewards[i:n], axis=0)
        med_actions.append(temp_actions)
        med_rewards.append(temp_rewards)
        n += 1

    recompiled_actions = np.asarray(med_actions)
    recomplied_rewards = np.asarray(med_rewards)
    
    return recompiled_actions, recomplied_rewards

