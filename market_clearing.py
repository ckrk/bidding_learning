"""
Created on Wed Mar 18 18:03:36 2020

@author: Claude
"""
import numpy as np
import numpy_groupies as npg
from numpy_groupies import aggregate_numpy as anp


#a = np.array([0,5,10])
#b = np.array([1,2,10])
#c = np.array([2,3,11])
#d = np.array([1,9,7])

#Flip
a = np.array([0,10, 2])
b = np.array([1,10,3])
c = np.array([2,11, 2])
d = np.array([3,7,9])


#Fringe Readout for Testing

#Readout fringe players from other.csv (m)
#read_out = np.genfromtxt("others.csv",delimiter=";",autostrip=True,comments="#",skip_header=1,usecols=(0,1))
#Readout fringe switched to conform with format; finge[0]=quantity fringe[1]=bid
#fringe = np.fliplr(read_out)
#fringe = np.pad(fringe,((0,0),(1,0)),mode='constant')

#test_array = np.stack((a,b,d,a,b,c,b,c))
#test_array = np.stack((a,b,c,a,d,c))
#test_array = fringe




#demand =27


#print(test_array)
def market_clearing(demand,bids):    
    """ 
    Implements a uniform pricing market clearing of several players price-quantity bids
    Requires the bids in a certain from and numbered
    
    Input:
    -   demand as scalar
    -   Expects bids as an (3xn) Numpy Array
    1st Row Player Name, 2nd Row Quantity bid, 3rd Row Price-bid 
    
    Output:
    -   Market Price
    -   Ordered result of each quantity assigned by bid
    -   Assigns sold quantities per player
    
    #Attention: player labels need to be integers
    """
    
    # Sort by 3rd Row (ie by Price-Bids)
    ind = np.argsort(bids[:,2]) 
    bids = bids[ind]
    #print(bids)
    
    #Consecutively add up 2nd Row (ie Quantity-Bids)
    bids[:,1]=np.cumsum(bids[:,1])
    #print(bids)
    
    #Restrict Quantity by 0 and Demand
    bids[:,1]=np.clip(bids[:,1],0,demand)
    #print(bids)
    
    #Determine Position of Price setting player and Marketprice
    cutoff = np.argmax(bids[:,1])
    market_price = bids[cutoff,2]
    #print(bids)
    
    #Convert CumSum to Differences
    #This sets all quantities above cutoff to 0 and gives sold quantities below cutoff
    bids[:,1]=np.hstack((bids[0,1],np.diff(bids[:,1])))
    
    #print(bids)
    
    
    #Aggregate quantities py player name
    
    #Labels are player names in a[:,0] and Values are quantities in a[:,1]
    
    #Attention: the labels need to be integers
    #bids[:,0]=bids[:,0].astype(int)
    
    #Attention: without dtype float in the values we get an overflow
    
    quantities = npg.aggregate(bids[:,0].astype(int),bids[:,1],func='sum',dtype=np.float)
    #print(quantities)
    
    return market_price, bids, quantities

#market_clearing(37,test_array)

def converter_new(suppliers, nmb_agents):
    
    sup_split = [0]*nmb_agents*2
    
    for n in range(nmb_agents):
        sup_split[n] = np.array([int(n), (suppliers[n,1]*suppliers[n,4]), suppliers[n,2], suppliers[n,5], suppliers[n,6]])
        sup_split[n+1] = np.array([int(n), (suppliers[n,1] - suppliers[n,1]*suppliers[n,4]), suppliers[n,2], suppliers[n,5], suppliers[n,6]])
        n += 1 # check if next n starts now at 3!!!!
    
    all_together = np.asarray(sup_split)
    
    return all_together

def combine_sold_quantities(split_quantities, nmb_agents):
    
    sold_quantities = [0]*nmb_agents
    
    for n in range(nmb_agents):
        sold_quantities[n] = spilt_quantities[n]+split_quantities[n+1]
        n += 1
    np.asarray(sold_quantities)
    
    return sold_quantities
        
    

def converter(Sup0, Sup1, Sup2):
    
    Sup0a = np.array([int(0), (Sup0[1]*Sup0[4]), Sup0[2], Sup0[5], Sup0[6]])
    Sup0b = np.array([int(1), (Sup0[1]-(Sup0[1]*Sup0[4])), Sup0[3], Sup0[5], Sup0[6]])
    
    Sup1a = np.array([int(2), (Sup1[1]*Sup1[4]), Sup1[2], Sup1[5], Sup1[6]])
    Sup1b = np.array([int(3), (Sup1[1]-(Sup1[1]*Sup1[4])), Sup1[3], Sup1[5], Sup1[6]])
    
    Sup2a = np.array([int(4), (Sup2[1]*Sup2[4]), Sup2[2], Sup2[5], Sup2[6]])
    Sup2b = np.array([int(5), (Sup2[1]-(Sup2[1]*Sup2[4])), Sup2[3], Sup2[5], Sup2[6]])
    
    All_together = np.stack((Sup0a, Sup0b, Sup1a, Sup1b, Sup2a, Sup2b))

    return All_together

''' 
Possible Testcases:
    Check if sales \leq supply
    Check if sales \geq demand
    Check 3 cases low demand, equal demanl and high demand
'''

  