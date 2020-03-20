"""
Created on Wed Mar 18 18:03:36 2020

@author: Claude
"""
import numpy as np

a = np.array([0,5,10])
b = np.array([1,2,10])
c = np.array([2,3,10])
d = np.array([1,9,10])
test_array = np.stack((a,b,d,a,b,c,b,c))
#test_array = np.stack((a,b,c))

print(test_array)
def market_clearing(self,demand,bids):    
    """ 
    Implements a uniform pricing market clearing of several players price-quantity bids
    Requires the bids in a certain from and numbered
    
    Input:
    -   demand as scalar
    -   Expects bids as an (3xn) Numpy Array
    1st Row Player Name, 2nd Row Price-bid, 3rd Row Quantity bid 
    
    Output:
    -   Market Price
    -   Assigns sold quantities
    """
    
    
    # Sort by 2nd Row (ie by Price-Bids)
    ind = np.argsort(bids[:,1]) 
    bids = bids[ind]
    print(bids)
    
    #Consecutively add up 3rd Row (ie Quantity-Bids)
    bids[:,2]=np.cumsum(bids[:,2])
    print(bids)
    
    #Restrict Quantity by 0 and Demand
    bids[:,2]=np.clip(bids[:,2],0,demand)
    print(bids)
    
    #Determine Position of Price setting player and Marketprice
    cutoff = np.argmax(bids[:,2])
    market_price = bids[cutoff,1]
    print(bids)
    
    #Set all quantities above cutoff to 0
    bids[cutoff+1:,2] = 0
    print(bids)
    
    #Split the bids according to players
    
    #Sort-back py player name
    #ind = np.argsort(bids[:,0]) 
    #bids = bids[ind]
    #print(bids)
    #print(np.diff(bids[:,0],axis=0))
    
    #print(market_price)
    return market_price,bids

market_clearing(_,37,test_array)

    