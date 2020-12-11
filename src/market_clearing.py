import numpy as np
import numpy_groupies as npg

from numpy_groupies import aggregate_numpy as anp


#a = np.array([0,5,10])
#b = np.array([1,2,10])
#c = np.array([2,3,11])
#d = np.array([1,9,7])

#Flip
a = np.array([1,50, 40,0,50])
b = np.array([0,50,40,0,50])
#a = np.array([0,2,9,0,2])
#b = np.array([1,3,9,0,3])
#c = np.array([2,6,9,0,6])
#d = np.array([3,9,9,0,9])
#e = np.array([4,20,9,0,20])
#f = np.array([5,7,9,0,7])

#Fringe Readout for Testing

#Readout fringe players from other.csv (m)
#read_out = np.genfromtxt("others.csv",delimiter=";",autostrip=True,comments="#",skip_header=1,usecols=(0,1))
#Readout fringe switched to conform with format; finge[0]=quantity fringe[1]=bid
#fringe = np.fliplr(read_out)
#fringe = np.pad(fringe,((0,0),(1,0)),mode='constant')

#test_array = np.stack((a,b,c,a,d,c,b))
#bids = np.vstack([a, fringe])
#bids = np.stack((a,b))
#test_array = fringe




demand = 70


#print(test_array)
def market_clearing(demand,bids,price_cap):    
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
    bids = bids.astype(float)
    
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

    # Tie Break
    
    if len(np.argwhere(bids[:,2] == np.amax(bids[:,2]))) > 1 or len(np.argwhere(bids[:,2] >= price_cap)) > 1:
        bids = tie_break(bids, price_cap)
    
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

def converter(suppliers, nmb_agents):
    
    sup_split = []

    for n in range(nmb_agents):
        sup_splitA = [int(n), (suppliers[n,1]*suppliers[n,4]), suppliers[n,2], suppliers[n,5]]
        sup_splitB = [int(n), (suppliers[n,1] - suppliers[n,1]*suppliers[n,4]), suppliers[n,3], suppliers[n,5]]
        sup_split.append(sup_splitA)
        sup_split.append(sup_splitB)
        
       
    all_combined = np.asarray(sup_split)

    return all_combined

# should work for all cases
def tie_break(bids, price_cap):
    # determine candidates who are in a tie break
    #tie_break_candidates = np.argwhere(bids[:,2] == np.amax(bids[:,2]))
    tie_break_candidates = np.argwhere(bids[:,2] >= price_cap)
    
    if len(tie_break_candidates) == 0:
        tie_break_candidates = np.argwhere(bids[:,2] == np.amax(bids[:,2]))
        
    # starting capacity for distributin
    overall_base_quantity = sum(bids[tie_break_candidates[0,0]:,1])   
    base_quantities = overall_base_quantity/len(tie_break_candidates)
    
    # parameters needed to start while-loop
    quantity_for_distribution = overall_base_quantity
    new_quantities = base_quantities
    more_cap_candidates = np.argwhere(bids[tie_break_candidates[0,0]:,4] > new_quantities)
    distributed_quantities = 0
    
    while quantity_for_distribution > 0:
        
        # check amount which is already distributed 
        quantity_for_distribution = overall_base_quantity -(distributed_quantities + new_quantities*len(more_cap_candidates))
        
        # determine those who sitll have free capacity to sell and those who are already satisfied
        more_cap_candidates = np.argwhere(bids[tie_break_candidates[0,0]:,4] > new_quantities) 
        less_cap_candidates = np.argwhere(bids[tie_break_candidates[0,0]:,4] <= new_quantities)
        
        # get how much is still left for distribution
        surplus= 0
        for i in range(len(less_cap_candidates)):
            surplus += new_quantities - bids[less_cap_candidates[i,0],4]
        
        # determine new amount for distribution
        if len(more_cap_candidates) > 0:
            new_quantities = new_quantities + (surplus/len(more_cap_candidates))
        
        # check amount of already satisfied candidates
        distributed_quantities = 0
        for i in range(len(less_cap_candidates)):
            distributed_quantities += bids[less_cap_candidates[i,0],4]
        

    bids[tie_break_candidates[0,0]:,1] = np.clip(new_quantities,0,bids[tie_break_candidates[0,0]:,4])
    
    return bids  



# not working for all cases
def simple_tie_break(bids):
    
  
    tie_break_candidates = np.argwhere(bids[:,2] == np.amax(bids[:,2]))
        
    quantity_for_distribution = sum(bids[tie_break_candidates[0,0]:,1])
        
    new_quantities = quantity_for_distribution/len(tie_break_candidates)
        
    less_cap_candidates = np.argwhere(new_quantities > bids[tie_break_candidates[0,0]:,4])
        
    surplus = 0
    for i in range(len(less_cap_candidates)):
        surplus +=  new_quantities - bids[tie_break_candidates[less_cap_candidates[i,0],0],4]
            
    surplus_quantities = new_quantities + (surplus / (len(tie_break_candidates)- len(less_cap_candidates)))

        
    bids[tie_break_candidates[0,0]:,1] = np.clip(surplus_quantities,0,bids[tie_break_candidates[0,0]:,4])
        
    return bids

''' 
Possible Testcases:
    Check if sales \leq supply
    Check if sales \geq demand
    Check 3 cases low demand, equal demanl and high demand
'''
