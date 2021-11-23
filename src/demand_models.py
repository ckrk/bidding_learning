# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:30:15 2021

@author: claude_kloeckl
"""
import numpy as np
import numpy.random as random

class demand_normal:
    def __init__(self, means, variances):
        '''
        
        Parameters
        ----------
        means       : np.vector
        variances   : np.vector
                        Input Variances (!) not standard deviations
        samples : int

        Returns
        -------
        
        vektor of n-samples drawn from the above distribution cycle
        '''
        assert means.shape == variances.shape, 'means and variances shapes dont match'
        assert max(means.shape) == means.size, 'means and variances are no vectors'
        
        self.means = means
        self.variances = variances        

        
    def generate(self,samples):
        '''
        
        Parameters
        ----------
        samples : int

        Returns
        -------
        
        vektor of n-samples drawn from the above distribution cycle
        '''
        
        
        
        demand_raw = random.randn(samples)
        mean_timeseries    = np.resize(self.means,(samples,))
        std_timeseries= np.sqrt(np.resize(self.variances,(samples,))) # Squareroot of Var for Std
        simulated_demand_timeseries = np.multiply(demand_raw, std_timeseries) + mean_timeseries 
        
        return simulated_demand_timeseries