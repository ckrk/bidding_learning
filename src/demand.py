# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:30:15 2021

@author: claude_kloeckl
"""
import numpy as np
import numpy.random as random

class demand:
    def generate_normal(means,variances,samples):
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
        
        
        demand_raw = random.randn(samples)
        print(demand_raw)
        mean_timeseries    = np.resize(means,(samples,))
        std_timeseries= np.sqrt(np.resize(variances,(samples,))) # Squareroot of Var for Std
        demand = np.multiply(demand_raw, std_timeseries) + mean_timeseries 
        
        return demand