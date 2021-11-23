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
        means       : np.array
        variances   : np.array
                        Input Variances (!) not standard deviations

        Returns
        -------
        Class demand_normal
        '''
        assert means.shape == variances.shape, 'means and variances shapes dont match'
        
        self.means = means
        self.variances = variances 
        self.scenarios = means.shape[0] # Scenarios in Zeilen!

        
    def generate(self,samples):
        '''
        
        Parameters
        ----------
        samples : int

        Returns
        -------
        np.array of n-samples drawn from the above distribution cycle
        '''
        
        
        if max(self.means.shape) == self.means.size: 
            # Vector-Single Scenario Case Output (samples,) timeseries
            demand_raw = random.randn(samples)
            mean_timeseries    = np.resize(self.means,(samples,))
            std_timeseries= np.sqrt(np.resize(self.variances,(samples,))) # Squareroot of Var for Std
            simulated_demand_timeseries = np.multiply(demand_raw, std_timeseries) + mean_timeseries 
        else:
            # Matrix-Multi Scenario Case Output (scenarios,samples) timeries
            demand_raw      = np.array([random.randn(samples) 
                                        for i in range(self.scenarios)])
            mean_timeseries = np.array([np.resize(self.means[i],(samples,)) 
                                        for i in range(self.scenarios)])
            std_timeseries  = np.array([np.sqrt(np.resize(self.variances,(samples,))) 
                                        for i in range(self.scenarios)])
            simulated_demand_timeseries = np.multiply(demand_raw, std_timeseries) + mean_timeseries
            
        return simulated_demand_timeseries