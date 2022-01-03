# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:30:15 2021

@author: claude_kloeckl
"""
import numpy as np
import numpy.random as random
from numpy.random import default_rng


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
    

def recoverOfferCurve(mu, sigmaSquare):
    '''
    Assume offer price differences are log-normally distributed with mu being the mean and sigmaSquare the empirical variance of the logs of each offer price step
    More precisely, define: y := log(y_k) if k =0
                                 log(y_k - y_{k-1}) if k > 0
    where [y_0, y_1, ... , y_K] are the offer prices for the K discretized steps
    
    mu and sigmaSquare must be 1-d np.arrays on a grid (ascending)
    output: offer curve (y-values, i.e., offer prices)  
    
    see https://math.stackexchange.com/questions/2409702/expected-value-of-a-lognormal-distribution
    
    example: mu = np.array([np.log(20.), np.log(30.-20.), np.log(50.-30.), np.log(70.-30.)]) 
             sigmaSquare = np.array([1., 1., 1., 1.]) 
             Note that the elements of sigmaSquare should be the variance of logs  
    
    '''
        
    return np.cumsum(np.exp(mu + .5 * sigmaSquare))
    
def sampleOfferCurves(loc, shape, N, seed=42):
    '''
    mu and sigmaSquare must be np.arrays of mean and variance of log normal distributed offer curve step increments
    More precisely, define: y := log(y_k) if k =0
                                 log(y_k - y_{k-1}) if k > 0
    where [y_0, y_1, ... , y_K] are the offer prices for the K discretized steps
    N is the number of random samples
    
    output is a 2-d array of offer price steps with dim (N x |loc|)
    
    example: loc = np.array([20., 10., 5., 50.]), i.e., [0] + list(np.cumsum(loc)) is the mean offer curve
             shape = loc / 10 is the step-wise standard deviation
    
    https://en.wikipedia.org/wiki/Log-normal_distribution#Arithmetic_moments
    Sampling for one offer (incremental) offer steps using moment conditions
    
    moment generating function: E[X^n] = exp(n * loc + .5 * n^2 * shape^2)
    E[X] = exp(loc + .5 * shape^2)
    E[X^2] = exp(2 * loc + 2 * shape^2)
    V[X] = E[X^2] - E[X]^2 = exp(2 * loc + shape^2) * exp(sigma^2 - 1)
    
    trace out loc and shape from first and second moment, resp. Var
    loc = log(E[X]^2/(E[X^2])^.5) = log(E[X]^2/((V[X] + E[X]^2)^.5) 
    shape = (log(1 + V[X]/E[X]^2))^.5 
    
    alternative: https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Sampling_Random_Variables
    cpp implementation https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Sampling_Random_Variables
    '''
    rng = default_rng(seed)
    return np.cumsum(np.vstack([rng.lognormal(np.log(loc[j]**2/(np.sqrt(loc[j]**2 + shape[j]**2))), np.sqrt(np.log(1+(shape[j]**2/loc[j]**2))), N) for j in range(len(loc))]).T, 1)
    