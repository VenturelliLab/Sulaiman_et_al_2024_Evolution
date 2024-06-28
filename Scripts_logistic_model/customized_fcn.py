import numpy as np
import scipy as sp
from scipy import integrate

def gLV(x, t, r, A):
    '''gLV dynamics with time-invariant parameters. Note that logistic growth is equivalent to gLV with 1 species.'''
    # A = num_species x num_species
    # r = num_species x 1
    
    num_species = len(x)
    
    dxdt = np.zeros(num_species)
    
    for i in range(num_species):
        interaction_temp = 0
        for j in range(num_species):
            interaction_temp = interaction_temp + A[i][j] * x[j]
        
        dxdt[i] = x[i] * (r[i] + interaction_temp)
    
    return dxdt

def sim_TIV_gLV(x0, para_dyn, time):
    '''simulate a single passage using time invariant gLV'''
    r = para_dyn[0]
    A = para_dyn[1]
    
    x_sol = sp.integrate.odeint(gLV, x0, time, args = (r, A))
    
    return x_sol
    