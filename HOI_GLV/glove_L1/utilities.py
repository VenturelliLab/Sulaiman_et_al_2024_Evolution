import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


### Function to process dataframes ###
def process_df_glove(df, species):

    # store measured datasets for quick access
    data = []
    for treatment, comm_data in df.groupby("Treatments"):

        # make sure comm_data is sorted in chronological order
        comm_data = comm_data.sort_values(by='Time', ascending=True).copy()

        # pull evaluation times
        t_eval = np.array(comm_data['Time'].values, float)

        # pull species data
        Y_measured = np.clip(np.array(comm_data[species].values, float), 0., np.inf)

        # append t_eval and Y_measured to data list
        data.append([treatment, t_eval, Y_measured])

    return data

# Function to process dataframes
def process_df(df, species):
    # return vector of evaluation times, t = [n]
    # matrix of species initial and final conditions, S = [n, 2, len(species)]
    t = []
    S = []

    # loop over each unique condition
    for treatment, comm_data in df.groupby("Treatments"):
        # make sure comm_data is sorted in chronological order
        comm_data.sort_values(by='Time', ascending=True, inplace=True)

        # pull evaluation times
        t_eval = np.array(comm_data['Time'].values, float)

        # pull species data
        species_data = np.array(comm_data[species].values, float)
        # species_data = np.clip(np.array(comm_data[species].values, float), 0., np.inf)

        # append data
        for i, tf in enumerate(t_eval[1:]):
            # append eval time
            t.append(tf)

            # append species data
            S.append(np.stack([species_data[0], species_data[i + 1]], 0))

    # return data
    return np.array(t), np.stack(S)

def lin_fit(x, a, b):
    return a + b * x

def check_convergence(f):

    # convert to numpy array
    f = np.array(f)

    # ignore nans
    f = f[~np.isnan(f)]

    p, cov = curve_fit(lin_fit, xdata=np.arange(len(f)), ydata=f/np.max(np.abs(f)), p0=[1., 0.])
    a, b, = p

    # return value of slope
    return b
