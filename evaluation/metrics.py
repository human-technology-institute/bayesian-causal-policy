
import numpy as np
from scipy.stats import entropy

from sklearn import metrics as sklearn_metrics

from sklearn import metrics

from sklearn.metrics import roc_auc_score

from scipy.stats import t


import pickle
from utils.graph_utils import update_graph_frequencies

import matplotlib.pyplot as plt
import  seaborn as sns
sns.set()

##
## kl_divergence
###########################################################################################
def kl_divergence(P : dict, Q : dict, epsilon: float = 1e-15):
    """computes the KL divergence between two distributions. Requires that the two distributions have the same length
    """

    # Ensure that the distributions have the same length
    # if the dist1 and dist2 do not have the same length, return an error and exit
    len_diff = np.abs(len(P) - len(Q))
    if len_diff > 0:
        print(f"Error: P and Q have different lengths: {len(P)} and {len(Q)}")
        return -1
    
    # Convert the distributions to lists (ensuring consistent order)
    p = np.array(list(P.values())) + epsilon
    q = np.array(list(Q.values())) + epsilon
    
    return entropy(p, q)

##
## jensen_shannon_divergence
###########################################################################################
def jensen_shannon_divergence(P : dict, Q : dict, epsilon: float = 1e-15):

    # Ensure the distributions have the same length
    len_diff = np.abs(len(P) - len(Q))
    if len_diff > 0:
        print(f"Error: P and Q have different lengths: {len(P)} and {len(Q)}")
        return -1
    
    # Convert the distributions to lists (ensuring consistent order)
    p = np.array(list(P.values())) + epsilon
    q = np.array(list(Q.values())) + epsilon

    # Normalize the distributions to ensure they are proper probability distributions
    p /= p.sum()
    q /= q.sum()

    # Compute M
    m = 0.5 * (p + q)
    
    # Compute the Jensen-Shannon divergence
    jsd = 0.5 * (entropy(p, m) + entropy(q, m))
    
    return jsd


def mean_squared_error(P : dict, Q : dict):
    
    P = np.array(list(P.values())).astype(float)
    Q = np.array(list(Q.values())).astype(float) 
    
    return np.mean((P - Q)**2)


def mean_absolute_error(P : dict, Q : dict):
    
    P = np.array(list(P.values())).astype(float)
    Q = np.array(list(Q.values())).astype(float)
    
    return np.mean(np.abs(P - Q))


def graph_model_averaging( mcmc_graph_list, num_nodes, prob = 0.5):
    # Stack the matrices vertically (along a new third axis), then sum over this new axis
    frequency_matrix = np.sum(np.array(mcmc_graph_list) != 0, axis=0)

    # Normalize by the number of samples to get frequencies
    frequency_matrix = frequency_matrix / len(mcmc_graph_list)

    # Create a matrix where entries greater than the threshold `prob` are 1, else 0
    prob_matrix = (frequency_matrix >= prob).astype(int)
    return prob_matrix
        
def compute_hamming_dist(B_est, B_true):

    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)

    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)

    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)

    return shd

# Monte Carlo Standard Error
def MC_se(x, B):
    # x is the array of values
    # B is the number of simulations
    
    t_value = t.ppf(0.975, B-1)
    std_dev = np.std(x, ddof=1)
    mc_se = t_value * std_dev / np.sqrt(B)
    return mc_se