import itertools

import numpy as np
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt


from scores.ScoreAbstract import Score

from utils.graph_utils import *

from collections import Counter

from utils.graph_utils import intersection


## gen_base_dag_dict
##################################################################################
def generate_all_dags_keys( num_nodes : int, get_adj_mat = False ):
    """
    Generate a dictionary of all unique DAGs with N nodes.

    Args:
        num_nodes (int): number of nodes in the graph
        node_labels (list): list of node labels. Each label is a string.

    Returns:
        dict: dictionary of all unique DAGs with N nodes.
    """
    
    # generate node labels for N nodes
    node_labels = [f"X{i}" for i in range(num_nodes)]
    
    # Dictionary to store all unique DAGs
    base_dag_lst = []
    
    # Generate all possible directed edges among the nodes
    all_possible_edges = list(itertools.permutations(node_labels, 2))
    
    # Iterate over all possible adjacency matrices
    # Iterate over the subsets of all possible edges to form directed graphs
    for r in range(len(all_possible_edges)+1):
        for subset in itertools.combinations(all_possible_edges, r):
            
            # Initialize an NxN matrix filled with zeros
            adj_matrix = create_zeros_matrix( num_nodes )
            
            # Set entries corresponding to the edges in the current subset to 1
            for edge in subset:
                source, target = edge
                adj_matrix[node_labels.index(source)][node_labels.index(target)] = 1
            
            if not has_cycle(adj_matrix):
                
                if get_adj_mat:
                    base_dag_lst.append(adj_matrix)
                else:
                    adj_matrix_key = generate_key_from_adj_matrix(adj_matrix)
                    base_dag_lst.append(adj_matrix_key)
                    
    return base_dag_lst



def generate_all_dags( data : pd.DataFrame, my_score : Score, rounding = False, isScoreLogSpace = True, gen_augmented_priors = True):
    
    num_nodes = data.shape[1]
    node_labels = list(data.columns)
        
    # Dictionary to store all unique DAGs
    base_dag_dict = {}

    # Generate all possible directed edges among the nodes
    all_valid_dag_keys = generate_all_dags_keys( num_nodes )
    
    for key in all_valid_dag_keys:
                
        # get the corresponding adjacency matrix
        adj_matrix = generate_adj_matrix_from_key( key, num_nodes )
        
        # start storing all information for DAG G
        base_dag_dict[key] = {}
        base_dag_dict[key]['DAG'] = generate_graph_from_key( key, node_labels )
        base_dag_dict[key]['adj_matrix'] = adj_matrix
        
        # store info about network structure
        base_dag_dict[key]['frequency'] = 1 # for the true distrib we generate 1 graph only
        base_dag_dict[key]['num_edges'] = len(base_dag_dict[key]['DAG'].edges())

        # compute score of the data given the graph using any score object
        my_score_object = my_score(data=data, incidence=adj_matrix, rounding = rounding)
        score_of_data_given_G = my_score_object.compute()
        
        try:
            # Attempt to take the log of the score if the score function is not in log space.
            base_dag_dict[key]["log_score"] = score_of_data_given_G["score"] if isScoreLogSpace else np.log(score_of_data_given_G["score"])
        except ValueError as e:
            # Handle cases where np.log() fails due to an incompatible score value.
            print("[ERROR] Unable to compute log of the score. Please check the score object [compute] method and ensure it's a valid numeric value.")
            print(f"Score value: {score_of_data_given_G['score']}")
            # Assign a fallback log score in case of error.
            base_dag_dict[key]["log_score"] = np.log(0.0000001)
            
    base_dag_dict_norm = normalise_scores( base_dag_dict )
        
    print(f"Total {num_nodes} node DAGs generated = {len(base_dag_dict.keys())}")
        
    return base_dag_dict_norm

    
def normalise_scores( base_dag_dict : dict, gen_augmented_priors = False ):
        
    # get the max scores of the log_scores 
    total_score = 0
    total_score_augmented = 0
    
    max_log_score = max([ base_dag_dict[key]["log_score"] for key in base_dag_dict.keys() ])

    for key in base_dag_dict.keys():
        base_dag_dict[key]["log_score_scaled"] = base_dag_dict[key]["log_score"]  - max_log_score
        base_dag_dict[key]["score"] = np.exp( base_dag_dict[key]["log_score_scaled"]  )  
        total_score = total_score + base_dag_dict[key]["score"]          


    # iterate of the dags and normalise the scores
    for key in base_dag_dict.keys():
        base_dag_dict[key]["score_norm"] = base_dag_dict[key]["score"] / total_score

    return base_dag_dict 



def plot_posterior_distribution(all_dags_dict, true_graph=None, true_graph_color="orange", fontsize=10, ylabel=r'Groundtruth P($G | Data$)', prob_threshold=0.001, figsize=(7, 5), with_aug_prior=False, title="Groundtruth posterior distribution", my_color='skyblue', alpha=1, ax=None, label=None, save_path=None):
    # Generate IDs for all DAGs, assume keys are unique identifiers
    all_ids = list(all_dags_dict.keys())
    key_to_id = {k: i for i, k in enumerate(all_ids)}
    id_to_key = {i: k for i, k in enumerate(all_ids)}

    # Filter all_dags_dict for scores greater than the probability threshold
    filtered_dags = {k: v for k, v in all_dags_dict.items() if v >= prob_threshold}
    filtered_ids = [key_to_id[k] for k in filtered_dags.keys()]
    
    # Create a list of positions for plotting, which are just consecutive integers
    plot_positions = list(range(len(filtered_dags)))
    scores = list(filtered_dags.values())

    # Create the matplotlib figure and axis if not provided
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    
    # Plotting
    bars = ax.bar(plot_positions, scores, color=my_color, alpha=alpha, label=label)

    # Generate the key for the true graph and highlight it
    if true_graph is not None:
        true_graph_key = generate_key_from_adj_matrix(true_graph)
        if true_graph_key in all_ids:
            true_graph_index = key_to_id[true_graph_key]
            if true_graph_index in filtered_ids:
                bars_index = filtered_ids.index(true_graph_index)
                bars[bars_index].set_color(true_graph_color)
                ax.legend([bars[bars_index]], ['True Graph'], loc='upper right')

    # Setting up axes labels and title
    ax.set_xticks(plot_positions)
    ax.set_xticklabels(filtered_ids, rotation=90, fontsize=fontsize)  # Showing original DAG indices
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Graph Index')
    ax.set_title(title)
    plt.grid(False)
    
    # Saving the plot if a save path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    
    # Show the plot if ax is not provided
    if ax is None:
        plt.show()
    return ax, key_to_id, id_to_key

def plot_approx_posterior_distribution(all_dags_dict, true_graph=None, true_graph_color="orange", prob_threshold=0.001, figsize=(7, 5), title="MCMC Approximate Posterior Distribution", ax=None, save_path=None):
    # Generate IDs for all DAGs, assume keys are unique identifiers
    all_ids = list(all_dags_dict.keys())
    key_to_id = {k: i for i, k in enumerate(all_ids)}
    
    # Filter all_dags_dict for scores greater than the probability threshold
    filtered_dags = {k: v for k, v in all_dags_dict.items() if v >= prob_threshold}
    filtered_ids = [key_to_id[k] for k in filtered_dags.keys()]
    
    # Create a list of positions for plotting, which are just consecutive integers
    plot_positions = list(range(len(filtered_dags)))
    scores = list(filtered_dags.values())

    # Create the matplotlib figure and axis if not provided
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    
    # Plotting
    bars = ax.bar(plot_positions, scores, color='skyblue', alpha=1)

    # Generate the key for the true graph and highlight it
    if true_graph is not None:
        true_graph_key = generate_key_from_adj_matrix(true_graph)
        if true_graph_key in all_ids:
            true_graph_index = key_to_id[true_graph_key]
            if true_graph_index in filtered_ids:
                bars_index = filtered_ids.index(true_graph_index)
                bars[bars_index].set_color(true_graph_color)
                # Add legend instead of text over the bar
                ax.legend([bars[bars_index]], ['True Graph'], loc='upper right')

    # Setting up axes labels and title
    ax.set_xticks(plot_positions)
    ax.set_xticklabels(filtered_ids, rotation=90, fontsize=10)  # Showing original DAG indices
    ax.set_ylabel(r'MCMC Approximate P($G | Data$)')
    ax.set_xlabel('Graph Index')
    ax.set_title(title)
    plt.grid(False)
    
    # Saving the plot if a save path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    
    # Show the plot if ax is not provided
    if ax is None:
        plt.show()
    
    return ax


    
def compute_true_distribution( all_dags, with_aug_prior = False ):
    true_distr_score = {}
    for k in all_dags.keys():
        if with_aug_prior:
            true_distr_score[k] = all_dags[k]['score_ordering_norm'] #*dag_orders 
        else:
            true_distr_score[k] = all_dags[k]['score_norm']
    
    return true_distr_score

def compute_approx_distribution(graph_list : list):
    graph_str_counter = Counter(generate_key_from_adj_matrix(graph) for graph in graph_list)
    
    # sum all values in graph_str_counter
    total = sum(graph_str_counter.values())

    # normalise graph_str_counter
    graph_str_counter = {k: v / total for k, v in graph_str_counter.items()}

    #graph_id_to_str = {i: k for i, k in enumerate(graph_str_counter.keys())}
    #graph_str_to_id = {v: k for k, v in graph_id_to_str.items()}
    
    # replace the keys of graph_str_counter with an integer
    #graph_id_counter = {graph_id_to_str[i]: graph_str_counter[k] for i, k in enumerate(graph_str_counter.keys())}
    
    return graph_str_counter #graph_id_counter, graph_id_to_str, graph_str_to_id



def compute_approx_distribution_index(graph_list: list, true_posterior: dict):
    """Given a list of graphs, returns a dictionary with the number of times a graph occurs

    Args:
        graph_list (list): List of graphs.
        dag_dict (dict): Dictionary to be updated with graph frequencies.

    Returns:
        dict: Updated dictionary with normalized frequencies.
    """
    # Count occurrences of each graph string
    result_index = {key: 0 for key in true_posterior.keys()}
    
    graph_str_counter = Counter(generate_key_from_adj_matrix(graph) for graph in graph_list)
    
    # convert graph_str_counter to a dictionary
    graph_str_dict = dict(graph_str_counter)
    result = intersection(graph_str_dict, result_index)

    return result


