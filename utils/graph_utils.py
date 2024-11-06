import itertools

import collections

import numpy as np
import pandas as pd

import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns
import re

# import counter
from collections import Counter

import matplotlib as mpl

import time
import scipy.stats as stats

sns.set()


import os

def convert_dataframe_to_nx_graph( df : pd.DataFrame):

    G = nx.from_pandas_adjacency(df, create_using = nx.DiGraph)
    return G


## save_graphs
##################################################################################
def save_graphs( graphs: list, filename : str, num_nodes : int):
    """_summary_

    Args:
        graphs (list): a list of graphs nx.DiGraph
        filename (str): a string representing the filename to be saved
        num_nodes (int): number of nodes in graphs
    """

    # Save each graph to a separate GraphML file
    for idx, graph in enumerate(graphs):
        nx.write_graphml(graph, f"./results/graph_generation/{num_nodes}nodes/graph_{idx}.graphml")

    # Create a ZIP archive containing all the GraphML files
    with zipfile.ZipFile(f"./results/graph_generation/{num_nodes}nodes/{filename}.zip", "w") as zipf:
        for idx in range(len(graphs)):
            zipf.write(f"./results/graph_generation/{num_nodes}nodes/graph_{idx}.graphml")
            # os.remove(f"graph_{idx}.graphml")


def update_graph_frequencies(graph_list: list, result_index: dict):
    """Given a list of graphs, returns a dictionary with the number of times a graph occurs

    Args:
        graph_list (list): List of graphs.
        dag_dict (dict): Dictionary to be updated with graph frequencies.

    Returns:
        dict: Updated dictionary with normalized frequencies.
    """
    # Count occurrences of each graph string
    graph_str_counter = Counter(generate_key_from_adj_matrix(graph) for graph in graph_list)
    
    # convert graph_str_counter to a dictionary
    graph_str_dict = dict(graph_str_counter)
    result = intersection(graph_str_dict, result_index)

    return result


def build_ground_truth_graph():
    
    # Create a directed graph
    G = nx.DiGraph()

    nodes = ["Unemployed", "LowIncome", "NotCompletedYr12", "SingleParent", "MentalHealth"]
    G.add_nodes_from(nodes)


    # Add edges
    edges = [
        ("NotCompletedYr12", "Unemployed"),
        ("NotCompletedYr12", "MentalHealth"),
        ("LowIncome", "NotCompletedYr12"),
        ("SingleParent", "LowIncome"),
        ("SingleParent", "NotCompletedYr12"),
        ("Unemployed", "MentalHealth")
    ]
    G.add_edges_from(edges)

    positions = {
        'NotCompletedYr12': (0.5, 1),
        'Unemployed': (-0.5, 0.8),
        'SingleParent': (1.5, 0.8),
        'LowIncome': (-0.2, 0.6),
        'MentalHealth': (1.1, 0.6)
    }
    
    return G, positions


def build_linear_regr_graph():
    
    # Create a directed graph
    G = nx.DiGraph()

    nodes = ["Unemployed", "LowIncome", "NotCompletedYr12", "SingleParent", "MentalHealth"]
    G.add_nodes_from(nodes)


    # Add edges
    edges = [
        ("Unemployed" , "NotCompletedYr12"),
        ("MentalHealth", "NotCompletedYr12" ),
        ("LowIncome", "NotCompletedYr12" ),
        ("SingleParent", "NotCompletedYr12" ),
    ]
    G.add_edges_from(edges)

    positions = {
        "NotCompletedYr12": (0.5, 1),    # top center
        "Unemployed": (0, 0.5),          # left middle
        "SingleParent": (1, 0.5),        # right middle
        "LowIncome": (0.25, 0.1),         # bottom left
        "MentalHealth": (0.75, 0.1)        # bottom right
    }
    
    return G, positions


def plot_graph(G : nx.DiGraph, pos = None, highlighted_edges = None, title=None, figsize = (5,3), node_size=2000, node_color="skyblue", k=5, save = False, filepath = None):
    
    if pos is None:
        pos = nx.spring_layout(G, k=k)

    plt.figure(figsize=figsize)
    
    
    nx.draw(G, with_labels=True, arrowsize=20, arrows=True, node_size=node_size, node_color=node_color, pos=pos)
    if highlighted_edges:
        nx.draw(G, edgelist=highlighted_edges,edge_color='red',width=5, arrowsize=25, alpha=0.4, with_labels=True, arrows=True, node_size=node_size, node_color=node_color, pos=pos)
    
    plt.gca().margins(0.20)
    
    if title:
        plt.title(title)
    plt.axis("off")
    
    if save:
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
    plt.show()
    
    
def plot_graph_from_adj_mat( adj_matrix : np.ndarray , node_labels : list,  pos = None, title="Graph", figsize = (5,3), node_size=2000, node_color="skyblue", k=5, save = False, filepath = None):
    
    key = generate_key_from_adj_matrix( adj_matrix )
    G = generate_graph_from_key( key, node_labels)
    
    plot_graph(G = G,  pos = pos, title=title, figsize = figsize, node_size=node_size, node_color=node_color, k=k, save = save, filepath=filepath)
    

# n - how many nodes
# p - probability of an edge
# labels - list of node labels
def rDAG(n : int, p : float , labels : str, random_seed: int = 42):
    np.random.seed(random_seed)
    adjmat = np.zeros((n, n))
    adjmat[np.tril_indices_from(adjmat, k=-1)] = np.random.binomial(1, p, size=int(n * (n - 1) / 2))
    return pd.DataFrame(adjmat, columns=labels,index=labels)


def edge_frequency_heatmap(dags, nodes=None, figsize=(5, 5), title="Edge Occurrence Probabilities from Sampled DAGs", save_path=None):
    if not dags:
        raise ValueError("The list of DAGs is empty.")
    
    if nodes is None:
        # Collect all unique nodes from all DAGs
        all_nodes = set(node for G in dags for node in G.nodes())
        nodes = sorted(all_nodes)  # Sort nodes to maintain consistent order
    
    num_nodes = len(nodes)
    frequency_matrix = np.zeros((num_nodes, num_nodes))
    
    # Populate frequency matrix
    for G in dags:
        for edge in G.edges():
            source, target = edge
            if source in nodes and target in nodes:  # Check if nodes are in the list
                source_index = nodes.index(source)
                target_index = nodes.index(target)
                frequency_matrix[source_index, target_index] += 1
    
    # Normalize by the number of samples to get frequencies
    frequency_matrix /= len(dags)
    frequency_matrix = np.around(frequency_matrix, decimals=2)
    
    # Visualize as heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(frequency_matrix, annot=True, cmap="YlGnBu", xticklabels=nodes, yticklabels=nodes)
    plt.ylabel("Source")
    plt.xlabel("Target")
    plt.title(title)

    plt.tight_layout()  # Adjust the layout to make sure nothing is cut off

    # Optionally save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save the figure with high resolution and ensure all content is included
    
    # Always show the plot
    plt.show()


def edge_frequency_heatmap_np(adjacency_matrices, nodes=None, figsize=(5, 5), title="Edge Occurrence Probabilities from Sampled DAGs", save_path=None):
    if not adjacency_matrices:
        raise ValueError("The list of adjacency matrices is empty.")
    
    if nodes is None:
        # Assuming nodes are represented by consecutive integers starting at 0 for each matrix
        num_nodes = adjacency_matrices[0].shape[0]
        nodes = list(range(num_nodes))
    else:
        num_nodes = len(nodes)

    frequency_matrix = np.zeros((num_nodes, num_nodes))
    
    # Populate frequency matrix based on adjacency matrices
    for adj_matrix in adjacency_matrices:
        # Iterate through each cell in the matrix
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:  # There is an edge from i to j
                    frequency_matrix[i, j] += 1
    
    # Normalize by the number of samples to get frequencies
    frequency_matrix /= len(adjacency_matrices)
    frequency_matrix = np.around(frequency_matrix, decimals=2)
    
    # Visualize as heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(frequency_matrix, annot=True, cmap="YlGnBu", xticklabels=nodes, yticklabels=nodes)
    plt.ylabel("Source")
    plt.xlabel("Target")
    plt.title(title)

    plt.tight_layout()  # Adjust layout to make sure nothing is cut off

    # Optionally save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save the figure with high resolution and ensure all content is included
    
    # Always show the plot
    plt.show()


# intersetion of two dictionaries
def intersection(d1, d2):
    
    d2_cp = d2.copy()
    d1_keys = set(d1.keys())
    d2_keys = set(d2_cp.keys())
    shared_keys = d1_keys.intersection(d2_keys)
        
    # sum the values of d1 
    total = sum(d1.values())
    
    for key in shared_keys:
        d2_cp[key] = d1[key] / total
    
    return d2_cp

def create_zeros_matrix(N):
    return np.zeros((N,N)).astype(int)

def has_cycle(adj_matrix):

    num_vertices = len(adj_matrix)

    visited = [False] * num_vertices
    rec_stack = [False] * num_vertices

    for node in range(num_vertices):
        if not visited[node]:
            if is_cyclic_util(node, visited, rec_stack, adj_matrix):
                return True

    return False

def is_cyclic_util(v, visited, rec_stack, adj_matrix):
    visited[v] = True
    rec_stack[v] = True

    # Consider all adjacent vertices
    for i in range(len(adj_matrix)):
        if adj_matrix[v][i] != 0:
            if not visited[i]:
                if is_cyclic_util(i, visited, rec_stack, adj_matrix):
                    return True
            elif rec_stack[i]:
                return True

    rec_stack[v] = False
    return False


def generate_key_from_adj_matrix( adj_matrix ):
    
    num_nodes = str(adj_matrix.shape[0])
    
    # Convert the graph to a string
    s = "".join(map(str, adj_matrix.flatten().astype(int)))

    # Insert a space every three characters
    key = re.sub("(.{" + num_nodes + "})", "\\1 ", s)
    key = key if key[-1] != " " else key[:-1]
    return key


def generate_adj_matrix_from_key( key, num_nodes ):
    # Split the string by spaces
    s_list = key.split()

    # Convert each string in the list to a list of integers
    int_list = [[int(char) for char in string] for string in s_list]

    # Convert the list of lists to a numpy array
    array = np.array(int_list)
    
    return array.reshape(num_nodes, num_nodes)


def generate_graph_from_key( key, node_labels):
    
    # Create a graph from the adjacency matrix
    adj_matrix = generate_adj_matrix_from_key( key, len(node_labels) )
    G =  nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    # Create a mapping from old labels to new labels
    mapping = {old_label: new_label for old_label, new_label in zip(G.nodes(), node_labels)}

    # Relabel the nodes
    G = nx.relabel_nodes(G, mapping)
    
    return G

def compare_graphs(candidate_graph, groundtruth_graph):
    added_edges = []    # Edges in candidate but not in groundtruth
    reversed_edges = [] # Edges that are reversed in the candidate relative to the groundtruth
    deleted_edges = []  # Edges not in candidate but in groundtruth

    # Check for added and reversed edges in the candidate graph
    for edge in candidate_graph.edges():
        if not groundtruth_graph.has_edge(*edge):  # Check if exact edge is missing in groundtruth
            if groundtruth_graph.has_edge(edge[1], edge[0]):
                reversed_edges.append((edge[0], edge[1]))  # Append only the candidate edge direction
            else:
                added_edges.append((edge[0], edge[1]))  # Edge is entirely new

    # Check for edges that are present in the groundtruth but missing in the candidate graph
    for edge in groundtruth_graph.edges():
        if not candidate_graph.has_edge(*edge) and not candidate_graph.has_edge(edge[1], edge[0]):
            deleted_edges.append((edge[0], edge[1]))  # Edge is missing in the candidate graph

    return {'Added Edges': added_edges, 'Deleted Edges': deleted_edges, 'Reversed Edges': reversed_edges}



def plot_approx_posterior_distribution(all_dags_dict, true_graph=None, prob_threshold=0.001, figsize=(12, 7), title="MCMC Approximate Posterior Distribution", algo1_scores=None, label1=None):
    mpl.rcParams.update({'font.size': 12, 'font.family': 'Arial'})  # Uniform font style

    # Filter all_dags_dict for scores greater than the threshold and limit to 200 graphs
    filtered_dags = {k: v for k, v in all_dags_dict.items() if v >= prob_threshold}
    if len(filtered_dags) > 200:
        filtered_dags = dict(list(filtered_dags.items())[:200])

    x_labels = list(filtered_dags.keys())
    scores = list(filtered_dags.values())
    
    if algo1_scores is not None:
        algo1_scores = [algo1_scores.get(k, 0) for k in x_labels]

    # Plotting
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(x_labels)), scores, color='lightblue', label='True Posterior Distribution', alpha=1)
    
    # Plot algorithm scores with different markers
    if algo1_scores is not None:
        plt.plot(range(len(x_labels)), algo1_scores, 's', color='darkred', markersize=3, label=label1, alpha=0.8)

    # Highlight the true graph in the plot if it's present
    if true_graph is not None:
        true_graph_key = generate_key_from_adj_matrix(true_graph)
        if true_graph_key in x_labels:
            true_graph_index = x_labels.index(true_graph_key)
            bars[true_graph_index].set_color('orange')
            plt.text(true_graph_index, scores[true_graph_index] + max(scores) * 0.05, 'True DAG', ha='center', va='bottom', fontsize=12, color='orange')
    plt.xlim(left=-0.5, right=len(x_labels)-0.5)  # Adjust x-axis limits to fit the range of the data

    plt.xticks(range(0, len(x_labels), 50), rotation=45)  # Adjust rotation and interval
    plt.xlabel('$G_j$')
    plt.ylabel('True Posterior Probability P($G | \mathcal{D}$)')
    plt.title(title, fontsize=16)
    # Adjusting the legend placement
    plt.legend(fontsize=14, loc='lower right', bbox_to_anchor=(1.0, 0.2))  # Shift the legend slightly higher
    # Remove background color
    #plt.gca().set_facecolor('none')
    #plt.gcf().set_facecolor('none')

    # Subtle grid
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # Save the figure before showing it
    plt.savefig("./results/approx_posterior_comparison.png", dpi=300)
    
    # Display the plot
    plt.show()




def mcmc_post_process(max_experiment: int, num_nodes: float):
    # Initialize the result dictionary outside the experiment loop
    expr_post = {'JSD': {}}
    
    # Load the true posterior distribution
    

    
    for exp_id in range(max_experiment):
        
        # load true distr
        true_posterior_path = f"./results/true_distr_{exp_id}.pkl"
        with open(true_posterior_path, 'rb') as f:
            true_posterior = pickle.load(f)
        
        # Load the graph list
        with open(f"./results/pmcmc_graph_list_{exp_id}.pkl", 'rb') as f:
            graph_list = pickle.load(f)

        # Calculate the number of samples per 1% of the graph list
        upper_bound = len(graph_list) // 100
        expr_post['JSD'][exp_id] = {}

        total_time_start = time.time()

        temp_true = {k: 0 for k in true_posterior.keys()}
        for j in range(1, 101):
            # Select the subset of the MCMC chain
            mcmc_chain_sample = graph_list[:j * upper_bound]
            
            # put all the values of true posterior to 0 in a new dictionary
            approx_posterior = update_graph_frequencies(mcmc_chain_sample, temp_true)

            # Compute and store the JSD for this subset
            expr_post['JSD'][exp_id][j] = jensen_shannon_divergence(approx_posterior,true_posterior)
            #print(f"JSD for experiment id {exp_id} at iteration {j}: {expr_post['JSD'][exp_id][j]}")

            if j % 50 == 0:
                print(f"Finished iteration {j} for experiment id {exp_id}")

        time_end = time.time()
        print(f"Total time for experiment id {exp_id}: {np.round(time_end - total_time_start)} seconds")

    # Saving the results
    save_path = f"./results/eval_curve.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(expr_post, f)

    print(save_path)
    return expr_post



def plot_mcmc_metric(num_experiments: int, data_path, title="MCMC Methods Comparison"):
    
    loss_JSD = []

    for i in range(0, num_experiments):
        jsd_path = f"{data_path}/eval_curve.pkl"
        with open(jsd_path, 'rb') as f:
            jsd = pickle.load(f)
            jsd = jsd['JSD']
        
        loss_JSD.append(list(jsd[i].values()))

    losses = np.array(loss_JSD)
    mean_losses = np.mean(losses, axis=0)
    std_losses = np.std(losses, axis=0)
    n = len(losses[0])
    confidence = 0.95

    upper_bound = mean_losses + stats.t.ppf((1 + confidence) / 2., n-1) * std_losses
    lower_bound = mean_losses - stats.t.ppf((1 + confidence) / 2., n-1) * std_losses

    plt.figure()  # Ensure a new figure is created for this plot
    plt.plot(range(1, len(mean_losses)+1), mean_losses, label="JSD Loss")
    plt.fill_between(range(1, len(mean_losses)+1), upper_bound, lower_bound, alpha=.2)

    plt.title(title)
    plt.xlabel("Iterations (in thousands)")
    plt.ylabel("JSD Loss")
    plt.legend(loc='best')
    plt.grid(False)

    # save figure with 300dpi before showing it
    save_path = f"{data_path}/eval_curve.png"
    plt.savefig(save_path, dpi=300)
    
    # Now display the plot
    plt.show()
