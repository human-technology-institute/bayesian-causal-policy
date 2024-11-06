from abc import ABC, abstractmethod
import networkx as nx

import pandas as pd
import numpy as np

from scipy.special import gammaln
from scipy.stats import invgamma
from scipy.stats import multivariate_normal
from scipy.special import loggamma as lgamma

import statsmodels.api as sm



from scores.ScoreAbstract import Score

class BGeScore(Score):
    
    ##
    ## Constructor
    ####################################################################
    def __init__(self, data : pd.DataFrame, incidence : np.ndarray, rounding = False, isLogSpace = True, edge_penalty = 1.0):
        """_summary_

        Args:
            data (pd.DataFrame): _description_
            graph (nx.DiGraph, optional): _description_. Defaults to None.
        """
        super().__init__(data, incidence, "BGe Score" )
        
        self.incidence = incidence
        self.node_labels = list(data.columns)
        
        self.rounding = rounding
        
        # node_label to node_indx
        self.node_label_to_indx = {node_label: indx for indx, node_label in enumerate(self.node_labels)}
        
        self.num_cols = data.shape[1] # number of variables
        self.num_obvs = data.shape[0] # number of observations
        self.mu0 = np.zeros(self.num_cols) 

        # Scoring parameters.
        self.am = 1
        self.aw = self.num_cols + self.am + 1
        T0scale = self.am * (self.aw - self.num_cols - 1) / (self.am + 1)
        self.T0 = T0scale * np.eye(self.num_cols)
        self.TN = (
            self.T0 + (self.num_obvs - 1) * np.cov(data.T) + ((self.am * self.num_obvs) / (self.am + self.num_obvs))
            * np.outer(
                (self.mu0 - np.mean(data, axis=0)), (self.mu0 - np.mean(data, axis=0))
            )
        )
        self.awpN = self.aw + self.num_obvs
        self.constscorefact = - (self.num_obvs / 2) * np.log(np.pi) + 0.5 * np.log(self.am / (self.am + self.num_obvs))
        self.scoreconstvec = np.zeros(self.num_cols)
        for i in range(self.num_cols):
            awp = self.aw - self.num_cols + i + 1
            self.scoreconstvec[i] = (
                self.constscorefact
                - lgamma(awp / 2)
                + lgamma((awp + self.num_obvs) / 2)
                + (awp + i) / 2 * np.log(T0scale) 
                - (i + 1) * np.log(edge_penalty)
            )
            
        self.isLogSpace = isLogSpace
        self.t = T0scale
        self.parameters = {}
        self.reg_coefficients = {}
    
    ##
    ## Compute the marginal likelihood for the data
    ####################################################################
    def compute(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.compute_BGe_with_graph() #if self.graph is not None else self.compute_marginal_log_likelihood_from_data()
    
    def compute_node(self, node : str):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.compute_BGe_with_node( node ) #if self.graph is not None else self.compute_marginal_log_likelihood_from_data()
    
    
    def compute_BGe_with_graph(self):
        
        total_log_ml = 0
        parameters = {}  # Dictionary to store the parameters for each node
        
        # Loop through each node in the graph
        for node in self.node_labels:
            
            log_ml_node = self.compute_BGe_with_node( node )['score']
            
            # Save the parameters for the node
            parameters[node] = {
                'score' : log_ml_node,
                'parents': self.find_parents(self.incidence, self.node_label_to_indx[node])
            }
            
            total_log_ml += log_ml_node
        
        # save the parameters
        self.parameters = parameters
        
        # Return the total marginal likelihood and the parameters
        if self.rounding:
                total_log_ml = np.round(total_log_ml, 5)
                
        score = {
            'score': total_log_ml,
            'parameters': parameters
        }
        return score
    
    def compute_BGe_with_node(self, node: str):
    
        parameters = {}  # Dictionary to store the parameters for each node
        
        node_indx = self.node_label_to_indx[node]
        parentnodes = self.find_parents(self.incidence, node_indx)
        num_parents = len(parentnodes)  # number of parents
        
        awpNd2 = (self.awpN - self.num_cols + num_parents + 1) / 2
        A = self.TN[node_indx, node_indx]
        
        if num_parents == 0:  # just a single term if no parents
            corescore = self.scoreconstvec[num_parents] - awpNd2 * np.log(A)

        elif num_parents == 1:  # no need for matrices
            D = self.TN[parentnodes[0], parentnodes[0]]
            logdetD = np.log(D)
            
            B = self.TN[node_indx, parentnodes[0]]
            logdetpart2 = np.log(A - B ** 2 / D)
            corescore = self.scoreconstvec[num_parents] - awpNd2 * logdetpart2 - logdetD / 2

        elif num_parents == 2:  # can do matrix determinant and inverse explicitly
            D = self.TN[np.ix_(parentnodes, parentnodes)]
            detD = np.linalg.det(D)
            logdetD = np.log(detD)

            B = self.TN[node_indx, parentnodes]
            
            det_D_B = np.linalg.det(D - np.outer(B, B) / A)
            logdetpart2 = np.log(det_D_B) + np.log(A) - logdetD
            
            corescore = self.scoreconstvec[num_parents] - awpNd2 * logdetpart2 - logdetD / 2
            
        else:  # otherwise we use Cholesky decomposition to perform both
            D = self.TN[np.ix_(parentnodes, parentnodes)]
            choltemp = np.linalg.cholesky(D)
            logdetD = 2 * np.sum(np.log(np.diag(choltemp)))

            B = self.TN[node_indx, parentnodes]
            logdetpart2 = np.log(A - np.sum(np.linalg.solve(choltemp, B.T) ** 2))
            corescore = self.scoreconstvec[num_parents] - awpNd2 * logdetpart2 - logdetD / 2
        
        # Save the parameters for the node
        parameters[node] = {
            'parents': parentnodes
        }
        self.parameters = parameters
        
        if self.rounding:
            corescore = np.round(corescore, 5)
        
        score = {
            'score': corescore,
            'parameters': parameters
        }

        return score

    
    def compute_BGe_with_edge(self, node : str, parents: list):
        
        parameters = {}  # Dictionary to store the parameters for each node
        
        node_indx = self.node_label_to_indx[node]
        parentnodes = [ self.node_label_to_indx[i]  for i in parents ] # get index of parents labels
        num_parents = len(parentnodes) # number of parents
        
        awpNd2 = (self.awpN - self.num_cols + num_parents + 1) / 2
        
        A = self.TN[node_indx, node_indx]
        
        if num_parents == 0:  # just a single term if no parents
            corescore = self.scoreconstvec[num_parents] - awpNd2 * np.log(A)
        else:
            D = self.TN[np.ix_(parentnodes, parentnodes)]
            choltemp = np.linalg.cholesky(D)
            logdetD = 2 * np.sum(np.log(np.diag(choltemp)))

            B = self.TN[np.ix_([node_indx], parentnodes)]
            logdetpart2 = np.log( A - np.sum(np.linalg.solve(choltemp, B.T)**2) )
            corescore = self.scoreconstvec[num_parents] - awpNd2 * logdetpart2 - logdetD / 2
        
        # Save the parameters for the node
        parameters[node] = {
            'parents': parentnodes
        }
        self.parameters = parameters
        
        score = {
            'score': corescore,
            'parameters': parameters
            }

        return score
    
    def find_parents(self, adj_matrix: np.ndarray, node: int):
        """
    Find the indices of parent nodes for a given node in a directed graph.

    :param adj_matrix: numpy array representing the adjacency matrix of the graph
    :param node: index of the node to find parents for
    :return: list of indices of parent nodes of the specified node
    """
        if node < 0 or node >= adj_matrix.shape[0]:
            raise ValueError("Node index out of bounds")

        # Find the indices of non-zero entries in the node's column
        parent_indices = np.nonzero(adj_matrix[:, node])[0]
        
        return parent_indices.tolist()
    
    
    # GETTERS AND SETTERS
    #####################################################
    
    def get_incidence(self):
        return self.incidence
    
    def set_incidence(self, adj_matrix):
        self.incidence = adj_matrix
    
    def get_am(self):
        return self.am
    
    def set_am(self, am):
        self.am = am
        
    def get_aw(self):
        return self.aw
    
    def set_aw(self, aw):
        self.aw = aw
        
    def get_t(self):
        return self.t
    
    def set_t(self, t):
        self.t = t
        
    def get_parameters(self):
        return self.parameters
    
    def set_parameters(self, parameters):
        self.parameters = parameters
        
    def get_reg_coefficients(self):
        return self.reg_coefficients
    
    def set_reg_coefficients(self, reg_coefficients):
        self.reg_coefficients = reg_coefficients
        
class WeightedBGeScore(Score):
    
    ##
    ## Constructor
    ####################################################################
    def __init__(self, data : pd.DataFrame, incidence : np.ndarray, penalty_coefficient = 0.01, isLogSpace = True):
        """_summary_

        Args:
            data (pd.DataFrame): _description_
            graph (nx.DiGraph, optional): _description_. Defaults to None.
        """
        super().__init__(data, incidence, "Weighted BGeScore" )
        
        self.penalty_coefficient = penalty_coefficient
        self.incidence = incidence
        self.node_labels = list(data.columns)
        
        # extract the number of edges from incidence
        self.num_edges = np.sum(np.sum(incidence)) + 1
        
        # node_label to node_indx
        self.node_label_to_indx = {node_label: indx for indx, node_label in enumerate(self.node_labels)}
        
        self.num_cols = data.shape[1] # number of variables
        self.num_obvs = data.shape[0] # number of observations
        self.mu0 = np.zeros(self.num_cols) 

        # Scoring parameters.
        self.am = 1
        self.aw = self.num_cols + self.am + 1
        T0scale = self.am * (self.aw - self.num_cols - 1) / (self.am + 1)
        self.T0 = T0scale * np.eye(self.num_cols)
        self.TN = (
            self.T0 + (self.num_obvs - 1) * np.cov(data.T) + ((self.am * self.num_obvs) / (self.am + self.num_obvs))
            * np.outer(
                (self.mu0 - np.mean(data, axis=0)), (self.mu0 - np.mean(data, axis=0))
            )
        )
        self.awpN = self.aw + self.num_obvs
        self.constscorefact = - (self.num_obvs / 2) * np.log(np.pi) + 0.5 * np.log(self.am / (self.am + self.num_obvs))
        self.scoreconstvec = np.zeros(self.num_cols)
        for i in range(self.num_cols):
            awp = self.aw - self.num_cols + i + 1
            self.scoreconstvec[i] = (
                self.constscorefact
                - lgamma(awp / 2)
                + lgamma((awp + self.num_obvs) / 2)
                + (awp + i) / 2 * np.log(T0scale)
            )
            
        self.isLogSpace = isLogSpace
        self.t = T0scale
        self.parameters = {}
        self.reg_coefficients = {}
    
    
    def calculate_complexity_penalty(self):
        """Calculate the complexity penalty based on the number of edges in the graph."""
        num_edges = np.sum(self.incidence > 0)  # Assuming the incidence matrix is binary
        complexity_penalty = self.penalty_coefficient * num_edges
        return complexity_penalty
    
    def calculate_node_complexity_penalty(self, num_parents):
        """Calculate the complexity penalty based on the number of parents a node has."""
        complexity_penalty = self.penalty_coefficient * num_parents
        return complexity_penalty

    
    ##
    ## Compute the marginal likelihood for the data
    ####################################################################
    def compute(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.compute_BGe_with_graph() #if self.graph is not None else self.compute_marginal_log_likelihood_from_data()
    
    def compute_node(self, node : str):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.compute_BGe_with_node( node ) #if self.graph is not None else self.compute_marginal_log_likelihood_from_data()
    
    
    def compute_BGe_with_graph(self):
        
        total_log_ml = 0
        parameters = {}  # Dictionary to store the parameters for each node
        
        # Loop through each node in the graph
        for node in self.node_labels:
            
            log_ml_node = self.compute_BGe_with_node( node )['score']
            
            # Save the parameters for the node
            parameters[node] = {
                'score' : log_ml_node,
                'parents': self.find_parents(self.incidence, self.node_label_to_indx[node])
            }
            
            total_log_ml += log_ml_node
        
        # calculate and subtract the complexity penalty
        complexity_penalty = self.calculate_complexity_penalty()
        total_log_ml -= complexity_penalty
        
        # save the parameters
        self.parameters = parameters
        
        # Return the total marginal likelihood and the parameters
        score = {
            'score': total_log_ml,
            'parameters': parameters
        }
        return score
    
    def compute_BGe_with_node(self, node : str):
        
        parameters = {}  # Dictionary to store the parameters for each node
        
        node_indx = self.node_label_to_indx[node]
        parentnodes = self.find_parents(self.incidence, node_indx)
        num_parents = len(parentnodes) # number of parents
        
        awpNd2 = (self.awpN - self.num_cols + num_parents + 1) / 2
        
        A = self.TN[node_indx, node_indx]
        
        if num_parents == 0:  # just a single term if no parents
            corescore = self.scoreconstvec[num_parents] - awpNd2 * np.log(A)
        else:
            D = self.TN[np.ix_(parentnodes, parentnodes)]
            choltemp = np.linalg.cholesky(D)
            logdetD = 2 * np.sum(np.log(np.diag(choltemp)))

            B = self.TN[np.ix_([node_indx], parentnodes)]
            logdetpart2 = np.log( A - np.sum(np.linalg.solve(choltemp, B.T)**2) )
            corescore = self.scoreconstvec[num_parents] - awpNd2 * logdetpart2 - logdetD / 2
        
        
        # Calculate and subtract the node-specific complexity penalty
        complexity_penalty = self.calculate_node_complexity_penalty(num_parents)
        corescore -= complexity_penalty  
    
        # Save the parameters for the node
        parameters[node] = {
            'parents': parentnodes
        }
        self.parameters = parameters
        
        score = {
            'score': corescore,
            'parameters': parameters
            }

        return score
    
    def find_parents(self, adj_matrix: np.ndarray, node: int):
        """
    Find the indices of parent nodes for a given node in a directed graph.

    :param adj_matrix: numpy array representing the adjacency matrix of the graph
    :param node: index of the node to find parents for
    :return: list of indices of parent nodes of the specified node
    """
        if node < 0 or node >= adj_matrix.shape[0]:
            raise ValueError("Node index out of bounds")

        # Find the indices of non-zero entries in the node's column
        parent_indices = np.nonzero(adj_matrix[:, node])[0]
        
        return parent_indices.tolist()
    
    
    # GETTERS AND SETTERS
    #####################################################
    
    def get_incidence(self):
        return self.incidence
    
    def set_incidence(self, adj_matrix):
        self.incidence = adj_matrix
    
    def get_am(self):
        return self.am
    
    def set_am(self, am):
        self.am = am
        
    def get_aw(self):
        return self.aw
    
    def set_aw(self, aw):
        self.aw = aw
        
    def get_t(self):
        return self.t
    
    def set_t(self, t):
        self.t = t
        
    def get_parameters(self):
        return self.parameters
    
    def set_parameters(self, parameters):
        self.parameters = parameters
        
    def get_reg_coefficients(self):
        return self.reg_coefficients
    
    def set_reg_coefficients(self, reg_coefficients):
        self.reg_coefficients = reg_coefficients
        
