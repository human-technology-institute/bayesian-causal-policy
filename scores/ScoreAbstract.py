from abc import ABC, abstractmethod
import networkx as nx

import pandas as pd

import numpy as np


class Score( ABC ):
    
    def __init__(self, data : pd.DataFrame, incidence : np.array, to_string : str, rounding = False, isLogSpace = True):
        """
        initialises the Score abstract class. All classes that inherit from this class must implement the compute method.

        Args:
            data (pd.DataFrame): dataset
            graph (nx.DiGraph, optional): graph structure. Defaults to None. The graph must be DAG.
        """
        
        self.data = data
        self.adj_matrix = incidence
        self.isLogSpace = isLogSpace
        self.rounding = rounding
        
        self.to_string = to_string
    
    # abstract method to be implemented by subclasses 
    @abstractmethod
    def compute(self):
        """
        implements a score function (e.g. BGe, Marginal Likelihood, etc)
        """
        pass
    
    abstractmethod
    def compute_node(self):
        pass
    
    # GETTERS AND SETTERS
    ####################################################

    def get_to_string(self):
        return self.to_string
    
    def set_to_string(self, to_string):
        self.to_string = to_string

    def get_data(self):
        return self.data
    
    def get_incidence(self):
        return self.incidence
    
    def set_incidence(self, incidence):
        self.incidence = incidence
        
    def set_data(self, data):
        self.data = data
        
    def get_isLogSpace(self):
        return self.isLogSpace
    
    def set_isLogSpace(self, isLogSpace):
        self.isLogSpace = isLogSpace
        
        
