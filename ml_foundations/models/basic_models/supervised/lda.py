import numpy as np 

class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.discriminants = None 
    
    def fit(self, X, y):
        pass 
    
    def transform(self, X):
        return np.dot(X, self.linear_discriminants.T)