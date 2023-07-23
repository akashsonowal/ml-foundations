"""Adaboost for classification"""

import numpy as np 

# Decision stump used as weak classifier
class DecisionStump: # single node tree
    def __init__(self):
        self.polarity = 1 # rightside of threshold
        self.feature_idx = None
        self.threshold = None 
        self.alpha = None  # performance weightage
    
    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]

        predictions = np.ones(n_samples) # initialize everything to right i.e., 1
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions

class Adaboost:
    def __init__(self, n_clf=6):
        self.n_clf = n_clf
        self.clfs = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape 
        

    


