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

        # Initialize weights to 1/N for each sample
        w = np.full(n_samples, (1 / n_samples))

        # Iterate through the classifiers
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float("-Inf")

            # greedy search to find threshold and feature
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)
                



    


