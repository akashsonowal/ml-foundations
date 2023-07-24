import numpy as np 

class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.discriminants = None 
    
    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # Within class scatter matrix
        # SW = sum((X_c - mean_X_c)^2 )
        SW = np.zeros((n_features, n_features))

        # Between class scatter
        # SB = sum( n_c * (mean_X_c - mean_overall)^2 )
        SB = np.zeros((n_features, n_features))

        mean_overall = np.mean(X, axis=0)
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            


    
    def transform(self, X):
        return np.dot(X, self.linear_discriminants.T)