import numpy as np 

class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None 
    
class XGBoost:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
    
    def _negative_gradient(self, y_true, y_pred):
        # Calculate the negative gradient (residuals) of the loss function
        return - (y_true - y_pred)
    
    def fit(self, X, y):
        predictions = np.zeros(len(y))

        for _ in range(self.n_estimators):
            # Calculate the negative gradient (residuals) of the current predictions
            residuals = self._negative_gradient(y, predictions)

            # Fit a new decision tree on the residuals
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Make predictions with the new tree and update the ensemble predictions
            tree_predictions = tree.predict(X)
            predictions += self.learning_rate * tree_predictions

            # Store the new tree in the ensemble
            self.trees.append(tree)

    
    def predict(self, X):
        # Make predictions by summing the predictions from all the trees in the ensemble
        predictions = np.zeros(X.shape[0])

        for tree in self.trees:
            tree_predictions = tree.predict(X)
            predictions += self.learning_rate * tree_predictions
        return predictions 
