import numpy as np 

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alpha = np.zeros(self.n_estimators) # defined at tree level
        self.trees = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        weights = np.ones(n_samples) / n_samples

        for t in range(self.n_estimators):
            tree = DecisionStump()
            error = 0.0

            for i in range(self.n_samples):
                prediction = tree.predict(X[i]) # where is fit?

                if prediction != y[i]:
                    error += weights[i]
            
            self.alpha[t] = 0.5 * np.log((1.0 - error) / max(error, 1e-10)) # higher alpha means more contributuon of this weak learner
            self.trees.append(tree)

            for i in range(self.n_samples):
                prediction = tree.predict(X[i])
                exponent = -self.alpha[t] * y[i] * prediction
                weights[i] = weights[i] * np.exp(exponent)
            
            weights /= np.sum(weights)

    
    def predict(self, X):
        predictions = np.zeros(len(X))

        for alpha, tree in zip(self.alpha, self.trees):
            predictions += alpha * np.array([tree.predict(x) for x in X])
        return np.sign(predictions)
