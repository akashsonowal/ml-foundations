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
            X_c = X[y == c] # (n_c, dim)
            mean_c = np.mean(X_c, axis=0) # (1, dim)
            SW += (X_c - mean_c).T.dot((X_c - mean_c)) 

            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1) # (1, dim) -> (dim, 1)?
            SB += n_c * (mean_diff).dot(mean_diff.T) # (dim, dim)
        
        # Determine SW^-1 * SB
        A = np.linalg.inv(SW).dot(SB)
        # Get eigenvalues and eigenvectors of SW^-1 * SB
        eigenvalues, eigenvectors = np.linalg.eig(A) # eigenvector v = [:,i] column vector, transpose for easier calculations
        eigenvectors = eigenvectors.T 
        # sort eigenvalues high to low
        idxs = np.argsort(abs(eigenvalues))[::-1] # descending order
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.linear_discriminants = eigenvectors[0 : self.n_components]

    def transform(self, X):
        return np.dot(X, self.linear_discriminants.T)

if __name__ == "__main__":
    from sklearn import datasets 
    data =  datasets.load_iris()
    X, y = data.data, data.target

    lda = LDA(2)
    lda.fit(X, y)

    X_projected = lda.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)