import numpy as np 

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None 
        self.mean = None 
    
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance, function needs samples as columns
        cov = np.cov(X.T)

        # eigenvalues, eigenvectors 
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # store first n eigenvectors
        self.components = eigenvectors[0 : self.n_components] # (n_components, dim)
    
    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)  

if __name__ == "__main__":
    from sklearn import datasets 

    data = datasets.load_iris()
    X = data.data
    y = data.target 

    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)