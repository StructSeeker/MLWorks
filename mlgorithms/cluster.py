import numpy as np

# The API is designed to be similar to scikit-learn

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, iter = None, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.iter = iter
        self.max_iter = max_iter
        
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        self.cluster_centers_ = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        iters = min(self.iter, self.max_iter) 
        for _ in range(iters):
            distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
            self.labels_ = np.argmin(distances, axis=1)
            new_centers = np.array([X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.linalg.norm(self.cluster_centers_ - new_centers) < self.tol:
                break
            self.cluster_centers_ = new_centers
        self.inertia_ = np.sum((X - self.cluster_centers_[self.labels_]) ** 2)

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
