import numpy as np


class LinearRegression():
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_, self.intercept_, _, _= np.linalg.lstsq(X, y)
        

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
    
    
class Ridge():
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        A = np.dot(X.T, X) + self.alpha * np.eye(n_features)
        b = np.dot(X.T, y)
        self.coef_ = np.linalg.solve(A, b)
        self.intercept_ = np.mean(y) - np.dot(np.mean(X, axis=0), self.coef_)

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_