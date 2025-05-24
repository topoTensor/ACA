import numpy as np

class KMeans:
    def __init__(self, k, method='random', max_iter=300):
        self.k = k
        self.method = method
        self.max_iter = max_iter
        pass
    
    def init_centers(self, X):
        if self.method == 'random':
            return X[np.random.choice(X.shape[0], self.k, replace=False)]
        if self.method == 'k-means++':
            p1= X[np.random.choice(X.shape[0], 1, replace=False)]
            p1_dist=np.sqrt(np.sum(np.square(X-np.asarray(p1)),axis=1))
            farthest_p=p1_dist.argmax()
            print(farthest_p)
            pass
    
    def fit(self, X):
        self.centroids = self.init_centers(X)
        for i in range(self.max_iter):
            self.clusters = self.expectation(X, self.centroids)
            new_centroids = self.maximization(X)
            if (new_centroids == self.centroids).all():
                break
            self.centroids = new_centroids
            
    def expectation(self, X, centroids):
        centroids=centroids[:, np.newaxis]
        #argmin returns the index of minimal value in array

        # distance between points and centroids
        # centroid1, centroid2, centroid3
        #x1   d11        d12       d13
        #x2
        #...
        clusters=np.sqrt(np.sum(np.square(centroids-X), axis=-1)).argmin(axis=0)
        return clusters

    def maximization(self, X):
        new_centroids=np.zeros(self.centroids.shape)
        for i in range(self.k):
            new_centroids[i]=X[self.clusters==i].mean(axis=0)
        return new_centroids
        
    def predict(self, X):
        return (self.centroids,self.expectation(X, self.centroids))
    
    def predict_proba(self, X):
        # ideas ?
        return 