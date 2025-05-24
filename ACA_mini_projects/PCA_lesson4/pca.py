import numpy as np
import pandas as pd

class PCA:
    def fit(self,X):
        self.X=np.array(X)
        sigma=np.cov(self.X,rowvar=False)
        self.eigenvals,self.eigenvecs=np.linalg.eig(sigma)

    def transform(self, dim=-1):
        if dim == -1:
            dim=self.X.shape[1]

        s=np.argsort(-self.eigenvals)
        print(self.eigenvecs.shape)

        vecs=self.eigenvecs[s][:dim]
        print(self.X.shape, vecs.shape)
        return self.X@vecs.T