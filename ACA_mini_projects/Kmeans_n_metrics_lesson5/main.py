import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import metrics
import kmeans

centers = [[1, 1], [-1, -1], [1, -1]]
X_blobs, y_blobs = make_blobs(
    n_samples=1000, centers=centers, cluster_std=0.3, random_state=42
)

# centers = [[1, 1], [-1, -1], [1, -1]]
# X_blobs, y_blobs = make_moons(n_samples=1000, noise=0.08, random_state=42)

# knn=KMeans(n_clusters=2)
# labels=knn.fit_predict(Xs, ys)

model=kmeans.KMeans(k=3,method='k-means++')
model.fit(X_blobs)
centroids,clusters=model.predict(X_blobs)

# my_precision=metrics.my_precision_bcubed(Xs,ys, labels)
# print(my_precision)

precision=metrics.precision_bcubed(y_blobs, clusters)
recall=metrics.recall_bcubed(y_blobs, clusters)
f1=metrics.f1_bcubed(y_blobs, clusters)

print(precision, recall,f1)

# plt.figure(figsize=(15,7))
# plt.scatter(Xs[:,0], Xs[:,1],c=ys)

# plt.figure(figsize=(15,7))
# plt.scatter(Xs[:,0], Xs[:,1],c=labels)

# plt.show()

figs, axs = plt.subplots(1,2,figsize=(15,7))

axs[0].scatter(X_blobs[:,0], X_blobs[:,1],c=y_blobs)

axs[1].scatter(X_blobs[:,0], X_blobs[:,1],c=clusters)

axs[1].scatter(centroids[:,0], centroids[:,1], marker='*',s=100, c='red')

plt.show()