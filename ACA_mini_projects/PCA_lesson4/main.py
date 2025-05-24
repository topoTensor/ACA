import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

import pca

data=pd.read_csv('Country-data.csv')
p=pca.PCA()
# data=data[['child_mort','exports']]
data=data.loc[:, data.columns != 'country']

std=sklearn.preprocessing.StandardScaler()
data_transformed=std.fit_transform(data)

p.fit(data_transformed)
X_pca=p.transform(dim=2)
print(data.shape, X_pca.shape, "\n", np.var(data_transformed),np.var(X_pca))

fig,axs=plt.subplots(2,1)
axs[0].scatter(x=data['child_mort'],y=data['exports'])
axs[1].scatter(x=X_pca[:,0],y=X_pca[:,1])

plt.show()