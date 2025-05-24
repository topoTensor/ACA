import numpy as np
import pandas as pd
import sklearn

import mydisc

def get_X_train_labeled(train, labels):
    X_train_labeled=[]
    for l in labels:
        X_train=train[train["Survived"]==l][["Sex", "Pclass"]]
        X_train["Sex"]=X_train["Sex"].apply(lambda sex: 1 if sex=="male" else 0)
        X_train_labeled.append(X_train)
    
    return X_train_labeled

data = pd.read_csv("titanic/train.csv")
(train, test) = sklearn.model_selection.train_test_split(data, test_size=0.2, random_state=42)

# X_train=train[["Sex", "Pclass"]]
# X_train["Sex"]=X_train["Sex"].apply(lambda sex: 1 if sex=="male" else 0)
# test = pd.read_csv("titanic/test.csv")

y_train=train[["Survived"]]
labels=np.unique(y_train)
X_train_labeled=get_X_train_labeled(train, labels)

print("______________________")

lda=mydisc.LDA()
lda.fit(X_train_labeled, y_train, labels)
lda.discriminant_func()