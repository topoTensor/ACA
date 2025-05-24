import numpy as np

def CL_matrices(labels, clusters):
    labels=labels[:, np.newaxis]
    clusters=clusters[:,np.newaxis]

    C=(clusters==clusters.T)-np.eye(len(labels))
    L=(labels==labels.T)-np.eye(len(labels))

    return (C,L)

def recall_bcubed(labels, clusters):
    C,L=CL_matrices(labels,clusters)
    corr=(C==L)-np.eye(len(labels))

    corr_and_L=corr*L
    nom=np.sum(corr_and_L, axis=0)
    denom=np.sum(L,axis=0)
    res=nom/denom
    res=res[~np.isnan(res)].mean()
    if np.isnan(res):
        return 1
    return res

def precision_bcubed(labels, clusters):
    C,L=CL_matrices(labels,clusters)
    corr=(C==L)-np.eye(len(labels))

    corr_and_C=corr*C
    nom=np.sum(corr_and_C, axis=0)
    denom=np.sum(C,axis=0)
    res=nom/denom
    res=res[~np.isnan(res)].mean()
    if np.isnan(res):
        return 1
    return res

def my_correctness(i,j, labels, clusters):
    p=clusters[i] == clusters[j]
    q=labels[i] == labels[j]
    if (p and q) or (not p and not q):
        return 1
    else:
        return 0

def my_precision_bcubed(xs,clusters, labels):
    i_accum=0

    for i in range(len(xs)):
        j_accum=0
        num_in_cluster=0
        for j in range(len(xs)):
            if i != j and clusters[i] == clusters[j]:
                num_in_cluster+=1
                j_accum+=my_correctness(i,j,clusters,labels)
        j_accum/=num_in_cluster
        i_accum+=j_accum
    return i_accum/len(xs)


def f1_bcubed(y_test, y_pred):
    precision = precision_bcubed(y_test,y_pred)
    recall = recall_bcubed(y_test,y_pred)
    return 2/(1/precision+1/recall)