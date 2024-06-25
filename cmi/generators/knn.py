import numpy as np
import torch
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import KDTree


# knn based permutation as described in K-Nearest-Neighbor Local Sampling Based
# Conditional Independence Testing Shuai et al
# https://openreview.net/forum?id=luyXPdkNSN&referrer=%5Bthe%20profile%20of%20Christina%20Dan%20Wang%5D(%2Fprofile%3Fid%3D~Christina_Dan_Wang1)
def sample_q_knn(x, y, z):
    """
    sample p(x,y)p(y|z) using k-th nearest neighbour
    """
    # 1-NN based sampling from p(y|z) with DV representation in classifier based KL estimator
    # x, y, z - 2dim arrays
    # with a even number of observations
    if x.shape[0] % 2:
        x = x[:-1]
        y = y[:-1]
        z = z[:-1]
    if len(x.shape) == 1:
        x=x.reshape(-1, 1)
    if len(y.shape) == 1:
        y=y.reshape(-1, 1)
    if len(z.shape) == 1:
        z=z.reshape(-1, 1)
        
    # shuffle samples
    data_perm = np.random.permutation(x.shape[0])
    x = x[data_perm]
    y = y[data_perm]
    z = z[data_perm]
    # split into v2 and v1
    v1_ix = np.array([True] * (x.shape[0] // 2) + [False] * (x.shape[0] // 2))
    v2_ix = ~v1_ix



    # make v1
    x1 = x[v1_ix].copy()
    y1 = y[v1_ix].copy()
    z1 = z[v1_ix].copy()
    v1 = np.column_stack([x1, y1, z1])
    # make v2
    x2 = x[v2_ix].copy()
    y2 = y[v2_ix].copy()
    z2 = z[v2_ix].copy()
    v2 = np.column_stack([x2, y2, z2])

    # generate V' 
    # calculate nearest observation from v1 for each v2
    v1_tree = KDTree(z1)
    distances, indices = v1_tree.query(z2, k=8)
    y_p = y[v1_ix][indices[:, 7]]

    # x2 y_p z2 make v'
    x_p = x2.copy()
    z_p = z2.copy()
    vp = np.column_stack([x_p, y_p, z_p])
    return v2, vp


