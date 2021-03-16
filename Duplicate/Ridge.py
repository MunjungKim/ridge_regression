from numpy.linalg import inv
import numpy as np


def standard(X):
    N = X.shape[0]
    p = X.shape[1]
    X_std = np.zeros(shape=(N,p))
    X_mean = np.zeros(shape=(1,))
    for i in range(p):
        mean = np.mean(X[:,i])
        #std = np.std(X[:,i])
        X_std[:,i] = X[:,i]-mean
        a = mean.reshape((1,-1))
        X_mean = np.vstack([X_mean, mean])
        
    X_mean = X_mean[1:]    
    return X_std, X_mean




def solve(X,y,lamb):
    N = X.shape[0]
    p = X.shape[1]
    
    X_trans = np.transpose(X)
    save = np.matmul(inv(np.matmul(X_trans,X)+lamb*np.identity(p)),X_trans)
    beta_ridge = np.matmul(save, y)
    beta_0 = np.sum(y)/N

    return beta_ridge, beta_0