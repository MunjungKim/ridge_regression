from numpy.linalg import inv
import numpy as np



def solve(X,y,lamb):
    N = X.shape[0]
    p = X.shape[1]
    
    X_trans = np.transpose(X)
    save = np.matmul(inv(np.matmul(X_trans,X)+lamb*np.identity(p)),X_trans)
    beta_ridge = np.matmul(save, y)
    beta_0 = np.sum(y)/N

    return beta_ridge, beta_0