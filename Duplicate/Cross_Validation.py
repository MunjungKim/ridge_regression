from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
import numpy as np
from scipy import stats
import Ridge


def cross_(c_x,c_y,alpha):
    ss = ShuffleSplit(n_splits=100, test_size=0.25, random_state=0) 
    Train = []
    Test = []
    for train_index, test_index in ss.split(c_x): 
        Train.append(train_index)
        Test.append(test_index)
    
    
    mse_l = []
    for i in range(len(Train)):
    #test data
        train = Train[i]
        test = Test[i]
    
        X_norm = stats.zscore(c_x[train],axis=0)
        X_std = np.std(c_x[train],axis=0)
        X_mean = np.mean(c_x[train],axis=0)

        beta_ridge, beta_0 = Ridge.solve(X_norm, c_y[train],0.1)
        #test data
        
        # test
        test_norm = (c_x[test]-X_mean)/X_std
        Y_pred = np.matmul(test_norm, beta_ridge)+beta_0
        
        mse = mean_squared_error(c_y[test], Y_pred)
        
        mse_l.append(np.sqrt(mse))
    return np.mean(mse_l)