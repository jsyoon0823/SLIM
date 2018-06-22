import numpy as np
from sklearn.linear_model import Lasso

#%% Fitting
def logit_fit (X,Y, alpha_val):
    
    # Change to matrix
    X = np.asarray(X)
    Y = np.asarray(Y)    
    
    Feature_No = len(X[0,:])
    
    # Y Change
    Temp_Y1 = float(1)/(Y+1e-8) - 1
    Temp_Y = -np.log(Temp_Y1)    
    
    model = Lasso(alpha_val)
    model.fit(X, Temp_Y)
    
    W = np.zeros([Feature_No+1,1])
    W[1:,0] = model.coef_     
    W[0,0] =  model.intercept_
    
    return W
    
#%% Prediction
    
def logit_predict (X, W):
    No = len(X[:,0])
    
    # X Add
    Ones = np.ones([No,1])
    Temp_X = np.concatenate((Ones, X),1)    
    
    Temp_Y = np.matmul(Temp_X,W)
    
    Y1 = np.exp(-Temp_Y)
    
    Y_Out = float(1) / (1+Y1+1e-8)
    
    return np.reshape(Y_Out, [No,])
    