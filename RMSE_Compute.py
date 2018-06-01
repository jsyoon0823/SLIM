"""
Jinsung Yoon (04/05/2018)
Synthetic Data Generation & Test on the Predictive Model
"""

## Import packages
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBRegressor
from tqdm import tqdm

## Import Function
import sys
sys.path.append('/home/jinsung/Documents/Jinsung/2018_Research/SLIM/MAGGIC')
import Data_Loading
import Final_SettingA


#%%
def kappa (Order_A, Order_B):
    L = len(Order_A)
    
    nom = 0    
    
    for i in tqdm(range(L)):
        for j in range(L):
            Temp_A = Order_A[i] - Order_A[j]
            Temp_B = Order_B[i] - Order_B[j]
            
            Temp = Temp_A * Temp_B
            
            if (Temp >= 0):
                nom = nom + 1
                
    kappa_stat = round(float(nom - L)  / (L*(L-1)),6)
    
    return kappa_stat
    
#%%

## Parameters for predictive models
year = 1
num_folds = 4
test_fraction = 0.25
alpha_val = 0.001

k_No = 5
thresh = [0.2,0.4,0.6,0.8]

## Diverse Predictive models
models = ['randomforest','gbm','xgb','NN']

L = len(models)

## Diverse datasets
datasets = ['maggic']
data_name = datasets[0]

## Output Initialization
MSE_Each_ar   = np.zeros([L, k_No])
Kappa_Each_ar   = np.zeros([L, k_No])
Division   = np.zeros([L, k_No])
MSE_all_ar   = np.zeros([L,])
Kappa_all_ar = np.zeros([L,])

##### Data
X_train, Y_train, X_test, Y_test = Data_Loading.Data_Loading(data_name, test_fraction, 0, year)  

X_all = pd.concat([X_train,X_test])
Y_all = pd.concat([Y_train,Y_test])

############ Each Method
for j in range(L):
     
    ### For each model
    print('method: ' + str(j+1))              
   
    model_name = models[j]            
        
    if model_name == 'linearregression':
        model         = LinearRegression()
   
    if model_name == 'NN':    
        model        = MLPClassifier(hidden_layer_sizes=(100,20))

    if model_name == 'randomforest':      
        model        = RandomForestClassifier(n_estimators = 1000)
       
    if model_name == 'gbm':         
        model         = GradientBoostingClassifier()    
        
    if model_name == 'xgb':
        model =   XGBRegressor()

    ## Train
    model.fit(X_all, Y_all)          
                            
    ### 4-cross validation
                
    if (model_name == 'linearregression' or model_name == 'xgb'):
        P_all = model.predict(X_all)             
                
    else:
        P_all = model.predict_proba(X_all)[:,1]
        
    P_all = np.asarray(P_all)
        
    Z_all, MSE_all_ar[j], MSE_Each_ar[j,:], Division[j,:], Kappa_Each_ar[j,:] = Final_SettingA.MSE_Split_Setting(X_all, P_all, k_No, alpha_val, thresh)  
   
    Z_all, MSE_all_ar[j], MSE_Each_ar[j,:], _, Kappa_Each_ar[j,:] = Final_SettingA.MSE_New_Setting(X_all, P_all, k_No, alpha_val, thresh)  

#    Kappa_all_ar[j] = kappa(P_all,Z_all)
    
                        
### Find the best model
### Save Results
file_name = '/home/jinsung/Documents/Jinsung/2018_Research/SLIM/MAGGIC/Result/RMSE_Result/' + data_name + '_Each_MSE.csv'
np.savetxt(file_name, MSE_Each_ar)

file_name = '/home/jinsung/Documents/Jinsung/2018_Research/SLIM/MAGGIC/Result/RMSE_Result/' + data_name + '_All_MSE.csv'
np.savetxt(file_name, MSE_all_ar)
    
file_name = '/home/jinsung/Documents/Jinsung/2018_Research/SLIM/MAGGIC/Result/RMSE_Result/' + data_name + '_Kappa.csv'
np.savetxt(file_name, Kappa_all_ar)

file_name = '/home/jinsung/Documents/Jinsung/2018_Research/SLIM/MAGGIC/Result/RMSE_Result/' + data_name + '_Each_Kappa.csv'
np.savetxt(file_name, Kappa_Each_ar)

file_name = '/home/jinsung/Documents/Jinsung/2018_Research/SLIM/MAGGIC/Result/RMSE_Result/' + data_name + '_Each_Division.csv'
np.savetxt(file_name, Kappa_Each_ar)
#%%

