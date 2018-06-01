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
import csv

## Import Function
import sys
sys.path.append('/home/jinsung/Documents/Jinsung/2018_Research/SLIM/MAGGIC')
import Data_Loading
import Final_SettingA

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

##### Data
X_train, Y_train, X_test, Y_test = Data_Loading.Data_Loading(data_name, test_fraction, 0, year)  

X_all = pd.concat([X_train,X_test])
Y_all = pd.concat([Y_train,Y_test])

Feature_No = len(X_all.columns)
Feature_Name = X_all.columns

#%%
Output   = [[0 for x in range(k_No+1)] for y in range(Feature_No)]

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
                            
    ### 1. Testing Data Generation              
            
    ## Predict
    if (model_name == 'linearregression' or model_name == 'xgb'):
        P_all = model.predict(X_all)            
                
    else:
        P_all = model.predict_proba(X_all)[:,1]
        
    P_all = np.asarray(P_all)
    
    _, _, _, Coef = Final_SettingA.MSE_New_Setting(X_all, P_all, k_No, alpha_val, thresh)  
        
    Ori_Coef = Coef.copy()
    Ori_Coef = Ori_Coef[1:,:]
    
    for it in range(k_No):
        
        for itt in range(Feature_No):
            
            Output[itt][0] = Feature_Name[itt]
            
            Output[itt][it+1] = Ori_Coef[itt,it]
    
    file_name = '/home/jinsung/Documents/Jinsung/2018_Research/SLIM/MAGGIC/Result/Coef_Result/' + data_name + '_model_' + model_name + '_Coef.csv'
    with open(file_name, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(Output)
                
      