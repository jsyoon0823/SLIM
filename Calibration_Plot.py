"""
Jinsung Yoon (05/11/2018)
Synthetic Data Generation & Test on the Predictive Model
"""

## Import packages
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

#%%
## Import Function
import sys
sys.path.append('/home/jinsung/Documents/Jinsung/2018_Research/SLIM/MAGGIC')
import Data_Loading

## Parameters for predictive models
year = 1
num_folds = 4
test_fraction = 0.25

## Diverse Predictive models
models = ['randomforest','gbm','xgb','NN']

L = len(models)

## Diverse datasets
datasets = ['maggic']
data_name = datasets[0]

## Output Initialization
X_train, Y_train, X_test, Y_test = Data_Loading.Data_Loading(data_name, test_fraction, 0, year)  

No = len(Y_train) + len(Y_test)

Predict_Out = np.zeros([2*No,L])
Label_Out = np.zeros([2*No,1])
idx = 0


#%%
##### Iteration Start
### For each CV Folds
for k in range(num_folds):
     
    ### For each imputed data matrix      
            
    ### 1. Testing Data Generation
    X_train, Y_train, X_test, Y_test = Data_Loading.Data_Loading(data_name, test_fraction, k, year)  

    # For each model
    for j in range(L):
      
        print('num_fold: ' + str(k+1) + ', Algo_num: ' + str(j+1))
    
        model_name = models[j]            
        
        if model_name == 'NN':    
            model        = MLPClassifier(hidden_layer_sizes=(100,10))

        if model_name == 'randomforest':      
            model        = RandomForestClassifier(n_estimators = 1000)
       
        if model_name == 'gbm':         
            model         = GradientBoostingClassifier()    
        
        if model_name == 'xgb':
            model =   XGBRegressor()

        ## Train and Predict
        if (model_name == 'linearregression' or model_name == 'xgb'):
            model.fit(X_train, Y_train)
            Predict = model.predict(X_test)    
                           
        else:
            model.fit(X_train, Y_train)
            Predict = model.predict_proba(X_test)[:,1]
            
        Predict_Out[idx:idx+len(Predict),j] = Predict            
            
        # Performance                                                     
    Label_Out[idx:idx+len(Predict),0] = Y_test
        
    idx = idx + len(Predict)            
    
Predict_Out = Predict_Out[:idx,:]
Label_Out = Label_Out[:idx,:]

#%% Calibration Plot

#%%  
Division = 20
Output1 = np.zeros([Division,L])   
   
for i in range(L):
    
    Score = Predict_Out[:,i]    
        
    Label = Label_Out
    
    for j in range(Division):
        
        thresh0 = float(1)/Division*j
        thresh1 = float(1)/Division*(j+1)
        
        idx0 = np.where(Score>=thresh0)[0]
        idx1 = np.where(Score < thresh1)[0]
        
        idx = np.intersect1d(idx0,idx1)
        
        Output1[j,i] = np.mean(Label[idx])
                
#%%
### Scatter Plot

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
#fig.tight_layout()

for i in range(4):
    plt.subplot(2, 2, i+1)
    X = np.asarray(range(Division)) * 1/float(Division) + 0.5/float(Division)
    
    plt.scatter(Output1[:,i], X, c='r')
    if i == 0:     
        plt.title('Random Forest')
    elif i == 1:
        plt.title('GBM')
    elif i == 2:
        plt.title('XgBoost')
    elif i == 3:
        plt.title('NN')
        
    plt.xlabel('Predicted Risk')
    plt.ylabel('Observed Risk')
    plt.xlim([0,1])
    plt.ylim([0,1])
    
    plt.grid()
    plt.plot((0,1),(0,1))
    
#%%
plt.savefig('/home/jinsung/Documents/Jinsung/2018_Research/SLIM/MAGGIC/Result/Calibration_Result/Calibration_MAGGIC.pdf')
plt.close(fig)


#%% Risk Distribution
Slot = 1000

X_Risk = np.asarray(range(Slot)) * 1/float(Slot) + 0.5/float(Slot)
Y_Risk = np.zeros([Slot,L])

for i in range(L):
    
    Score = Predict_Out[:,i]
    
    for j in range(Slot):
        
        Y_Risk[j,i] = np.sum(Score < ( j/float(Slot) + 0.5 * 1/float(Slot) ) ) / float(len(Score))
                
#%%
        
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
#fig.tight_layout()

for i in range(4):
    plt.subplot(2, 2, i+1)
    X = X_Risk
    
    plt.plot(X, Y_Risk[:,i], c='r')
    if i == 0:     
        plt.title('Random Forest')
    elif i == 1:
        plt.title('GBM')
    elif i == 2:
        plt.title('XgBoost')
    elif i == 3:
        plt.title('NN')
        
    plt.xlabel('Predicted Risk')
    plt.ylabel('Cumulative Percentile')
    plt.xlim([0,1])
    plt.ylim([0,1])
    
    plt.grid()
    
#%%
plt.savefig('/home/jinsung/Documents/Jinsung/2018_Research/SLIM/MAGGIC/Result/Calibration_Result/Risk_Strata_MAGGIC.pdf')
plt.close(fig)

