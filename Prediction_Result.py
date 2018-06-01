"""
Jinsung Yoon (05/11/2018)
Synthetic Data Generation & Test on the Predictive Model
"""

## Import packages
import numpy as np
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBRegressor
from sklearn import svm
from lifelines import CoxPHFitter

from tqdm import tqdm

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
models = ['svmlin','coxregression','randomforest','gbm','xgb','NN']

L = len(models)

## Diverse datasets
datasets = ['maggic']
data_name = datasets[0]

## Output Initialization
AUC_ar   = [[0 for x in range(num_folds)] for y in range(L)]
AUPRC_ar   = [[0 for x in range(num_folds)] for y in range(L)]
Cind_ar   = [[0 for x in range(num_folds)] for y in range(L)]


#%% C-index 
def C_index(Y_Val,Pred_Val):
    
    Pred_Val = np.asarray(Pred_Val)
    Y_Val = np.asarray(Y_Val)
    
    idx1 = np.where(Y_Val==1)[0]
    idx0 = np.where(Y_Val==0)[0]
    
    num = 0
    den = 0
    
    for i in tqdm(range(10000)):
        idx_ran = np.random.permutation(len(idx0))[:len(idx1)]
        
        for j in range(len(idx1)):
            if (Pred_Val[idx1[j]] >= Pred_Val[idx0[idx_ran[j]]]):
                num = num+1
            den = den + 1
            
    C_Prob = float(num)/float(den)    
    
    return C_Prob

#%%
##### Iteration Start
### For each CV Folds
for k in range(num_folds):
     
    ### For each imputed data matrix      
            
    ### 1. Testing Data Generation
    X_train, Y_train, X_test, Y_test = Data_Loading.Data_Loading(data_name, test_fraction, k, year)  

    # For Cox Regression
    Train_All = Data_Loading.Data_Loading_Cox(data_name, test_fraction, k, year)       

    # For each model
    for j in range(L):
      
        print('num_fold: ' + str(k+1) + ', Algo_num: ' + str(j+1))
    
        model_name = models[j]            
        
        if model_name == 'coxregression':
            model         = CoxPHFitter()
            
        if model_name == 'svmlin':         
            model        = svm.LinearSVC()
   
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
            
        elif(model_name=='svmlin'): 
            model.fit(X_train, Y_train)
            Predict = model.decision_function(X_test)
           
        elif (model_name == 'coxregression'):
            if data_name == 'maggic':
                model.fit(Train_All, duration_col='days_to_fu', event_col='death_all')               
                Predict = model.predict_partial_hazard(X_test)  
            elif (data_name == 'heart_trans' or 'heart_wait'):
                model.fit(Train_All, duration_col="'Survival'", event_col="'Censor'")               
                Predict = model.predict_partial_hazard(X_test)  
                
        else:
            model.fit(X_train, Y_train)
            Predict = model.predict_proba(X_test)[:,1]       
            
        # Performance    
        AUC_ar[j][k] = metrics.roc_auc_score(Y_test, Predict)
        AUPRC_ar[j][k] = metrics.average_precision_score(Y_test, Predict)
        Cind_ar[j][k] = C_index(Y_test, Predict)
                                               
    
#%%
Output = np.zeros([L,6])

for j in range(L):
    Output[j,0] = round(np.mean(AUC_ar[j]),4)
    Output[j,1] = round((2*np.std(AUC_ar[j])/np.sqrt(num_folds)),4)
    
    Output[j,2] = round(np.mean(AUPRC_ar[j]),4)
    Output[j,3] = round((2*np.std(AUPRC_ar[j])/np.sqrt(num_folds)),4)
    
    Output[j,4] = round(np.mean(Cind_ar[j]),4)
    Output[j,5] = round((2*np.std(Cind_ar[j])/np.sqrt(num_folds)),4)

print(Output)

### Save Results
file_name = '/home/jinsung/Documents/Jinsung/2018_Research/SLIM/MAGGIC/Result/Prediction_Result/'+data_name+'_AUROC_Cind.csv'
np.savetxt(file_name, Output)
