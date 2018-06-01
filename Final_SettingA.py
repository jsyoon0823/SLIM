# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import Lasso
from sklearn import metrics
import sys
sys.path.append('/home/jinsung/Documents/Jinsung/2018_Research/PLOS_Medicine/Logit')
import Logit_Regress

from tqdm import tqdm


def MSE_New_Setting(X_all, P_all, k_No, alpha_val, thresh):
    
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
    
    # Change to Array
    X_all = np.asarray(X_all)
    P_all = np.asarray(P_all)   

    P_all = P_all - np.min(P_all)
    P_all = P_all / (np.max(P_all) + 1e-5)
    P_all = P_all + 1e-8           
    
    # Output initialization
    Z_all = np.zeros([len(P_all),])    
    Output_each = np.zeros([k_No,])      
    Kappa_each = np.zeros([k_No,])       
    Coef = np.zeros([len(X_all[0,:])+1,k_No])
        
    # For each subgroup
    for i in range(k_No):
        # Find the threshold
        if i == 0:
            threshold1 = -0.1
            threshold2 = np.percentile(P_all, thresh[0]*100)
        elif i == 1:
            threshold1 = np.percentile(P_all, thresh[0]*100)
            threshold2 = np.percentile(P_all, thresh[1]*100)
        elif i == 2:
            threshold1 = np.percentile(P_all, thresh[1]*100)
            threshold2 = np.percentile(P_all, thresh[2]*100)
        elif i == 3:
            threshold1 = np.percentile(P_all, thresh[2]*100)
            threshold2 = np.percentile(P_all, thresh[3]*100)
        elif i == 4:
            threshold1 = np.percentile(P_all, thresh[3]*100)
            threshold2 = 1.1
        
        # Train index
                
        
        idx_all1 = np.where(P_all >= threshold1)[0]
        idx_all2 = np.where(P_all < threshold2)[0]
        idx_all = np.intersect1d(idx_all1, idx_all2)
        
        ## Linear Model Training
        model = Lasso(alpha = alpha_val)
        model.fit(X_all[idx_all,:], P_all[idx_all])
                
        # Test prediction
        if (len(idx_all)>0):
            Z_all[idx_all] = model.predict(X_all[idx_all,:])  
                
            Output_each[i] = np.sqrt(metrics.mean_squared_error(P_all[idx_all], Z_all[idx_all])) 
            
        Coef[1:,i] = model.coef_
        Coef[0,i] = model.intercept_    
            

            #%%
        
        model = Lasso(alpha = alpha_val)
        model.fit(X_all[idx_all,:], P_all[idx_all])
        
        Temp_Predict1 = model.predict(X_all[idx_all,:])
        Temp_Perform1 = np.sqrt(metrics.mean_squared_error(P_all[idx_all], Temp_Predict1)) 
        
        ## Logit Model Training
        
        ## Linear Model Training
        W = Logit_Regress.logit_fit(X_all[idx_all,:], P_all[idx_all], alpha_val)
        
        Temp_Predict2 = Logit_Regress.logit_predict(X_all[idx_all,:], W) 
        Temp_Perform2 = np.sqrt(metrics.mean_squared_error(P_all[idx_all], Temp_Predict2))                 
        
        # Test prediction
        if (len(idx_all)>0):
            if (Temp_Perform1 <= Temp_Perform2):
                Z_all[idx_all] = model.predict(X_all[idx_all,:])  
            else:
                Z_all[idx_all] = Logit_Regress.logit_predict(X_all[idx_all,:], W)    
                
            Output_each[i] = np.sqrt(metrics.mean_squared_error(P_all[idx_all], Z_all[idx_all])) 
            Kappa_each[i] = kappa(P_all[idx_all], Z_all[idx_all])  
            
            
            
    Output_all = np.sqrt(metrics.mean_squared_error(P_all, Z_all))             
                        
    return Z_all, Output_all, Output_each, Coef, Kappa_each
    
    
#%%