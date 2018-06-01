# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split

#%%
def Data_Loading(data_name, test_fraction, k, year):
    
    # MAGGIC
    if data_name == 'maggic':
        File_name = '/home/jinsung/Documents/Jinsung/2018_Research/SLIM/MAGGIC/Data/Maggic.csv'
        Maggic_imputed = pd.read_csv(File_name, sep=',')
        # Remover 'days_to_fu' == NA
        Maggic_imputed = Maggic_imputed.dropna(axis=0, how='any')
        indice = range(len(Maggic_imputed))
        Train, Test, Train_idx, Test_idx  = train_test_split(Maggic_imputed, indice, test_size=test_fraction, random_state=k)
                                                
        Time_hor = 365*year

        Test = Test[(Test['days_to_fu']>Time_hor) | ((Test['days_to_fu']<=Time_hor) & (Test['death_all']==1))]
        N  = Test.shape[0]
        Test['Label_death_horizon']=[0]*N
        Test.loc[((Test['days_to_fu']<=Time_hor) & (Test['death_all']==1)), 'Label_death_horizon'] =1   
        X_test = Test.drop(['death_all', 'days_to_fu', 'Label_death_horizon'], axis=1)
        Y_test = Test['Label_death_horizon']
            
        ### 2. Training Data Generation            
        Train = Train[(Train['days_to_fu']>Time_hor) | ((Train['days_to_fu']<=Time_hor) & (Train['death_all']==1))]
        N  = Train.shape[0]
        Train['Label_death_horizon']=[0]*N
        Train.loc[((Train['days_to_fu']<=Time_hor) & (Train['death_all']==1)), 'Label_death_horizon'] =1   
        X_train = Train.drop(['death_all', 'days_to_fu', 'Label_death_horizon'], axis=1)
        Y_train = Train['Label_death_horizon']            
            
    return X_train, Y_train, X_test, Y_test

#%% 
def Data_Loading_Cox(data_name, test_fraction, k, year):
    
    # MAGGIC
    if data_name == 'maggic':
        File_name = '/home/jinsung/Documents/Jinsung/2018_Research/SLIM/MAGGIC/Data/Maggic.csv'
        Maggic_imputed = pd.read_csv(File_name, sep=',')
        # Remover 'days_to_fu' == NA
        Maggic_imputed = Maggic_imputed.dropna(axis=0, how='any')
        indice = range(len(Maggic_imputed))
        Train, Test, Train_idx, Test_idx  = train_test_split(Maggic_imputed, indice, test_size=test_fraction, random_state=k)
                                                 
    return Train
