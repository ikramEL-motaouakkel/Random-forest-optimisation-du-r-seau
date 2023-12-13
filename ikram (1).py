#!/usr/bin/env python
# coding: utf-8

# In[177]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[178]:


data_train=pd.read_csv('desktop\ML Training Dataset - Nemo Handy.csv')


# In[179]:


data_train


# In[180]:


data_train.shape


# In[181]:


data_train.head()


# In[182]:


data_train.tail()


# In[126]:


#data_train = data_train.values


# In[184]:


#X_train = data_train[:, :-1]
#y_train = data_train[:, -1]
X_train=data_train[['D','AZ','ELEV','TILT','CLASS','FQ','OBS']]
Y_train=data_train['RSRP']


# In[185]:


X_train.isnull().sum()
#valeurs_manquantes = np.isnan(X_train)
#somme_valeurs_manquantes_par_colonne = np.sum(valeurs_manquantes, axis=0)
#somme_valeurs_manquantes_par_colonne


# In[186]:


Y_train.isnull().sum()


# In[307]:


from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=7,random_state=0)
regressor.fit(X_train,Y_train)


# In[298]:


data_test=pd.read_csv('desktop\SUT_P11_L7_131031_B00010_CID22_TILT_2022b.csv')


# In[290]:


data_test


# In[243]:


#data_test = data_test.values


# In[308]:


#X_test = data_train[:, :-1]
#y_test = data_train[:, -1]
Y_test=data_test['RSRP']
X_test=data_test[['D','AZ','ELEV','TILT','CLASS','FQ','OBS']]


# In[309]:


Y_pred = regressor.predict(X_test)


# In[293]:


ik=data_test["RSRP"].iloc[1]
ik
#type(Y_test[1])
#data_test.dtype



# In[258]:


X_test.isnull().sum()
#valeurs_manquantes = np.isnan(X_test)
#somme_valeurs_manquantes_par_colonne = np.sum(valeurs_manquantes, axis=0)
#somme_valeurs_manquantes_par_colonne


# In[294]:


#valeurs_manquantes = np.isnan(Y_test)
#somme_valeurs_manquantes_par_colonne = np.sum(valeurs_manquantes, axis=0)
#somme_valeurs_manquantes_par_colonne
Y_test.isnull().sum()


# In[295]:


print(len(Y_test), len(Y_pred))


# In[310]:


from sklearn.metrics import mean_squared_error
MSEvalue=mean_squared_error(Y_test,Y_pred,multioutput="uniform_average")
MSEvalue
#from sklearn.metrics import mean_absolute_error
#MSEvalue=mean_absolute_error(Y_test,y_pred,multioutput="uniform_average")


# In[313]:


RMSE=np.sqrt(MSEvalue)
RMSE


# In[311]:


from sklearn.metrics import r2_score
r2 = r2_score(Y_test, Y_pred)
r2

