#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv('ipl.csv')


# In[2]:


df.head()


# In[3]:


columns_to_remove = ['venue','batsman','bowler','striker','non-striker','mid']
df.drop(columns_to_remove,axis=1,inplace=True)
df['bat_team'].unique()


# In[4]:


## keeping consistent teams
consistent_team = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
       'Mumbai Indians','Kings XI Punjab',
       'Royal Challengers Bangalore', 'Delhi Daredevils',
        'Sunrisers Hyderabad']
df = df[(df['bat_team'].isin(consistent_team)) & (df['bowl_team'].isin(consistent_team))]
df.head()


# In[5]:


df = df[df['overs']>=5]


# In[6]:


from datetime import datetime
df['date'] = df['date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))


# In[7]:


encoded_df = pd.get_dummies(data=df,columns=['bat_team','bowl_team'])
encoded_df.head()


# In[8]:


x_train = encoded_df.drop(labels='total',axis=1)[encoded_df['date'].dt.year <= 2016]
x_test = encoded_df.drop(labels='total',axis=1)[encoded_df['date'].dt.year >= 2017]


# In[9]:


y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values


# In[10]:


x_train.drop(labels='date',axis=1,inplace=True)
x_test.drop(labels='date',axis=1,inplace=True)


# # Linear Regression

# In[11]:


from sklearn.linear_model import LinearRegression
lr_regression = LinearRegression()
lr_regression.fit(x_train,y_train)


# In[12]:


y_pred_lr = lr_regression.predict(x_test)


# In[13]:


import seaborn as sns
sns.distplot(y_test-y_pred_lr)


# In[14]:


from sklearn import metrics
import numpy as np
print('MAE: ',metrics.mean_absolute_error(y_test,y_pred_lr))
print('MSE: ',metrics.mean_squared_error(y_test,y_pred_lr))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_pred_lr)))


# In[15]:


import pickle
file = open('first-innings-score-lr-model.pkl','wb')
pickle.dump(lr_regression,file)


# # Ridge Regression

# In[16]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[17]:


ridge = Ridge()
parameters = {'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
ridge_regressor = GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(x_train,y_train)


# In[18]:


y_pred_rg = ridge_regressor.predict(x_test)


# In[19]:


ridge_regressor.best_params_


# In[20]:


ridge_regressor.best_score_


# In[21]:


sns.distplot(y_test-y_pred_rg)


# In[22]:


from sklearn import metrics
import numpy as np
print('MAE: ',metrics.mean_absolute_error(y_test,y_pred_rg))
print('MSE: ',metrics.mean_squared_error(y_test,y_pred_rg))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_pred_rg)))


# In[ ]:





# # Lasso regression

# In[23]:


from sklearn.linear_model import Lasso
lasso = Lasso()
regression_lasso = GridSearchCV(lasso,parameters,scoring = 'neg_mean_squared_error',cv=5)
regression_lasso.fit(x_train,y_train)


# In[24]:


y_pred_lasso = regression_lasso.predict(x_test)


# In[25]:


sns.distplot(y_test-y_pred_lasso)


# In[26]:


from sklearn import metrics
import numpy as np
print('MAE: ',metrics.mean_absolute_error(y_test,y_pred_lasso))
print('MSE: ',metrics.mean_squared_error(y_test,y_pred_lasso))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_pred_lasso)))


# In[27]:


file = open('first-innings-score-ls-model.pkl','wb')
pickle.dump(regression_lasso,file)


# In[ ]:




