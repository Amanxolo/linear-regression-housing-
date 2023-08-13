#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df=pd.read_csv('USA_Housing.csv')


# In[5]:


df.head(10)


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.columns


# In[9]:


sns.pairplot(df)


# In[10]:


sns.displot(df['Price'])


# In[12]:


df.corr()#correlation wrt each other
#sns.heatmap(df.corr())


# In[13]:


sns.heatmap(df.corr(),annot=True)


# In[14]:


df.columns


# In[25]:


X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[26]:


y=df['Price']#prediction variable


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


#SHIFT+TAB to see documentation


# In[29]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=101) #splits the array into subsets for training and testing


# In[30]:


from sklearn.linear_model import LinearRegression


# In[31]:


lm= LinearRegression()


# In[32]:


lm.fit(X_train,y_train)


# In[33]:


print(lm.intercept_)


# In[34]:


lm.coef_


# In[35]:


X_train.columns


# In[36]:


cdf=pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])


# In[37]:


cdf

# PREDICTIONS
# In[38]:


predictions=lm.predict(X_test)


# In[39]:


predictions


# In[40]:


y_test


# In[41]:


plt.scatter(y_test,predictions)


# In[42]:


sns.displot(y_test-predictions)


# In[43]:


from sklearn import metrics


# In[44]:


metrics.mean_absolute_error(y_test,predictions)


# In[45]:


metrics.mean_squared_error(y_test,predictions)


# In[46]:


np.sqrt(metrics.mean_squared_error(y_test,predictions))


# In[ ]:




