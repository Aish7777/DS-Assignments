#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


d = pd.read_csv('file:///C:/Users/Aishwarya/Desktop/Datasets/calories_consumed.csv')
d.head()


# In[3]:


x = d['Calories Consumed']
y = d['Weight gained (grams)']


# In[4]:


from numpy.polynomial.polynomial import polyfit

b,m= polyfit(x,y,1)
plt.scatter(x,y)
plt.plot(x,y, '.', color='c')
plt.plot(x, b+m*x, '-', color='purple')
plt.xlabel('Calories Consumed')
plt.ylabel('Weight Gained')


# - Correlation Check

# In[5]:


cor = np.corrcoef(x,y)
print(cor)


# # Model Building

# In[9]:


import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


# In[10]:


model = sm.OLS(y ,x).fit()
predicted_model = model.predict(x)


# In[11]:


model.summary()


#  - Log transformation of X

# In[13]:


lg_x = np.log(d['Calories Consumed'])


# In[14]:


model = sm.OLS(y, lg_x).fit()
predicted_model = model.predict(lg_x)


# In[15]:


model.summary()


# - Log Transformation of Y

# In[16]:


lg_y = np.log(d['Weight gained (grams)'])
model = sm.OLS(lg_y, x).fit()
predicted_model = model.predict(x)


# In[19]:


model.summary()


# - Log Transformation of x and y

# In[22]:


model = sm.OLS(lg_x,lg_y).fit()
predicted_model = model.predict(lg_x)
model.summary()


# - We can use Log transformation of X and Y model as it gives the best r2 value.

# In[ ]:




