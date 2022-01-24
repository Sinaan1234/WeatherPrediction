#!/usr/bin/env python
# coding: utf-8

# In[40]:


#importing the libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cufflinks as cf
cf.go_offline()
from plotly import __version__
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot


# In[41]:


#reading the dataframe
weatherdata = pd.read_excel ("weather.xlsx")


# In[42]:


#Showing the data
weatherdata


# In[43]:


weatherdata.head(10)


# In[44]:


#information about our data
weatherdata.describe()


# In[45]:


#3D graph for Max Temp, Min Temp, Avg Temp
weatherdata[['Max.Temp(°C)','Min.Temp(°C)','Avg.Temp(°C)']].iplot(kind='surface',colorscale = "rdylbu")


# In[46]:


#Graph for Rainfall and Evaporation
weatherdata[['Rainfall (mm)','Evaporation (mm)']].iplot(kind='scatter')


# In[47]:


temp=weatherdata['Avg.Temp(°C)']
evaporation=weatherdata['Evaporation (mm)']


# In[48]:


x = np.array(temp).reshape(-1, 1)
y = np.array(evaporation)


# In[49]:


#train and test the model
from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split( x, y, test_size=0.3, random_state=101 )


# In[50]:


#showing the xtrain
xtrain 


# In[51]:


#showing the xtest
xtest


# In[52]:


#fit the model
from sklearn.linear_model import LinearRegression 
lm = LinearRegression()
lm.fit( xtrain, ytrain )


# In[53]:


#coefficient of the regression line
lm.coef_ 


# In[54]:


#intercept of the regression line
lm.intercept_


# In[55]:


#predicting our ytrain by passing the xtrain values and comparing it with the actual ytrain values
actualValue = ytrain
predictedValue = lm.predict(xtrain) 
xtrain[0], actualValue[0] , predictedValue[0]


# In[58]:


#Mathematical way to find the predicted ytrain values i.e y=mx+c
Y = lm.coef_ * xtrain[0] + lm.intercept_ 
Y


# In[59]:


from sklearn import metrics 


# In[60]:


#finding rmse
np.sqrt(metrics.mean_squared_error(actualValue,predictedValue))


# In[61]:


#plotting the actual values of xtrain and ytrain
plt.scatter(xtrain, actualValue, color='green')
#comparing the same with the predicting ytrain i.e regressive line
prediction = lm.predict(xtrain)
plt.plot(xtrain, prediction , color = 'black') 

plt.title ("Prediction for Training Dataset")
plt.xlabel("Temperature in degree"), plt.ylabel("Evaporation")
plt.show()


# In[35]:


#plotting the actual values of xtest and ytest
plt.scatter(xtest, ytest, color= 'green')

#comparing the same with the predicting ytrain i.e regressive line
plt.plot(xtrain, lm.predict(xtrain), color = 'black')

plt.title ("Training Dataset")
plt.xlabel("Tempertaure in degree"), plt.ylabel("Evaporation")
plt.show()


# In[ ]:





# In[ ]:




