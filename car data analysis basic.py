#!/usr/bin/env python
# coding: utf-8

# In[93]:


#importing libarary
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# In[94]:


#loading uploaded csv
car = pd.read_csv('car.csv')

# Display the first few rows of the dataframe
car.head()


# In[95]:


car.dtypes #checking data types


# In[96]:


#checking columns
car.columns


# In[97]:


#analysis of dataset
car.describe(include='all')


# In[98]:


#showing in histrogram
car.hist(figsize=(20,30))


# In[99]:


sns.pairplot(car)


# In[ ]:


#relaltionship betwwen categorical and continous variable
sns.boxplot (x="title", y="year", data=car)


# In[ ]:


#Renaming the columns name
car=car.rename(columns={"mileage":"MILEAGE", "title_status":"title"})


# In[ ]:


car.head()


# In[101]:


#counting rows and coloums
car.shape


# In[102]:


#checking duplicate data
duplicate_row = car [car.duplicated()]
print("duplicate rows are: " ,duplicate_row.shape)


# In[103]:


#counting row to remove duplicate value
car.count()


# In[104]:


#drop duplicates
car = car.drop_duplicates()
car.head()


# In[105]:


#finding null values
print(car.isnull().sum())


# In[106]:


#drop  missing values
car =car.dropna()
car.count()


# In[107]:


#checking outliers
sns.boxplot(x=car['price'])


# In[109]:


#plottting a hsitrogram number of car per brand
car.brand.value_counts().nlargest(40).plot(kind='bar',figsize=(10,5))
plt.title('number of car per brand')
plt.ylabel('number of car')
plt.xlabel('brand name')


# In[ ]:





# In[ ]:





# In[ ]:




