#!/usr/bin/env python
# coding: utf-8

# 1.What is the use of matplotlib package in Data Science?

# Matplotlib is an essential Python library utilised extensively in data science for the construction of a variety of visualisations. It plays a crucial role in data exploration and analysis by allowing the creation of static, animated, and interactive charts and graphs. These visualisations include line plots, scatter plots, bar charts, histograms, and pie charts, among others, and serve as potent instruments for discovering insights and effectively communicating data-driven findings. Matplotlib facilitates exploratory data analysis (EDA) during the initial phases of data analysis by facilitating the examination of data distributions, patterns, anomalies, and trends. Supporting ROC curves, confusion matrices, and learning curves, it can also be utilised to evaluate model performance metrics in machine learning and statistical analyses. Matplotlib is indispensable for displaying temporal patterns, trends, and seasonality in time series data. Additionally, it is adaptable to geospatial data visualisation, accommodating maps, heatmaps, and other geographical visualisations when integrated with Basemap or Cartopy libraries. The library's extensive customization options, which include colours, labels, inscriptions, and typefaces, ensure that visualisations adhere to particular project specifications or brand aesthetics. Furthermore, Matplotlib excels in generating publication-ready plots and figures, facilitating export in various file formats. Additionally, it can be combined with other libraries to produce interactive visualisations for web applications and presentations. Matplotlib therefore plays a crucial role in data storytelling, allowing data scientists to create compelling narratives by employing visualisations to support findings and communicate complex information. Its integration with other data analysis tools such as NumPy, Pandas, and SciPy facilitates the importation, analysis, and visualisation of all data. Matplotlib is a flexible and indispensable tool for data professionals working with Python, offering extensive charting capabilities to investigate, analyse, and effectively communicate data.

# 2. downloading and loading the data from github

# In[13]:


import pandas as pd

file_path = "scottish_hills.csv"

df = pd.read_csv("scottish_hills.csv")


# In[14]:


df_head = df.head(10)

# Printing DataFrame
print(df_head)


# 3. Sort the heights of the mountains in the descending order. (Hint: You can sort
# the values in the DataFrame using the sort_values method.

# In[16]:


# Sort the DataFrame by the "Height" column in descending order
df_sorted = df.sort_values(by="Height", ascending=False)

# Display the sorted DataFrame
print(df_sorted)


# 4. Draw a scatter plot “latitude” vs “height”. Clearly label the axes for full
# credit.

# In[24]:


import matplotlib.pyplot as plt


# Creating a scatter plot
plt.scatter(df['Latitude'], df['Height'])

#  labels for the x and y axes
plt.xlabel('Latitude')
plt.ylabel('Height (meters)')

# the title of the plot
plt.title('Scatter Plot of Latitude vs. Height')

# Displaying the plot
plt.show()


# 2a. What is machine learning?

# Machine learning, a branch of artificial intelligence (AI), is dedicated to the advancement of algorithms and statistical models. These tools empower computer systems to enhance their performance in a given activity by leveraging data-driven learning, without the need for explicit programming. Machine learning is a computational approach that enables computers to autonomously acquire knowledge and provide predictions or determinations by discerning patterns and extracting information from data.
# 

# In[32]:


from scipy import stats

# Defining the data 
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

# Performing linear regression analysis
slope, intercept, r, p, std_err = stats.linregress(x, y)

# Defining a function for the linear model
def myfunc(x):
    return slope * x + intercept

# Calculating the speed for x = 10 using the linear model
speed = myfunc(10)

# Printing the result
print("Predicted speed for x = 10:", speed)


# c.Predict the speed of an 8 year old car.

# In[33]:


# Predicting the speed of an 8-year-old car
car_age = 8
predicted_speed = myfunc(car_age)

# Printing the predicted speed
print("Predicted speed for an 8-year-old car:", predicted_speed)


# In[ ]:




