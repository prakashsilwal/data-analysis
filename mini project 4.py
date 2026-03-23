#!/usr/bin/env python
# coding: utf-8

"""
Mini Project 4: Scottish Hills Data Analysis
This script demonstrates basic data manipulation and visualization techniques
using Scottish hills data.
"""

# ==============================================================================
# QUESTION 1: What is the use of matplotlib package in Data Science?
# ==============================================================================

# ANSWER:
# Matplotlib is an essential Python library utilized extensively in data science
# for creating various visualizations. It plays a crucial role in data exploration
# and analysis by enabling the creation of static, animated, and interactive charts
# and graphs. These visualizations include:
# - Line plots, scatter plots, bar charts, histograms, and pie charts
# - Exploratory Data Analysis (EDA) visualizations for examining distributions,
#   patterns, anomalies, and trends
# - Machine learning metrics visualization (ROC curves, confusion matrices,
#   learning curves)
# - Time series analysis (temporal patterns, trends, seasonality)
# - Geospatial visualizations (maps, heatmaps) when integrated with Basemap
#   or Cartopy
#
# Key benefits:
# - Extensive customization options (colors, labels, fonts)
# - Publication-ready plots with export to various file formats
# - Integration with NumPy, Pandas, and SciPy
# - Essential tool for data storytelling and communicating complex information

# ==============================================================================
# QUESTION 2: Downloading and loading data from GitHub
# ==============================================================================

# Import required library for data manipulation
import pandas as pd

# Load the Scottish hills dataset from CSV file
print("Loading Scottish hills data...")
file_path = "scottish_hills.csv"
df = pd.read_csv(file_path)

# Display the first 10 rows to understand the dataset structure
print("\nFirst 10 rows of the dataset:")
df_head = df.head(10)
print(df_head)


# ==============================================================================
# QUESTION 3: Sort the heights of mountains in descending order
# ==============================================================================

# Sort the DataFrame by the "Height" column in descending order
# This shows the tallest mountains first
df_sorted = df.sort_values(by="Height", ascending=False)

print("\n" + "="*60)
print("Scottish Hills sorted by height (tallest first):")
print("="*60)
print(df_sorted)


# ==============================================================================
# QUESTION 4: Draw a scatter plot "latitude" vs "height"
# ==============================================================================

import matplotlib.pyplot as plt

# Create a scatter plot to visualize the relationship between latitude and height
# This helps identify any geographic patterns in mountain heights
plt.figure(figsize=(10, 6))
plt.scatter(df['Latitude'], df['Height'], alpha=0.6, edgecolors='k', linewidth=0.5)

# Add clear labels for the axes
plt.xlabel('Latitude (degrees)', fontsize=12)
plt.ylabel('Height (meters)', fontsize=12)

# Add a descriptive title
plt.title('Scatter Plot: Latitude vs. Height of Scottish Hills', fontsize=14, fontweight='bold')

# Add a grid for better readability
plt.grid(True, alpha=0.3)

# Display the plot
plt.tight_layout()
plt.savefig('latitude_vs_height.png', dpi=300, bbox_inches='tight')
print("\nScatter plot saved as 'latitude_vs_height.png'")
plt.show()


# ==============================================================================
# QUESTION 2a: What is machine learning?
# ==============================================================================

# ANSWER:
# Machine learning is a branch of artificial intelligence (AI) dedicated to
# developing algorithms and statistical models that enable computer systems to
# improve their performance on specific tasks through data-driven learning,
# without explicit programming. It is a computational approach that allows
# computers to autonomously:
# - Learn from data
# - Identify patterns
# - Make predictions or decisions
# - Extract meaningful information
#
# Key characteristics:
# - Uses algorithms that learn from experience
# - Improves performance over time with more data
# - Can handle complex problems without explicit rules
# - Applications include classification, regression, clustering, and more


# ==============================================================================
# LINEAR REGRESSION EXAMPLE: Predicting Speed Based on Age
# ==============================================================================

from scipy import stats

# Define sample data
# x represents car age (years)
# y represents car speed (arbitrary units)
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

# Perform linear regression analysis
# This creates a linear model to predict speed based on age
slope, intercept, r, p, std_err = stats.linregress(x, y)

# Define a function for the linear model
# Model equation: y = slope * x + intercept
def myfunc(x):
    """
    Predict speed based on car age using linear regression.

    Args:
        x (float): Car age in years

    Returns:
        float: Predicted speed
    """
    return slope * x + intercept

# Calculate the predicted speed for a 10-year-old car
speed_10 = myfunc(10)
print("\n" + "="*60)
print("LINEAR REGRESSION PREDICTIONS")
print("="*60)
print(f"Predicted speed for a 10-year-old car: {speed_10:.2f}")

# Display the model parameters
print(f"\nModel equation: Speed = {slope:.4f} * Age + {intercept:.4f}")
print(f"R-squared value: {r**2:.4f}")
print(f"P-value: {p:.4e}")


# ==============================================================================
# QUESTION c: Predict the speed of an 8-year-old car
# ==============================================================================

# Predict the speed for an 8-year-old car using our linear model
car_age = 8
predicted_speed_8 = myfunc(car_age)

print("\n" + "="*60)
print(f"Predicted speed for an {car_age}-year-old car: {predicted_speed_8:.2f}")
print("="*60)

# Create a visualization of the linear regression
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', alpha=0.6, s=100, edgecolors='k', label='Actual data')

# Plot the regression line
x_line = range(0, 20)
y_line = [myfunc(i) for i in x_line]
plt.plot(x_line, y_line, color='red', linewidth=2, label='Regression line')

# Highlight the predictions
plt.scatter([10], [speed_10], color='green', s=150, marker='^',
            edgecolors='k', linewidth=2, label='Predicted (age=10)', zorder=5)
plt.scatter([8], [predicted_speed_8], color='orange', s=150, marker='s',
            edgecolors='k', linewidth=2, label='Predicted (age=8)', zorder=5)

plt.xlabel('Car Age (years)', fontsize=12)
plt.ylabel('Speed', fontsize=12)
plt.title('Linear Regression: Car Age vs Speed', fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('car_age_speed_regression.png', dpi=300, bbox_inches='tight')
print("\nRegression plot saved as 'car_age_speed_regression.png'")
plt.show()

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)




