#!/usr/bin/env python
# coding: utf-8

"""
Car Data Analysis Script
This script performs basic exploratory data analysis on car sales data.
"""

# Import necessary libraries for data analysis and visualization
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import seaborn as sns  # For statistical visualizations
import matplotlib.pyplot as plt  # For creating plots

# Configure matplotlib for display
# Note: Removed get_ipython() call for standalone execution
# plt.ion()  # Uncomment if you want interactive mode
sns.set(color_codes=True)


def load_and_explore_data(file_path='car.csv'):
    """
    Load and perform initial exploration of car sales data.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded car data
    """
    # Load the car sales data from CSV file
    print("Loading car sales data...")
    car = pd.read_csv(file_path)

    # Display the first few rows to understand the dataset structure
    print("\nFirst few rows of the dataset:")
    print(car.head())

    return car


def analyze_data_types(car):
    """Display data types of each column."""
    print("\n" + "="*50)
    print("DATA TYPES")
    print("="*50)
    print(car.dtypes)


def display_columns(car):
    """Display all column names in the dataset."""
    print("\n" + "="*50)
    print("COLUMN NAMES")
    print("="*50)
    print(car.columns.tolist())


def statistical_summary(car):
    """Display comprehensive statistical summary."""
    print("\n" + "="*50)
    print("STATISTICAL SUMMARY")
    print("="*50)
    print(car.describe(include='all'))


def visualize_distributions(car):
    """Create histograms for all numerical variables."""
    print("\n" + "="*50)
    print("Creating histograms for numerical variables...")
    print("="*50)
    car.hist(figsize=(20, 30))
    plt.tight_layout()
    plt.savefig('car_histograms.png', dpi=300, bbox_inches='tight')
    print("Histograms saved as 'car_histograms.png'")
    plt.show()


def create_pairplot(car):
    """Create pairplot to show relationships between variables."""
    print("\n" + "="*50)
    print("Creating pairplot...")
    print("="*50)
    sns.pairplot(car)
    plt.savefig('car_pairplot.png', dpi=300, bbox_inches='tight')
    print("Pairplot saved as 'car_pairplot.png'")
    plt.show()


def analyze_categorical_continuous(car):
    """Analyze relationship between categorical and continuous variables."""
    print("\n" + "="*50)
    print("Analyzing relationship between title status and year...")
    print("="*50)

    # Check if required columns exist
    if 'title' in car.columns and 'year' in car.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="title", y="year", data=car)
        plt.title('Relationship between Title Status and Year')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('title_year_relationship.png', dpi=300, bbox_inches='tight')
        print("Boxplot saved as 'title_year_relationship.png'")
        plt.show()
    else:
        print("Required columns not found. Skipping this analysis.")


def rename_columns(car):
    """Rename columns for better readability."""
    print("\n" + "="*50)
    print("Renaming columns...")
    print("="*50)

    # Rename columns if they exist
    rename_dict = {}
    if "mileage" in car.columns:
        rename_dict["mileage"] = "MILEAGE"
    if "title_status" in car.columns:
        rename_dict["title_status"] = "title"

    if rename_dict:
        car = car.rename(columns=rename_dict)
        print(f"Renamed columns: {rename_dict}")
    else:
        print("No columns to rename.")

    print("\nUpdated column names:")
    print(car.columns.tolist())

    return car


def check_shape(car):
    """Display the shape of the dataset."""
    print("\n" + "="*50)
    print("DATASET SHAPE")
    print("="*50)
    print(f"Rows: {car.shape[0]}, Columns: {car.shape[1]}")
    return car.shape


def check_duplicates(car):
    """Check for and display duplicate rows."""
    print("\n" + "="*50)
    print("DUPLICATE CHECK")
    print("="*50)
    duplicate_row = car[car.duplicated()]
    print(f"Number of duplicate rows: {duplicate_row.shape[0]}")

    # Display row count before removing duplicates
    print(f"\nRow count before removing duplicates: {car.count().iloc[0]}")

    return duplicate_row


def remove_duplicates(car):
    """Remove duplicate rows from the dataset."""
    print("\n" + "="*50)
    print("REMOVING DUPLICATES")
    print("="*50)

    initial_shape = car.shape[0]
    car = car.drop_duplicates()
    final_shape = car.shape[0]

    print(f"Removed {initial_shape - final_shape} duplicate rows")
    print("\nFirst few rows after removing duplicates:")
    print(car.head())

    return car


def check_missing_values(car):
    """Find and display missing values."""
    print("\n" + "="*50)
    print("MISSING VALUES CHECK")
    print("="*50)
    print(car.isnull().sum())


def handle_missing_values(car):
    """Drop rows with missing values."""
    print("\n" + "="*50)
    print("HANDLING MISSING VALUES")
    print("="*50)

    initial_shape = car.shape[0]
    car = car.dropna()
    final_shape = car.shape[0]

    print(f"Removed {initial_shape - final_shape} rows with missing values")
    print(f"\nRemaining rows: {car.count().iloc[0]}")

    return car


def check_outliers(car):
    """Check for outliers in the price variable."""
    print("\n" + "="*50)
    print("OUTLIER DETECTION")
    print("="*50)

    if 'price' in car.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=car['price'])
        plt.title('Price Distribution - Outlier Detection')
        plt.tight_layout()
        plt.savefig('price_outliers.png', dpi=300, bbox_inches='tight')
        print("Outlier boxplot saved as 'price_outliers.png'")
        plt.show()
    else:
        print("'price' column not found in dataset.")


def brand_analysis(car):
    """Analyze and visualize car distribution by brand."""
    print("\n" + "="*50)
    print("BRAND ANALYSIS")
    print("="*50)

    if 'brand' in car.columns:
        plt.figure(figsize=(10, 5))
        car.brand.value_counts().nlargest(40).plot(kind='bar')
        plt.title('Number of Cars per Brand (Top 40)')
        plt.ylabel('Number of Cars')
        plt.xlabel('Brand Name')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('brand_distribution.png', dpi=300, bbox_inches='tight')
        print("Brand distribution chart saved as 'brand_distribution.png'")
        plt.show()
    else:
        print("'brand' column not found in dataset.")


def main():
    """Main function to run the complete analysis pipeline."""
    print("="*50)
    print("CAR DATA ANALYSIS")
    print("="*50)

    try:
        # Load and explore data
        car = load_and_explore_data('car.csv')

        # Perform analysis steps
        analyze_data_types(car)
        display_columns(car)
        statistical_summary(car)

        # Visualizations
        visualize_distributions(car)
        create_pairplot(car)

        # Data cleaning and transformation
        car = rename_columns(car)
        analyze_categorical_continuous(car)
        check_shape(car)
        check_duplicates(car)
        car = remove_duplicates(car)
        check_missing_values(car)
        car = handle_missing_values(car)

        # Advanced analysis
        check_outliers(car)
        brand_analysis(car)

        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)
        print(f"\nFinal dataset shape: {car.shape}")
        print("All visualizations have been saved as PNG files.")

        return car

    except FileNotFoundError:
        print("\nError: 'car.csv' file not found!")
        print("Please ensure the file exists in the current directory.")
        return None
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return None


# Run the analysis if this script is executed directly
if __name__ == "__main__":
    car_data = main()




