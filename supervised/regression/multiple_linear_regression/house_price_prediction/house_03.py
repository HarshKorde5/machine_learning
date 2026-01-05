import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
PRINT_SEPARATOR = lambda : print(100*'-')

"""
target variable 'price' analysis
price distribution using histplot
price outliers using boxplot

distribution of numerical features
correlation analysis for features using heatmap

"""
def housePricePredictor():
    df = pd.read_csv(r'data/Housing.csv')
    print(df.head())
    PRINT_SEPARATOR()
    print(df.shape)
    print(df.describe())
    PRINT_SEPARATOR()
    print(df.info())
    PRINT_SEPARATOR()    
    
    #target variable 'price' analysis
        
    print(df['price'].info())
    PRINT_SEPARATOR()
    print(df['price'].describe())
    PRINT_SEPARATOR()
    
    #price distribution
    plt.figure(figsize=(6,4))
    sns.histplot(df['price'], kde=True)
    plt.title("Price Distribution")
    plt.savefig('data/price_distribution.png')
    
    #price outliers
    plt.figure(figsize=(5,3))
    sns.boxplot(x=df["price"])
    plt.title("Price Outliers")
    plt.savefig('data/price_outliers.png')
    
    #distribution of numerical features
    df.hist(figsize=(12,8), bins=20)
    plt.savefig('data/distribution_numerical_features.png')
    
    #correlation analysis
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation matrix")
    plt.savefig('data/correlation_matrix.png')

def main():
    PRINT_SEPARATOR()
    print("House price predicition with help of Multiple Linear Regression")
    PRINT_SEPARATOR()
    housePricePredictor()
    
    
if __name__ == "__main__":
    main()