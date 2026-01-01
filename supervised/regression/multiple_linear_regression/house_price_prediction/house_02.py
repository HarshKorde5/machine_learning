import pandas as pd

PRINT_SEPARATOR = lambda : print(100*'-')

"""
Data description using:
head()
shape()
describe()
info()
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
    
    
def main():
    PRINT_SEPARATOR()
    print("House price predicition with help of Multiple Linear Regression")
    PRINT_SEPARATOR()
    housePricePredictor()
    
    
if __name__ == "__main__":
    main()