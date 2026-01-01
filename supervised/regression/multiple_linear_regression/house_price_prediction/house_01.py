import pandas as pd

PRINT_SEPARATOR = lambda : print(100*'-')

def housePricePredictor():
    data = pd.read_csv(r'data/Housing.csv')
    print(data.head(5))    
    
def main():
    PRINT_SEPARATOR()
    print("House price predicition with help of Multiple Linear Regression")
    PRINT_SEPARATOR()
    housePricePredictor()
    
    
if __name__ == "__main__":
    main()