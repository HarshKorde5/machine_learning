import pandas as pd

PRINT_SEPARATOR = lambda : print(100*'-')

"""
ordinal encoding for 'furnishingstatus'
binary encoding for other categorical columns
"""
def housePricePredictor():
    df = pd.read_csv(r'data/Housing.csv')    

    binary_columns = ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]
    for col in binary_columns:
        df[col] = df[col].map({"yes": 1, "no": 0})    

    furnishing_map = {
        "unfurnished" : 0,
        "semi-furnished" : 1,
        "furnished" : 2,
    }

    df["furnishingstatus"] = df["furnishingstatus"].map(furnishing_map)

    X = df.drop('price',axis=1)
    Y = df['price']
    print(X)


def main():
    PRINT_SEPARATOR()
    print("House price predicition with help of Multiple Linear Regression")
    PRINT_SEPARATOR()
    housePricePredictor()
    
    
if __name__ == "__main__":
    main()