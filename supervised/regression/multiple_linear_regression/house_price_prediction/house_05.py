"""
Let's just test on 'area' feature of the dataset
It shows that using only 'area' feature model tends to be weak predictor as we can see from te low R2 value.
The model is underfitting, meaning it is unable to capture the underlying relationships in the data effectively.
This proves other features are necessary for this case study.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

PRINT_SEPARATOR = lambda : print(100*'-')

def binaryEncoder(df):
    binary_columns = ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]
    for col in binary_columns:
        df[col] = df[col].map({"yes": 1, "no": 0})

def ordinalEncoder(df):
    furnishing_map = {
        "unfurnished" : 0,
        "semi-furnished" : 1,
        "furnished" : 2,
    }
    df["furnishingstatus"] = df["furnishingstatus"].map(furnishing_map)

def housePricePredictor():
    df = pd.read_csv(r'data/Housing.csv')    

    binaryEncoder(df)
    ordinalEncoder(df)

    X = df['area'].values
    Y = df['price']
    X = X.reshape(-1,1)
    
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
    
    regModel = LinearRegression()
    regModel.fit(x_train, y_train)
    y_pred = regModel.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    print("Goodness of fit with area feature is : ",r2) #low r2 means bad at predicting, model is underfitting


def main():
    PRINT_SEPARATOR()
    print("House price predicition with help of Multiple Linear Regression")
    PRINT_SEPARATOR()
    housePricePredictor()
    
    
if __name__ == "__main__":
    main()