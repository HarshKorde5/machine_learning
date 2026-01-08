
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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
    
def scaled_data(x_train, x_test):
    numerical_columns_to_scale = ["area", "bedrooms", "bathrooms", "stories", "parking"]        #not included 0/1 columns like airconditioning,guestroom,etc as they're already on a scale of 0-1

    scaler = StandardScaler()

    x_train[numerical_columns_to_scale] = scaler.fit_transform(x_train[numerical_columns_to_scale])
    x_test[numerical_columns_to_scale] = scaler.transform(x_test[numerical_columns_to_scale])

def housePricePredictor():
    df = pd.read_csv(r'data/Housing.csv')    

    binaryEncoder(df)
    ordinalEncoder(df)

    X = df.drop('price', axis=1)
    Y = df['price']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    scaled_data(x_train, x_test)

    regModel = LinearRegression()
    regModel.fit(x_train, y_train)

    y_pred = regModel.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    print("Goodness of fit using r squared method is : ",r2)


def main():
    PRINT_SEPARATOR()
    print("House price predicition with help of Multiple Linear Regression")
    PRINT_SEPARATOR()
    housePricePredictor()
    
    
if __name__ == "__main__":
    main()