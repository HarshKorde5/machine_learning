"""
Model evaluation:
To evaluate a linear regression model we need to calculate multiple metrics such as
r_square : R-squared / Goodness of fit should be maximized to 1
mse : Mean Squared Error must be minimum
mae : Mean Absolute Error must be lower

We store all these metrics in a structure called model_evalution_results which contains:
train and test mse,rmse,mae,r2
"""

import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
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

def model_evaluation(y_train, y_test, y_train_pred, y_test_pred):
    model_evaluation_results = pd.DataFrame(columns=['Model', 'Train_MSE', 'Train_RMSE','Train_MAE', 'Train_R2', 'Test_MSE', 'Test_RMSE','Test_MAE','Test_R2'])

    
    metrics = [
        "Linear Regression Model with Area + Numerical Features"
        ,mean_squared_error(y_train, y_train_pred)
        ,mean_squared_error(y_test, y_test_pred)
        ,root_mean_squared_error(y_train, y_train_pred)
        ,root_mean_squared_error(y_test, y_test_pred)
        ,mean_absolute_error(y_train, y_train_pred)
        ,mean_absolute_error(y_test, y_test_pred)
        ,r2_score(y_train, y_train_pred)
        ,r2_score(y_test, y_test_pred)
    ]

    model_evaluation_results.loc[len(model_evaluation_results)] = metrics

    print(model_evaluation_results)


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

    y_test_pred = regModel.predict(x_test)
    y_train_pred = regModel.predict(x_train)

    model_evaluation(y_train, y_test, y_train_pred, y_test_pred)


def main():
    PRINT_SEPARATOR()
    print("House price predicition with help of Multiple Linear Regression")
    PRINT_SEPARATOR()
    housePricePredictor()
    
    
if __name__ == "__main__":
    main()