from math import sqrt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PRINT_SEPARATOR = lambda : print('-'*100)

def fit(X,Y):
        
    #least square method
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    
    print("Mean of X is : ",mean_x)
    print("Mean of Y is : ",mean_y)
    
    n = len(X)
    
    numerator = 0
    denominator = 0
    
    #y = wx + b
    
    #w = (x - x')*(y - y') / (x - x')**2
    
    for i in range(n):
        numerator += (X[i] - mean_x)*(Y[i] - mean_y)
        denominator += (X[i] - mean_x)**2
        
        
    w = numerator/denominator
    print("The slope of line is : ",w)
    
    b = mean_y - ( w * mean_x)
    print("Bias ( Y - Intercept ) is : ",b)
    
    return w,b
    
def predict(X,w,b):
    
    y_pred = []
    
    for i in range(len(X)):
        y_pred.append(w*X[i] + b)
        
    # print("Testing inputs : ",X)
    # print("Predicted outputs : ",y_pred)
        
    return y_pred
    

def r_square(y_pred, y_test):    
    
    if len(y_pred) != len(y_test):
        print("Error : Unequal lengths for predict and test...")
        return 0
    
    mean_y = np.mean(y_test)
    
    ss_r = 0        #residual sum of squares
    ss_t = 0        #total variance in data
    for i in range(len(y_test)):
        ss_t += (y_test[i] - mean_y)**2
        ss_r += (y_test[i] - y_pred[i])**2
        
    if ss_t == 0:
        print("Error : Zero Variance in y_test")        
        return 0
    
    r2 = 1 - (ss_r/ss_t)
    return r2    
        
def meanAbsoluteError(y_pred, y_test):
    
    n = len(y_pred)
    l1 = 0
    for i in range(n):
        l1 += abs(y_test[i] - y_pred[i])
        
    print("L1 Loss is : ",l1)
    
    mae = l1 / n
    return mae
    
def meanSquaredError(y_pred, y_test):
    
    n = len(y_pred)
    l2 = 0
    for i in range(n):
        l2 += (y_test[i] - y_pred[i])**2
        
    print("L2 Loss is : ",l2)
    
    mse = l2 / n
    return mse
    
def rootMeanSquaredError(y_pred, y_test):
    mse = meanSquaredError(y_pred, y_test)
    rmse = sqrt(mse)
    return rmse
    
def responseTimePredictor():
    
    data = pd.read_csv("data/cloud_load_vs_response_time.csv")
    PRINT_SEPARATOR()    
    print("Dataset top 10 record : \n",data.head(10))
    PRINT_SEPARATOR()    
    print("\nDataset bottom 10 records : \n",data.tail(10))
    PRINT_SEPARATOR()    
    print("Type of data : ",type(data))
    PRINT_SEPARATOR()    
    print("Shape of data : \n",data.shape)
    PRINT_SEPARATOR()
    
    
    X = data['Concurrent_Users'].values
    Y = data['Response_Time_ms'].values
    
    # print("Independent variable 'X' is the count of concurrent users at the server : ", X)
    # print("Dependent variable 'Y' is the response time for the server : ", Y)        
    
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3)   
    
    w,b = fit(x_train,y_train)
    y_pred = predict(x_test, w, b)
    PRINT_SEPARATOR()    
    print("Goodness of Fit for the model using R-Squared method")
    r2 = r_square(y_pred, y_test)
    print("R-squared goodness of fit :: ",r2)
    
    PRINT_SEPARATOR()    
    print("Mean Absolute Error")
    mae = meanAbsoluteError(y_pred, y_test)
    print("Mean absolute error for the model is : ",mae)
    
    PRINT_SEPARATOR()    
    print("Mean Sqaured Error")    
    mse = meanSquaredError(y_pred, y_test)
    print("Mean squared error for the model is : ",mse)
    
    PRINT_SEPARATOR()    
    print("Root mean squared error")    
    rmse = rootMeanSquaredError(y_pred, y_test)    
    print("Root mean squared error is : ", rmse)    
    PRINT_SEPARATOR()    
    
    
def main():
    PRINT_SEPARATOR()
    print("Concurrent Users V/S API Response Time")
    PRINT_SEPARATOR()
    responseTimePredictor()
    PRINT_SEPARATOR()
    
    
if __name__ == "__main__":
    main()