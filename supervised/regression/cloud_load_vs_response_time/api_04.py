import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def responseTimePredictor():
    data = pd.read_csv("data/cloud_load_vs_response_time.csv")
    print("Dataset top 10 record : \n",data.head(10))
    print("\nDataset bottom 10 records : \n",data.tail(10))
    print("Type of data : ",type(data))
    print("Shape of data : \n",data.shape)
    
    
    X = data['Concurrent_Users'].values
    Y = data['Response_Time_ms'].values
    
    # print("Independent variable 'X' is the count of concurrent users at the server : ", X)
    # print("Dependent variable 'Y' is the response time for the server : ", Y)        
    
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3)
        
    #least square method
    mean_x = np.mean(x_train)
    mean_y = np.mean(x_test)
    
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
    
    y_pred = []
    
    for i in range(len(x_test)):
        y_pred.append(w*x_test[i] + b)
        
    print("Testing inputs : ",x_test)
    print("Predicted outputs : ",y_pred)
    
        
    
    
    
    
def main():
    print("------------------------------Concurrent Users V/S API Response Time------------------------------")
        
    responseTimePredictor()
    
    
if __name__ == "__main__":
    main()