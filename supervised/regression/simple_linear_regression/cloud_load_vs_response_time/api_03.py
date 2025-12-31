import numpy as np
import pandas as pd

def main():
    print("------------------------------Concurrent Users V/S API Response Time------------------------------")
    
    data = pd.read_csv("data/cloud_load_vs_response_time.csv")
    print("Dataset top 10 record : \n",data.head(10))
    print("\nDataset bottom 10 records : \n",data.tail(10))
    print("Type of data : ",type(data))
    print("Shape of data : \n",data.shape)
    
    
    X = data['Concurrent_Users'].values
    Y = data['Response_Time_ms'].values
    
    # print("Independent variable 'X' is the count of concurrent users at the server : ", X)
    # print("Dependent variable 'Y' is the response time for the server : ", Y)    
    
    
    
if __name__ == "__main__":
    main()