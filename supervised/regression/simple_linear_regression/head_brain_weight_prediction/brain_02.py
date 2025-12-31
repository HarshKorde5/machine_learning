import pandas as pd

PRINT_SEPARATOR = lambda : print('-'*100)

def brainWeightPredictor():
    
    data = pd.read_csv(r'data/HeadBrain.csv')    
    print(data.head(5))
    print(data.shape)
    PRINT_SEPARATOR()
    
    X = data['Head Size(cm^3)'].values
    Y = data['Brain Weight(grams)'].values
    
    # print("Independent Variable X : Head Size in cms : ",X)
    # print("Dependent Variable Y : Brain Weight in gms : ",Y)
    
    #reshape X : Independent variable as LinearRegression expects 2D array (n_samples, n_features) 
    print("X shape before reshape : ",X.shape) #(237,)
    X = X.reshape(-1,1)    
    print("X shape after reshape : ",X.shape) #(237,1)
    
    
    
    
def main():
    PRINT_SEPARATOR()
    print("Brain weight prediction with the help of brain size")
    PRINT_SEPARATOR()
    brainWeightPredictor()
    
    
    
if __name__ == "__main__":
    main()