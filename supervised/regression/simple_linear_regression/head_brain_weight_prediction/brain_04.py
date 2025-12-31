import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

PRINT_SEPARATOR = lambda : print('-'*100)

def plotModel(X, Y, y_pred):
    """
    Plots and saves 4 important Linear Regression graphs:
    1. Actual Data (Scatter)
    2. Regression Line
    3. Residual Plot
    4. Actual vs Predicted
    """
    
    if not os.path.exists('data'):
        os.mkdir('data')
                
                
    #1. Scatter plot
    plt.figure()
    plt.scatter(X,Y, color="blue", label="Actual Data")
    plt.xlabel("Head Size (cm^3)")
    plt.ylabel("Brain Weight (grams)")
    plt.title("Actual Data Distribution")
    plt.legend()
    plt.savefig('data/data_scatter.png')
    plt.close()
    
    #2. Regression Line Plot    
    plt.figure()
    plt.scatter(X,Y, color="blue", label="Actual Data")
    plt.plot(X,y_pred, color="red", linewidth=2, label="Regression Line")
    plt.xlabel("Head Size (cm^3)")
    plt.ylabel("Brain Weight (grams)")
    plt.title("Linear Regression Line")
    plt.legend()
    plt.savefig('data/regression_line.png')
    plt.close()
    
    #3. Residual Plot
    residuals = Y - y_pred
    plt.figure()
    plt.scatter(X, residuals, color="green", label="Residuals")
    plt.axhline(y=0, color="black", linestyle="--", label="Zero Error Line")
    plt.xlabel("Head Size (cm^3)")
    plt.ylabel("Residuals")    
    plt.legend()
    plt.savefig('data/residual_plot.png')
    plt.close()
    
    #4. Actual vs Predicted
    plt.figure()
    plt.scatter(Y, y_pred, color="purple", label="Predicted vs Actual")
    plt.xlabel("Actual Y")
    plt.ylabel("Predicted Y")
    plt.title("Actual vs Predicted")
    plt.legend()
    plt.savefig('data/actual_vs_predicted.png')
    plt.close()
    
    print(f"All plots saved successfully in '/data' folder")
    
    
def brainWeightPredictor():
    
    data = pd.read_csv(r'data/HeadBrain.csv')    
    print(data.head(5))
    print(data.shape)
    PRINT_SEPARATOR()
    
    X = data['Head Size(cm^3)'].values
    Y = data['Brain Weight(grams)'].values
    
    # print("Independent Variable X : Head Size in cm^3 : ",X)
    # print("Dependent Variable Y : Brain Weight in gms : ",Y)
    
    #reshape X : Independent variable as LinearRegression expects 2D array (n_samples, n_features) 
    print("X shape before reshape : ",X.shape) #(237,)
    X = X.reshape(-1,1)    
    print("X shape after reshape : ",X.shape) #(237,1)
    
    reg = LinearRegression()
    reg = reg.fit(X,Y)
    
    y_pred = reg.predict(X)
    
    r2 = reg.score(X,Y)
    
    print("Goodness of fit using r-squared method is : ",r2)
    plotModel(X,Y,y_pred)
    
    
def main():
    PRINT_SEPARATOR()
    print("Brain weight prediction with the help of brain size")
    PRINT_SEPARATOR()
    brainWeightPredictor()
    PRINT_SEPARATOR()
    
    
    
if __name__ == "__main__":
    main()