import numpy as np
import matplotlib.pyplot as plt

def linearPredictor():
    
    X = [1,2,3,4,5]
    Y = [3,4,2,3,5]
    
    print("Values of Independent variable x : ",X)
    print("Values of Dependent variable y : ",Y)
    
    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    print("Mean of Independent variable X is : ",mean_x)
    print("Mean of Dependent variable Y is : ",mean_y)
    
    n = len(X)
    
    numerator = 0
    denominator = 0
    
    #Equation of line is y = mx+c
    
    for i in range(n):
        numerator += (X[i] - mean_x) * (Y[i] - mean_y)
        denominator += (X[i] - mean_x)**2
        
    m = numerator/ denominator
    
    #c = y' - mx'
    
    c = mean_y - (m*mean_x)
    
    print("Slope of Regression line is : ", m)
    print("Y intercept of Regression line is : ", c)
    
    #display plotting of above points
    x = np.linspace(1,6,n)    
    
    y = m* x + c
    yp = []
    
    for i in range(n):
        y_pred = c + m * X[i]
        yp.append(y_pred)
        
    print(f"Y Predicted for {X} is {yp}")
        
    plt.plot(x,y, color = "#58b970", label="Regression Line")
    
    plt.scatter(X,Y, color="#23caef", label="Scatter Plot")
    plt.scatter(X,yp, color="red", label= "Scatter Plot - Predicted")

    plt.xlabel("X - Independent Variable")
    plt.ylabel("Y - Dependent Variable")
    
    plt.legend()
    plt.savefig(fname="sample.png")
        
    
def main():
    print("-------------------------Simple Linear Regression----------------------")
    
    linearPredictor()
    
    
if __name__ == "__main__":
    main()
    