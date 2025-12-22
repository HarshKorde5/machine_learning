
from sklearn.datasets import fetch_openml
from tensorflow.keras.datasets import mnist

"""
consumes time to load the dataset
requires internet connectivity
"""
def fetch_from_sklearn():
    mnist = fetch_openml('mnist_784', version=1)
    print(type(mnist))
    print(mnist)
    
    
"""
fast compared to fetch_openml
requires space for installing tensorflow
"""
def fetch_from_tensorflow():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # sample check
    for i in range (28):
        for j in range (28):
            print(x_test[0][i][j],"\t", end=" ")
        print()
    
    
def main():
    # fetch_from_sklearn()
    fetch_from_tensorflow()
    
    
    
if __name__ == "__main__":
    main()
