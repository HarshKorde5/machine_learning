from tensorflow.keras.datasets import mnist

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    """
    KNN requires 2D data :[[0...27],[],[],...]
    Original data is 3D :
    [
        [[],
         [],
         .
         .
         28
        ],
        [
         [],
         [],
         .
         .
         28
        ],
        .
        .
        .
        70000
    ]
    
    Reshape / flatten the dataset
    
    divide by 255.0 to normalize the data as calculations for values in range 0-255 will be huge and consume time
    
    """
    print("Before reshape : \n",x_test)
        
    x_train = x_train.reshape(len(x_train), -1) / 255.0
    x_test = x_test.reshape(len(x_test), -1) / 255.0

    print("After reshape : \n",x_test)
    print(x_test)
    
    
    """
    Use small set of data first as size of dataset is huge
    training => 10000
    testing => 1000
    """

    x_train_small = x_train[:10000]
    y_train_small = y_train[:10000]

    x_test_small = x_test[:1000]
    y_test_small = y_test[:1000]
    
    
if __name__ == "__main__":
    main()