from tensorflow.keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier

def main():
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(len(x_train), -1) / 255.0
    x_test = x_test.reshape(len(x_test), -1) /255.0

    x_train_small = x_train[:10000]
    y_train_small = y_train[:10000]

    x_test_small = x_test[:1000]
    y_test_small = y_test[:1000]


    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(x_train_small, y_train_small)

    accuracy = knn.score(x_test_small,y_test_small)

    print("Accuracy : ",accuracy * 100,"%")


if __name__ == "__main__":
    main()