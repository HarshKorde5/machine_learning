import time
from tensorflow.keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier

"""
Testing with different values of k to find better accuracy
on whole dataset
"""
def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)

    start_time = time.perf_counter()

    for k in [3, 5, 7, 9, 11]:
        
        """
        n_jobs = -1 : uses all cores in the cpu if set to -1, and uses n cores if set to n
        for sample splits calculations using multiprocessing to speed up
        """
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs= -1)
        knn.fit(x_train, y_train)
        print(f"Accuracy with n_neighbors = {k} is : {knn.score(x_test, y_test) * 100}%")
        
    end_time = time.perf_counter()

    # Requires more than 160seconds to complete all calculations which is expensive
    print(f"Completed work in {end_time - start_time} seconds")

if __name__ == "__main__":
    main()

