import time
import multiprocessing as mp
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier

X_TRAIN = None
Y_TRAIN = None
X_TEST = None
Y_TEST = None

def init_globals(x_train, y_train, x_test, y_test):
    global X_TRAIN, Y_TRAIN, X_TEST, Y_TEST
    X_TRAIN = x_train
    Y_TRAIN = y_train
    X_TEST = x_test
    Y_TEST = y_test
    


def evaluate_k(k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_TRAIN, Y_TRAIN)
    accuracy = knn.score(X_TEST, Y_TEST)*100 
    return k, accuracy
    
    
def main():
    global X_TRAIN, Y_TRAIN, X_TEST, Y_TEST

    (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = mnist.load_data()

    X_TRAIN = X_TRAIN.reshape(len(X_TRAIN), -1) / 255.0
    X_TEST = X_TEST.reshape(len(X_TEST), -1) / 255.0
        

    """
    PCA : Principal Component Analysis
    Converts each vector of 784 elements to 50 meaningful components
    This will significantly reduce the required number of total calculations
    And so the execution time performs well and accuracy remains same.
    """            
    pca = PCA(n_components=50)
    X_TRAIN_PCA = pca.fit_transform(X_TRAIN)
    X_TEST_PCA = pca.transform(X_TEST)
    # print("Number of cores : ",mp.cpu_count())
    
    k_values = [3, 5, 7, 9, 11]
    
    start_time = time.perf_counter()
    
    with mp.Pool(processes=mp.cpu_count(),
                 initializer=init_globals,
                 initargs=(X_TRAIN_PCA, Y_TRAIN, X_TEST_PCA, Y_TEST)
                )as pool:
        results = pool.map(evaluate_k, k_values)
        
    end_time = time.perf_counter()
        
    for k,acc in results:
        print(f"Accuracy with n_neighbors = {k} is : {acc}")
    
    print(f"Execution time : {end_time - start_time}")
    
if __name__ == "__main__":
    
    main()