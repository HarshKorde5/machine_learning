import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(points):
    q = np.random.rand(points.shape[1])
    return np.linalg.norm(points - q, axis = 1)

def manhattan_distance(points):
    q = np.random.rand(points.shape[1])
    return np.sum(np.abs(points - q), axis = 1)

def main():
    print("--------------------Comparison between Euclidean and Manhattan Distances when dimensions increase-----------------------")
    
    
    # Synthetic dataset
    np.random.seed(42)
    n_points  = 600
    
    points_2d = np.random.rand(n_points, 2)
    points_100d = np.random.rand(n_points, 100)
    
    # print(points_100d)
    
    dist_2d_euc = euclidean_distance(points_2d)
    dist_2d_man = manhattan_distance(points_2d)

    dist_100d_euc = euclidean_distance(points_100d)
    dist_100d_man = manhattan_distance(points_100d)
    
    plt.figure()
    plt.hist(dist_2d_euc, bins=30)
    plt.title("Euclidean Distance Distribution (2D)")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.savefig("data/euclidean_2d.png", bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.hist(dist_100d_euc, bins=30)
    plt.title("Euclidean Distance Distribution (100D)")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.savefig("data/euclidean_100d.png", bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.hist(dist_2d_man, bins=30)
    plt.title("Manhattan Distance Distribution (2D)")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.savefig("data/manhattan_2d.png", bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.hist(dist_100d_man, bins=30)
    plt.title("Manhattan Distance Distribution (100D)")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.savefig("data/manhattan_100d.png", bbox_inches='tight')
    plt.close()

    
    
if __name__ == "__main__":
    main()