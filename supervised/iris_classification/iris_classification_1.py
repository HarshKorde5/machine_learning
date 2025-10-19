from sklearn import tree
from sklearn.datasets import load_iris

def main():
    print("----------------------Iris Classification Case Study----------------------")
    
    _iris = load_iris()
    
    print(type(_iris))
    
    features = _iris.data
    labels = _iris.target
    
    print("Features are : ")
    print(features)
    
    print("Labels are : ")
    print(labels)
            
    
if __name__ == "__main__":
    main()