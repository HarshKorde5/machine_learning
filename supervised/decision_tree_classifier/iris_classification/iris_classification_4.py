import os
from sklearn import tree
import graphviz
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    print("----------------------Iris Classification Case Study----------------------")
    
    _iris = load_iris()
    
    features = _iris.data
    labels = _iris.target
    
    data_train, data_test, target_train, target_test = train_test_split(features, labels, test_size = 0.5)
    
    obj = tree.DecisionTreeClassifier()
    
    obj = obj.fit(data_train, target_train)
    
    tree.plot_tree(obj)
    dot_data = tree.export_graphviz(obj, 
                        feature_names=_iris.feature_names,
                           class_names=_iris.target_names,
                           filled=True,
                           rounded=True,
                           out_file=None)
    graph = graphviz.Source(dot_data)
    
    png_path = os.path.join("data", "iris_tree.png")
    graph.render(png_path, format="png", cleanup=True)
    
    
    output = obj.predict(data_test)
    
    accuracy = accuracy_score(target_test, output)
    
    print("Accuracy of the model is : ",accuracy*100,"%")
    
    
    
if __name__ == "__main__":
    main()