import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    
    # if os.path.exists(fname):
    #     fobj = open(fname, "r")
    #     print("File successfully opened!")
    #     print(fobj)
    # else:
    #     print("Unable to open file")
    
    
    fname = os.path.join("data","iris.csv")    
    data = pd.read_csv(fname)
    print(data)
    print(type(data))
    print(data.dtypes)
    print(data.columns)
    
    print("Range of sepal length of Iris-setosa is : ",data['sepal_length'].iloc[:50].min()," -> ",data['sepal_length'].iloc[:50].max())
    print("Range of sepal width of Iris-setosa is : ",data['sepal_width'].iloc[:50].min()," -> ",data['sepal_width'].iloc[:50].max())
    print("Range of petal length of Iris-setosa is : ",data['petal_length'].iloc[:50].min()," -> ",data['petal_length'].iloc[:50].max())
    print("Range of petal width of Iris-setosa is : ",data['petal_width'].iloc[:50].min()," -> ",data['petal_width'].iloc[:50].max())

    print("Range of sepal length of Iris-versicolor is : ",data['sepal_length'].iloc[51:100].min()," -> ",data['sepal_length'].iloc[51:100].max())
    print("Range of sepal width of Iris-versicolor is : ",data['sepal_width'].iloc[51:100].min()," -> ",data['sepal_width'].iloc[51:100].max())
    print("Range of petal length of Iris-versicolor is : ",data['petal_length'].iloc[51:100].min()," -> ",data['petal_length'].iloc[51:100].max())
    print("Range of petal width of Iris-versicolor is : ",data['petal_width'].iloc[51:100].min()," -> ",data['petal_width'].iloc[51:100].max())

    print("Range of sepal length of Iris-virginica is : ",data['sepal_length'].iloc[101:150].min()," -> ",data['sepal_length'].iloc[101:150].max())
    print("Range of sepal width of Iris-virginica is : ",data['sepal_width'].iloc[101:150].min()," -> ",data['sepal_width'].iloc[101:150].max())
    print("Range of petal length of Iris-virginica is : ",data['petal_length'].iloc[101:150].min()," -> ",data['petal_length'].iloc[101:150].max())
    print("Range of petal width of Iris-virginica is : ",data['petal_width'].iloc[101:150].min()," -> ",data['petal_width'].iloc[101:150].max())

    plt.figure()
    sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=data)
    plt.savefig("data/iris_scatterplot_sepal.png", bbox_inches='tight')
    plt.close()

    plt.figure()
    sns.scatterplot(x='petal_length', y='petal_width', hue='species', data=data)
    plt.savefig("data/iris_scatterplot_petal.png", bbox_inches='tight')
    plt.close()
    
    
if __name__ == "__main__":
    main()