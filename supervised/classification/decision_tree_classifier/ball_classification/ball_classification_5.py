from sklearn import tree

def MyClassifier(_weight,_surface):
    #Feature Encoding
    Features = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1]]

    #Label Encoding
    Labels = [1,1,2,1,2,1,2,1,1,1]

    #Decide the algorithm
    obj = tree.DecisionTreeClassifier()

    #Train the model
    obj = obj.fit(Features,Labels)

    #Test the model
    ret = obj.predict([[_weight,_surface]])
    if ret == 1:
        print("Your object looks like tennis ball")
    else:
        print("Your object looks like cricket ball")


def main():

    print("------------Ball type classification case study------------\n")

    print("Please enter the information about the object that you want to test : ")

    print("Please enter the weight of your object in grams : ")
    _weight = int(input())

    print("Please mention the type of surface : Rought/Smooth : ")
    _surface = input()

    if _surface.lower() == "rough":
        _surface = 1
    elif _surface.lower() == "smooth":
        _surface = 0
    else:
        print("Invalid type of surface")
        exit()

    MyClassifier(_weight,_surface)


if __name__ == "__main__":
    main()