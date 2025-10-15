from sklearn import tree

def MyClassifier(_weight, _surface):
    
    features = [[35,"Rough"],[47,"Rough"],[90,"Smooth"],[48,"Rough"],[90,"Smooth"],[35,"Rough"],[92,"Smooth"],[35,"Rough"],[35,"Rough"],[35,"Rough"]]
    labels = ["Tennis","Tennis","Cricket","Tennis","Cricket","Tennis","Cricket","Tennis","Tennis","Tennis"]
    
    print("Dataset before encoding : ")
    print("Features : ",features)
    print("Labels",labels)
    
    #Rough : 1
    #Smooth : 0
    
    #Tennis : 1
    #Cricket : 2
    
    #Feature Encoding (manually)
    encoded_features = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1]]

    #Label Encoding (manually)
    encoded_labels = [1,1,2,1,2,1,2,1,1,1]
    print("Dataset after encoding : ")
    print("Features : ",encoded_features)
    print("Labels",encoded_labels)
    
    
    #Decide the algorithm
    obj = tree.DecisionTreeClassifier()
    
    #Train the model
    obj = obj.fit(encoded_features, encoded_labels)
    
    #Test the model
    ret = obj.predict([[_weight,_surface]])
    
    if ret == 1 : 
        print("Your object looks like a Tennis Ball")
    else : 
        print("Your object looks like a Cricket Ball")


def main():
    print("---------------------------Ball Classification Case Study---------------------------")
    
    print("Please enter the information about object you want to test : ")
    print("Enter the weight of the ball : ")
    _weight = int(input())
    
    print("Enter the surface type of the ball (Rough / Smooth) :")
    _surface = input()
    
    #Label encoding
    if _surface.lower() == "rough":
        _surface = 1
    elif _surface.lower() == "smooth":
        _surface = 0
    else:
        print("Invalid surface type entered. Please enter either 'Rough' or 'Smooth'.")
        exit()
        
    MyClassifier(_weight,_surface)
    
    
if __name__ == "__main__":
    main()