import pandas as pd

PRINT_SEPARATOR = lambda : print('-'*100)

def brainWeightPredictor():
    data = pd.read_csv(r'data/HeadBrain.csv')    
    print(data.head(5))
    print(data.shape)
    
def main():
    PRINT_SEPARATOR()
    print("Brain weight prediction with the help of brain size")
    PRINT_SEPARATOR()
    brainWeightPredictor()
    
    
    
if __name__ == "__main__":
    main()