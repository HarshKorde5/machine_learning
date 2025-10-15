# Ball Classification Case Study

## 1. Problem Statement
Ball classification case study is to classify a ball as either a tennis ball or a cricket ball based on it's weight and the surface type (smooth / rough).

--- 
## 2. Dataset   
We're using a small custom dataset of balls with two features:

| Weight (grams) | Surface | Label        |
|----------------|---------|-------------|
| 35             | Rough   | Tennis Ball |
| 47             | Rough   | Tennis Ball |
| 90             | Smooth  | Cricket Ball |
| 48             | Rough   | Tennis Ball |
| 92             | Smooth  | Cricket Ball |
| 35             | Rough   | Tennis Ball |
| 90             | Smooth  | Cricket Ball |
| 35             | Rough   | Tennis Ball |
| 35             | Rough   | Tennis Ball |
| 35             | Rough   | Tennis Ball |

**Feature encoding**
- Surface : `Rough = 1`, `Smooth = 0`

**Label encoding**
- Tennis ball : 1
- Cricket ball : 2

---

## 3. Approach
- We use **Decision Tree Classifier** from `sklearn` (Scikit-learn: an open source Python library which provides various machine learning algorithms).
- Features (weight and surface) are used to **train the model.**
- The trained model **predicts the type of ball** based on user input.

- Two classes can be defined here:
1. More weight and smooth surface can be classified as a Cricket Ball.
2. Less weight and rough surface can be classified as a Tennis Ball.

**What other features of a ball exist?**
- Color : Red or Green
- Size : Radius 

**Why these features of a ball are not considered?**
- The feature color does not classify a ball type, in other words both cricket and tennis ball, both can have a red color or a green color which make confusion to classify. 
- The feature size also does not classify a ball type, size of both the balls might be same irrespective of their types.
- This is also known as **feature extraction**. (Not exact but similar)

This simple case study helps us to understand
- How **features and labels** are used in ML.
- How to **train a classifier** and **make predictions**.

---

## 4. Model training
The training is done directly inside the code using inbuilt function `fit()`.

```python
from sklearn import tree

#Features and Labels
features = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1]]

labels = [1,1,2,1,2,1,2,1,1,1]

#Decision Tree
obj = tree.DecisionTreeClassifier()
obj = obj.fit(features, labels)

```

---

## 5. How to execute
- Make sure **Python** and the **scikit-learn** library are installed.  
  You can install scikit-learn using:
  ```bash
  pip install scikit-learn
- Run the program using:
    ```bash
    python ball_classification.py

---
## 6. Conclusion
- This case study shows how a simple Machine Learning model using a **Decision Tree Classifier** **learns from data (training)** and **makes predictions (classification)**.  
- It demonstrates the basic idea of how ML works using a small dataset and simple features.
- Ball classification is a simple case study made for initial learning purpose of how machine learning or specifically classification works.
- This case study does not have any relevance with actual use case in industry or other places.

