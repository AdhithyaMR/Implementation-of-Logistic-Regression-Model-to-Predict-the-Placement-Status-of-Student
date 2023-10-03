# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Find the null and duplicate values.
3. Using logistic regression find the predicted values of accuracy , confusion matrices
4. Display the results

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Adhithya M R
RegisterNumber:  212222240002
*/
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```

## Output:
### 1.PLACEMENT DATA

![240539700-3aaf36a3-3225-4e02-894b-679fae92571c](https://github.com/AdhithyaMR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118834761/2e17abd1-c666-43f4-bd81-346137cee657)
### 2.SALARY DATA
![240539869-bfb3690b-0316-4a6a-8d41-eec8657f928c](https://github.com/AdhithyaMR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118834761/6dd2e70f-bb17-4071-b0a2-1923a1d6a239)
### 3.CHECKING THE NULL FUNCTIONAL()


![240540376-821980ec-0828-4d23-a523-210744cfe740](https://github.com/AdhithyaMR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118834761/d3008bde-bbc2-494f-af50-875b617e6bdc)
### 4.DATA DUPLICATE

![240540445-7dc60349-0dc5-465b-8865-346198a9851d](https://github.com/AdhithyaMR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118834761/df71898d-43ae-4068-b5f8-c141a7ae310e)
### 5.PRINT DATA
![240540511-973ca213-b6a0-47d6-a04c-42f6152a4fb3](https://github.com/AdhithyaMR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118834761/bd2f6763-ea9f-4110-8734-8ae27d431094)


![240540555-b82f1f41-495c-4284-9d63-7dd9a8f1f74e](https://github.com/AdhithyaMR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118834761/bba169d9-e1ad-4dfc-9efd-ab7ead8f9101)
### 6.DATA STATUS


![240541249-a308db6f-3645-4a26-bad3-854a0dd8df13](https://github.com/AdhithyaMR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118834761/fe0e84a5-7631-4979-a58a-bb6b996dc6dc)
### 7.Y_PREDICTION ARRAY
![240541939-68313695-f848-4c88-9b28-cbc943b46b2c](https://github.com/AdhithyaMR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118834761/8e921de9-7e42-471d-8527-ec8fb8bc09f7)
### 8.ACCURACY VALUE
![240542082-4fd084e5-a1a4-4f2b-90f9-2272afecb5a1](https://github.com/AdhithyaMR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118834761/abf5bee4-6cbf-436d-8a34-2adaf4ab83ca)

### 9.CONFUSION MATRIX
![240542348-e89e4d54-4861-49a2-ae42-3855314d2c65](https://github.com/AdhithyaMR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118834761/fd4125fb-e8d5-42b2-bf6d-225d72c4045a)

### 10. CLASSIFICATION REPORT

![240542586-11220784-7d11-4818-86f1-529911748b78](https://github.com/AdhithyaMR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118834761/f8a73758-3ab8-40ff-8e85-7a0cd8120b4c)
### 11. PREDICTION OF LR
![240542709-8c568904-1ce8-492c-8294-4cc087c0cdcc](https://github.com/AdhithyaMR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118834761/f5416691-bb1e-4971-ac71-4a82e6e64695)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
