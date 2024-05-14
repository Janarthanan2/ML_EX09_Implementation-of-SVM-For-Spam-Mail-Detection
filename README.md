# EX-09 Implementation of SVM For Spam Mail Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.

## Program:

```
Program to implement the SVM For Spam Mail Detection..
Developed by: JANARTHANAN V K 
RegisterNumber: 212222230051

```
```python

import chardet 
file="CSVs/spam.csv"
with open(file,'rb')as rawdata: 
    result = chardet.detect(rawdata.read(100000)) 
result
import pandas as pd 
data=pd.read_csv("CSVs/spam.csv",encoding="'Windows-1252") 
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values 
y=data["v2"].values
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()
x_train=cv.fit_transform(x_train) 
x_test=cv.transform(x_test)
from sklearn.svm import SVC 
svc=SVC() 
svc.fit(x_train,y_train) 
y_pred=svc.predict(x_test) 
y_pred
from sklearn import metrics 
accuracy=metrics.accuracy_score(y_test,y )  
accuracy
```

## Output:
## Head:
<img src="https://github.com/deepikasrinivasans/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393935/666a2fbe-b1e9-4389-bf89-a54ee4fe1de3" width=50%>

## Kernel Model:
<img src="https://github.com/deepikasrinivasans/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393935/72448a19-ec6f-425c-8f14-34d4125032e1">

## Accuracy and Classification report:
<img src="https://github.com/deepikasrinivasans/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393935/5894ab20-ef10-45ee-91f1-f099cb3733da" width=40%>

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
