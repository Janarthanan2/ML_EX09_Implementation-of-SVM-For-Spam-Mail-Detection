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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score

df=pd.read_csv('/content/spam.csv',encoding='ISO-8859-1')
df.head()

vectorizer = CountVectorizer()
X=vectorizer.fit_transform(df['v2'])
y=df['v1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model=svm.SVC(kernel='linear')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))
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
