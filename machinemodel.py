import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('C:/Users/BankNoteAuthentication.csv')
df
X = df.iloc[:,0:4]
y = df.iloc[:,4]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_test
y_pred
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
pickle.dump(model,open('note_model.pkl','wb')) 