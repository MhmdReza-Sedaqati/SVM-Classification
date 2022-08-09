import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn import svm





# Importing the dataset
df = pd.read_csv('import your csv file here')
df.drop(df.columns[[0, 3, 17, 18, 19, 20, 21, 22, 23, 24, 31, 32, 33, 42, 49, 55, 56, 57, 58, 59, 60, 73, 74, 75, 76]], axis=1, inplace=True)


pd.set_option("display.max.columns", None)
# pd.set_option("display.max.rows", None)


X = df.iloc[:, [2, 3]].values
y = df.iloc[:, -1].values



# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)




# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




# Training the SVM model on the Training set

classifier = svm.SVC(gamma='auto')
classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)



# Making the Confusion Matrix
Cofusion_Matrix = confusion_matrix(y_test, y_pred)
Accuracy_Score = accuracy_score(y_test, y_pred)
Report = classification_report(y_test, y_pred)

print('Cofusion_matrix is: \n', Cofusion_Matrix)
print('The accuracy of SVM is: ', Accuracy_Score)
print('Classification report is: \n', Report)
