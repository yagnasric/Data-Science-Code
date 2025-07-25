import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

train  = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['Age'].fillna(train['Age'].median(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
train.drop(['Cabin', 'Ticket', 'Name'], axis = 1, inplace = True)

train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
train = pd.get_dummies(train, columns = ['Embarked'], drop_first = True)

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

X = train.drop(['Survived', 'PassengerId'], axis =1)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =0)



# lr_model = LogisticRegression()
# lr_model.fit(X_train, y_train)

# y_pred = lr_model.predict(X_test)

# print(confusion_matrix(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

kn_model = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean', p =2)

kn_model.fit(X_train, y_train)

y_pred = kn_model.predict(X_test)

# print(confusion_matrix(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

dtr_model = DecisionTreeClassifier(criterion = 'entropy', random_state =0)

dtr_model.fit(X_train, y_train)

y_pred = dtr_model.predict(X_test)


# print(confusion_matrix(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

nb_model = GaussianNB()

nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))