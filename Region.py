import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv("Data/clean-salaries-by-region.csv")
data = data.values

X = data[:, 0:4]
y_temp = data[:, 4:].tolist()

y = []
for i in range(len(y_temp)):
    y.append(y_temp[i].index(1))
y = np.array(y)

print(X[0:3])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666)

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
score_dev = logistic_model.score(X_test, y_test)
print('Accuracy on dev set:', score_dev)

svm_clf = svm.SVC(gamma=2, C=1)
svm_clf.fit(X_train, y_train)
score_dev = svm_clf.score(X_test, y_test)
print('Accuracy on dev set:', score_dev)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X, y)
score_dev = neigh.score(X_test, y_test)
print('Accuracy on dev set:', score_dev)
