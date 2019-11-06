from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sklearn

print(sklearn.__version__)
iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=0)
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, y_train)
print(clf.score)

preds_test = clf.predict(X_test)
print(accuracy_score(y_test, preds_test))
