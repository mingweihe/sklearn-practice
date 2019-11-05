from sklearn import svm
from sklearn import datasets

clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()
# print(iris.data.shape, iris.target.shape)
# print(iris.data)
# print(iris.target)
X, y = iris.data, iris.target
clf.fit(X, y)
# way 1
import pickle
filename = 'svc.pkl'
with open(filename, 'wb') as f:
    pickle.dump(clf, f)
with open(filename, 'rb') as f:
    clf2 = pickle.load(f)
print(clf2.predict(X[0:1]))
print(y[0])
# way 2, faster
from joblib import dump, load
filename = 'svc.joblib'
dump(clf, filename)
clf3 = load(filename)
print(clf3.predict(X[0:1]))
print(y[0])