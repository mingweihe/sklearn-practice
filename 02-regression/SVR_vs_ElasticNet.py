from sklearn.svm import SVR
from sklearn import datasets
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score # accuracy_score is for classification only
import matplotlib.pyplot as plt

boston = datasets.load_boston()
# print(type(boston), boston.keys())
# print(boston.feature_names, boston.DESCR, boston.filename)
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=22)
print(X.shape, y.shape)
# plot the price 
# plt.plot(y)
# plt.show()
model = SVR(gamma='scale')
model.fit(X_train, y_train)
preds = model.predict(X_test)
plt.plot(y_test, label='y_test')
plt.plot(preds, label='y_predict')
plt.legend()
plt.show()
print(model.score(X_test, y_test))
# compare with ElasticNet linear model
from sklearn.linear_model.coordinate_descent import ElasticNet
model = ElasticNet()
model.fit(X_train, y_train)
preds = model.predict(X_test)
plt.plot(y_test, label='y_test')
plt.plot(preds, label='y_predict')
plt.legend()
plt.show()
print(model.score(X_test, y_test))
