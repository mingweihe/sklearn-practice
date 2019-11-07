import sys
import pickle
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

digits = load_digits()
X_digits = digits.data / digits.data.max()
y_digits = digits.target
# import matplotlib.pyplot as plt
# plt.imshow(digits.data[0].reshape(-1,8))
# plt.show()

n_sapmles = len(X_digits)

X_train = X_digits[:int(.9 * n_sapmles)]
y_train = y_digits[:int(.9 * n_sapmles)]
X_test = X_digits[int(.9 * n_sapmles):]
y_test = y_digits[int(.9 * n_sapmles):]

knn = KNeighborsClassifier()
logistic = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial')

# calculate size of models
s_knn = pickle.dumps(knn)
print(sys.getsizeof(s_knn))
s_logistic = pickle.dumps(logistic)
print(sys.getsizeof(s_logistic))

print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
print('LogisticRegression score: %f' % logistic.fit(X_train, y_train).score(X_test, y_test))


# calculate size of models
s_knn = pickle.dumps(knn)
print(sys.getsizeof(s_knn))
s_logistic = pickle.dumps(logistic)
print(sys.getsizeof(s_logistic))