import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import ExtraTreeRegressor
import matplotlib.pyplot as plt
from collections import defaultdict

# optional: sklearn.datasets.make_regression(n_samples=xxx,n_features=xxx,n_targets=xxx,noise=xxx)
def load_data():
    f = lambda x1, x2: .5 * np.sin(x1) + .5 * np.cos(x2) + .1 * x1 + 3
    x1_raw = np.linspace(0, 50, 500)
    x2_raw = np.linspace(-10, 10, 500)
    state = np.random.RandomState(12)
    data = np.array([[x1, x2, f(x1, x2) + state.rand(1)[0]-.5] for x1, x2 in zip(x1_raw, x2_raw)])
    return data[:, :2], data[:, 2]

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=99, shuffle=False)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
names = ['Decision Tree', 'Linear Regression', 'SVR', 'KNN', 'RFR', 'Ada Boost', 
    'Gradient Boost', 'Bagging', 'Extra Tree']
regressors = [
    DecisionTreeRegressor(),
    LinearRegression(),
    SVR(gamma='scale'),
    KNeighborsRegressor(),
    RandomForestRegressor(n_estimators=20),
    AdaBoostRegressor(n_estimators=50),
    GradientBoostingRegressor(n_estimators=100),
    BaggingRegressor(),
    ExtraTreeRegressor()
]
performance = []
plt.figure(figsize=(15, 8))
for i, model in enumerate(regressors, 1):
    ax = plt.subplot(3, 3, i)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    ax.plot(y_train, '-g', label='y_train')
    ax.plot(np.arange(400, 500) ,y_test, '-b', label='y_test')
    ax.plot(np.arange(400, 500) ,preds, '-r', label='y_predict')
    neg_mse = model.score(X_test, y_test)
    performance.append([neg_mse, names[i-1]])
    ax.text(.5, .5, '%.2f' % neg_mse, horizontalalignment='center', verticalalignment='center')
    ax.set_title(names[i-1])
    ax.legend()
plt.tight_layout()
plt.show()
performance.sort()
data = list(zip(*performance))
plt.plot(data[1], data[0], '-.o', figure=plt.figure(figsize=(15, 7)))
for a, b in performance: plt.text(b, a, '%.2f' % a, ha='center', va='bottom', fontsize=9)
plt.grid()
plt.show()
