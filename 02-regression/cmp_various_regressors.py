import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
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

model_decision_tree = DecisionTreeRegressor()
model_decision_tree.fit(X_train, y_train)
preds = model_decision_tree.predict(X_test)

plt.figure()
# plt.plot(y_train, '-go', label='y_train')
plt.plot(np.arange(400, 500) ,y_test, '-bo', label='y_test')
plt.plot(np.arange(400, 500) ,preds, '-ro', label='y_predict')
plt.legend()
plt.show()

# TBC