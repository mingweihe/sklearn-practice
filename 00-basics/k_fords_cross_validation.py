import pandas
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import datasets

boston = datasets.load_boston()
X, y = boston.data, boston.target
kford = model_selection.KFold(n_splits=10, random_state=7)
regressors = [LinearRegression(), Ridge(), Lasso(), ElasticNet(),
    KNeighborsRegressor(), DecisionTreeRegressor(), SVR(gamma='scale')]
for model in regressors:
    res = model_selection.cross_val_predict(model, X, y, cv=kford)
    print(res.mean())
