import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from collections import defaultdict

h = .02
names = ['KNN', 'Linear SVM', 'RBF SVM', 'Gaussian Process',
    'Decision Tree', 'Random Forest', 'Neural Net', 'AdaBoost', 'Naive Bayes', 'QDA']
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel='linear', C=.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.*RBF(1.)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
    random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separate = (X, y)
datasets = [make_moons(noise=.3, random_state=0), make_circles(noise=.2, factor=.5,
    random_state=1), linearly_separate]
# quick note for different values of fmt parameter in plt.plot
# . , o v ^ < > 1 2 3 4 s p * h H + x D d | _
# - -- -. :
# b g r c m y k w
# data = list(zip(*datasets[0][0]))
# plt.plot(data[0], data[1], '-.og')
# plt.show()
figure=plt.figure(figsize=(27, 8))
i = 1
performances = defaultdict(float)
# iterate over these three datasets
for ds_cnt, ds in enumerate(datasets):
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # plot dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers)+1, i)
    if ds_cnt == 0: ax.set_title('Input data')
    # training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    # test points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=.6, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    # no ticks on x and y axis
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        performances[name] += score
        #plot the decision boundaries
        if hasattr(clf, 'decision_function'): Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else: Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        # put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        # training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
        # test points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=.6, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        # no ticks on x and y axis
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0: ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
        i += 1
plt.tight_layout()
plt.show()
# calculate overall performance
for k, v in performances.items(): performances[k] = v/3
data = list(zip(*sorted(performances.items(), key=lambda x:x[1])))
plt.plot(data[0], data[1], '-.o', figure=plt.figure(figsize=(15, 7)))
for a, b in zip(data[0], data[1]): plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=9)
plt.grid()
plt.show()




