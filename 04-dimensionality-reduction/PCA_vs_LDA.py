import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = load_iris()
X, y = iris.data, iris.target

pca = PCA(n_components=2)
# X_r = pca.fit(X).transform(X)
X_r = pca.fit_transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
# X_r2 = lda.fit(X, y).transform(X)
X_r2 = lda.fit_transform(X, y=y)

print(f'explained variance ratio (first two components): {pca.explained_variance_ratio_}')

colors = ['navy', 'turquoise', 'darkorange']
data = [(X_r, 'PCA'), (X_r2, 'LDA')]
plt.figure(figsize=(15, 7))
for i, (X_transformed, title) in enumerate(data, 1):
    ax = plt.subplot(1, 2, i)
    for color, j, target_name in zip(colors, [0, 1, 2], iris.target_names):
        ax.scatter(X_transformed[y==j, 0], X_transformed[y==j, 1],
            color=color, alpha=.8, lw=2, label=target_name)
    ax.set_title(f'{title} of IRIS dataset')
    ax.legend(loc='best', shadow=False, scatterpoints=1)
plt.tight_layout()
plt.show()