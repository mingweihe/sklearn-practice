import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, IncrementalPCA

iris = load_iris()
X, y = iris.data, iris.target
# print(X.shape)
# print(X)
n_components = 2
ipca = IncrementalPCA(n_components=n_components, batch_size=10)
X_ipca = ipca.fit_transform(X)
# print(X_ipca.shape)
# print(X_ipca)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
# print(X_pca.shape)
# print(X_pca)

colors = ['navy', 'turquoise', 'darkorange']
data = [(X_ipca, 'Incremental PCA'), (X_pca, 'PCA')]
plt.figure(figsize=(15, 7))
for i, (X_transformed, title) in enumerate(data, 1):
    ax = plt.subplot(1, 2, i)
    for color, j, target_name in zip(colors, [0, 1, 2], iris.target_names):
        ax.scatter(X_transformed[y==j, 0], X_transformed[y==j, 1],
            color=color, lw=2, label=target_name)
    if 'Incremental' in title:
        err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
        ax.set_title(f'{title} of iris dataset\nMean absolute unsinged error {err:.6f}')
    else:
        ax.set_title(f'{title} of iris dataset')
    ax.legend(loc='best', shadow=False, scatterpoints=1)
    ax.axis([-4, 4, -1.5, 1.5])
plt.tight_layout()
plt.show()

