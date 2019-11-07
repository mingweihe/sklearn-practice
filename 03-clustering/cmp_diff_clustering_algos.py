import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.cluster import MeanShift
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture

np.random.seed(0)
n_samples = 1500

# generate 6 datasets
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

random_state = 170
transformation = [[.6, -.6], [-.4, .8]]
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1., 2.5, .5], random_state=random_state)

# set up cluster parameters
plt.figure(figsize=(15, 8.2))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
plot_num = 1
default_base = {
    'quantile': .3,
    'eps': .3,
    'damping': .9,
    'preference': -200,
    'n_neighbors': 10,
    'n_clusters': 3,
    'min_samples': 20,
    'xi': .05,
    'min_cluster_size': .1
}
datasets = [
    (noisy_circles, {'damping': .77, 'preference': -240, 'quantile': .2, 
        'n_clusters': 2, 'min_samples': 20, 'xi': .25}),
    (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
    (varied, {'eps': .18, 'n_neighbors': 2, 'min_samples': 5,
        'xi': .035, 'min_cluster_size': .2}),
    (aniso, {'eps': .15, 'n_neighbors': 2, 'min_samples': 20,
        'xi': .1, 'min_cluster_size': .2}),
    (blobs, {}),
    (no_structure, {})
]

for i_dataset, (dataset, algo_param) in enumerate(datasets):
    params = default_base.copy()
    params.update(algo_param)
    X, y = dataset
    # normalize dataset for easir parameter selection
    X = StandardScaler().fit_transform(X)
    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])
    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params['n_neighbors'], include_self=False
    )
    # make connectivity symmetric
    connectivity = .5 * (connectivity + connectivity.T)
    # create clustering models
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = MiniBatchKMeans(n_clusters=params['n_clusters'])
    ward = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward', connectivity=connectivity)
    spectral = SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack', affinity='nearest_neighbors')
    dbscan = DBSCAN(eps=params['eps'])
    optics = OPTICS(min_samples=params['min_samples'], xi=params['xi'], min_cluster_size=params['min_cluster_size'])
    affinity_propagation = AffinityPropagation(damping=params['damping'], preference=params['preference'])
    average_linkage = AgglomerativeClustering(linkage='average', affinity='cityblock',
        n_clusters=params['n_clusters'], connectivity=connectivity)
    birch = Birch(n_clusters=params['n_clusters'])
    gmm = GaussianMixture(n_components=params['n_clusters'], covariance_type='full')

    clustering_models = (
        ('MiniBatchKMeans', two_means),
        ('AffinityPropagation', affinity_propagation),
        ('MeanShift', ms),
        ('SpectralClustering', spectral),
        ('Ward', ward),
        ('AgglomerativeClustering', average_linkage),
        ('DBSCAN', dbscan),
        ('OPTICS', optics),
        ('Birch', birch),
        ('GaussianMixture', gmm)
    )

    for name, model in clustering_models:
        t0 = time.time()
        # catch warnings ralated to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message='the number of connected components of the ' +
                'connectivity matrix is [0-9]{1,2}' +
                ' > 1. Completing it to avoid stopping the tree early.',
                category=UserWarning
            )
            warnings.filterwarnings(
                'ignore', message='Graph is not fully connected, spectral embedding' +
                ' may not work as expected.',
                category=UserWarning
            )
            model.fit(X)
        t1 = time.time()
        if hasattr(model, 'labels_'): y_pred = model.labels_.astype(np.int)
        else: y_pred = model.predict(X)
        plt.subplot(len(datasets), len(clustering_models), plot_num)
        if i_dataset == 0: plt.title(name, fontsize=10)
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
            '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']), int(max(y_pred)+1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ['#000000'])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1-t0)).lstrip('0'), transform=plt.gca().transAxes,
            size=15, horizontalalignment='right')
        plot_num += 1
# plt.tight_layout()
plt.show()
