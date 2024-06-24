''''
20240410: Programa para testar as metricas de inter e intra cluster
'''

import numpy as np
from sklearn.metrics import pairwise_distances_chunked
from sklearn.preprocessing import LabelEncoder
import functools
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from scipy.sparse import issparse
import pandas as pd


# functions ----
def _silhouette_reduce(D_chunk, start, labels, label_freqs):
    """Accumulate silhouette statistics for vertical chunk of X.

    Parameters
    ----------
    D_chunk : {array-like, sparse matrix} of shape (n_chunk_samples, n_samples)
        Precomputed distances for a chunk. If a sparse matrix is provided,
        only CSR format is accepted.
    start : int
        First index in the chunk.
    labels : array-like of shape (n_samples,)
        Corresponding cluster labels, encoded as {0, ..., n_clusters-1}.
    label_freqs : array-like
        Distribution of cluster labels in ``labels``.
    """
    n_chunk_samples = D_chunk.shape[0]
    # accumulate distances from each sample to each cluster
    cluster_distances = np.zeros(
        (n_chunk_samples, len(label_freqs)), dtype=D_chunk.dtype
    )

    if issparse(D_chunk):
        if D_chunk.format != "csr":
            raise TypeError(
                "Expected CSR matrix. Please pass sparse matrix in CSR format."
            )
        for i in range(n_chunk_samples):
            indptr = D_chunk.indptr
            indices = D_chunk.indices[indptr[i] : indptr[i + 1]]
            sample_weights = D_chunk.data[indptr[i] : indptr[i + 1]]
            sample_labels = np.take(labels, indices)
            cluster_distances[i] += np.bincount(
                sample_labels, weights=sample_weights, minlength=len(label_freqs)
            )
    else:
        for i in range(n_chunk_samples):
            sample_weights = D_chunk[i]
            sample_labels = labels
            cluster_distances[i] += np.bincount(
                sample_labels, weights=sample_weights, minlength=len(label_freqs)
            )

    # intra_index selects intra-cluster distances within cluster_distances
    end = start + n_chunk_samples
    intra_index = (np.arange(n_chunk_samples), labels[start:end])
    # intra_cluster_distances are averaged over cluster size outside this function
    intra_cluster_distances = cluster_distances[intra_index]
    # of the remaining distances we normalise and extract the minimum
    cluster_distances[intra_index] = np.inf
    cluster_distances /= label_freqs
    inter_cluster_distances = cluster_distances.min(axis=1)
    return intra_cluster_distances, inter_cluster_distances


X, y = make_blobs(n_samples=50, random_state=42)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

print(labels)

le = LabelEncoder()
labels = le.fit_transform(labels)
n_samples = len(labels)
label_freqs = np.bincount(labels)
# check_number_of_labels(len(le.classes_), n_samples)

metric = "euclidean"
kwds = {}

kwds["metric"] = metric
reduce_func = functools.partial(
    _silhouette_reduce, labels=labels, label_freqs=label_freqs
)
results = zip(*pairwise_distances_chunked(X, reduce_func=reduce_func, **kwds))
intra_clust_dists, inter_clust_dists = results
intra_clust_dists = np.concatenate(intra_clust_dists)
inter_clust_dists = np.concatenate(inter_clust_dists)

df = pd.DataFrame({'label': labels, 'inter': inter_clust_dists, 'intra': intra_clust_dists })
print(df)

result = df.groupby(['label']).mean()

print(result)

