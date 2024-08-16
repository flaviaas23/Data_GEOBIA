#funcoes para a clusterizacao dos segmentos
import numpy as np
from math import ceil

import pandas as pd
import psutil
import time
import sys
import os
import glob
import copy

from itertools import product

import pickle
import random
from tqdm import tqdm

import dask.array as da
import dask.dataframe as dd 

from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances_chunked
from sklearn.preprocessing import LabelEncoder
import functools
from scipy.sparse import issparse

from sklearn_extra.cluster import CLARA

def optimal_number_of_clusters(wcss,min_cl, max_cl, dist=False):
    #20240803: 
    #para retornar o numero de clusters ótimo, soma-se o min_cl
    #mas se for a posicao da chave do melhor cluster nao deve-se somar
    #ou subtrai na recebimento da função
    import math
    x1, y1 = min_cl, wcss[0]
    x2, y2 = max_cl, wcss[len(wcss)-1]
    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    if dist:
        return distances.index(np.max(distances)) + min_cl, distances
    else:
        return distances.index(np.max(distances)) + min_cl

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
    #print (f'start: {start}')
    end = start + n_chunk_samples
    intra_index = (np.arange(n_chunk_samples), labels[start:end])
    # intra_cluster_distances are averaged over cluster size outside this function
    intra_cluster_distances = cluster_distances[intra_index]
    # of the remaining distances we normalise and extract the minimum
    cluster_distances[intra_index] = np.inf
    cluster_distances /= label_freqs
    inter_cluster_distances = cluster_distances.min(axis=1)
    return intra_cluster_distances, inter_cluster_distances

def calc_inter_intra_cluster(arraybands_sel, dic_cluster, n_opt, sh_print=False):
    
    le = LabelEncoder()
    labels = le.fit_transform(dic_cluster[n_opt])
    n_samples = len(labels)
    label_freqs = np.bincount(labels)
    # check_number_of_labels(len(le.classes_), n_samples)
    
    metric = "euclidean"
    kwds = {}
    
    kwds["metric"] = metric
    reduce_func = functools.partial(
        _silhouette_reduce, labels=labels, label_freqs=label_freqs
    )
    print (f'arraybands_sel : {arraybands_sel}') if sh_print else None
    results = zip(*pairwise_distances_chunked(arraybands_sel, reduce_func=reduce_func, **kwds))
    intra_clust_dists, inter_clust_dists = results
    intra_clust_dists = np.concatenate(intra_clust_dists)
    inter_clust_dists = np.concatenate(inter_clust_dists)
    print (f'intra {intra_clust_dists}') if sh_print else None
    df = pd.DataFrame({'label': labels, 'inter': inter_clust_dists, 'intra': intra_clust_dists })
    return df


def gen_matrix_sim_da(n_opt, dic_cluster, perc_min=0, perc_max=1, k_selected=[], chunks=0):
    ''''
    gen matriz similarity dask
    '''
    #n_ini =2 if not specified a perc_min for n_opt
    keys_clusters = list(dic_cluster.keys())
    if not k_selected:
        n_ini = 2 if not perc_min else ceil(perc_min*n_opt)        
        max_n = len(keys_clusters) - 1       
        max_n = ceil(perc_max*n_opt) if perc_max*n_opt <= max_n else max_n 
        print (n_ini, max_n)
        #n_selected = list(range(n_ini,ceil(n_opt*1.2)+1))
        n_selected = list(range(n_ini,ceil(max_n)+1))
        k_selected = [keys_clusters[n] for n in n_selected]
    # else:
    #     n_selected = n_sel          
    print (f'keys_clusters[n_opt]: {keys_clusters[n_opt]}')
    if not chunks:
        #calculate the number of elements per chunk of 100MB
        chunk_size_bytes = 100 * 1024 * 1024  # 100 MB in bytes
        element_size_bytes = np.array(dic_cluster[k_selected[0]]).itemsize
        elements_per_chunk = chunk_size_bytes // element_size_bytes
        # Calculate chunk size (approximate)
        chunk_size = int(np.sqrt(elements_per_chunk))
        print (chunk_size)
        # Adjust chunk size to fit the shape
        chunks = (chunk_size,)
        
    #n_selected = [2,3,4]
    time_ini = time.time()
    #for i, n in enumerate(n_selected):
    i=0
    for k in k_selected:                    
        #k =  keys_clusters[n] 
        da_arr = da.from_array(np.array(dic_cluster[k]),chunks=chunks)
        if (i==0):
            i+=1
            matrix_sim = (da_arr[:, None] == da_arr[None, :]).astype(np.int16)  #estava só com int, que é int64
            continue
        matrix_sim = (da_arr[:, None] == da_arr[None, :]).astype(np.int16)+matrix_sim
    matrix_sim=matrix_sim/len(k_selected)    
    time_fim = time.time()
    print (time_fim-time_ini)    
    return matrix_sim