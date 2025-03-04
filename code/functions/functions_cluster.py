#funcoes para a clusterizacao dos segmentos
import numpy as np
from math import ceil

import pandas as pd
import psutil
import time
import datetime
import sys
import os
import gc
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

# for hierarchical clustering
from scipy.cluster.hierarchy import single, complete, average, ward, dendrogram
from scipy.cluster.hierarchy import fcluster,to_tree, leaders


import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.colorbar import Colorbar

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        #20250221: como retorna o indice do max nao soma min_cl
    if dist:
        return distances.index(np.max(distances)) + min_cl, distances
    else:
        return distances.index(np.max(distances)) + min_cl


# pode usar o import abaixo no lugar de copiar o codigo da função aqui
# from sklearn.metrics.cluster._unsupervised import _silhouette_reduce
def _silhouette_reduce(D_chunk, start, labels, label_freqs, sh_print =False):
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
        print ('issparse')
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
        print ("entrei no else _silhouette_reduce")
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
    # intra_cluster of each point here is the sum of distances of it to others points of the same label
    print (f'1. cluster_distance = \n{cluster_distances}\nlabel_freqs= {label_freqs}') if sh_print else None#inseri 20241119
    intra_cluster_distances = cluster_distances[intra_index]
    print (f'intra_cluster_distance = \n{intra_cluster_distances}') if sh_print else None #inseri 20241119
    intra_cluster_distances /= label_freqs #20241119: inclui esta linha : nao funcionou com o calculo sem matriz de distancia
    # of the remaining distances we normalise and extract the minimum
    print (f'2. cluster_distance = \n{cluster_distances}\nlabel_freqs= {label_freqs}') if sh_print else None #inseri 20241119
    cluster_distances[intra_index] = np.inf
    # Para o cáclulo do inter cluster faz a média da distancias do ponto 
    # para os outros pontos de outros clusters, e fica com o menor valor entre os
    # labels
    cluster_distances /= label_freqs
    print (f'3. cluster_distance = \n{cluster_distances}\nlabel_freqs= {label_freqs}') if sh_print else None #inseri 20241119
    # obtem a menor distancia
    inter_cluster_distances = cluster_distances.min(axis=1)
    return intra_cluster_distances, inter_cluster_distances

from sklearn.metrics.cluster._unsupervised import _silhouette_reduce
def calc_inter_intra_cluster(arraybands_sel, dic_cluster, n_opt, metric = "euclidean", sh_print=False): 
    le = LabelEncoder()
    labels = le.fit_transform(dic_cluster[n_opt])
    n_samples = len(labels)
    label_freqs = np.bincount(labels)
    print (f'n_opt = {n_opt} n_samples = {n_samples}, label_freqs = {label_freqs}')
    # check_number_of_labels(len(le.classes_), n_samples)    
    # metric = "euclidean"  'precomputed' if arraybands_sel is a distance matrix
    kwds = {}    
    kwds["metric"] = metric
    reduce_func = functools.partial(
        _silhouette_reduce, labels = labels, label_freqs = label_freqs
    )
    print (f'arraybands_sel : {arraybands_sel}') if sh_print else None
    results = zip(*pairwise_distances_chunked(arraybands_sel, reduce_func=reduce_func, **kwds))
    intra_clust_dists, inter_clust_dists = results
    #para obter a frequencia de cada label e fazer a media do intra
    #subtrai um pq o valor é a media de um pto com os outros pontos dentro de um mesmo
    # label. Qdo um label tem um ponto só causa 0/0 e retorna nan
    denom = (label_freqs-1).take(labels, mode="clip")
    print (f'denom = {len(denom)}')#\nintra_clust_dists={len(intra_clust_dists)}')
    #comentei em 20241216, pq estava dando erro qdo usado para fazer o calculo com
    # a primeira clusterizacao, acho que o valor do inrta de cada ponto precisa ser 
    #dividido pela qtde de elementos do label para se ter a média da distancia do
    #ponto dentro do grupo
    if metric=='precomputed':
        with np.errstate(divide="ignore", invalid="ignore"):  
            intra_clust_dists /= denom
    intra_clust_dists = np.concatenate(intra_clust_dists)
    inter_clust_dists = np.concatenate(inter_clust_dists)
    print (f'intra {intra_clust_dists}') if sh_print else None
    df = pd.DataFrame({'label': labels, 'inter': inter_clust_dists, 'intra': intra_clust_dists })
    if metric=='euclidean':
        df["intra"] = df["intra"] / label_freqs[df["label"] - 1]
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

# 20240828: Funcao gerada a partir da gen_matrix_sim usando o np.memmap
def gen_matrix_sim_npmem(n_opt, dic_cluster, tmp_dir, perc_min=0, 
                         perc_max=1, k_selected=[], chunk_size=5000, 
                         logger='', sh_print=0):
    ''''
    gen matriz similarity com np.memmap
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
    # if not chunks:
    #     #calculate the number of elements per chunk of 100MB
    #     chunk_size_bytes = 100 * 1024 * 1024  # 100 MB in bytes
    #     element_size_bytes = np.array(dic_cluster[k_selected[0]]).itemsize
    #     elements_per_chunk = chunk_size_bytes // element_size_bytes
    #     # Calculate chunk size (approximate)
    #     chunk_size = int(np.sqrt(elements_per_chunk))
    #     print (chunk_size)
    #     # Adjust chunk size to fit the shape
    #     chunks = (chunk_size,)
        
    #n_selected = [2,3,4]
    time_ini = time.time()
    #for i, n in enumerate(n_selected):
    i=0
    # for k in k_selected:                    
    #     #k =  keys_clusters[n] 
    #     #da_arr = da.from_array(np.array(dic_cluster[k]),chunks=chunks)
    #     print (f'k: {k}')
    #     da_arr = np.array(dic_cluster[k])
    #     if (i==0):
    #         i+=1
    #         matrix_sim = (da_arr[:, None] == da_arr[None, :]).astype(np.int16)
    #         #estava só com int, que é int64
    #         continue
    #     matrix_sim = (da_arr[:, None] == da_arr[None, :]).astype(np.int16)+matrix_sim
    # matrix_sim=matrix_sim/len(k_selected)    
    # time_fim = time.time()
    # print (time_fim-time_ini)    
    # return matrix_sim

    # 20241216: cria dir tmp da matriz senao existir
    os.makedirs(os.path.dirname(tmp_dir), exist_ok=True)
    # File to store the main matrix
    filename = tmp_dir + 'matrix_sim.dat'
    # File to store the temporary matrices
    temp_filename = tmp_dir + 'temp_matrix.dat'
    len_cluster = len(dic_cluster[keys_clusters[n_opt]])
    matrix_shape = (len_cluster, len_cluster)
    #chunk_size = 5000  # Define a chunk size that your memory can handle coloquei como param da funcao

    # Create the main memmap matrix
    # memory used:
    # number of elements = 110168 * 110168 = 12,127,001,824 
    # suze of each elem , float16 = 2 bytes
    # mem(in bytes) =~ 12,127,001,824 *2 = 24,254,003,648 bytes
    # mem(in GiB) = 24,254,003,648/1024ˆ3 = 22.6 GiB
    
    matrix_sim = np.memmap(filename, dtype='float16', mode='w+', shape=matrix_shape)
    t2 = time.time()
    print (f'Tempo para gerar a matrix sim inicial np.memmap: {t2-time_ini}')
    print (f'shape/chunk: {matrix_shape[0]}/{chunk_size}= {matrix_shape[0]/chunk_size}')
    i=0
    for k in k_selected:
        print(f'k: {k}') if sh_print else None
        da_arr = np.array(dic_cluster[k])
        for start in range(0, matrix_shape[0], chunk_size):
            end = min(start + chunk_size, matrix_shape[0])        
            temp_matrix = np.memmap(temp_filename, dtype='int8', mode='w+', shape=(end-start, matrix_shape[1]))
            temp_matrix[:] = (da_arr[start:end, None] == da_arr[None, :]).astype(np.int8)
            # Create the temp_matrix as a memmap
            # temp_matrix = np.memmap(temp_filename, dtype='int8', mode='r+', shape=matrix_shape)
            # temp_matrix[:] = (da_arr[:, None] == da_arr[None, :]).astype(np.int8)         
            #temp_matrix = (da_arr[:, None] == da_arr[None, :]).astype(np.float32)
            # print(f'1. type: {type(matrix_sim)}') 
            if i == 0:
                i+=1
                # Initialize the memmap matrix with the first computation
                #matrix_sim[:] = temp_matrix
                matrix_sim[start:end, :] = temp_matrix
                print(f'{i} matrix_sim: {matrix_sim}') if sh_print else None
            else:
                i+=1
                # Add subsequent matrices to the existing memmap matrix
                #matrix_sim[:] += temp_matrix
                matrix_sim[start:end, :] += temp_matrix
                print(f'{i} {matrix_sim}') if sh_print else None             
            # print(f'Iteration {i+1}, type of matrix_sim: {type(matrix_sim)}')
            # print(matrix_sim) if sh_print else None
            del temp_matrix
            gc.collect()
        del da_arr
        gc.collect()
    # Final division operation
    matrix_sim[:] /= len(k_selected)
    time_fim = time.time()
    print (f'Tempo para gerar matrix sim: {(time_fim-time_ini):.2f}s, {(time_fim-time_ini)/60:.2f}m') 
    print(f'Type matrix: {type(matrix_sim)}') 
    
    return matrix_sim


# 20240828: Funcao gerada a partir da gen_matrix_sim
def gen_matrix_sim_np(n_opt, dic_cluster, perc_min=0, perc_max=1, k_selected=[], sh_print=0):
    ''''
    gen matriz similarity com np.memmap
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
    # if not chunks:
    #     #calculate the number of elements per chunk of 100MB
    #     chunk_size_bytes = 100 * 1024 * 1024  # 100 MB in bytes
    #     element_size_bytes = np.array(dic_cluster[k_selected[0]]).itemsize
    #     elements_per_chunk = chunk_size_bytes // element_size_bytes
    #     # Calculate chunk size (approximate)
    #     chunk_size = int(np.sqrt(elements_per_chunk))
    #     print (chunk_size)
    #     # Adjust chunk size to fit the shape
    #     chunks = (chunk_size,)
        
    #n_selected = [2,3,4]
    time_ini = time.time()
    #for i, n in enumerate(n_selected):
    i=0
    for k in k_selected:                    
        #k =  keys_clusters[n] 
        #da_arr = da.from_array(np.array(dic_cluster[k]),chunks=chunks)
        print (f'k: {k}')
        da_arr = np.array(dic_cluster[k])
        if (i==0):
            i+=1
            matrix_sim = (da_arr[:, None] == da_arr[None, :]).astype(np.int16)
            #estava só com int, que é int64
            continue
        matrix_sim = (da_arr[:, None] == da_arr[None, :]).astype(np.int16)+matrix_sim
    matrix_sim=matrix_sim/len(k_selected)    
    time_fim = time.time()
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a}: Tempo para gerar a matrix de sim em np: {(time_fim-time_ini)}s, {(time_fim-time_ini)/60}m')    
    return matrix_sim

    

#20240829: Funcao para ler snic centroid df from pickle
def read_snic_centroid(file_to_open, id=0,i=''):
    t1 = time.time()
    file_to_open = '/scratch/flavia/pca/pca_snic/n_110000_comp_2_snic_centroid_df_0.pkl'
    with open(file_to_open, 'rb') as handle:    
        b = pickle.load(handle)
    snic_centroid_df= b[id]['centroid_df']
    t2 = time.time()
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a}: {i}.1 Tempo para ler snic_centroid_df: {t2-t1}s {(t2-t1)/60}m')
    print (f'{a}: {i}.2 infos no arquivo com snic centroids: {b[id].keys()}, shape: {snic_centroid_df.shape}')
    del b
    print (f'{a}: {i}.3 snic_centroid_df.head():\n{snic_centroid_df.head()}')

    return snic_centroid_df

#20241016: Funcao para ler snic centroid df dos quadrantes e gerar o inteiro 
#           from pickle
def read_qs_gen_snic_centroid(pca_snic_dir, q_number=4, id=0,quad=1, i=''):
    t1 = time.time()
    
    t_1 = time.time()
    for q in range (1, q_number+1):
        t1 = time.time()
        q = quad if q_number==1 else q 
        print (f'q = {q}')
        # file_to_open = '/scratch/flavia/pca/pca_snic/n_110000_comp_2_snic_centroid_df_0.pkl'
        pca_snic_dirq = pca_snic_dir+'Quad_'+str(q)+'/'
        file_to_open = f'{pca_snic_dirq}quad{str(q)}_n_30000_comp_2_snic_centroid_df_0.pkl'
        with open(file_to_open, 'rb') as handle:    
            b = pickle.load(handle)
        print (f'b.keys {b[0].keys()}')
        if (q_number == 1) or (q == 1):
            id = list(b.keys())[0]
            print (f'b[{id}].keys {b[id].keys()}')

            snic_centroid_df = b[id]['centroid_df']
            del b
            t2 = time.time()
            continue
        #else:
        max_label = snic_centroid_df['label'].max()
        print (f'max_label quad{q}: {max_label}')
        df = b[id]['centroid_df']
        df['label'] = df['label'] + max_label + 1
        snic_centroid_df = pd.concat([snic_centroid_df, df], axis=0)
        
        del b,df
        t2 = time.time()

        a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        print (f'{i}.1. quad{q} Tempo leitura centroids df: {t2-t1}s, {(t2-t1)/60}m')
        print (f'{i}.1.1 quad{q} snic_centroid_df: \n{snic_centroid_df.head()}')

    snic_centroid_df.reset_index(drop=True, inplace=True)
    t_2 = time.time()
    print (f'{i}.1.2 Tempo para gerar snic_centroids df: {t_2-t_1}s, {(t2-t1)/60}m')
    print (f'{i}.1.3 snic_centroid_df: \n{snic_centroid_df.tail()}')

    return snic_centroid_df

#20240829: Funcao para gerar array to be used in cluster clara
def gen_arrayToCluster(snic_centroid_df, cols_sel='', calc_nans=1, ind='', sh_print=False):

    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    if not cols_sel:
        cols_snic_centroid = snic_centroid_df.columns[4:]
        comps = [x.split('_')[-1] for x in cols_snic_centroid]
        comps = sorted(list(set(comps)))
        print (f'{a}: {ind}. {cols_snic_centroid,} {comps}, {type(comps)}')
        cols_sel = ['avg_'+str(i) for i in comps]
    
        #get nan rows in snic_centroid
    if calc_nans:    
        #calculate the nans        
        nan_count=[]
        #20241030: acho que o cols_snic_centroid deve ser cols_sel
        nan_counts = snic_centroid_df[cols_snic_centroid].isna().sum()
        print (f'{a}: {ind}.1 nan_counts per column\n{nan_counts}')
        snic_centroid_df_nan = snic_centroid_df[snic_centroid_df.isnull().any(axis=1)]
        print (f'{a}: {ind}.2\n{snic_centroid_df_nan}') if sh_print else None
      
    #fazer o drop do nans no snic_centroid_df
    t1 = time.time()
    snic_centroid_df = snic_centroid_df.dropna()
    t2 = time.time()
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a}: {ind}.3 Tempo drop rows with nan in snic_centroid_df: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
    print (f'{a}: {ind}.4 snic_centroid_df_nan shape:{snic_centroid_df.shape}')

    #converte para numpy array
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a}: {ind}.5 cols_sel of snic_centroid_df: {cols_sel}')
    t1 = time.time()
    arraybands_sel = snic_centroid_df[cols_sel].to_numpy()
    t2 = time.time()
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a}: {ind}.6 Tempo para obter arraybands_sel do snic_centroid_df: {t2-t1}s {(t2-t1)/60}m')

    return arraybands_sel


#Function to plot imagem with clara cluster
#20240904: copied from Test_PCA_img_full_working-Copy notebook
def plot_clustered_clara(img_sel_norm, n_opt_df, list_cluster_n_opt, \
                         n_cols=3, cl_map='tab20', plot_orig_img=1, plot_centroids=1, chart_size=(12, 12)):
    ''''
    funcion to plot clustered images in cols and rows with the original in 
    first position
    '''
    img_sel_norm = np.clip(img_sel_norm, 0, 1) # fazer isso para remover valores negativos 
    if plot_orig_img:
        elemento='original_image'
        list_cluster_n_opt = np.insert(list_cluster_n_opt, 0, elemento)
    
    num_plots=len(list_cluster_n_opt)
    n_rows = np.ceil(num_plots/n_cols).astype(int)
    
    #print (f'num plots = {num_plots} , n_rows = {n_rows}')
    
    sub_titles=[]
    for k in list_cluster_n_opt:
        # segms=f"{params_test_dic[k]['segms']}/{props_df_sel[k].shape[0]}"
        # compact=params_test_dic[k]['compactness']
        # sigma=params_test_dic[k]['sigma']
        # conect=params_test_dic[k]['connectivity']
        subt=f'{k}'
        sub_titles.append(subt)
    
    #cl = {0: 'red', 1: 'green', 2: 'blue', 3:'white', 4:'orange', \
    #      5:'yellow', 6:'magenta', 7:'cyan'}
 
    cl = plt.get_cmap(cl_map)      # 'tab20'
    fig,axes = plt.subplots(nrows=int(n_rows), ncols=n_cols, sharex=True, sharey=True,figsize=chart_size)
    
    #axes = axes.flatten()

    if (plot_centroids):
        x_centroids = n_opt_df['centroid-0']
        y_centroids = n_opt_df['centroid-1']

    else:
        
        df_exploded = n_opt_df.explode('coords')
        x_pixels = [p[1] for p in list(df_exploded['coords'])]
        y_pixels = [p[0] for p in list(df_exploded['coords'])]
        
    for i, n in enumerate(list_cluster_n_opt):
                
        #df = props_df_sel[id_test][['std_B11','std_B8A','std_B02']]
        r = (i//n_cols)# + 1-1
        c = int(i%n_cols)#+1-1

        if n_rows==1:
            ax = axes[c]
        else:
            ax = axes[r,c]
        #print (r,c)
        if (r==0) & (i==0):
            ax.imshow(img_sel_norm)
            ax.set_title(f'Original Image', fontsize=7)
            ax.axis('off')
            continue

        #colors=[cl(x) for x in n_opt_df[n]]
        #n_opt_df['cor'] = n_opt_df[n].apply(lambda row: cl(row))
        
        if (plot_centroids):
            colors=[cl(x) for x in n_opt_df[n]]
            ax.scatter(x_centroids, y_centroids, s=1, color=colors)
            
            #n_opt_df['cor'] = n_opt_df[n].apply(lambda row: cl(row))
            #ax.scatter(x_centroids, y_centroids, s=1, color=n_opt_df['cor'])
        else:
            
            #df_exploded = n_opt_df.explode('coords')

            #estava usando esta opcao
            # colors=[cl(x) for x in df_exploded[n]] #tempo similar a criar uma coluna
            # ax.scatter(x_pixels, y_pixels, s=1, color=colors)
            
            # x_pixels = [p[1] for p in list(df_exploded['coords'])]
            # y_pixels = [p[0] for p in list(df_exploded['coords'])]
                
            #df_exploded['cor'] = df_exploded[n].apply(lambda row: cl(row))                        
            #ax.scatter(x_pixels, y_pixels, s=1, color=df_exploded['cor'])
            
            n_opt_df['cor'] = n_opt_df[n].apply(lambda row: cl(row))
            df_exploded = n_opt_df.explode('coords')
            ax.scatter(x_pixels, y_pixels, s=1, color=df_exploded['cor'])
            del df_exploded            
        
        ax.imshow(img_sel_norm)
        #axes[r, c].imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
        ax.set_title(f'Image clustered n_opt {n}', fontsize=7)
               
        # Customize subplot title
        #axes[r,c].set_title(sub_titles[i], fontsize=7)

        # Hide axis ticks and labels
        ax.axis('off')
        
    # tem que setar visible to false the subs plots to complement the grid
    # of subplots
    num_subs = n_cols*n_rows
    if (num_plots)< num_subs:
        for cc in range(c+1,n_cols):
            #print (r,cc)
            ax.axis('off')
            ax.set_visible(False)

    #fig.update_layout(showlegend=True, title_font=dict(size=10), width=chart_size[0], height=chart_size[1])

    # # Update subtitle font size for all subplots
    # for annotation in fig['layout']['annotations']:
    #     annotation['font'] = dict(size=10)

    # Adjust layout
    plt.tight_layout()
    plt.show()
    #fig.show()

# 20241103: function to use distance matrix, copied from: 
# /Users/flaviaschneider/Documents/flavia/Doutorado/ESG/Dash_MHTS/programs/clustering_hierararchical.py
#calculate hierarchical clustering
def hierarchical_clustering(distance_matrix, method='average'):
    if method == 'complete':
        Z = complete(distance_matrix)
    if method == 'single':
        Z = single(distance_matrix)
    if method == 'average':
        Z = average(distance_matrix)
    if method == 'ward':
        Z = ward(distance_matrix)
    
    # fig = plt.figure(figsize=(16, 8))
    # dn = dendrogram(Z)
    # plt.title(f"Dendrogram for {method}-linkage with correlation distance")
    # plt.show()
    
    return Z