#20240807: codigo para testar a geracao da matriz de distancia que vai ser usada na cluserizacao
#           com clara
#20240807: estava dando um erro com compute, tentei usar o kmeans do dask, mas estava
#           com erro tb, parei por enquanto para tengtar migrar para o sparky

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
from dask_ml.decomposition import PCA

from pysnic.algorithms.snic import snic
#For snic SP
from pysnic.algorithms.snic import compute_grid
from pysnic.ndim.operations_collections import nd_computations
from itertools import chain
from pysnic.metric.snic import create_augmented_snic_distance
from timeit import default_timer as timer

import gc
import datetime
from dask_ml.decomposition import PCA
from dask_ml.preprocessing import StandardScaler
from dask.diagnostics import ProgressBar
from dask.array import from_zarr

#for slic SP
from skimage.segmentation import slic
from skimage.measure import regionprops, regionprops_table

#for clustering
from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances_chunked
from sklearn.preprocessing import LabelEncoder
import functools
from scipy.sparse import issparse

from sklearn_extra.cluster import CLARA

import matplotlib.pyplot as plt

ProgressBar().register()

####### Funcoes
def calc_avg_array(arr, img_dic, c, med='med', pca=1):
    # Map array elements to dictionary values if they are not nan (-9999, -32768)
    #pixel_values_map = [pixel_band_value[elem[0], elem[1]] for elem in arr]
    if not pca:
        pixel_values_map = [img_dic[b][elem[0], elem[1]] for elem in arr if img_dic[b][elem[0], elem[1]] not in [-9999, -32768]]
    else:
        pixel_values_map = [img_dic[c][elem[0], elem[1]] for elem in arr if img_dic[c][elem[0], elem[1]]]
    #print (f'{b}\nmapped array: {pixel_values_map}')
    # Calculate median deviation of mapped values
    #median or mean: average
    
    arr_avg = np.median(pixel_values_map) if pixel_values_map else np.nan
    return arr_avg

def calc_avg_array_pca(arr, img_dic, c, med='med' ):
    # Map array elements to dictionary values if they are not nan (-9999, -32768)
    #pixel_values_map = [pixel_band_value[elem[0], elem[1]] for elem in arr]
    #20240803: comentei linha abaixo e passando a imagem em vez do img_dic
    #pixel_values_map = [img_dic[c][elem[0], elem[1]] for elem in arr if img_dic[c][elem[0], elem[1]]]
    pixel_values_map = [img_dic[elem[0], elem[1]] for elem in arr if img_dic[elem[0], elem[1]]]
    #print (f'{b}\nmapped array: {pixel_values_map}')
    # Calculate median deviation of mapped values
    #median or mean: average
    
    if med == 'med':
        arr_calc = np.median(pixel_values_map) if len(pixel_values_map) else np.nan
    elif med == 'avg':    
        arr_calc = np.mean(pixel_values_map) if len(pixel_values_map) else np.nan
    elif med == 'std':
        arr_calc = np.std(pixel_values_map) if len(pixel_values_map) else np.nan        
    else:
        arr_calc = pixel_values_map[0]#np.mean(pixel_values_map) if pixel_values_map else np.nan
    return arr_calc

def gen_coords_snic_df(segments_snic_sel_sp, sh_print=False):
    #create dictionaries for coords and num pixels
    t1=time.time()
    coords_snic_dic={}
    for i in range(len(segments_snic_sel_sp)):
        for j in range(len(segments_snic_sel_sp[i])):
            current = segments_snic_sel_sp[i][j]
            if current in coords_snic_dic:
                coords_snic_dic[current].append([i , j ])
            else:
                coords_snic_dic[current] = [[i , j ]]
    t2=time.time()    
    #count number of pixels if each group
    num_pixels_dic={}
    for l in coords_snic_dic.keys():
        num_pixels_dic[l] = len(coords_snic_dic[l])
    t3=time.time()    
    print (t2-t1, t3-t2) if sh_print else None
    # create a df using the dictionaries coords_snic_dic and num_pixels_dic
    t4=time.time()
    coords_snic_df =pd.DataFrame({'label':coords_snic_dic.keys(), \
                                  'coords':coords_snic_dic.values(), \
                                  'num_pixels': num_pixels_dic.values()},
                                index=coords_snic_dic.keys())
    coords_snic_df.sort_values('label', inplace=True)
    #coords_snic_df['label'] = coords_snic_dic.keys()
    #coords_snic_df=coords_snic_df.set_index('label')
    t5=time.time()
    print (t5-t1, t5-t4) if sh_print else None
    # n_part = round(coords_snic_df.shape[0] * coords_snic_df.shape[1]/1000)    
    # dd_coords_snic_df = dd.from_pandas(coords_snic_df,npartitions=n_part)    
    return coords_snic_df

#20240730: versao ok
def gen_centroid_snic_dask(image_band_dic, centroids_snic_sp, coords_snic_df, \
                         bands_sel, ski=True, stats=True, sh_print=True):
    '''
    Function to gen a df from snic centroids
    Ski: 
        True: considere the avg values of snic centroid 
              and do avg only of bands not used in snic
        False: do avg of all bands 
    '''   
    # gen a dictionary with num_pixels, centroids and bands/pca 
    # values and do a df with results of snic segmentation    
    t1=time.time()
    snic_n_segms = len(centroids_snic_sp)
    #centroids_snic_only = [subar[0] for subar in centroids_snic_sp]
    centroids_snic_sp_dic={}
    centroids_snic_sp_dic['label'] = [i for i in range(snic_n_segms)]
    centroids_snic_sp_dic['num_pixels'] = [subar[2] for subar in centroids_snic_sp]
    centroids_snic_sp_dic['centroid-0'] = [subar[0][0] for subar in centroids_snic_sp]
    centroids_snic_sp_dic['centroid-1'] = [subar[0][1] for subar in centroids_snic_sp]
    if ski:
        for i,b in enumerate(bands_sel):
            centroids_snic_sp_dic['avg_'+b] = [subar[1][i] for subar in centroids_snic_sp]
    t2=time.time()
    print (f'time to gen centroids dictionary: {t2-t1}') if sh_print else None
    snic_centroid_df = pd.DataFrame(centroids_snic_sp_dic)
    del centroids_snic_sp_dic     

    n_part = round(snic_centroid_df.shape[0] * snic_centroid_df.shape[1]/10000)        
    dd_snic_centroid_df = dd.from_pandas(snic_centroid_df,npartitions=n_part)  
    dd_snic_centroid_df = dd_snic_centroid_df.repartition(partition_size="100MB")
    dd_coords_snic_df =  dd.from_pandas(coords_snic_df,npartitions=n_part)
    dd_coords_snic_df = dd_coords_snic_df.repartition(partition_size="100MB")
    del coords_snic_df, snic_centroid_df
    
    #criar as stats das bandas e SPs
    if stats:                
        for c in image_band_dic.keys():            
            # no nome da banda vai ser a mediana do sp 
            t1=time.time()
            dd_snic_centroid_df['med_'+c] = dd_coords_snic_df['coords'].apply(calc_avg_array_pca, img_dic=image_band_dic[c],\
                                                              c=c, med='med', meta=dd_coords_snic_df['coords'])
            t2=time.time()
            dd_snic_centroid_df['std_'+c] = dd_coords_snic_df['coords'].apply(calc_avg_array_pca, img_dic=image_band_dic[c],\
                                                              c=c, med='std', meta=dd_coords_snic_df['coords'])
            t3=time.time()
            print (f'{c} tempo calc mediana: {t2-t1}s, std: {t3-t2}s ') if sh_print else None
            # Não precisa fazer a média pq o snic retorna as medias das bandas de cada sp (grupo)
            # valores incluidos acima no if do ski
            # testei e vi que os valores sao iguais
            # testar se o avg da banda já nao existe e incluir
            if not ski:
                print (c, ski, "do avg") if sh_print else None
                dd_snic_centroid_df['avg_'+c] = dd_coords_snic_df['coords'].apply(calc_avg_array_pca, img_dic=image_band_dic[c],\
                                                                   c=c, med='avg', meta=dd_coords_snic_df['coords'])
            elif (c not in bands_sel): 
                print (c, "do avg") if sh_print else None
                dd_snic_centroid_df['avg_'+c] = dd_coords_snic_df['coords'].apply(calc_avg_array_pca, img_dic=image_band_dic[c],\
                                                                   c=c, med='avg', meta=dd_coords_snic_df['coords'])
                
                #dd_centroid_df[[c, 'avg_'+c, 'std_'+c]] = dd_coords_SP_df['coords'].compute().apply(calc_avg_array, img_dic=image_band_dic[c],\
                                                           #   c=c).apply(pd.Series)
    return dd_snic_centroid_df, dd_coords_snic_df  

#for clustering
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
        print (f'chunk_size: {chunk_size}')
        # Adjust chunk size to fit the shape
        chunks = (chunk_size*20,)
        print (f'chunks: {chunks}')
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
    matrix_sim=matrix_sim.astype(np.float32)
    time_fim = time.time()
    print (time_fim-time_ini)    
    return matrix_sim

def update_rows(target, source):
    for idx in source.index:
        if idx in target.index:
            target.loc[idx] = source.loc[idx]
    return target

def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


#from sklearn.cluster import KMeans
from dask_ml.cluster import KMeans
def cria_grupos_kmeans(n, dist_sim_matrix):
        ''''
        Cria cluster based on distance matrix
        '''
        KMeans(n_clusters=8, init='k-means||', oversampling_factor=2, max_iter=300,
                tol=0.0001, precompute_distances='auto', random_state=None, copy_x=True,
                  n_jobs=1, algorithm='full', init_max_iter=None, n_init='auto')
        kmeans = KMeans(n_clusters=n, random_state=0, n_init="auto")
        kmeans_fit = kmeans.fit(dist_sim_matrix)
        clusters_centers = kmeans_fit.cluster_centers_
        cluster_labels = kmeans_fit.labels_
        cluster_inertia = kmeans_fit.inertia_
        return cluster_labels, clusters_centers, cluster_inertia
#%%

sh_print_n1=1
sh_print_n2=1
# Read PCA images
cur_dir = os.getcwd()
print (f'{cur_dir}')


######## read cluster clara
#read clusters
read_path = cur_dir + '/data/pca_snic/'
# read_path = '/scratch/flavia/pca/'
# '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/pca_snic/'
f_name = 'n110k_c2_s1_con0_'
f_read = f_name+'snic_clara.pkl'
print (f_read)
#n110k_c2_s1_con0_segments_snic_sel_pca_sp.pkl
with open(read_path+f_read, 'rb') as handle: 
    clara_obj = pickle.load(handle)

sse_ski = clara_obj[id]['sse_ski']
dic_cluster = clara_obj[id]['dic_cluster']
keys_ski=list(dic_cluster.keys())
n_clusters = int(keys_ski[-1].split('_')[0])
####### fim read
n_opt_ski, distances = optimal_number_of_clusters(sse_ski, 2, 3*n_clusters, dist=True)

print (n_opt_ski)
#47

print (keys_ski[n_opt_ski])
#'17_612'
print (keys_ski[n_opt_ski-2])
#'17_611'
print (distances.index(np.max(distances)), len(distances))
#(45, 87)
#o np_opt_ski como indice deve-se subtrair 2 , explicado o pq na funcao

#obter as keys das maiores distancias
dist_key={}
dist_key = {keys_ski[i]: distances[i] for i in range(len(keys_ski))}
#dist_key[:10] dicionario , este  comando nao funciona

sorted_keys = [k for k, v in sorted(dist_key.items(), key=lambda item: item[1], reverse=True)]
print (sorted_keys[:3])
#['17_611', '16_917', '12_837']
for k in sorted_keys[:5]:
    print (f'{k}: {dist_key[k]}')

# 2. Calculate Similarity Matriz

##### Calculate distance matrix choosing the clusters groups

a = round((n_opt_ski-2)*1.5)
b = len(distances)-1
max_ind = a if a<=b else b
min_ind = round(n_opt_ski*0.5) - 1
n_elem=max_ind-min_ind
print (n_elem)
# 45
ind_opts=[]
keys_n_opt=[]

k_selected=sorted_keys[:n_elem]
#gerar matriz similaridade

t1=time.time()
matrix_sim_dask = gen_matrix_sim_da(n_opt_ski-2, dic_cluster, k_selected=k_selected, chunks=(95000,))
t2=time.time()
print (f'1. tempo para gerar matrix similaridade com clusters: {t2-t1}')
#matrix_dist_dask
matrix_dist_dask = 1 - matrix_sim_dask
del matrix_sim_dask

t1=time.time()
matrix_dist= matrix_dist_dask.compute()
t2=time.time()
del matrix_dist_dask
print (f'2. tempo para gerar matrix distancia com compute: {t2-t1}')

save_path=read_path
t1=time.time()
save_to_pickle(matrix_dist, save_path+'matrix_dist_snic_clara.pkl')
t2=time.time()

print (f'3. tempo para salvar matrix de distancia: {t2-t1}')

n_clusters = 3
dic_cl_km_mskm = {}
sse_cl_km_mskm = []
for n in tqdm(range(2, n_clusters+1)):
    #print ("n_clusters= ", n_clusters)
    #cluster_labels[n_cluster],silhouette_avg[n_clusters],sample_silhouette_values[n_clusters]=cria_grupos(n_clusters, df_cluster_array_1k, "Kmeans", "dtw", seed=997)
    dic_cl_km_mskm[n], clusters_center, sse = cria_grupos_kmeans(n, matrix_dist_dask)
    sse_cl_km_mskm.append(sse)

#calculo do n_opt
n_opt_cl_km_mskm = optimal_number_of_clusters(sse_cl_km_mskm, 2, n_clusters, dist=False)
print (f'n_opt_cl_km_mskm: {n_opt_cl_km_mskm}')