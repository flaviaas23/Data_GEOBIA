# Programa para ler resultado do cluster snic pca com ms, faz a matriz de 
# similaridade 
# 20240828
# 20241016: baixado do exacta e alterado para rodar local no laptop
#           e ler e gerar o snic_centroid_df de cada quadrante
# 20241215: inclusao do log e do df com os tempos de processamento
#           clusterizacao da segmentação de 1/4 do tile, nao juntar os 
#           quadrantes
# 20241215: passagem de argumentos para o programa
# 20250131: gerado copia do spk_gen2_2cluster_snic_pa.py 
#         : para gerar matriz de similaridade do cluster feito com a matriz de 
#         : similaridade gerada a partir do cluster simples, 
# 20250226: gerado copia do spk_gen2_2cluster_snic_pa.py , faz selecao dos clusters que serao \
#           considerados na construcao da matriz de similaridade se pelas maiores distancias do cotovelo
#           ou todos a partir do n_opt

import os
import gc
import time
import datetime
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

import pickle
import random
from tqdm import tqdm
#from sklearn_extra.cluster import CLARA
import zarr

from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances_chunked
from sklearn.preprocessing import LabelEncoder
import functools
from scipy.sparse import issparse

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.colorbar import Colorbar

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from functions.functions_segmentation import gen_coords_snic_df
from functions.functions_pca import save_to_pickle, cria_logger, update_procTime_file
from functions.functions_cluster import optimal_number_of_clusters, gen_matrix_sim_np, \
                                        gen_matrix_sim_npmem, calc_inter_intra_cluster,\
                                        read_snic_centroid, read_qs_gen_snic_centroid,\
                                        gen_arrayToCluster

ti = time.time()

# 20241212 bloco de parse incluido
# Inicializa o parser
parser = argparse.ArgumentParser(description="Program cluster segmented image")

# Define os argumentos
parser.add_argument("-bd", '--base_dir', type=str, help="Diretorio base", default='')
parser.add_argument("-sd", '--save_dir', type=str, help="Dir base para salvar saidas de cada etapa", default='data/tmp2/')
# parser.add_argument("-td", '--tif_dir', type=str, help="Dir dos tiffs", default='data/Cassio/S2-16D_V2_012014_20220728_/')
parser.add_argument("-q", '--quadrante', type=int, help="Numero do quadrante da imagem [0-all,1,2,3,4]", default=1)
parser.add_argument("-ld", '--log_dir', type=str, help="Dir do log", default='code/logs/')
# parser.add_argument("-i", '--name_img', type=str, help="Nome da imagem", default='S2-16D_V2_012014')
parser.add_argument("-sp", '--sh_print', type=int, help="Show prints", default=0)
parser.add_argument("-pd", '--process_time_dir', type=str, help="Dir para df com os tempos de processamento", default='data/tmp2/')
# parser.add_argument("-rf", '--READ_df_features', type=int, help="Read or create df with features", default=0 )
# parser.add_argument("-nc", '--num_components', type=int, help="number of PCA components", default=4 )
# parser.add_argument("-dsi", '--image_dir', type=str, help="Diretorio da imagem pca", default='data/tmp/spark_pca_images/')
# parser.add_argument("-p", '--padrao', type=str, help="Filtro para o diretorio ", default='*')
parser.add_argument("-k", '--knn', type=str, help="Use KNN", default=False)
parser.add_argument("-md", '--max_dist_cotov', type=int, help="Sel os clusters pela dist cotovelo ou do n_opt em diante", default=1)
args = parser.parse_args()

base_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/'
# base_dir = args.base_dir
save_etapas_dir = base_dir + args.save_dir if base_dir else args.save_dir + args.name_img +'/'
# tif_dir = base_dir + args.tif_dir if base_dir else args.tif_dir
read_quad = args.quadrante 
log_dir = base_dir + args.log_dir if base_dir else args.log_dir
# name_img = args.name_img
process_time_dir = base_dir + args.process_time_dir
sh_print = args.sh_print
print (f"sh_print {sh_print}")
# n_components = args.num_components
# img_dir = args.image_dir
# padrao = args.padrao
KNN = args.knn
# indica o tipo de selecao dos clusters que serao usados na matriz de similaridade, se por distancia
# ou considerar todos a partir do n_opt
MaxDist = args.max_dist_cotov

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
t = datetime.datetime.now().strftime('%Y%m%d_%H%M_')

nome_prog = os.path.basename(__file__)
print (f'{a}: 0. INICIO {nome_prog}') if sh_print else None

# base_dir = '/scratch/flavia/pca/'
# base_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/'
#cria logger
nome_log = t + nome_prog.split('.')[0]+'.log'
logger = cria_logger(log_dir, nome_log)
logger.info(f'######## INICIO {nome_prog} ##########')
logger.info(f'args: sd={save_etapas_dir} q={read_quad} ld={log_dir} pd={process_time_dir} sp={args.sh_print}')

ri=0        #indice do dicionario com os tempos de cada subetapa
proc_dic = {}
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = f'Gen similarity matriz from Cluster segmented image for quad {read_quad} with n_opt to nmax'

# 1. read snic df
id=0
t1 = time.time()
# file_to_open = base_dir + 'pca_snic_cluster/clara_0.pkl
# KNN = 0 # passado via argumento
if KNN:
    file_to_open = base_dir + 'data/tmp/pca_snic_cluster/clara_knn'+str(id)+'.pkl'
else:
    # file_to_open = base_dir + 'data/tmp/pca_snic_cluster/clara_'+str(id)+'.pkl'
    d_name = save_etapas_dir + 'pca_snic_cluster/'
    if MaxDist:
        file_to_open = d_name+'clara_ms_'+str(id)+'_quad_'+str(read_quad)+'.pkl'
    else:    
        file_to_open = d_name+'clara_ms_'+str(id)+'_30_quad_'+str(read_quad)+'.pkl'

with open(file_to_open, 'rb') as handle:    
    b = pickle.load(handle)

t2 = time.time()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 1. Tempo leitura cluster clara ms: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m') if sh_print else None 
print (f'{a}: 1.1 Clara ms file keys: {b.keys()}') if sh_print else None
logger.info(f'1. Tempo leitura cluster clara ms: {t2-t1}s, {(t2-t1)/60}m')  
logger.info(f'1.1 Clara ms file keys: {b.keys()}')

proc_dic[ri]['subetapa'] = f'read Cluster Clara with distance matrix/sim for quad {read_quad}'
proc_dic[ri]['tempo'] = t2-t1

sse_ski_ms = b['sse_ski']
n_clusters_ms = len(sse_ski_ms)
# sse_rd = b['sse_rd'],
# rd_state = b['rd_state']
# dic_cluster_rd = b['dic_cluster_rd']
dic_cluster_ski_ms = b['dic_cluster_ski']

del b
gc.collect()

# 2. Calculate n_opt
t1 = time.time()
n_opt_ski_ms, distances_ms = optimal_number_of_clusters(sse_ski_ms, 2, 1*n_clusters_ms, dist=True)
keys_ski_ms = list(dic_cluster_ski_ms.keys())
t2 = time.time()

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
#olhar este print
print ('*********************************')
print (f'{a}:2. n_opt_ski_ms = {n_opt_ski_ms-2}\ndistances_ms = {distances_ms}')
print (f'{a}: 2. keys_ski_ms: {keys_ski_ms}') if sh_print else None  
print (f'{a}: 2.1 n_opt_ski_ms: {n_opt_ski_ms-2} , keys_ski[n_opt_ski_ms-2]: {keys_ski_ms[n_opt_ski_ms-2]}, keys_ski_ms[n_opt_ski]:{keys_ski_ms[n_opt_ski_ms-2]}') if sh_print else None
logger.info(f'2. keys_ski_ms: {keys_ski_ms}')  
logger.info(f'2.1 n_opt_ski: {n_opt_ski_ms-2} , keys_ski_ms[n_opt_ski_ms-2]: {keys_ski_ms[n_opt_ski_ms-2]}, keys_ski_ms[n_opt_ski]:{keys_ski_ms[n_opt_ski_ms-2]}')

# #como está retornando o indice da max distancia, 
# n_opt_ski_ms = n_opt_ski_ms - 2

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = f'Gen similarity matriz from Cluster segmented image {read_quad}'
proc_dic[ri]['subetapa'] = f'Get optimal number of clusters for quad {read_quad}'
proc_dic[ri]['tempo'] = t2-t1

# 3. para obter a lista com indices das maiores distancias e chaves do dic_cluster
#distances.index(np.max(distances)) + min_cl
#ordenar pelas maximas distancias
dist_copy = distances_ms.copy()
dist_copy.sort(reverse=True)
print (f'{a}: distancias: {dist_copy}') if sh_print else None
logger.info(f'distancias: {dist_copy}')

if MaxDist:
    #seleciona os clusters que tiveram as maiores distancias de acordo com o grafico do cotovelo
    ind_opts_ms=[]
    keys_n_opt_ms=[]
    max_ind_ms = round((n_opt_ski_ms-2)*1.5)          # 
    for x in dist_copy[:max_ind_ms]:
        ind = distances_ms.index(x)
        #print (x, distances.index(x), keys_ski[ind])
        ind_opts_ms.append(distances_ms.index(x))
        keys_n_opt_ms.append(keys_ski_ms[ind])

    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a}: 3.1 max_ind_ms: {max_ind_ms =}\nind_opts_ms: {ind_opts_ms}\nkeys_n_opt_ms: {keys_n_opt_ms}') if sh_print else None  
    logger.info(f'3.1 max_ind_ms: {max_ind_ms =}\nind_opts_ms: {ind_opts_ms}\nkeys_n_opt_ms: {keys_n_opt_ms}')  

    k_selected = keys_n_opt_ms
else:
    #seleciona todos a partir do n_opt até o final
    n_opt_ind_ms = n_opt_ski_ms - 2
    k_selected = keys_ski_ms[n_opt_ind_ms:]

print (f'{a}: 3.2 k_selected: {k_selected}') if sh_print else None  
logger.info (f'{a}: 3.2 k_selected: {k_selected}')  

# 3. Plot distances cluster
SH_PLOT=1
if SH_PLOT:
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a}: 3. Plot distances cluster') if sh_print else None 
    plt.plot(range(2,len(dic_cluster_ski_ms)+2),sse_ski_ms)
    plt.plot([2, len(dic_cluster_ski_ms)+1], [sse_ski_ms[0], sse_ski_ms[-1]], 'r--', label='Linha de Referência')

    plt.show()

# 4. Calculate Intra and Inter Cluster
# 4.1 Read snic_centroid_df
id=0
t1 = time.time()
# pca_snic_dir = base_dir +'data/tmp/spark_pca_snic/' #20241215 commented
pca_snic_dir = save_etapas_dir +'spark_pca_snic/' 
# snic_centroid_df = read_snic_centroid(file_to_open, id=0, i='4.1')

#como está retornando o indice da max distancia, deve-se subtrair 2 pq o n varia de 2 a 30
n_opt_ski_ms = n_opt_ski_ms - 2

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = f'Gen similarity matriz from Cluster using matrix sim {read_quad}'
proc_dic[ri]['subetapa'] = f'salve cluster clara Inter Intra info of cluster simple'
proc_dic[ri]['tempo'] = t2-t1

id=0

# 6. Calculate similarity Matrix in np 
t1 = time.time()
SIM_NP = 0 # 1, if should calculate numpy only
if (SIM_NP):
    # 6.1 Calculate similarity Matrix com numpy
    # matrix_sim_ski_sel = gen_matrix_sim(n_opt_ski, dic_cluster_ski, perc_min=0.5, perc_max=2,\
    #                                             k_selected=ind_opts)
    t0 = time.time()
    matrix_sim_sel_ms = gen_matrix_sim_np(n_opt_ski_ms, dic_cluster_ski_ms, k_selected=k_selected)
    t1 = time.time()
    print (f'6.0 Tempo to gen matrix_sim_sel in np for quad {read_quad} {t1-t0:.2f}') if sh_print else None
    logger.info(f'6.0 Tempo to gen matrix_sim_sel in np: {t2-t1:.2f}, {(t2-t1)/60:.2f}') 
    
    ri+=1
    proc_dic[ri]={} if ri not in proc_dic else None
    proc_dic[ri]['etapa'] = f'Gen similarity matriz from Cluster segmented image {read_quad}'
    proc_dic[ri]['subetapa'] = f'en matrix_sim_sel in np for quad {read_quad}'
    proc_dic[ri]['tempo'] = t2-t1
    #save matrix
    #base_dir = '/scratch/flavia/pca/'
    t1 = time.time()
    # matrix_path = base_dir + '/spark_pca_matrix_sim/matrix_similarity_np_zarr' #commented
    matrix_path = save_etapas_dir + 'spark_pca_matrix_sim/matrix_similarity_np_zarr_ms'
    zarr.save(matrix_path, matrix_sim_sel_ms)
    t2 = time.time()

    print (f'6.1 Tempo funcao para salvar matrix sim np com zarr: {t2-t1:.2f}, {(t2-t1)/60:.2f}') if sh_print else None 
    print (f'6.2 matrix sim np com zarr dir: {matrix_path}') if sh_print else None 
    logger.info(f'6.1 Tempo funcao para salvar matrix sim np com zarr: {t2-t1:.2f}, {(t2-t1)/60:.2f}') 
    logger.info(f'6.2 matrix sim np com zarr dir: {matrix_path}') 

    ri+=1
    proc_dic[ri]={} if ri not in proc_dic else None
    proc_dic[ri]['etapa'] = f'Gen similarity matriz from Cluster segmented image {read_quad}'
    proc_dic[ri]['subetapa'] = f'salve matrix sim np with zarr for quad {read_quad}'
    proc_dic[ri]['tempo'] = t2-t1

    # #salvar em pickle
    # t1 = time.time()
    # file_to_save = base_dir + '/spark_pca_matrix_sim/matrix_similarity_np.pkl'
    # save_to_pickle(matrix_sim_sel, file_to_save)
    # t2 = time.time()
    # print (f'6.3 Tempo funcao para salvar matrix sim np com pickle: {t2-t1:.2f}, {(t2-t1)/60:.2f}') 
    # print (f'6.4 matrix sim np com pickle: {file_to_save}')
    del matrix_sim_sel_ms


# 7. Calculate similarity Matrix com numpy memmap

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 7. Calculating matrixs_sim with numpy memmap') if sh_print else None
logger.info(f'7. Calculating matrixs_sim with numpy memmap')
t1 = time.time()
# tmp_dir='/scratch/flavia/pca/'
# tmp_dir = base_dir + 'data/tmp/pca/' #20241215 commented 
tmp_dir = save_etapas_dir + 'memmap/'

#chunk_size_in_bytes = 800 * 1024 * 1024  # 800 MB = 20480
matrix_sim_sel_ms = gen_matrix_sim_npmem(n_opt_ski_ms, dic_cluster_ski_ms, tmp_dir,\
                                      k_selected=k_selected, chunk_size=20480,\
                                      logger=logger)

# matrix_sim_ski_sel.shape
t2 = time.time()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 7.1 Tempo funcao para gerar matrix sim em np.memmap: {t2-t1:.2f}, {(t2-t1)/60:.2f}') if sh_print else None 
logger.info(f'7.1 Tempo funcao para gerar matrix sim em np.memmap: {t2-t1:.2f}, {(t2-t1)/60:.2f}') 

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = f'Gen similarity matriz from Cluster using MS {read_quad}'
proc_dic[ri]['subetapa'] = f'gen matrix sim for cluster MS np.memmap with zarr for quad {read_quad}'
proc_dic[ri]['tempo'] = t2-t1

#save matrix
# calculate the chunk to decrease the number of files
tam_arq_zarr = 800
chunk_size_in_bytes = tam_arq_zarr * 1024 * 1024  # 800 MB

# Tamanho de cada elemento no array (em bytes)
type_elem = matrix_sim_sel_ms.dtype
element_size = matrix_sim_sel_ms.dtype.itemsize  # Para float32 =4 bytes, float16 = 2 bytes
print (f'{a}: 7.2 type elem: {type_elem}, size: {element_size}') if sh_print else None
logger.info(f'7.2 type elem: {type_elem}, size: {element_size}')

# Número de elementos que cabem em um chunk de 100 MB
n_elements_per_chunk = chunk_size_in_bytes // element_size

# Assumindo chunks quadrados, calcule a dimensão de cada chu nk
chunk_dim = int(np.sqrt(n_elements_per_chunk))
print (f'{a}: 7.3 chunk_dim: {chunk_dim}') if sh_print else None
logger.info(f'7.3 chunk_dim: {chunk_dim}')
# Certifique-se de que o chunk_dim seja um divisor das dimensões da matriz
chunk_dim = min(chunk_dim, matrix_sim_sel_ms.shape[0], matrix_sim_sel_ms.shape[1])
print (f'{a}: 7.4 chunk_dim: {chunk_dim}') if sh_print else None
logger.info(f'{a}: 7.4 chunk_dim: {chunk_dim}')
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 7.5 Saving in zarr, "Tamanho dos chunks: {chunk_dim} x {chunk_dim} ') if sh_print else None
logger.info(f'{a}: 7.5 Saving in zarr, "Tamanho dos chunks: {chunk_dim} x {chunk_dim} ')
# # Salvar com chunks ajustados
# zarr.save('data.zarr', data, chunks=(chunk_dim, chunk_dim))

# print(f"Tamanho dos chunks: {chunk_dim} x {chunk_dim}")

# matrix_path = base_dir + '/spark_pca_matrix_sim/matrix_similarity_npmem_job'
if KNN:
    # matrix_path = base_dir + 'data/tmp/spark_pca_matrix_sim/matrix_similarity_npmem_job_knn'
    matrix_path = save_etapas_dir + 'spark_pca_matrix_sim/matrix_similarity_npmem_job_knn'
else:
    if MaxDist:
        matrix_path = base_dir + 'data/tmp/spark_pca_matrix_sim/matrix_similarity_npmem_job'
    else:
        matrix_path = save_etapas_dir + 'spark_pca_matrix_sim/matrix_similarity_npmem_job_ms_30_Quad_'+str(read_quad)
print (f'{a}: 7.6 matrix sim npmem com zarr dir: {matrix_path}') if sh_print else None 
logger.info(f'7.6 matrix sim npmem com zarr dir: {matrix_path}') 
t1 = time.time()
zarr.save(matrix_path, matrix_sim_sel_ms, chunks=(chunk_dim, chunk_dim))
t2 = time.time()

del matrix_sim_sel_ms
gc.collect()

#delete tmp files from np.memmap it they still exists
[f.unlink() for f in Path(tmp_dir).glob("*.dat") if f.is_file()]
print (f'{a}: memmap .dat files deleted from {tmp_dir}') if sh_print else None
logger.info(f'{a}: memmap .dat files deleted from {tmp_dir}')

tf = time.time()

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 7.7 Tempo funcao para salvar matrix sim em np: {t2-t1:.2f}, {(t2-t1)/60:.2f}') if sh_print else None
logger.info(f'7.7 Tempo para salvar matrix sim em np.memmap: {t2-t1:.2f}, {(t2-t1)/60:.2f}') 

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = f'Gen similarity matriz from Cluster using MS {read_quad}'
proc_dic[ri]['subetapa'] = f'save matrix sim np.memmap with zarr for quad {read_quad}'
proc_dic[ri]['tempo'] = t2-t1

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = f'Gen similarity matriz from Cluster using MS {read_quad}'
proc_dic[ri]['subetapa'] = f'{nome_prog} time execution total for quad {read_quad}'
proc_dic[ri]['tempo'] = tf-ti

time_file = process_time_dir + "process_times.pkl"
update_procTime_file(proc_dic, time_file)

# t1 = time.time()
# file_to_save = base_dir + '/spark_pca_matrix_sim/matrix_similarity_npmem.pkl'
# save_to_pickle(matrix_sim_sel, file_to_save)
# t2 = time.time()
# print (f'7.3 Tempo funcao para salvar matrix sim np com pickle: {t2-t1:.2f}, {(t2-t1)/60:.2f}') 
# print (f'7.4 matrix sim np com pickle: {file_to_save}')

print (f'{a}: 0. Fim {nome_prog} tempo total {tf-ti:.2f}s {(tf-ti)/60:.2f}') if sh_print else None
logger.info(f'0. Fim {nome_prog} tempo total {tf-ti:.2f}s {(tf-ti)/60:.2f}') 
