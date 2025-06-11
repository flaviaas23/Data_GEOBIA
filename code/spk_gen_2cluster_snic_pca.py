# Programa para ler resultado do cluster snic pca , faz a matriz de 
# similaridade 
# 20240828
# 20241016: baixado do exacta e alterado para rodar local no laptop
#           e ler e gerar o snic_centroid_df de cada quadrante
# 20241215: inclusao do log e do df com os tempos de processamento
#           clusterizacao da segmentação de 1/4 do tile, nao juntar os 
#           quadrantes
# 20241215: passagem de argumentos para o programa
# 20250226: inclusao de arg para selecao dos n's para calcular matriz de similaridade
#           -md =1 se seleciona clusters pelas maiores distancias 
#           -md 0 seleciona do n_opt a'te o max n  
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
parser.add_argument("-pfi", '--pca_fullImg', type=int, help="usar pca da imagem full", default=0 )
args = parser.parse_args()

base_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/'
if args.base_dir: 
    base_dir = args.base_dir
    print (f'base_dir: {base_dir}') 

save_etapas_dir = base_dir + args.save_dir if base_dir else args.save_dir + args.name_img +'/'
# tif_dir = base_dir + args.tif_dir if base_dir else args.tif_dir
read_quad = args.quadrante 
log_dir = base_dir + args.log_dir if base_dir else args.log_dir
# name_img = args.name_img
process_time_dir = base_dir + args.process_time_dir
sh_print = args.sh_print
pca_fullImg = args.pca_fullImg

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
logger.info(f'args: sd={save_etapas_dir} ld={log_dir} pd={process_time_dir} sp={args.sh_print}')

ri=0        #indice do dicionario com os tempos de cada subetapa
proc_dic = {}
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = f'Gen similarity matriz from Cluster segmented image for quad {read_quad}'

# 1. read snic df
id=0
t1 = time.time()
# file_to_open = base_dir + 'pca_snic_cluster/clara_0.pkl
# KNN = 0 # passado via argumento
if KNN:
    file_to_open = base_dir + 'data/tmp/pca_snic_cluster/clara_knn'+str(id)+'.pkl'
else:
    # file_to_open = base_dir + 'data/tmp/pca_snic_cluster/clara_'+str(id)+'.pkl'
    if pca_fullImg:
        d_name = save_etapas_dir + 'pca_snic_cluster/PCAFullImg/'
    else:
        d_name = save_etapas_dir + 'pca_snic_cluster/'
    file_to_open = d_name+'clara_'+str(id)+'_quad_'+str(read_quad)+'.pkl'

with open(file_to_open, 'rb') as handle:    
    b = pickle.load(handle)

t2 = time.time()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 1. Tempo leitura cluster clara: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m') if sh_print else None 
print (f'{a}: 1.1 Clara file keys: {b.keys()}') if sh_print else None
logger.info(f'1. Tempo leitura cluster clara: {t2-t1}s, {(t2-t1)/60}m')  
logger.info(f'1.1 Clara file keys: {b.keys()}')

proc_dic[ri]['subetapa'] = f'read Cluster Clara of segmented image for quad {read_quad}'
proc_dic[ri]['tempo'] = t2-t1

sse_ski = b['sse_ski']
n_clusters = len(sse_ski)
sse_rd = b['sse_rd'],
rd_state = b['rd_state']
dic_cluster_rd = b['dic_cluster_rd']
dic_cluster_ski = b['dic_cluster_ski']

del b
gc.collect()

# 2. Calculate n_opt
t1 = time.time()
n_opt_ski, distances = optimal_number_of_clusters(sse_ski, 2, 1*n_clusters, dist=True)
t2 = time.time()

keys_ski = list(dic_cluster_ski.keys())

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 2. keys_ski: {keys_ski}') if sh_print else None  
print (f'{a}: 2.1 n_opt_ski: {n_opt_ski} , keys_ski[n_opt_ski-2]: {keys_ski[n_opt_ski-2]}, keys_ski[n_opt_ski]:{keys_ski[n_opt_ski]}') if sh_print else None
logger.info(f'2. keys_ski: {keys_ski}')  
logger.info(f'2.1 n_opt_ski: {n_opt_ski} , keys_ski[n_opt_ski-2]: {keys_ski[n_opt_ski-2]}, keys_ski[n_opt_ski]:{keys_ski[n_opt_ski]}')

# #como está retornando o indice da max distancia, deve-se subtrair 2
# n_opt_ski = n_opt_ski - 2

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = f'Gen similarity matriz from Cluster segmented image {read_quad}'
proc_dic[ri]['subetapa'] = f'Get optimal number of clusters for quad {read_quad}'
proc_dic[ri]['tempo'] = t2-t1

# 3. para obter a lista com indices das maiores distancias e chaves do dic_cluster
#distances.index(np.max(distances)) + min_cl
#ordenar pelas maximas distancias
dist_copy = distances.copy()
dist_copy.sort(reverse=True)
print (f'{a}: distancias: {dist_copy}') if sh_print else None
logger.info(f'distancias: {dist_copy}')

if MaxDist:
    #seleciona os clusters que tiveram as maiores distancias de acordo com o grafico do cotovelo
    ind_opts=[]
    keys_n_opt=[]
    max_ind = round((n_opt_ski-2)*1.5)          # 
    for x in dist_copy[:max_ind]:
        ind = distances.index(x)
        #print (x, distances.index(x), keys_ski[ind])
        ind_opts.append(distances.index(x))
        keys_n_opt.append(keys_ski[ind])

    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a}: 3.1 max_ind: {max_ind =}\nind_opts: {ind_opts}\nkeys_n_opt: {keys_n_opt}') if sh_print else None  
    logger.info(f'3.1 max_ind: {max_ind =}\nind_opts: {ind_opts}\nkeys_n_opt: {keys_n_opt}')  

    k_selected = keys_n_opt

else:
    #seleciona todos a partir do n_opt até o final
    n_opt_ind = n_opt_ski - 2
    k_selected = keys_ski[n_opt_ind:]

print (f'{a}: 3.2 k_selected: {k_selected}') if sh_print else None  
logger.info (f'{a}: 3.2 k_selected: {k_selected}')  

# 3. Plot distances cluster
SH_PLOT=1
if SH_PLOT:
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a}: 3. Plot distances cluster') if sh_print else None 
    plt.plot(range(2,len(dic_cluster_ski)+2),sse_ski)
    plt.plot([2, len(dic_cluster_ski)+1], [sse_ski[0], sse_ski[-1]], 'r--', label='Linha de Referência')

    plt.show()

# 4. Calculate Intra and Inter Cluster
# 4.1 Read snic_centroid_df
id=0
t1 = time.time()
# pca_snic_dir = base_dir +'data/tmp/spark_pca_snic/' #20241215 commented
if pca_fullImg:
    pca_snic_dir = save_etapas_dir + 'spark_pca_snic/PCAFullImg/' #Quad_' + str(q) +'/'
else:
    pca_snic_dir = save_etapas_dir +'spark_pca_snic/' 
# snic_centroid_df = read_snic_centroid(file_to_open, id=0, i='4.1')
if KNN:
    #read snic_centroid_df
    file_to_open = base_dir + 'data/tmp/spark_pca_snic_centroid_df_knn/snic_centroid_df_knn'+str(id)+'.pkl'
    with open(file_to_open, 'rb') as handle:    
        snic_centroid_df = pickle.load(handle)

    cols_snic_centroid = snic_centroid_df.columns[1:]
    comps = [x.split('_')[-1] for x in cols_snic_centroid]
    comps = sorted(list(set(comps)))
    print (f'{a}: {ind}. {cols_snic_centroid,} {comps}, {type(comps)}') if sh_print else None
    cols_sel = ['avg_'+str(i) for i in comps]
    arraybands_sel = snic_centroid_df[cols_sel].to_numpy()
else:
    # snic_centroid_df = read_qs_gen_snic_centroid(pca_snic_dir, q_number=4, id=0,i='4.1')
    t1 = time.time()
    snic_centroid_df = read_qs_gen_snic_centroid(pca_snic_dir, q_number=1, id=0,quad=read_quad, i='4.1')
    t2 = time.time()
    ri+=1
    proc_dic[ri]={} if ri not in proc_dic else None
    proc_dic[ri]['etapa'] = f'Gen similarity matriz from Cluster segmented image {read_quad}'
    proc_dic[ri]['subetapa'] = f'read snic centroid df for quad {read_quad}'
    proc_dic[ri]['tempo'] = t2-t1
        # 4.2 Get the arraybands_sel from snic_centroid_df to calculate intra/inter cluster

    # cols_sel = ['avg_'+str(i) for i in comps]
    # a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    # print (f'{a}: cols_sel of snic_centroid_df: {cols_sel}')
    t1 = time.time()
    arraybands_sel = gen_arrayToCluster(snic_centroid_df, cols_sel='', calc_nans=1, ind='4.2')
    t2 = time.time()
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a}: 4.2 Tempo total funcao para obter arraybands_sel do snic_centroid_df: {t2-t1:.2f}s {(t2-t1)/60:.2f}m') if sh_print else None
    logger.info(f'4.2 Tempo total funcao para obter arraybands_sel do snic_centroid_df: {t2-t1:.2f}s {(t2-t1)/60:.2f}m')
    ri+=1
    proc_dic[ri]={} if ri not in proc_dic else None
    proc_dic[ri]['etapa'] = f'Gen similarity matriz from Cluster segmented image {read_quad}'
    proc_dic[ri]['subetapa'] = f'gen arraybands_sel df snic centroid df for quad {read_quad}'
    proc_dic[ri]['tempo'] = t2-t1

# 4.3 Calculate intra and inter cluster
#como está retornando o indice da max distancia, deve-se subtrair 2 pq o n varia de 2 a 30
n_opt_ski = n_opt_ski - 2

t1=time.time()
df_ski = calc_inter_intra_cluster(arraybands_sel, dic_cluster_ski, keys_ski[n_opt_ski], sh_print =True)
print (f'4.3 intra and insta cluster shape: {df_ski.shape}') if sh_print else None 
logger.info(f'4.3 intra and insta cluster shape: {df_ski.shape}') 
t2=time.time()

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 4.3.2 Tempo para calcular inter e intra cluster: {t2-t1:.2f}s {(t2-t1)/60:.2f}m') if sh_print else None
print (f'{a}: 4.3.3 df with inter and Intra Cluster:\n{df_ski.head()}') if sh_print else None
logger.info(f'4.3.3 df with inter and Intra Cluster:\n{df_ski.head()}')

# 5. Stats of intra and inter cluster
result = df_ski.groupby('label').agg({'inter':['max', 'min','median', 'mean', 'std'],'intra':['max', 'min','median', 'mean', 'std']})
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 5. stats of intra and inter cluster:\n{result}') if sh_print else None
logger.info(f'5. stats of intra and inter cluster:\n{result}')

stats_IntraInter_df = df_ski[['inter','intra']].agg(['max', 'min','median', 'mean', 'std'])
print (f'{a}: 5.1 stats of intra and inter cluster:\n({stats_IntraInter_df}') if sh_print else None
logger.info(f'{a}: 5.1 stats of intra and inter cluster:\n({stats_IntraInter_df}')

# salvar resultados
obj_dic={}
obj_dic = {
    # "matrix_dist_file": matrix_path,
    "df_inter_intra": df_ski,
    "stats_inter_intra_per_cluster": result,
    "stats_inter_intra": stats_IntraInter_df,
    "n_opt_ind": n_opt_ski,
    "n_opt_key": keys_ski[n_opt_ski]
    
}
ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = f'Gen similarity matriz from Cluster segmented image {read_quad}'
proc_dic[ri]['subetapa'] = f'salve cluster clara Inter Intra info of cluster simple'
proc_dic[ri]['tempo'] = t2-t1

id=0
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
t1=time.time()
if pca_fullImg:
    file_to_save = save_etapas_dir + 'pca_snic_cluster/PCAFullImg/clara_'+str(id)+'_InterIntra_quad_'+str(read_quad)+'.pkl'
else:
    file_to_save = save_etapas_dir + 'pca_snic_cluster/clara_'+str(id)+'_InterIntra_quad_'+str(read_quad)+'.pkl'
save_to_pickle(obj_dic, file_to_save)
t2=time.time()
print (f'3.2 Tempo para salvar cluster clara ms Inter Intra info of distance matrix : {t2-t1:.2f}, {(t2-t1)/60:.2f}') if sh_print else None
print (f'3.3 clara com matriz de similaridade salvo {file_to_save}') if sh_print else None

logger.info(f'3.2 Tempo para salvar cluster clara ms Inter Intra info of distance matrix : {t2-t1:.2f}, {(t2-t1)/60:.2f}')
logger.info(f'3.3 clara com matriz de similaridade salvo {file_to_save}')

del obj_dic
gc.collect()

# 6. Calculate similarity Matrix in np 
t1 = time.time()
SIM_NP = 0 # 1, if should calculate numpy only
if (SIM_NP):
    # 6.1 Calculate similarity Matrix com numpy
    # matrix_sim_ski_sel = gen_matrix_sim(n_opt_ski, dic_cluster_ski, perc_min=0.5, perc_max=2,\
    #                                             k_selected=ind_opts)
    t0 = time.time()
    matrix_sim_sel = gen_matrix_sim_np(n_opt_ski-2, dic_cluster_ski, k_selected=k_selected)
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
    matrix_path = save_etapas_dir + 'spark_pca_matrix_sim/matrix_similarity_np_zarr'
    zarr.save(matrix_path, matrix_sim_sel)
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
    del matrix_sim_sel


# 7. Calculate similarity Matrix com numpy memmap

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 7. Calculating matrixs_sim with numpy memmap') if sh_print else None
logger.info(f'7. Calculating matrixs_sim with numpy memmap')
t1 = time.time()
# tmp_dir='/scratch/flavia/pca/'
# tmp_dir = base_dir + 'data/tmp/pca/' #20241215 commented 
tmp_dir = save_etapas_dir + 'memmap/'

#chunk_size_in_bytes = 800 * 1024 * 1024  # 800 MB = 20480
matrix_sim_sel = gen_matrix_sim_npmem(n_opt_ski, dic_cluster_ski, tmp_dir,\
                                      k_selected=k_selected, chunk_size=20480,\
                                      logger=logger)

# matrix_sim_ski_sel.shape
t2 = time.time()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 7.1 Tempo funcao para gerar matrix sim em np.memmap: {t2-t1:.2f}, {(t2-t1)/60:.2f}') if sh_print else None 
logger.info(f'7.1 Tempo funcao para gerar matrix sim em np.memmap: {t2-t1:.2f}, {(t2-t1)/60:.2f}') 

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = f'Gen similarity matriz from Cluster segmented image {read_quad}'
proc_dic[ri]['subetapa'] = f'gen matrix sim np.memmap with zarr for quad {read_quad}'
proc_dic[ri]['tempo'] = t2-t1

#save matrix
# calculate the chunk to decrease the number of files
tam_arq_zarr = 800
chunk_size_in_bytes = tam_arq_zarr * 1024 * 1024  # 800 MB

# Tamanho de cada elemento no array (em bytes)
type_elem = matrix_sim_sel.dtype
element_size = matrix_sim_sel.dtype.itemsize  # Para float32 =4 bytes, float16 = 2 bytes
print (f'{a}: 7.2 type elem: {type_elem}, size: {element_size}') if sh_print else None
logger.info(f'7.2 type elem: {type_elem}, size: {element_size}')

# Número de elementos que cabem em um chunk de 100 MB
n_elements_per_chunk = chunk_size_in_bytes // element_size

# Assumindo chunks quadrados, calcule a dimensão de cada chunk
chunk_dim = int(np.sqrt(n_elements_per_chunk))
print (f'{a}: 7.3 chunk_dim: {chunk_dim}') if sh_print else None
logger.info(f'7.3 chunk_dim: {chunk_dim}')
# Certifique-se de que o chunk_dim seja um divisor das dimensões da matriz
chunk_dim = min(chunk_dim, matrix_sim_sel.shape[0], matrix_sim_sel.shape[1])
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
    # matrix_path = base_dir + 'data/tmp/spark_pca_matrix_sim/matrix_similarity_npmem_job'
    if pca_fullImg:
        path_matrix_sim = save_etapas_dir + 'spark_pca_matrix_sim/PCAFullImg/'
    else:
        path_matrix_sim = save_etapas_dir + 'spark_pca_matrix_sim/'
    if MaxDist:
        # matrix_path = save_etapas_dir + 'spark_pca_matrix_sim/matrix_similarity_npmem_job_Quad_'+str(read_quad)
        matrix_path = path_matrix_sim + 'matrix_similarity_npmem_job_Quad_'+str(read_quad)
    else:
        # matrix_path = save_etapas_dir + 'spark_pca_matrix_sim/matrix_similarity_npmem_job_30_Quad_'+str(read_quad)
        matrix_path = path_matrix_sim + 'matrix_similarity_npmem_job_30_Quad_'+str(read_quad)

print (f'{a}: 7.6 matrix sim npmem com zarr dir: {matrix_path}') if sh_print else None 
logger.info(f'7.6 matrix sim npmem com zarr dir: {matrix_path}') 
t1 = time.time()
zarr.save(matrix_path, matrix_sim_sel, chunks=(chunk_dim, chunk_dim))
t2 = time.time()

del matrix_sim_sel
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
proc_dic[ri]['etapa'] = f'Gen similarity matriz from Cluster segmented image {read_quad}'
proc_dic[ri]['subetapa'] = f'save matrix sim np.memmap with zarr for quad {read_quad}'
proc_dic[ri]['tempo'] = t2-t1

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = f'Gen similarity matriz from Cluster segmented image {read_quad}'
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

print (f'{a}: 0. Fim {nome_prog} tempo total {tf-ti:.2f}') if sh_print else None
logger.info(f'0. Fim {nome_prog} tempo total {tf-ti:.2f} s') 
