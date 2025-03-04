# Programa para ler a matriz de distancia e calcular intra e inter cluster 
# 
# 20240903
# 20241117: Baixado do exacta 
# 20241217: inclusao do log e do df com os tempos de processamento
#           clusterizacao da segmentação de 1/4 do tile, nao juntar os 
#           quadrantes
# 20241217: passagem de arqumentos para o programa
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
from sklearn_extra.cluster import CLARA
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
                                        read_snic_centroid, gen_arrayToCluster
ti = time.time()

# 20241215 bloco de parse incluido
# Inicializa o parser
parser = argparse.ArgumentParser(description="Program segment image")

# Define os argumentos
parser.add_argument("-bd", '--base_dir', type=str, help="Diretorio base", default='')
parser.add_argument("-sd", '--save_dir', type=str, help="Dir base para salvar saidas de cada etapa", default='data/tmp2/')
# parser.add_argument("-td", '--tif_dir', type=str, help="Dir dos tiffs", default='data/Cassio/S2-16D_V2_012014_20220728_/')
parser.add_argument("-q", '--quadrante', type=int, help="Numero do quadrante da imagem [0-all,1,2,3,4]", default=1)
parser.add_argument("-ld", '--log_dir', type=str, help="Dir do log", default='code/logs/')
parser.add_argument("-i", '--name_img', type=str, help="Nome da imagem", default='S2-16D_V2_012014')
parser.add_argument("-sp", '--sh_print', type=int, help="Show prints", default=0)
parser.add_argument("-pd", '--process_time_dir', type=str, help="Dir para df com os tempos de processamento", default='data/tmp2/')
# parser.add_argument("-rf", '--READ_df_features', type=int, help="Read or create df with features", default=0 )
# parser.add_argument("-nc", '--num_components', type=int, help="number of PCA components", default=4 )
# parser.add_argument("-dsi", '--image_dir', type=str, help="Diretorio da imagem pca", default='data/tmp2/spark_pca_images/')
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
name_img = args.name_img
process_time_dir = base_dir + args.process_time_dir
sh_print = args.sh_print
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
print (f'{a}: 0. INICIO {nome_prog}')

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
proc_dic[ri]['etapa'] = 'intra and inter cluster calculation'

# read snic df
id=0
t1 = time.time()
# file_to_open = base_dir + 'pca_snic_cluster/clara_ms'+str(id)+'.pkl'
# KNN=0 #20241217 passado como arg
if KNN:
    file_to_open = base_dir + 'data/tmp/pca_snic_cluster/clara_ms_knn_'+str(id)+'_20241117.pkl'
else:
    if MaxDist:
        file_to_open = save_etapas_dir + 'pca_snic_cluster/clara_ms_'+str(id)+'_quad_'+str(read_quad)+'.pkl'
    else:
        file_to_open = save_etapas_dir + 'pca_snic_cluster/clara_ms_'+str(id)+'_30_quad_'+str(read_quad)+'.pkl'


with open(file_to_open, 'rb') as handle:    
    b = pickle.load(handle)

t2 = time.time()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 0.1 Tempo leitura cluster clara: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m') if sh_print else None
print (f'{a}: 0.2 b.keys(): {b.keys()}') if sh_print else None
logger.info(f'0.1 Tempo leitura cluster clara: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
logger.info(f'0.2 b.keys(): {b.keys()}')

sse_ski_ms = b['sse_ski']
n_clusters_ms = len(sse_ski_ms)
dic_cluster_ski_ms = b['dic_cluster_ski']

proc_dic[ri]['subetapa'] = f'read Cluster Clara MS for quad {read_quad}'
proc_dic[ri]['tempo'] = t2-t1

### #### Calculate n_opt
t1 =time.time()
n_opt_ski_ms, distances_ms = optimal_number_of_clusters(sse_ski_ms, 2, 1*n_clusters_ms, dist=True)
keys_ski_ms=list(dic_cluster_ski_ms.keys())
t2 =time.time()

print (f'n_opt_ski_ms = {n_opt_ski_ms-2}\ndistances_ms = {distances_ms}')
print (f'0.3 Tempo para ober n_optimal {t2-t1}') if sh_print else None
ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'intra and inter cluster calculation'
proc_dic[ri]['subetapa'] = 'get n_optimal'
proc_dic[ri]['tempo'] = t2-t1

#distances.index(np.max(distances)) + min_cl
dist_copy=distances_ms.copy()
dist_copy.sort(reverse=True)
dist_copy

if MaxDist: #este bloco nao está sendo usado neste programa ... 20250226
    #seleciona os clusters que tiveram as maiores distancias de acordo com o grafico do cotovelo
    #para obter a lista com indices das maiores distancias e chaves do dic_cluster
    ind_opts_ms=[]
    keys_n_opt_ms=[]
    a = round((n_opt_ski_ms-2)*1.5)
    max_ind_ms= a if a <= len(dist_copy) else len(dist_copy)
    for x in dist_copy[:max_ind_ms]:
        ind = distances_ms.index(x)
        #print (x, distances.index(x), keys_ski[ind])
        ind_opts_ms.append(distances_ms.index(x))
        keys_n_opt_ms.append(keys_ski_ms[ind])

    #ind_opts_ms, keys_n_opt_ms
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a}: 3.1 max_ind: {max_ind_ms =}\nind_opts: {ind_opts_ms}\nkeys_n_opt: {keys_n_opt_ms}') if sh_print else None  
    logger.info(f'3.1 max_ind: {max_ind_ms =}\nind_opts: {ind_opts_ms}\nkeys_n_opt: {keys_n_opt_ms}')  

# 3. Plot distances cluster
SH_PLOT=1
if SH_PLOT:
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a}: 3. Plot distances cluster') if sh_print else None 
    plt.plot(range(2,len(dic_cluster_ski_ms)+2),sse_ski_ms)
    plt.plot([2, len(dic_cluster_ski_ms)+1], [sse_ski_ms[0], sse_ski_ms[-1]], 'r--', label='Linha de Referência')

    plt.show()

#######
#tem que ler a matrix de distancia
#read the sim matrix s
# matrix_path = base_dir + '/spark_pca_matrix_sim/matrix_similarity_npmem_job'
# KNN=0
if KNN:
    matrix_path = base_dir + 'data/tmp/spark_pca_matrix_sim/matrix_similarity_npmem_job_knn'
else:
    if MaxDist:
        matrix_path = save_etapas_dir + 'spark_pca_matrix_sim/matrix_similarity_npmem_job_Quad_'+str(read_quad)
    else:    
        matrix_path = save_etapas_dir + 'spark_pca_matrix_sim/matrix_similarity_npmem_job_30_Quad_'+str(read_quad)

t1 = time.time()
zarr_group = zarr.open(matrix_path, mode='r')
t2 = time.time()

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 1. Tempo leitura similarity matrix : {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')  
#print (f'{a}: 1.1 matrix_sim_sel.shape : {matrix_sim_sel.shape} ')
print (f'{a}: 1.1 zarr_group type : {type(zarr_group)} ')
print (f'{a}: 1.2 zarr_group.array_keys: {list(zarr_group.array_keys())}')
print(f'{a}: 1.3 zarr_group.group_keys: {list(zarr_group.group_keys())}')

# Iterar sobre os arrays no grupo e exibir seus nomes e formas
for array_name in zarr_group.array_keys():
    array = zarr_group[array_name]
    print(f'1.4 Array: {array_name}, Shape: {array.shape}') if sh_print else None

matrix_sim_sel = zarr_group['arr_0']
print (f'1.5 matrix_sim_sel.shape: {matrix_sim_sel.shape}') if sh_print else None
print (f'1.5 matrix_sim_sel.shape: {matrix_sim_sel.shape}') if sh_print else None

# 2. Gen distance matrix
# Criar um novo array Zarr para armazenar o resultado
t1 = time.time()
# matrix_dist_sel = zarr_group.create_dataset(
#     'matrix_dist_sel', 
#     shape=matrix_sim_sel.shape, 
#     dtype=matrix_sim_sel.dtype
# )
if 'matrix_dist_sel' not in list(zarr_group.array_keys()):
    print (f'2.0 matrix_dist_sel not in zarr_group and created there') if sh_print else None
    logger.info(f'2.0 matrix_dist_sel not in zarr_group and created there')
    # del zarr_group['matrix_dist_sel']  # Remove o dataset existente

    matrix_dist_sel = zarr_group.create_dataset(
        'matrix_dist_sel', 
        shape=matrix_sim_sel.shape, 
        dtype=matrix_sim_sel.dtype
    )

matrix_dist_sel = zarr_group['matrix_dist_sel']
t2 = time.time()

# print (f'{a}: 2.1 Tempo para create matrix de distancia: {(t2-t1)}, {(t2-t1)/60}')
print (f'{a}: 2.1 Tempo para read matrix de distancia: {(t2-t1):.2f}, {(t2-t1)/60:.2f}') if sh_print else None
print (f'{a}: 2.2 Matrix de distancia, Type: {type(matrix_dist_sel)}, shape: {matrix_dist_sel.shape}') if sh_print else None
logger.info(f'{a}: 2.1 Tempo para read matrix de distancia: {(t2-t1):.2f}, {(t2-t1)/60:.2f}') if sh_print else None
logger.info(f'{a}: 2.2 Matrix de distancia, Type: {type(matrix_dist_sel)}, shape: {matrix_dist_sel.shape}') if sh_print else None

#### Calculate intra and Inter Cluster
#como está retornando o indice da max distancia, deve-se subtrair 2 pq o n varia de 2 a 30
n_opt_ski_ms = n_opt_ski_ms - 2
t1=time.time()
# df_ski_ms = calc_inter_intra_cluster(matrix_dist_sel, dic_cluster_ski_ms, keys_ski_ms[n_opt_ski_ms])
# df_ski_ms = calc_inter_intra_cluster(matrix_dist_memmap, dic_cluster_ski_ms, keys_ski_ms[n_opt_ski_ms])
df_ski_ms = calc_inter_intra_cluster(matrix_dist_sel, dic_cluster_ski_ms, keys_ski_ms[n_opt_ski_ms], metric='precomputed')
t2=time.time()

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: Tempo para calcular inter e intra cluster: {t2-t1:.2f}s {(t2-t1)/60:.2f}m') if sh_print else None
print (f'{a}: df_ski_ms shape: {df_ski_ms.shape}, \n{df_ski_ms.head()}') if sh_print else None
logger.info(f'Tempo para calcular inter e intra cluster: {t2-t1:.2f}s {(t2-t1)/60:.2f}m')
logger.info (f'df_ski_ms shape: {df_ski_ms.shape}, \n{df_ski_ms.head()}')

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'intra and inter cluster calculation'
proc_dic[ri]['subetapa'] = f'calculate cluster clara ms Inter Intra info of distance matrix'
proc_dic[ri]['tempo'] = t2-t1

#Stats per cluster
t1=time.time()
result_ms = df_ski_ms.groupby('label').agg({'inter':['max', 'min','median', 'mean', 'std'],'intra':['max', 'min','median', 'mean', 'std']})
t2=time.time()
print (f'{a}: t2-t1={t2-t1}\nresult_ms:\n{result_ms}') if sh_print else None

# 
stats_IntraInter_df_ms = df_ski_ms[['inter','intra']].agg(['max', 'min','median', 'mean', 'std'])
print (f'{a}: stats_IntraInter_df_ms:\n{stats_IntraInter_df_ms}')

#save the results
obj_dic={}
obj_dic = {
    "matrix_dist_file": matrix_path,
    "df_inter_intra": df_ski_ms,
    "stats_inter_intra_per_cluster": result_ms,
    "stats_inter_intra": stats_IntraInter_df_ms,
    "n_opt_ind": n_opt_ski_ms,
    "n_opt_key": keys_ski_ms[n_opt_ski_ms]
    
}
id=0
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
if MaxDist:
    file_to_save = save_etapas_dir + 'pca_snic_cluster/clara_ms_'+str(id)+'_InterIntra_quad_'+str(read_quad)+'.pkl'
else:
    file_to_save = save_etapas_dir + 'pca_snic_cluster/clara_ms_'+str(id)+'_30_InterIntra_quad_'+str(read_quad)+'.pkl'
t1=time.time()
save_to_pickle(obj_dic, file_to_save)
t2=time.time()
print (f'3.2 Tempo para salvar cluster clara ms Inter Intra info of distance matrix : {t2-t1:.2f}, {(t2-t1)/60:.2f}') if sh_print else None
print (f'3.3 clara com matriz de similaridade salvo {file_to_save}') if sh_print else None

logger.info(f'3.2 Tempo para salvar cluster clara ms Inter Intra info of distance matrix : {t2-t1:.2f}, {(t2-t1)/60:.2f}')
logger.info(f'3.3 clara com matriz de similaridade salvo {file_to_save}')

del obj_dic
gc.collect()

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'intra and inter cluster calculation'
proc_dic[ri]['subetapa'] = f'salve cluster clara ms Inter Intra info of distance matrix'
proc_dic[ri]['tempo'] = t2-t1

tf=time.time()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: Tempo de execução do {nome_prog}: {tf-ti:.2f} s {(tf-ti)/60:.2f} m')

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'intra and inter cluster calculation'
proc_dic[ri]['subetapa'] = f'{nome_prog} time execution total'
proc_dic[ri]['tempo'] = tf-ti

time_file = process_time_dir + ' process_times.pkl'
update_procTime_file(proc_dic, time_file)