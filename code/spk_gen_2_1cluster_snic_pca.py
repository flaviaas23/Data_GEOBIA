# Programa para ler a matriz de similaridade calcular a distancia e rodar o cluster com 
# a matriz de distancia
# 20240901
# 20241016: baixado do exacta e alterado para rodar no laptop
# 20241215: inclusao do log e do df com os tempos de processamento
#           clusterizacao da segmentação de 1/4 do tile, nao juntar os 
#           quadrantes
# 20241215: passagem de arqumentos para o programa
# 20250226: inclusao de arg md para selecionar o tipo de matriz de sim para clusterizar, 
#           se com max distancias ou a gerada com n_opt até max n
# 20250406: inclusao do argumento sm to read similarity matriz from simple cluster or
#            with similarity matriz 2
#20250529: inclusao de argumento pfi pca_fullImg para usar a matriz de similaridade do pca da imagem full
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

#20241104 comentei abaixo
#from functions.functions_segmentation import gen_coords_snic_df
from functions.functions_pca import save_to_pickle, cria_logger, update_procTime_file
from functions.functions_cluster import optimal_number_of_clusters, gen_matrix_sim_np, \
                                        gen_matrix_sim_npmem, calc_inter_intra_cluster,\
                                        read_snic_centroid, gen_arrayToCluster,\
                                        hierarchical_clustering
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
parser.add_argument("-i", '--name_img', type=str, help="Nome da imagem", default='')
parser.add_argument("-sp", '--sh_print', type=int, help="Show prints", default=0)
parser.add_argument("-pd", '--process_time_dir', type=str, help="Dir para df com os tempos de processamento", default='data/tmp2/')
# parser.add_argument("-rf", '--READ_df_features', type=int, help="Read or create df with features", default=0 )
# parser.add_argument("-nc", '--num_components', type=int, help="number of PCA components", default=4 )
# parser.add_argument("-dsi", '--image_dir', type=str, help="Diretorio da imagem pca", default='data/tmp/spark_pca_images/')
# parser.add_argument("-p", '--padrao', type=str, help="Filtro para o diretorio ", default='*')
parser.add_argument("-k", '--knn', type=str, help="Use KNN", default=False)
parser.add_argument("-md", '--max_dist_cotov', type=int, help="Sel os clusters pela dist cotovelo ou do n_opt em diante", default=1)
parser.add_argument("-sm", '--sim_matrix', type=int, help="sim matrix number to cluster", default='')
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
name_img = args.name_img
process_time_dir = base_dir + args.process_time_dir
sh_print = args.sh_print
pca_fullImg = args.pca_fullImg
# n_components = args.num_components
# img_dir = args.image_dir
# padrao = args.padrao
KNN = args.knn
# indica o tipo de selecao dos clusters que serao usados na matriz de similaridade, se por distancia
# ou considerar todos a partir do n_opt
MaxDist = args.max_dist_cotov
# indica se vai usar a primeira ou segunda matrix de similaridade
sm = args.sim_matrix

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
t = datetime.datetime.now().strftime('%Y%m%d_%H%M_')

# base_dir = '/scratch/flavia/pca/'

# base_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/'
nome_prog = os.path.basename(__file__)
print (f'{a}: 0. INICIO {nome_prog}') if sh_print else None

#cria logger
nome_log = t + nome_prog.split('.')[0]+'.log'
logger = cria_logger(log_dir, nome_log)
logger.info(f'######## INICIO {nome_prog} ##########')
logger.info(f'args: sd={save_etapas_dir} ld={log_dir} pd={process_time_dir} sp={args.sh_print}')


ri=0        #indice do dicionario com os tempos de cada subetapa
proc_dic = {}
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'cluster with distance matriz'
#read the sim matrix s
# matrix_path = base_dir + '/spark_pca_matrix_sim/matrix_similarity_npmem_job'
# KNN=0  #passado por args
# read matrix sim from zarr for clustering
if KNN:
    matrix_path = base_dir + 'data/tmp/spark_pca_matrix_sim/matrix_similarity_npmem_job_knn'
else:
    # matrix_path = base_dir + 'data/tmp/spark_pca_matrix_sim/matrix_similarity_npmem_job'
    # matrix_path = save_etapas_dir + 'spark_pca_matrix_sim/matrix_similarity_npmem_job_Quad_'+str(read_quad)
    if pca_fullImg:
        path_matrix_sim = save_etapas_dir + 'spark_pca_matrix_sim/PCAFullImg/'
    else:
        path_matrix_sim = save_etapas_dir + 'spark_pca_matrix_sim/'

    if MaxDist:
        if sm == 2: # read sim matrix(2) from cluster using sim matrix 
            # matrix_path = save_etapas_dir + 'spark_pca_matrix_sim/matrix_similarity_npmem_job_ms_Quad_'+str(read_quad)
            matrix_path = path_matrix_sim + 'matrix_similarity_npmem_job_ms_Quad_'+str(read_quad)
        else:   # read sim matrix from cluster of simple cluster
            # matrix_path = save_etapas_dir + 'spark_pca_matrix_sim/matrix_similarity_npmem_job_Quad_'+str(read_quad)
            matrix_path = path_matrix_sim + 'matrix_similarity_npmem_job_Quad_'+str(read_quad)
    else:
        # matrix_path = save_etapas_dir + 'spark_pca_matrix_sim/matrix_similarity_npmem_job_30_Quad_'+str(read_quad)
        matrix_path = path_matrix_sim + 'matrix_similarity_npmem_job_30_Quad_'+str(read_quad)


t1 = time.time()
zarr_group = zarr.open(matrix_path, mode='a')
t2 = time.time()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')

print (f'{a}: 1. Tempo leitura similarity matrix : {t2-t1}s, {(t2-t1)/60}m') if sh_print else None  
#print (f'{a}: 1.1 matrix_sim_sel.shape : {matrix_sim_sel.shape} ')
print (f'{a}: 1.1 zarr_group type : {type(zarr_group)} ')
print (f'{a}: 1.2 zarr_group.array_keys: {list(zarr_group.array_keys())}') if sh_print else None
print(f'{a}: 1.3 zarr_group.group_keys: {list(zarr_group.group_keys())}') if sh_print else None

logger.info(f'1. Tempo leitura similarity matrix : {t2-t1}s, {(t2-t1)/60}m')  
logger.info(f'1.1 zarr_group type : {type(zarr_group)} ')
logger.info(f'1.2 zarr_group.array_keys: {list(zarr_group.array_keys())}')
logger.info(f'1.3 zarr_group.group_keys: {list(zarr_group.group_keys())}')

proc_dic[ri]['subetapa'] = 'read similarity matrix'
proc_dic[ri]['tempo'] = t2-t1

# Iterar sobre os arrays no grupo e exibir seus nomes e formas
for array_name in zarr_group.array_keys():
    array = zarr_group[array_name]
    print(f'1.4 Array: {array_name}, Shape: {array.shape}') if sh_print else None
    logger.info(f'1.4 Array: {array_name}, Shape: {array.shape}')
    if array_name == 'matrix_dist_sel':
        print (f'array_name = {array_name}, Shape: {array.shape}')

matrix_sim_sel = zarr_group['arr_0']
print (f'{a}: 1.5 matrix_sim_sel.shape: {matrix_sim_sel.shape}') if sh_print else None
logger.info(f'1.5 matrix_sim_sel.shape: {matrix_sim_sel.shape}')

proc_dic[ri]['size'] = matrix_sim_sel.shape

# 2. Gen distance matrix
# Criar um novo array Zarr para armazenar o resultado
# t1 = time.time()
# 20241016: se matriz_dist_sel nao estiver sido criada nos arquivos zarr 
#            tem que criar como abaixo, tem que acertar o codigo para em caso de ainda 
#            ter executar linhas abaixo
# if 'matriz_dist_sel' not in zarr_group:
print (list(zarr_group.array_keys()))
# if 'matrix_dist_sel' in list(zarr_group.array_keys()):
#     print (f'2.0 matrix_dist_sel in zarr_group and created there') if sh_print else None

mat_dist_name = 'matrix_dist_sel2' if sm == 2 else 'matrix_dist_sel'
print (f'2.0 mat_dist_name: {mat_dist_name}') if sh_print else None

if mat_dist_name not in list(zarr_group.array_keys()):
    print (f'2.0 {mat_dist_name} not in zarr_group and created there') if sh_print else None
    logger.info(f'2.0 {mat_dist_name} not in zarr_group and created there')
    # del zarr_group['matrix_dist_sel']  # Remove o dataset existente

    matrix_dist_sel = zarr_group.create_dataset(
        mat_dist_name, 
        shape=matrix_sim_sel.shape, 
        dtype=matrix_sim_sel.dtype
    )

t1 = time.time()
matrix_dist_sel = zarr_group[mat_dist_name]
t2 = time.time()
# print (f'{a}: 2.1 Tempo para create matrix de distancia: {(t2-t1)}, {(t2-t1)/60}')
print (f'{a}: 2.1 Tempo para read empty matrix de distancia: {(t2-t1):.2f}, {(t2-t1)/60:.2f}') if sh_print else None
print (f'{a}: 2.1.1 matrix_dist_sel.shape: {matrix_dist_sel.shape}, {matrix_dist_sel.dtype}') if sh_print else None
logger.info(f'2.1 Tempo para read empty matrix de distancia: {(t2-t1):.2f}, {(t2-t1)/60:.2f}')
logger.info(f'2.1.1 matrix_dist_sel.shape: {matrix_dist_sel.shape}, {matrix_dist_sel.dtype}')

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = f'cluster with distance matriz {sm}'
proc_dic[ri]['subetapa'] = 'read empty distance matrix'
proc_dic[ri]['tempo'] = t2-t1

# t1 = time.time()
# matrix_dist_sel[:] = 1 - matrix_sim_sel[:]
# matrix_dist_sel = matrix_dist_sel[:10000, 0:10000]

t1 = time.time()
matrix_dist_sel[:] = 1 - matrix_sim_sel[:]  #[:] [:10000, 0:10000] 
t2=time.time()

# matrix_dist_sel = matrix_dist_sel[:50000, 0:50000]
# matrix_dist_sel = 1 - matrix_sim_sel
t2=time.time()
print (f'{a}: 2.2 Tempo para gerar matrix de distancia: {(t2-t1):.2f}, {(t2-t1)/60:.2f}') if sh_print else None #1110.08, 1.83
logger.info(f'{a}: 2.2 Tempo para gerar matrix de distancia: {(t2-t1):.2f}, {(t2-t1)/60:.2f}')
del matrix_sim_sel
gc.collect()
t2_1=t2-t1
t1 = time.time()
matrix_dist_sel = matrix_dist_sel.astype(np.float32)
t2=time.time()
print (f'{a}: 2.2 Tempo para converter matrix de distancia para float32: {(t2-t1):.2f}, {(t2-t1)/60:.2f}') if sh_print else None
logger.info(f'{a}: 2.2 Tempo para converter matrix de distancia para float32: {(t2-t1):.2f}, {(t2-t1)/60:.2f}') 

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = f'cluster with distance matriz {sm}'
proc_dic[ri]['subetapa'] = 'gen distance matrix'
proc_dic[ri]['tempo'] = t2-t1+t2_1
proc_dic[ri]['size'] = matrix_dist_sel.shape

print (f'{a}: 2.3 matrix_dist_sel.shape: {matrix_dist_sel.shape}') if sh_print else None
logger.info(f'{a}: 2.3 matrix_dist_sel.shape: {matrix_dist_sel.shape}')

#Run Clara cluster for dist matrix
n_clusters=30 #30 
n_sample= 200 #40+2*n
#max_iter=300 default
rd_state={}#[0,50,99]
dic_cluster_ski_ms2={}
sse_ski_ms2=[]
dic_cluster_rd_ms2={}
sse_rd_ms2={}
time_i = time.time()
# for n in tqdm(range (2, n_clusters+1)):
print ("Entrar no loop do cluster") if sh_print else None
for n in range (2, n_clusters+1):
    #clara = timedcall(CLARA(n_clusters=n, random_state=0).fit(arraybands_sel))
    t1 = time.time()
    rd_state[n] = random.sample(range(999),1)
    # print (f'n = {n}') if sh_print else None
    for rd in rd_state[n]:
        # print (f'rd = {rd}')
        t2=time.time()
        cl_d = CLARA(n_clusters=n,n_sampling=n_sample, \
                      n_sampling_iter=5, max_iter=3000, random_state=rd) #3000
        
        clara = cl_d.fit(matrix_dist_sel)
        
        clusters_sel = clara.labels_
        dic_cluster_ski_ms2[str(n)+'_'+str(rd)] = clusters_sel.tolist()
        
        t3 = time.time()
        # print (f'3.1 Cluster Clara, tempo de execucao para {n} {rd}: {t3-t2:.2f}s {(t3-t2)/60:.2f}m')
        ri+=1
        proc_dic[ri]={} if ri not in proc_dic else None
        proc_dic[ri]['etapa'] = f'cluster with distance matriz {sm}'
        proc_dic[ri]['subetapa'] = f'Cluster Clara para {n} {rd}'
        proc_dic[ri]['tempo'] = t3-t2   

        sse_ski_ms2.append(clara.inertia_)
        
        if rd in dic_cluster_rd_ms2:
            dic_cluster_rd_ms2[rd][n]=clusters_sel.tolist()
            sse_rd_ms2[rd].append(clara.inertia_)
        else:
            dic_cluster_rd_ms2[rd]={}
            dic_cluster_rd_ms2[rd][n]=clusters_sel.tolist()
            sse_rd_ms2[rd]=[]
            sse_rd_ms2[rd].append(clara.inertia_)
        #adiciona a info do cluster no df
        # não vou adicionar ao df agora
        #snic_centroid_ski_df_dic[id_]['cluster_'+str(n)+'_'+str(rd)] = clusters_sel
    t4 = time.time()
    print (f'{n}_{rd}: {t4-t1}')
    
    ri+=1
    proc_dic[ri]={} if ri not in proc_dic else None
    proc_dic[ri]['etapa'] = f'cluster with distance matriz {sm}'
    proc_dic[ri]['subetapa'] = f'Cluster Clara para {n}'
    proc_dic[ri]['tempo'] = t4-t1   

time_f = time.time()
print (f'3.2 Tempo de execucao total Clara: {time_f-time_i:.2f}') if sh_print else None
logger.info(f'3.2 Tempo de execucao total Clara: {time_f-time_i:.2f}')

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = f'cluster with distance matriz {sm}'
proc_dic[ri]['subetapa'] = 'Clara execution'
proc_dic[ri]['tempo'] = time_f-time_i

print (f'3.2 Tempo de execucao total Clara: {time_f-time_i:.2f}') if sh_print else None
logger.info(f'3.2 Tempo de execucao total Clara: {time_f-time_i:.2f}')
#calcular e incluir n_opt no arquivo
obj_dic={}
obj_dic = {
    "dic_cluster_ski": dic_cluster_ski_ms2,
    "sse_ski": sse_ski_ms2,
    "dic_cluster_rd": dic_cluster_rd_ms2,
    "sse_rd": sse_rd_ms2,
    "rd_state":rd_state 
    #"n_opt": n_opt
}

# #com kmedoids para testat no terminal
#https://github.com/kno10/python-kmedoids/blob/main/README.md
# import kmedoids
# matrix_dist_sel = matrix_dist_sel.astype(np.float32) 
# a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
# print (a)
# t1=time.time()
# #c = kmedoids.fasterpam(matrix_dist_sel[:100000, 0:100000], 5)
# dm = kmedoids.dynmsc(matrix_dist_sel[:95000,:95000],30,2)
# t2=time.time()
# print (f'TEMPO: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
# a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
# print (a)
# # print("Loss is:", c.loss)
# print("Optimal number of clusters according to the Medoid Silhouette:", dm.bestk)
# print("Medoid Silhouette over range of k:", dm.losses)
# print("Range of k:", dm.rangek)
######

if KNN:
    ### 
    #20241104: teste com hierarchical clustering
    # from https://stackoverflow.com/questions/77691074/clustering-data-using-scipy-and-a-distance-matriz-in-python
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster #, ward
    from scipy.spatial.distance import squareform

    # Create a linkage matrix from the distance matrix
    #para condensar a matrix antes de passar para o linkage
    #condensar é converter para o formato matrix triangular superior ou inferior
    #o retorno é um vetor
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html

    t1=time.time()
    dist_vector_matrix_dist_sel = squareform(matrix_dist_sel)
    t2=time.time()
    print (f'Tempo squareform {t2-t1}s, {(t2-t1)/60:.2f}m') #1.86m

    t1=time.time()
    linkage_matrix = linkage(dist_vector_matrix_dist_sel, method='ward')
    t2=time.time()
    print (f'Tempo linkage hierarchical clustering {t2-t1}s, {(t2-t1)/60:.2f}m') #12.11m

    # Obtain cluster assignments
    t1=time.time()
    clusters = fcluster(linkage_matrix, 30, criterion='distance')
    t2=time.time()
    print (f'Tempo fcluster hierarchical clustering {t2-t1}s, {(t2-t1)/60:.2f}m') 

    n_clusters=30
    dic_cluster_ski_ms2 = {}
    sse_ski_ms2 = []
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (a)
    t1=time.time()
    for n in range (2, n_clusters+1):   
        dic_cluster_ski_ms2[str(n)] = fcluster(linkage_matrix, n, criterion='maxclust')        


    t2=time.time()
        # print (f'n = {n}')    
        # clusters_sel = fcluster(linkage_matrix, n, criterion='distance')        
        # clusters_sel = fcluster(linkage_matrix, n, criterion='maxclust')        
        # dic_cluster_ski_ms2[str(n)] = clusters_sel.tolist()


    t2=time.time()
    print (f'Tempo fcluster clustering {t2-t1}s, {(t2-t1)/60:.2f}m') #2.654675245285034

    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (a)
    #calculo do SSE

    for n in dic_cluster_ski_ms2.keys():
        print (f'n = {n}')        
        # Calcular o SSE
        sse = 0
        clusters_sel = dic_cluster_ski_ms2[str(n)]
        unique_labels = np.unique(clusters_sel)
        # print (f'n={n}, num labels = {len(unique_labels)}')
        ts1 = time.time()
        for label in unique_labels:
            # Obter os índices dos pontos no cluster atual
            cluster_indices = np.where(clusters_sel == label)[0]        
            # Calcular SSE usando distâncias internas do cluster
            # for i in cluster_indices:
            #     for j in cluster_indices:
            #         if i < j:  # Evitar cálculos duplicados
            #             sse += matrix_dist_sel[i, j] ** 2 / (2 * len(cluster_indices))    
            # # Extrair as distâncias internas do cluster da matriz de distâncias
            intra_cluster_distances = matrix_dist_sel[np.ix_(cluster_indices, cluster_indices)]
            
            # # Calcular o SSE somando o quadrado das distâncias e dividindo pelo número de pares
            sse += np.sum(intra_cluster_distances ** 2) / 2  # Dividir por 2 para evitar contagem duplicada

        ts2 = time.time()
        print(f"n = {n}, Tempo sse {ts2-ts1}s, SSE: {sse}")
        sse_ski_ms2.append(sse)
    t2=time.time()                  
        # sse = 0
        # for cluster_id in np.unique(clusters_sel):
        #     # Selecionar os pontos que pertencem ao cluster atual
        #     pontos_cluster = dados[clusters_sel == cluster_id]
            
        #     # Calcular o centróide do cluster
        #     centroide = pontos_cluster.mean(axis=0)
            
        #     # Somar as distâncias quadradas dos pontos ao centróide
        #     sse += ((pontos_cluster - centroide) ** 2).sum()

        # sse_ski_ms2.append(clara.inertia_)

        #     if rd in dic_cluster_rd_ms2:
        #         dic_cluster_rd_ms2[rd][n]=clusters_sel.tolist()
        #         sse_rd_ms2[rd].append(clara.inertia_)
        #     else:
        #         dic_cluster_rd_ms2[rd]={}
        #         dic_cluster_rd_ms2[rd][n]=clusters_sel.tolist()
        #         sse_rd_ms2[rd]=[]
        #         sse_rd_ms2[rd].append(clara.inertia_)
        #     #adiciona a info do cluster no df
        #     # não vou adicionar ao df agora
        #     #snic_centroid_ski_df_dic[id_]['cluster_'+str(n)+'_'+str(rd)] = clusters_sel
        # t4 = time.time()
        # print (f'{n}_{rd}: {t4-t1}')
    print (f'3.2 Tempo de execucao total hierarchical cluster: {t2-t1:.2f}')
    # time_f = time.time()
    t2=time.time()    

    # print (f'3.2 Tempo de execucao total Clara: {time_f-time_i:.2f}')
    print (f'3.2 Tempo de execucao total hierarchical cluster: {t2-t1:.2f}')



    #from meus programas
    t1=time.time()
    Z_ward = hierarchical_clustering(matrix_dist_sel,method='ward') 
    t2=time.time()
    print (f'Tempo hierarchical clustering {t2-t1}s, {(t2-t1)/60:.2f}m')
    t1=time.time()
    cluster_labels_ward_10 = fcluster(Z_ward, 10, criterion='maxclust')
    t2=time.time()
    print (f'Tempo fcluster {t2-t1}s, {(t2-t1)/60:.2f}m')


    #calcular e incluir n_opt no arquivo
    obj_dic={}
    obj_dic = {
        "dic_cluster_ski": dic_cluster_ski_ms2,
        "sse_ski": sse_ski_ms2,
        "dic_cluster_rd": dic_cluster_rd_ms2,
        "sse_rd": sse_rd_ms2,
        "rd_state":rd_state 
        #"n_opt": n_opt
    }

id=0
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
t1=time.time()
# file_to_save = base_dir + 'pca_snic_cluster/clara_ms'+str(id)+'.pkl'
if KNN:
    file_to_save = base_dir + 'data/tmp/pca_snic_cluster/clara_ms_knn_'+str(id)+'_20241117.pkl'
else:
    if pca_fullImg:
        d_name = save_etapas_dir + 'pca_snic_cluster/PCAFullImg/'
    else:
        d_name = save_etapas_dir + 'pca_snic_cluster/'
    if MaxDist:
        if sm==2:
            # file_to_save = save_etapas_dir + 'pca_snic_cluster/clara_ms'+str(sm)+'_'+str(id)+'_quad_'+str(read_quad)+'.pkl'
            file_to_save = d_name + 'clara_ms'+str(sm)+'_'+str(id)+'_quad_'+str(read_quad)+'.pkl'
        else:    
            # file_to_save = save_etapas_dir + 'pca_snic_cluster/clara_ms_'+str(id)+'_quad_'+str(read_quad)+'.pkl'
            file_to_save = d_name + 'clara_ms_'+str(id)+'_quad_'+str(read_quad)+'.pkl'
    else:
        # file_to_save = save_etapas_dir + 'pca_snic_cluster/clara_ms_'+str(id)+'_30_quad_'+str(read_quad)+'.pkl'
        file_to_save = d_name + 'clara_ms_'+str(id)+'_30_quad_'+str(read_quad)+'.pkl'


save_to_pickle(obj_dic, file_to_save)
t2=time.time()
print (f'{a}: 3.2 Tempo para salvar cluster clara: {time_f-time_i:.2f}, {(time_f-time_i)/60:.2f}') if sh_print else None
print (f'{a}: 3.3 clara com matriz de similaridade salvo {file_to_save}') if sh_print else None  
logger.info(f'3.2 Tempo para salvar cluster clara: {time_f-time_i:.2f}, {(time_f-time_i)/60:.2f}')
logger.info(f'3.3 clara com matriz de similaridade salvo {file_to_save}')
del obj_dic
gc.collect()
ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = f'cluster with distance matriz {sm}'
proc_dic[ri]['subetapa'] = 'salvar Clara results'
proc_dic[ri]['tempo'] = t2-t1
tf=time.time()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: Tempo de execução do {nome_prog}: {tf-ti:.2f} s {(tf-ti)/60:.2f} m')
logger.info(f'{a}: Tempo de execução do {nome_prog}: {tf-ti:.2f} s {(tf-ti)/60:.2f} m')

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = f'cluster with distance matriz {sm}'
proc_dic[ri]['subetapa'] = f'{nome_prog} time execution total'
proc_dic[ri]['tempo'] = tf-ti

time_file = process_time_dir + ' process_times.pkl'
update_procTime_file(proc_dic, time_file)