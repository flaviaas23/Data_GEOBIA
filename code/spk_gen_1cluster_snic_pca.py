# Programa para ler resultado do snic , fazer a clusterização 
# 20240827
# 20241015: Programa baixado do exacta spk_gen_1cluster_snic_pca
# 20241015: Programa alterado para funcionar no laptop
# 20241015: A imagem foi dividida e sgmentada em 4 quadrantes na fase anterior
#           as segmentacoes sao lidas e agrupadas novamente para serem clusterizadas
# 20241212: inclusao do log e do df com os tempos de processamento
#           clusterizacao da segmentação de 1/4 do tile, nao juntar os 
#           quadrantes
# 20241213: passagem de arqumentos para o programa
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

from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances_chunked
from sklearn.preprocessing import LabelEncoder
import functools
from scipy.sparse import issparse

from functions.functions_segmentation import gen_coords_snic_df
from functions.functions_pca import save_to_pickle, cria_logger, update_procTime_file

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
parser.add_argument("-i", '--name_img', type=str, help="Nome da imagem", default='')
parser.add_argument("-sp", '--sh_print', type=int, help="Show prints", default=0)
parser.add_argument("-pd", '--process_time_dir', type=str, help="Dir para df com os tempos de processamento", default='data/tmp2/')
# parser.add_argument("-rf", '--READ_df_features', type=int, help="Read or create df with features", default=0 )
# parser.add_argument("-nc", '--num_components', type=int, help="number of PCA components", default=4 )
parser.add_argument("-dsi", '--image_dir', type=str, help="Diretorio da imagem pca", default='data/tmp/spark_pca_images/')
parser.add_argument("-p", '--padrao', type=str, help="Filtro para o diretorio ", default='*')
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
img_dir = args.image_dir
padrao = args.padrao

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
t = datetime.datetime.now().strftime('%Y%m%d_%H%M_')

nome_prog = os.path.basename(__file__)
print (f'{a}: 0. INICIO {nome_prog}') if sh_print else None

#cria logger
nome_log = t + nome_prog.split('.')[0]+'.log'
logger = cria_logger(log_dir, nome_log)
logger.info(f'######## INICIO {nome_prog} ##########')
logger.info(f'args: sd={save_etapas_dir} ld={log_dir} i={name_img} pd={process_time_dir} sp={args.sh_print}')

ri=0        #indice do dicionario com os tempos de cada subetapa
proc_dic = {}
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'Cluster segmented image'

#base_dir = '/scratch/flavia/pca/'  #for exacta
# base_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/'

# read snic df
id=0
#number of quadrants if read_quad=0: 4 quadrants, else 1 quadrant
q_number = 1 if read_quad else 4 #number of quadrants of image, then dfs
t_1 = time.time()
for q in range (1, q_number+1):
    t1 = time.time()
    q = read_quad if q_number == 1 else q
    # file_to_open = '/scratch/flavia/pca/pca_snic/n_110000_comp_2_snic_centroid_df_0.pkl'
    # pca_snic_dir = base_dir +'data/tmp/spark_pca_snic/'+'Quad_'+str(q)+'/'
    pca_snic_dir = save_etapas_dir +'spark_pca_snic/'+'Quad_'+str(q)+'/'
    file_to_open = f'{pca_snic_dir}quad{str(q)}_n_30000_comp_2_snic_centroid_df_0.pkl'
    print (f'file to open = {file_to_open}') if sh_print else None
    with open(file_to_open, 'rb') as handle:    
        b = pickle.load(handle)
    if q == 1 or q_number ==1:
        print (f'q = {q}, q_number = {q_number}') if sh_print else None
        id = list(b.keys())[0]
        snic_centroid_df = b[id]['centroid_df']
        del b
        gc.collect()
        t2 = time.time()
        proc_dic[ri]['subetapa'] = f'ler snic_centroids df for quad {read_quad}'
        proc_dic[ri]['tempo'] = t2-t1
        logger.info(f'Tempo para ler snic_centroids df for quad {read_quad}: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
        continue
    #else:
    max_label = snic_centroid_df['label'].max()
    print (f'max_label quad{q}: {max_label}') if sh_print else None
    df = b[id]['centroid_df']
    df['label'] = df['label'] + max_label + 1
    snic_centroid_df = pd.concat([snic_centroid_df, df], axis=0)
    
    del b,df
    gc.collect()
    t2 = time.time()

    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a}: 1. quad{q} Tempo leitura e concat do centroids df: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m') if sh_print else None
    print (f'{a}: 1.1 quad{q} snic_centroid_df: \n{snic_centroid_df.head()}') if sh_print else None
    logger.info(f'Tempo quad{q} Tempo leitura e concat do centroids df: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
    logger.info(f'1.1 quad{q} snic_centroid_df: \n{snic_centroid_df.head()}')
    
    proc_dic[ri]['subetapa'] = f'ler e concat snic_centroids df for quad {read_quad}'
    proc_dic[ri]['tempo'] = t2-t1

snic_centroid_df.reset_index(drop=True, inplace=True)
t_2 = time.time()
print (f'{a}: 1.2 Tempo para gerar snic_centroids df: {t_2-t_1}s, {(t2-t1)/60}m') if sh_print else None
print (f'{a}: 1.3 snic_centroid_df: \n{snic_centroid_df.tail()}') if sh_print else None
logger.info(f'1.2 Tempo para gerar snic_centroids df: {t_2-t_1}s, {(t2-t1)/60}m')
logger.info(f'1.3 snic_centroid_df: \n{snic_centroid_df.tail()}')
ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = f'Cluster segmented image for quad {read_quad}'
proc_dic[ri]['subetapa'] = f'gerar snic_centroids df for quad {read_quad}'
proc_dic[ri]['tempo'] = t2-t1

t_2 = time.time()
for q in range (1, q_number+1):
    t2 = time.time()
    # file_to_open = '/scratch/flavia/pca/pca_snic/n_110000_comp_2_snic_centroid_df_0.pkl'
    # pca_snic_dir = base_dir +'data/tmp/spark_pca_snic/'+'Quad_'+str(q)+'/'
    pca_snic_dir = save_etapas_dir +'spark_pca_snic/Quad_'+str(q)+'/'
        #read segments to gen snic_coords_df
    # file_to_open = '/scratch/flavia/pca/pca_snic/n_110000_comp_2_snic_segments_0.pkl'
    file_to_open = f'{pca_snic_dir}quad{str(q)}_n_30000_comp_2_snic_segments_0.pkl'
    with open(file_to_open, 'rb') as handle:    
        b = pickle.load(handle)
    snic_segments = b[id]['segments']
    del b
    gc.collect()
    if q == 1:       
        # gen snic_coords_df
        t1 = time.time()
        snic_coords_df = gen_coords_snic_df(snic_segments, sh_print=True)
        t2 = time.time()
        del snic_segments
        gc.collect()
        print (f'{a}: 2.1 Tempo gerar coords df: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m') if sh_print else None
        print (f'{a}: 2.2 snic_coords_df: \n{snic_coords_df.head()}') if sh_print else None
        logger.info(f'2.1 Tempo gerar coords df: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m') 
        logger.info(f'2.2 snic_coords_df: \n{snic_coords_df.head()}')
        
        ri+=1
        proc_dic[ri]={} if ri not in proc_dic else None
        proc_dic[ri]['etapa'] = f'Cluster segmented image for quad {read_quad}'
        proc_dic[ri]['subetapa'] = f'gerar coords df for quad {read_quad}'
        proc_dic[ri]['tempo'] = t2-t1
        continue
    # else:
                
    # gen snic_coords_df
    t1 = time.time()
    df = gen_coords_snic_df(snic_segments, sh_print=True)
    # snic_coords_df
    t2 = time.time()
    print (f'{a}: 2.1 Tempo gerar coords df: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m') if sh_print else None
    logger.info(f'2.1 Tempo gerar coords df: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
    max_label = snic_coords_df['label'].max()
    print (f'max_label quad{q}: {max_label}') if sh_print else None
    logger.info(f'max_label quad{q}: {max_label}')
    
    del snic_segments
    gc.collect()
    ri+=1
    proc_dic[ri]={} if ri not in proc_dic else None
    proc_dic[ri]['etapa'] = f'Cluster segmented image for quad {read_quad}'
    proc_dic[ri]['subetapa'] = f'gerar coords df for quad {read_quad}'
    proc_dic[ri]['tempo'] = t2-t1

    df['label'] = df['label'] + max_label + 1
    snic_coords_df = pd.concat([snic_coords_df, df], axis=0)
    print (f'{a}: 2.2 snic_coords_df: \n{snic_coords_df.head()}') if sh_print else None
    logger.info(f'2.2 snic_coords_df: \n{snic_coords_df.head()}')

    del df
    gc.collect()
    
    t3 = time.time()
    print (f'{a}: 2.3 quad{q} Tempo concat coords df: {t3-t2}s, {(t3-t2)/60}m') if sh_print else None
    logger.info(f'2.3 quad{q} Tempo concat coords df: {t3-t2}s, {(t3-t2)/60}m')
    ri+=1
    proc_dic[ri]={} if ri not in proc_dic else None
    proc_dic[ri]['etapa'] = f'Cluster segmented image for quad {read_quad}'
    proc_dic[ri]['subetapa'] = f'gerar concat coords df for quad {read_quad}'
    proc_dic[ri]['tempo'] = t3-t2

snic_coords_df.reset_index(drop=True, inplace=True)
t_2 = time.time()

print (f'{a}: 2.4 quad{q} Tempo gerar snic_coords_df a partir de segments: {t_2-t_1}s, {(t_2-t_1)/60}m') if sh_print else None
logger.info(f'2.4 quad{q} Tempo gerar snic_coords_df a partir de segments: {t_2-t_1}s, {(t_2-t_1)/60}m')
ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = f'Cluster segmented image for quad {read_quad}'
proc_dic[ri]['subetapa'] = f'gerar snic_coords_df a partir de segments for quad {read_quad}'
proc_dic[ri]['tempo'] = t_2-t_1

#separar a linha com label -1
t2 = time.time()
snic_coords_df_neg = snic_coords_df[snic_coords_df['label'] == -1]
t3 = time.time()
snic_coords_df = snic_coords_df[snic_coords_df['label'] != -1]
t4 = time.time()

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 2.1 Tempo gerar coords df: {t_2-t_1:.2f}s, {(t_2-t_1)/60:.2f}m') if sh_print else None
print (f'{a}: 2.2 Tempo gerar coords df label -1: {t3-t2:.2f}s, {(t3-t2)/60:.2f}m') if sh_print else None
print (f'{a}: 2.3 Tempo remover linha -1 do coords df: {t4-t3:.2f}s, {(t4-t3)/60:.2f}m') if sh_print else None
print (f'{a}: 2.4 snic_coords_df: \n{snic_coords_df.head()}') if sh_print else None
print (f'{a}: 2.5 snic_coords_df_neg: \n{snic_coords_df_neg.head()}') if sh_print else None

logger.info(f'2.1 Tempo gerar coords df: {t_2-t_1:.2f}s, {(t_2-t_1)/60:.2f}m')
logger.info(f'2.2 Tempo gerar coords df label -1: {t3-t2:.2f}s, {(t3-t2)/60:.2f}m') 
logger.info(f'2.3 Tempo remover linha -1 do coords df: {t4-t3:.2f}s, {(t4-t3)/60:.2f}m') 
logger.info(f'2.4 snic_coords_df: \n{snic_coords_df.head()}') 
logger.info(f'2.5 snic_coords_df_neg: \n{snic_coords_df_neg.head()}') 

#get nan rows in snic_centroid
#comps = ['c'+str(i) for i in range(6)]
cols_snic_centroid = snic_centroid_df.columns[4:]
comps = [x.split('_')[-1] for x in cols_snic_centroid]
comps = sorted(list(set(comps)))
cols_sel = ['avg_'+str(i) for i in comps]
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 2.6 components pca: {comps}, cols_selected: {cols_sel}') if sh_print else None
logger.info(f'{a}: 2.6 components pca: {comps}, cols_selected: {cols_sel}') 

#verifies if there is nans in dataframe
nan_counts = snic_centroid_df[cols_snic_centroid].isna().sum()
print (f'{a}: 3.0 snic_centroid_df nan_counts: \n{nan_counts}') if sh_print else None
logger.info(f'3.0 snic_centroid_df nan_counts: \n{nan_counts}') 
#get the nan rows in snic_centroid_df
t1 = time.time()
snic_centroid_df_nan= snic_centroid_df[snic_centroid_df.isnull().any(axis=1)]
t2 = time.time()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 3.1 Tempo pegar as rows with nan in snic_centroid_df: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m') if sh_print else None
print (f'{a}: 3.2 snic_centroid_df_nan shape:{snic_centroid_df_nan.shape},\nsnic_centroid_df_nan: \n{snic_centroid_df_nan}') if sh_print else None
logger.info(f'3.1 Tempo pegar as rows with nan in snic_centroid_df: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m') 
logger.info(f'3.2 snic_centroid_df_nan shape:{snic_centroid_df_nan.shape},\nsnic_centroid_df_nan: \n{snic_centroid_df_nan}') 

#fazer o drop do nans no snic_centroid_df
t1 = time.time()
snic_centroid_df = snic_centroid_df.dropna()
t2 = time.time()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 3.3 Tempo drop rows with nan in snic_centroid_df: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m') if sh_print else None
print (f'{a}: 3.4 snic_centroid_df_nan shape:{snic_centroid_df.shape}') if sh_print else None
logger.info(f'3.3 Tempo drop rows with nan in snic_centroid_df: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m') 
logger.info(f'3.4 snic_centroid_df after dropna shape:{snic_centroid_df.shape}')

#run Clara to cluster
arraybands_sel = snic_centroid_df[cols_sel].to_numpy()

n_clusters=30 
n_sample= 200 #40+2*n
n_randoms = 1 #numero de random que serao gerados
rd_state={}#[0,50,99]
dic_cluster_ski={}
sse_ski=[]
dic_cluster_rd={}
sse_rd={}
time_i = time.time()
for n in tqdm(range (2, n_clusters+1)):
    #clara = timedcall(CLARA(n_clusters=n, random_state=0).fit(arraybands_sel))
    t1 = time.time()
    rd_state[n] = random.sample(range(999),n_randoms)
    for rd in rd_state[n]:
        clara = CLARA(n_clusters=n,n_sampling=n_sample, \
                      n_sampling_iter=5, random_state=rd).fit(arraybands_sel)
        #Entender a diferenca do predict para labels_
        #clusters_sel = clara.predict(arraybands_sel)
        clusters_sel = clara.labels_
        
        #print (f'tempo de execucao para {n}: {time_fim-time_ini}')
        dic_cluster_ski[str(n)+'_'+str(rd)] = clusters_sel.tolist()        
        sse_ski.append(clara.inertia_)
        
        if rd in dic_cluster_rd:
            dic_cluster_rd[rd][n]=clusters_sel.tolist()
            sse_rd[rd].append(clara.inertia_)
        else:
            dic_cluster_rd[rd]={}
            dic_cluster_rd[rd][n]=clusters_sel.tolist()
            sse_rd[rd]=[]
            sse_rd[rd].append(clara.inertia_)
        #adiciona a info do cluster no df
        # não vou adicionar ao df agora
        #snic_centroid_ski_df_dic[id_]['cluster_'+str(n)+'_'+str(rd)] = clusters_sel
    t2 = time.time()
    # print (f'{n}_{rd}: {t2-t1}') if sh_print else None
time_f = time.time()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: Tempo de execucao total Clara: {time_f-time_i:.2f}') if sh_print else None
logger.info(f'Tempo de execucao total Clara: {time_f-time_i:.2f}') 
ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = f'Cluster segmented image for quad {read_quad}'
proc_dic[ri]['subetapa'] = f'Clara execution for quad {read_quad}'
proc_dic[ri]['tempo'] = time_f-time_i

obj_dic={}
obj_dic = {
    "dic_cluster_ski": dic_cluster_ski,
    "sse_ski": sse_ski,
    "dic_cluster_rd": dic_cluster_rd,
    "sse_rd": sse_rd,
    "rd_state":rd_state 

    #"n_opt": n_opt
}
# file_to_save = base_dir + 'pca_snic_cluster/clara_'+str(id)+'.pkl'
# file_to_save = base_dir + 'data/tmp/pca_snic_cluster/clara_'+str(id)+'.pkl' #20241213 commented
d_name = save_etapas_dir + 'pca_snic_cluster/'
file_to_save = d_name+'clara_'+str(id)+'_quad_'+str(read_quad)+'.pkl'
diretorio = Path(d_name)
diretorio.mkdir(parents=True, exist_ok=True)
print (f'files to save: {file_to_save}') if sh_print else None
logger.info(f'files to save: {d_name} {file_to_save}')
t1 = time.time()
save_to_pickle(obj_dic, file_to_save)
t2 = time.time()
print (f'Tempo to save: {t2-t1:.2f}') if sh_print else None
logger.info(f'Tempo to save: {t2-t1:.2f}') 

del obj_dic, snic_centroid_df, snic_coords_df   #,snic_segments #20241213 commented
gc.collect()

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = f'Cluster segmented image for quad {read_quad}'
proc_dic[ri]['subetapa'] = f'Save cluster Clara in pkl for quad {read_quad}'
proc_dic[ri]['tempo'] = t2-t1

#fim
tf = time.time()
print (f'Tempo total do {nome_prog} for quad {read_quad}: {tf-ti:.2f}s, {(tf-ti)/60:.2f}m') 
logger.info(f'Tempo total do {nome_prog} for quad {read_quad}: {tf-ti:.2f}s, {(tf-ti)/60:.2f}m') 

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
#b = datetime.datetime.now().strftime('%H:%M:%S')
nome_prog = os.path.basename(__file__)

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = f'Cluster segmented image for quad {read_quad}'
proc_dic[ri]['subetapa'] = f'{nome_prog} time execution total for quad {read_quad}'
proc_dic[ri]['tempo'] = tf-ti

time_file = process_time_dir + "process_times.pkl"
update_procTime_file(proc_dic, time_file)

print (f'{a}: 0. Fim {nome_prog}') if sh_print else None
logger.info(f'0. Fim {nome_prog}') 
