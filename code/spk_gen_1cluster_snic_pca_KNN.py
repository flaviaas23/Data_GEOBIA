# Programa para ler resultado do snic , fazer a clusterização 
# imports
# 20240827
# 20241015: Programa baixado do exacta spk_gen_1cluster_snic_pca
# 20241015: Programa alterado para funcionar no laptop
# 20241015: A imagem foi dividida e sgmentada em 4 quadrantes na fase anterior
#           as segmentacoes sao lidas e agrupadas novamente para serem clusterizadas
# 20241212: inclusao do log e do df com os tempos de processamento
#           clusterizacao da segmentação de 1/4 do tile, nao juntar os 
#           quadrantes

import os
import gc
import time
import datetime
import pandas as pd
import numpy as np
import pickle
import random
from tqdm import tqdm
from sklearn_extra.cluster import CLARA

from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances_chunked
from sklearn.preprocessing import LabelEncoder
import functools
from scipy.sparse import issparse

from functions.functions_segmentation import gen_coords_snic_df
from functions.functions_pca import save_to_pickle, update_procTime_file

ti = time.time()

# 20241212 bloco de parse incluido
# Inicializa o parser
parser = argparse.ArgumentParser(description="Program segment image")

# Define os argumentos
parser.add_argument("-bd", '--base_dir', type=str, help="Diretorio base", default='')
parser.add_argument("-sd", '--save_dir', type=str, help="Dir base para salvar saidas de cada etapa", default='data/tmp2/')
# parser.add_argument("-td", '--tif_dir', type=str, help="Dir dos tiffs", default='data/Cassio/S2-16D_V2_012014_20220728_/')
parser.add_argument("-q", '--quadrante', type=int, help="Numero do quadrante da imagem [0-all,1,2,3,4]", default=1)
parser.add_argument("-ld", '--log_dir', type=str, help="Dir do log", default='code/logs/')
parser.add_argument("-i", '--name_img', type=str, help="Nome da imagem", default='S2-16D_V2_012014')
parser.add_argument("-sp", '--sh_print', type=str, help="Show prints", default=True)
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
n_components = args.num_components
img_dir = args.image_dir
padrao = args.padrao

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
#b = datetime.datetime.now().strftime('%H:%M:%S')
nome_prog = os.path.basename(__file__)
print (f'{a}: 0. INICIO {nome_prog}')

#base_dir = '/scratch/flavia/pca/'
# base_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/' #20241212 commented

# read snic df
id=0
q_number = 4 #number of quadrants of image, then dfs
t_1 = time.time()
for q in range (1, q_number+1):
    t1 = time.time()
    # file_to_open = '/scratch/flavia/pca/pca_snic/n_110000_comp_2_snic_centroid_df_0.pkl'
    pca_snic_dir = base_dir +'data/tmp/spark_pca_snic/'+'Quad_'+str(q)+'/'
    file_to_open = f'{pca_snic_dir}quad{str(q)}_n_30000_comp_2_snic_centroid_df_0.pkl'
    with open(file_to_open, 'rb') as handle:    
        b = pickle.load(handle)
    if q == 1:
        id = list(b.keys())[0]
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
    print (f'{a}: 1. quad{q} Tempo leitura centroids df: {t2-t1}s, {(t2-t1)/60}m')
    print (f'{a}: 1.1 quad{q} snic_centroid_df: \n{snic_centroid_df.head()}')

snic_centroid_df.reset_index(drop=True, inplace=True)
t_2 = time.time()
print (f'{a}: 1.2 Tempo para gerar snic_centroids df: {t_2-t_1}s, {(t2-t1)/60}m')
print (f'{a}: 1.3 snic_centroid_df: \n{snic_centroid_df.tail()}')

t_2 = time.time()
for q in range (1, q_number+1):
    t2 = time.time()
    # file_to_open = '/scratch/flavia/pca/pca_snic/n_110000_comp_2_snic_centroid_df_0.pkl'
    pca_snic_dir = base_dir +'data/tmp/spark_pca_snic/'+'Quad_'+str(q)+'/'
        #read segments to gen snic_coords_df
    # file_to_open = '/scratch/flavia/pca/pca_snic/n_110000_comp_2_snic_segments_0.pkl'
    file_to_open = f'{pca_snic_dir}quad{str(q)}_n_30000_comp_2_snic_segments_0.pkl'
    with open(file_to_open, 'rb') as handle:    
        b = pickle.load(handle)
    snic_segments = b[id]['segments']
    del b
    if q == 1:       
        # gen snic_coords_df
        t1 = time.time()
        snic_coords_df = gen_coords_snic_df(snic_segments, sh_print=True)
        t2 = time.time()
        del snic_segments
        print (f'{a}: 2.1 Tempo gerar coords df: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
        print (f'{a}: 2.2 snic_coords_df: \n{snic_coords_df.head()}')
        continue
    # else:
                
    # gen snic_coords_df
    t1 = time.time()
    df = gen_coords_snic_df(snic_segments, sh_print=True)
    # snic_coords_df
    t2 = time.time()
    print (f'{a}: 2.1 Tempo gerar coords df: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
    max_label = snic_coords_df['label'].max()
    print (f'max_label quad{q}: {max_label}')
    
    df['label'] = df['label'] + max_label + 1
    snic_coords_df = pd.concat([snic_coords_df, df], axis=0)
    print (f'{a}: 2.2 snic_coords_df: \n{snic_coords_df.head()}')

    del df
 
    t3 = time.time()
    print (f'{a}: 2.3 quad{q} Tempo leitura segments df: {t3-t2}s, {(t3-t2)/60}m')

snic_coords_df.reset_index(drop=True, inplace=True)
t_2 = time.time()
print (f'{a}: 2.4 quad{q} Tempo gerar snic_coords_df a partir de segments: {t_2-t_1}s, {(t_2-t_1)/60}m')

#separar a linha com label -1
t2 = time.time()
snic_coords_df_neg = snic_coords_df[snic_coords_df['label'] == -1]
t3 = time.time()
snic_coords_df = snic_coords_df[snic_coords_df['label'] != -1]
t4 = time.time()

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 2.1 Tempo gerar coords df: {t_2-t_1:.2f}s, {(t_2-t_1)/60:.2f}m')
print (f'{a}: 2.2 Tempo gerar coords df label -1: {t3-t2:.2f}s, {(t3-t2)/60:.2f}m')
print (f'{a}: 2.3 Tempo remover linha -1 do coords df: {t4-t3:.2f}s, {(t4-t3)/60:.2f}m')
print (f'{a}: 2.4 snic_coords_df: \n{snic_coords_df.head()}')
print (f'{a}: 2.5 snic_coords_df_neg: \n{snic_coords_df_neg.head()}')

#get nan rows in snic_centroid
#comps = ['c'+str(i) for i in range(6)]
cols_snic_centroid = snic_centroid_df.columns[4:]
comps = [x.split('_')[-1] for x in cols_snic_centroid]
comps = sorted(list(set(comps)))
cols_sel = ['avg_'+str(i) for i in comps]
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 2.6 components pca: {comps}, cols_selected: {cols_sel}')

#verifies if there is nans in dataframe
nan_counts = snic_centroid_df[cols_snic_centroid].isna().sum()
print (f'{a}: 3.0 snic_centroid_df nan_counts: \n{nan_counts}')
#get the nan rows in snic_centroid_df
t1 = time.time()
snic_centroid_df_nan= snic_centroid_df[snic_centroid_df.isnull().any(axis=1)]
t2 = time.time()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 3.1 Tempo pegar as rows with nan in snic_centroid_df: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
print (f'{a}: 3.2 snic_centroid_df_nan shape:{snic_centroid_df_nan.shape},\nsnic_centroid_df_nan: \n{snic_centroid_df_nan}')


#fazer o drop do nans no snic_centroid_df
t1 = time.time()
snic_centroid_df = snic_centroid_df.dropna()
t2 = time.time()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 3.3 Tempo drop rows with nan in snic_centroid_df: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
print (f'{a}: 3.4 snic_centroid_df_nan shape:{snic_centroid_df.shape}')

# Fazer o knn e agrupar os SPs que sao os mais proximos entre si
# simetricamente, por ex, sp1 mais proximo de sp2 e sp2 mais proximo de sp1
from sklearn.neighbors import NearestNeighbors

ti = time.time()
k = 2  # Mudar para 2 ou 3 conforme necessário
t1 = time.time()
# Cria o modelo KNN
knn = NearestNeighbors(n_neighbors=k)
t2 = time.time()
print (f'{a}: 4.1 Tempo de execucao knn nearestNeighbors: {t2-t1}')

t1 = time.time()
# Treina o modelo com os dados do dataframe
knn.fit(snic_centroid_df[cols_sel])
t2 = time.time()
print (f'{a}: 4.2 Tempo de execucao knn fit: {t2-t1}')

t1 = time.time()
# Encontra os índices dos vizinhos mais próximos para cada exemplo
distances, indices = knn.kneighbors(snic_centroid_df[cols_sel])
t2 = time.time()
print (f'{a}: 4.3 Tempo para obter as dinstancias e indices do knn: {t2-t1}')

tf = time.time()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 4.4 Tempo de execucao total knn: {tf-ti}')

# agrupar os pares que sao os mais proximos entre si simetricamente
# opcao1 criando array
# t1=time.time()
# knn_label=[]
# for pair in indices:
#      print ( f'{pair[0]} ,{ indices[pair[1]][1]}')
#      if  pair[0] == indices[pair[1]][1]:
#             print (f'pair0 = {pair[0]}, pair1 = {pair[1]}, indices pair1 = {indices[pair[1]]}')
#             pair_label = str(pair[0]) + '_' + str(pair[1])
#             knn_label.append(pair_label)
#      else:
#             knn_label.append(pair[0])

# t2=time.time()
# print (f'Tempo agrupar knn {t2-t1}s, {(t2-t1)/60:.2f}m')

# opcao2 criando dicionario
t1=time.time()
knn_label={}
for pair in indices:
     #print ( f'{pair[0]} ,{ indices[pair[1]][1]}')
     if  pair[0] == indices[pair[1]][1]:
        # print (f'pair0 = {pair[0]}, pair1 = {pair[1]}, indices pair1 = {indices[pair[1]]}')
        pair_label = str(pair[0]) + '_' + str(pair[1])
        knn_label.setdefault(pair[0], pair_label)
        knn_label.setdefault(pair[1], pair_label)
        # knn_label[pair[0]] = pair_label
        # knn_label[pair[1]] = pair_label
            
     else:
        knn_label.setdefault(pair[0], str(pair[0]))
        knn_label.setdefault(pair[1], str(pair[1]))
        

t2=time.time()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 4.5.1 Tempo gerar dic knn {t2-t1}s, {(t2-t1)/60:.2f}m')

#incluir info do knn label no df
t1=time.time()
snic_centroid_df['knn_label'] = snic_centroid_df['label'].map(knn_label)
t2=time.time()

print (f'{a}: 4.5.2 Tempo incluir knn label no df {t2-t1}s, {(t2-t1)/60:.2f}m')
print (f'{a}: snic_centroid_df: shape{snic_centroid_df.shape}\n{snic_centroid_df.head(3)}')

#agrupar pelo knn_label , fazer média dos grupos
t1=time.time()
cols_sel_gr = ['label', 'knn_label'] + cols_sel
# snic_centroid_df_group = snic_centroid_df[cols_sel_gr].groupby('knn_label')
snic_centroid_df_group = snic_centroid_df.groupby('knn_label')[cols_sel].mean()
snic_centroid_df_group = snic_centroid_df_group.reset_index()  #nao sei se preciso fazer isso ...
snic_centroid_df_group = snic_centroid_df_group.sort_values(by='knn_label', key = lambda x: x.str.replace('_','.').astype(float)).reset_index()
snic_centroid_df_group= snic_centroid_df_group.drop(columns='index')
t2=time.time()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 4.6.1 Tempo para gerar snic_centroid_group: {t2-t1}s')
print (f'{a}: 4.6.2 snic_centroid_df_group:shape {snic_centroid_df_group.shape}\n{snic_centroid_df_group.head()}')

#save to pickle file
t1=time.time()
file_to_save = base_dir + 'data/tmp/spark_pca_snic_centroid_df_knn/snic_centroid_df_knn'+str(id)+'.pkl'
with open(file_to_save, 'wb') as file:
    pickle.dump(snic_centroid_df_group, file, protocol=pickle.HIGHEST_PROTOCOL)
#run Clara to cluster
# arraybands_sel = snic_centroid_df[cols_sel].to_numpy()
t2=time.time()
print (f'{a}: 4.7 Tempo para salvar snic_centroid_group: {t2-t1}s')
print (f'{a}: 4.8 arquivo snic_centroid_group: {file_to_save}')

arraybands_sel = snic_centroid_df_group[cols_sel].to_numpy()

n_clusters = 30 #30 
n_sample= 200 #40+2*n
n_randoms = 1 #numero de random que serao gerados
rd_state = {}#[0,50,99]
dic_cluster_ski = {}
sse_ski = []
dic_cluster_rd = {}
sse_rd = {}
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
    print (f'{a}: {n}_{rd}: {t2-t1}')

time_f = time.time()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}: 5.1 Tempo de execucao total Clara: {time_f-time_i}')

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
# file_to_save = base_dir + 'data/tmp/pca_snic_cluster/clara_'+str(id)+'.pkl'
file_to_save = base_dir + 'data/tmp/pca_snic_cluster/clara_knn'+str(id)+'.pkl'
print (f'{a}: 5.2 files to save: {file_to_save}')
save_to_pickle(obj_dic, file_to_save)

del obj_dic, snic_centroid_df, snic_coords_df, snic_segments
#fim
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
t_f = time.time()
#b = datetime.datetime.now().strftime('%H:%M:%S')
nome_prog = os.path.basename(__file__)
print (f'{a}: 6. Fim {nome_prog} {(t_f-t_1)/60:.2f}')
