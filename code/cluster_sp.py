
#%%
import os
import matplotlib.pyplot as plt
import numpy as np

#from skimage.data import astronaut,lily, immunohistochemistry
# from skimage.color import rgb2gray
# from skimage.filters import sobel
# from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
# from skimage.segmentation import mark_boundaries
# from skimage.util import img_as_float

#import imageio.v2 as imageio
from skimage.measure import regionprops, regionprops_table
import pandas as pd
#import matplotlib.pyplot as plt
#import plotly.express as px
import psutil
import time
import sys
from itertools import product

import pickle
import random
from tqdm import tqdm

from plotly.subplots import make_subplots

#%%
from pyclustering.cluster.clarans import clarans
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import timedcall;

seed = random.seed(999) # para gerar sempre com a mesma seed

def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def cria_SimilarityMatrix_freq(dic_cluster):
        '''
        Generate a frequency/similarity/Co-association matrix based on 
        frequency of point are together in the clustering of pixels
        '''
        n_clusters = list(dic_cluster.keys())  
        nrow = len(dic_cluster[n_clusters[0]])
        print ("nrow= {}, len nclusters = {}".format(nrow, len(n_clusters)))
        s = (nrow, nrow)
        freq_matrix= np.zeros(s)
        for n in n_clusters:
            #print ("n = ",n)
            #sil = dic_cluster[n]['sample_silhouette_values']
            cluster = dic_cluster[n]
            #print ("sil= ",sil,"\ncluster = ",cluster)
            for i in range(0, (nrow)):            
                #print ("i = ",i)
                for j in range(0, nrow):
                    #print ("j = ",j , cluster[i], cluster[j], sil[i], sil[j])
                    if cluster[i] == cluster[j]:

                        #freq = (sil[i]+sil[j]+2)/4
                        freq_matrix[i,j] += 1 # freq
                        #print ("j = ",j , cluster[i], cluster[j], sil[i], sil[j], freq)

            #print ("freq_matrix = \n", freq_matrix)
        freq_matrix= freq_matrix/len(n_clusters)
        #print ("freq_matrix = \n", freq_matrix)
        return freq_matrix        

def read_props_df_sel(ids, open_path, obj_to_read='props_df_sel',output=True):
    ''''
    read a list of props_df_sel and returns them as a dicionary
    '''
    
    props_df_sel={}
    for id in tqdm(ids):      #ids #ids_file
        if obj_to_read == 'props_df_sel':
            file_to_open = open_path + '_'+str(id)+'.pkl'
        elif obj_to_read == "segments_slic_sel":
            file_to_open = open_path + '_segments_'+str(id)+'.pkl'

        print (file_to_open) if output else None
        with open(file_to_open, 'rb') as handle: 
            b = pickle.load(handle)
        props_df_sel[id] = b[id][obj_to_read][id]
        
    return props_df_sel     
    

def get_labels(clusters, len_arraybands_list):
    ''''
    retorna os labels de cada elemento baseado no seu indice
    '''
    labels = []
    for elemento_procurado in range(len_arraybands_list): 
        for i, subarray in enumerate(clusters):
            if elemento_procurado in subarray:
                #indice_subarray = i
                labels.append(i)
                break
    
    return np.array(labels)

# getting dir and file name
# Get the current working directory
current_directory = os.getcwd()

# Construct the path to the upper-level directory
upper_level_directory = os.path.join(current_directory, '../data/git/')
#upper_level_directory = os.path.join(current_directory, 'data/test_segm_results/')

# Specify the filename and path within the upper-level directory
filename = 'SENTINEL-2_MSI_20LMR_RGB_2022-07-16'
save_path = os.path.join(upper_level_directory, filename)   
########################################################
#       Cluster                                        #
########################################################
#%% # ler arquivos props selecionados
id_test=1 # 314
ids = [id_test]
props_df_sel=read_props_df_sel(ids, save_path)
#%%
#segments_slic_sel=read_props_df_sel(ids,save_path, obj_to_read='segments_slic_sel')
#%%
# 5. Fazer cluster da imagem
# seleciona as bandas que Vão ser usadas na clusterizacao e converte para numpy 
# e depois para list
 # 304, 302
bands_to_cluster = ['NBR','EVI','NDVI']
#arraybands_sel = props_df_sel[['NBR','EVI','NDVI']].to_numpy()
arraybands_sel = props_df_sel[id_test][bands_to_cluster].to_numpy()
arraybands_list_sel = arraybands_sel.tolist()


#%%
dic_cluster = {}
#%%
n_clusters =  8 #number of clusters

for n in tqdm(range(2, n_clusters+1)):
    clarans_instance_img_sel = clarans(arraybands_list_sel, n ,6, 4)

    (ticks, result) = timedcall(clarans_instance_img_sel.process)
    print("Execution time : ", ticks, "\n")

    #returns the clusters 
    clusters_sel = clarans_instance_img_sel.get_clusters()

    #returns the medoids 
    medoids_sel = clarans_instance_img_sel.get_medoids()
    
    labels_sel = get_labels(clusters_sel, len(arraybands_list_sel))
    dic_cluster[n] = labels_sel
    
    #adiciona a info do cluster no df
    props_df_sel[id_test]['cluster_'+str(n)]= labels_sel[props_df_sel[id_test].index]
#%%
matrix_sim={}
#%%
matrix_sim[id_test] = cria_SimilarityMatrix_freq(dic_cluster)
obj_dic={}
obj_dic = {
    "props_df_sel_cluster": props_df_sel,
    "dic_labels_cluster": dic_cluster,
    "matrix_sim":matrix_sim
}

file_to_save = save_path + '_cluster_'+str(id_test)+'.pkl'
save_to_pickle(obj_dic, file_to_save)