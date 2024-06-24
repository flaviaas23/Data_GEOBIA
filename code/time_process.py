''''
Programa para calcular os tempos de cada parte do processo 
1. leitura do arquivos de banda tif
2. geracao de imagem com as bandas RGB
3. Geracao de mascara caso tenha valores Nan na bandas RGB
4. Gerar imagem normalizada
5. Carregar a imagem normalizada de um arquivo
6. Para a segmentacao SLIC, Gerar dataframe e dicionario com as coordenadas 
'''
#%%
# import matplotlib.pyplot as plt
# from matplotlib import gridspec
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.ticker import MaxNLocator, FormatStrFormatter
# from matplotlib.colorbar import Colorbar

import numpy as np
from math import ceil

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

import imageio.v2 as imageio
from skimage.measure import regionprops, regionprops_table
import pandas as pd
import pyspark.pandas as ps
from pyspark.sql import SparkSession

import matplotlib.pyplot as plt
import plotly.express as px
import psutil
import time
import sys
import os

from itertools import product

import pickle
import h5py
import random
from tqdm import tqdm

from plotly.subplots import make_subplots

seed = random.seed(999) # para gerar sempre com a mesma seed

#%%
def gen_files_to_read(name_img, bands, read_dir):
    ''''
    gen array with tif files name to read
    '''
    files_to_read = []
    for b in bands:
        files_to_read.append(read_dir+name_img+b+'.tif')
    
    return files_to_read

def load_image_files2(files_name,pos=-2):
    ''''
    load tiff image bands files of a timestamp
    '''
    #files_name=[file_nbr,file_evi, file_ndvi, file_red,file_green,file_blue]
    bands_name =[]
    image_band_dic={}
    for f in files_name:
        f_name = f.split('.')  #para remover o .tif
        #print ('f_name:',f_name)
        band = f_name[0].split("_") # para obter onome da banda do arquivo
        #print (band[pos])
        bands_name.append(band[pos])
        image_band_dic[band[pos]] = imageio.imread(f)

    return image_band_dic

def load_image_norm():
    read_path = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/test_segm_results/S2-16D_V2_012014_20220728_/S2-16D_V2_012014_20220728_'
    file_to_open = read_path + 'img_sel_norm.pkl'
    with open(file_to_open, 'rb') as handle:    
        img_sel_norm = pickle.load(handle)
    return img_sel_norm

def calc_std_array(arr,b):
    # Map array elements to dictionary values
    #pixel_values_map = [pixel_band_value[elem[0], elem[1]] for elem in arr]
    pixel_values_map = [image_band_dic[b][elem[0], elem[1]] for elem in arr]
    #print (f'{b}\nmapped array: {pixel_values_map}')
    # Calculate standard deviation of mapped values
    return np.std(pixel_values_map), np.mean(pixel_values_map), \
            pixel_values_map,

#%%
def img_slic_segment_gen_df(bands_sel, image_band_dic, img_sel='', n_segms=600, sigma=2, \
                             compactness = 5, mask=None, conectivity=False):
    ''''
    receives the bands to generate image and image bands dictionary
    slic only accepts 3 bands
    generate image with selected bands and returns df of segmented image
    '''
    # 1.
    #criando imagem RGB para segmentar com SLIC 
    #img_rgb = np.dstack((image_band_dic['B11'], image_band_dic['B8A'], image_band_dic['B02']))
    if not len(img_sel):
        print ("img_sel vazia")
        img_sel = np.dstack((image_band_dic[bands_sel[0]], image_band_dic[bands_sel[1]], image_band_dic[bands_sel[2]]))
    # n_segms = 600
    # compactness = 5
    # sigma=2
    # conectivity=False
    #print ("params for slic: ", n_segms, sigma, compactness, conectivity )
    time_ini = time.time()
    segments_slic_sel = slic(img_sel, n_segments=n_segms, compactness=compactness, sigma=sigma, \
                             start_label=1,mask=mask, enforce_connectivity=conectivity)
    #print(f'SLIC RGB number of segments : {len(np.unique(segments_slic_sel))}')
    
    # 2. 
    # props_dic = regionprops_table(segments_slic_sel, img_sel, \
    #                             properties=['label','coords', 'centroid','local_centroid'])
   
    time_slic = time.time()
    #26/02/2024: separando 
    
    props_dic = regionprops_table(segments_slic_sel, img_sel, \
                                properties=['label','centroid','local_centroid']) 
    
    props_dic_SP = regionprops_table(segments_slic_sel, img_sel, \
                                properties=['label','coords'])   


    # 3.
    
    props_df = pd.DataFrame(props_dic)
    #props_df['num_pixels'] = props_df['coords'].apply(len)

        
    # 4. adiciona os valores das bandas do centroids (x,y) como colunas no df, 
    #    for each sp (superpixel), adds as column: number of pixels, calculates std and average 
    #    of each band and the value of pixels for each band 
    for b in image_band_dic.keys():
        props_df[b] = props_df.apply(lambda row: image_band_dic[b][round(row['centroid-0']), round(row['centroid-1'])], axis=1)
        #props_df['desvio_'+b]= props_df['coords'].apply(lambda arr: [image_band_dic[b][elem[0],elem[1]] for elem in arr])
        #pixel_band_value = image_band_dic[b]
        #props_df[['std_'+b, 'mean_'+b, 'seg_'+b ]] = props_df['coords'].apply(calc_std_array, b=b).apply(pd.Series)
    time_df = time.time()
    time_proc={}
    time_proc['slic'] = time_slic-time_ini
    time_proc['gen_slic_df'] = time_df-time_slic
    time_proc['slic_df'] = time_df-time_ini
    if len(img_sel):
        return props_df, props_dic_SP, segments_slic_sel, time_proc
    else:
        return props_df, img_sel, segments_slic_sel, time_proc

def optimal_number_of_clusters(wcss,min_cl, max_cl):
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
    
    return distances.index(np.max(distances)) + min_cl

#%%
read_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/Cassio/S2-16D_V2_012014_20220728_/'
name_img = 'S2-16D_V2_012014_20220728_'

current_directory = os.getcwd()
#for new image 23/02/2024
upper_level_directory = os.path.join(current_directory, '../data/test_segm_results/S2-16D_V2_012014_20220728_/')

# Specify the filename and path within the upper-level directory
#filename = 'SENTINEL-2_MSI_20LMR_RGB_2022-07-16'
#for new image
save_path = os.path.join(upper_level_directory, name_img)

all_bands = ['B04', 'B03', 'B02', 'B08', 'EVI', 'NDVI']
bands_sel = ['B04', 'B03', 'B02'] # R,G,B bands selection for slic segmentation


#%%
# generate an array with tiff files name to be loaded
tif_names=[]
tif_names=gen_files_to_read(name_img, all_bands, read_dir)
#%%
time_ini= time.time()
image_band_dic = {}
image_band_dic = load_image_files2(tif_names, pos=-1)
time_band = time.time()
#%%
time_proc ={}
time_proc['load_band_files'] = time_band - time_ini

time_ini = time.time()
img_sel = np.dstack((image_band_dic[bands_sel[0]], image_band_dic[bands_sel[1]], image_band_dic[bands_sel[2]]))
time_fim = time.time()
time_proc['gen_img_rgb'] = time_fim - time_ini

# gen mask
mask=None
contains_nan={}

# Check if the matrix contains the element -9999
time_ini = time.time()
for i,b  in enumerate(bands_sel):
# Check if the matrix contains the element -9999
    contains_nan[i] = np.any(image_band_dic[bands_sel[i]] == -9999)
    if contains_nan[i]:
        #print ("true")
        if i==0:
            mask = (image_band_dic[bands_sel[i]] != -9999) 
        else:
            mask = (image_band_dic[bands_sel[i]] != -9999) & mask
time_mask= time.time()

del contains_nan
time_proc['gen_mask']=time_mask-time_ini

#%%
#getting the image with values normalized
time_ini=time.time()
image_band_dic_norm={}
for k in image_band_dic.keys():
    image_band_dic_norm[k]=image_band_dic[k].astype(float)/np.max(image_band_dic[k])
time_norm=time.time()

img_sel_norm = np.dstack((image_band_dic_norm[bands_sel[0]], image_band_dic_norm[bands_sel[1]], image_band_dic_norm[bands_sel[2]]))
time_norm=time.time()
time_proc['gen_img_norm'] = time_norm-time_ini


#%%
time_ini = time.time()
img_sel_norm = load_image_norm()
time_load_img_norm = time.time()
time_proc['load_img_norm'] = time_load_img_norm - time_ini 
for k,v in time_proc.items():
    print (f'{k}: {v}')

#%%
#for new_image
segms = [110000]
compactness = [1, 2]
sigmas = [0.1, 0.5]
connectivity = [False]
# making all possible parameters combinations 
parameter_combinations = list(product(segms, compactness, sigmas, connectivity))
#%%
params_test_dic = {}
for id,comb in enumerate(parameter_combinations, start=1):
    if id not in params_test_dic:
        params_test_dic[id] = {}
    params_test_dic[id]['segms'] = comb[0]
    params_test_dic[id]['compactness'] = comb[1]
    params_test_dic[id]['sigma'] = comb[2]
    params_test_dic[id]['connectivity'] = comb[3]

#%%    
#running tests for all combinations
ids = list(params_test_dic.keys())
stats_df_dic={}
props_df_sel = {}
props_dic_SP={} 
segments_slic_sel ={}
#%%    
ids=[3]
time_slic={}
for id in tqdm(ids):
    n_segms = params_test_dic[id]['segms'] #600
    sigma = params_test_dic[id]['sigma']  #0.1
    compact = params_test_dic[id]['compactness']
    conectivity= params_test_dic[id]['connectivity'] #False
    print ("test and params id: ",id, n_segms, sigma, compact, conectivity)
    time_start = time.time()
    props_df_sel[id], props_dic_SP, \
        segments_slic_sel, time_slic = img_slic_segment_gen_df(bands_sel, \
                                                image_band_dic, img_sel=img_sel_norm,\
                                                n_segms=n_segms, sigma=sigma, \
                                                compactness = compact, mask=mask,\
                                                conectivity=conectivity)
    time_end = time.time()
    print(f'SLIC RGB number of segments : {props_df_sel[id].shape[0]} Time elapsed: {round(time_end - time_start, 2)}s')
#%%
time_proc.update(time_slic)
for k,v in time_proc.items():
    print (f'{k}: {v}')

# %%
time_proc_df = pd.DataFrame({'Processo' : time_proc.keys() , 'time(s)' : time_proc.values() })

#%%
### Para cluster
from sklearn_extra.cluster import CLARA
import dask.array as da
import dask.dataframe as dd 
from math import ceil

id_test=3
bands_to_cluster = ['B08','EVI','NDVI'] #new image
#arraybands_sel = props_df_sel[['NBR','EVI','NDVI']].to_numpy()
time_ini_t = time.time()
arraybands_sel = props_df_sel[id_test][bands_to_cluster].to_numpy()
arraybands_list_sel = arraybands_sel.tolist()

n_clusters=30 
dic_cluster={}
sse=[]
time_ini_clara=time.time()
for n in range (2, n_clusters+1):
    #clara = timedcall(CLARA(n_clusters=n, random_state=0).fit(arraybands_sel))
    time_ini = time.time()
    clara = CLARA(n_clusters=n,n_sampling=40+2*n,n_sampling_iter=5, random_state=0).fit(arraybands_sel)
    clusters_sel = clara.predict(arraybands_sel)
    time_fim = time.time()
    print (f'tempo de execucao para {n}: {time_fim-time_ini}')
    #15/02/2024: nao me lembro pq preciso fazer o get_lebels aqui
    # labels_sel = get_labels(clusters_sel.tolist(), len(arraybands_sel))
    # dic_cluster[n] = labels_sel
    dic_cluster[n] = clusters_sel.tolist()
    sse.append(clara.inertia_)
    #adiciona a info do cluster no df
    #props_df_sel[id_test]['cluster_'+str(n)]= labels_sel[props_df_sel[id_test].index]

    props_df_sel[id_test]['cluster_'+str(n)]=clusters_sel

#calculo do num otimo de cluster pelo metodo do cotovelo
n_opt = optimal_number_of_clusters(sse, 2, n_clusters)
n_opt
time_fim_t= time.time()
time_proc['time_clara'] = time_fim - time_ini_clara
time_proc['cluster total'] = time_fim_t - time_ini_t
#%%
#calculo matrix similaridade
n_ini=2
n_selected = list(range(n_ini,ceil(n_opt*1.2)+1))
#n_selected = [2,3,4]
time_ini = time.time()
for i, n in enumerate(n_selected):
    da_arr = da.from_array(np.array(dic_cluster[n]),chunks=1000)
    if (i==0):
        matrix_sim = (da_arr[:, None] == da_arr[None, :]).astype(int)
        continue
    matrix_sim = (da_arr[:, None] == da_arr[None, :]).astype(int)+matrix_sim
matrix_sim=matrix_sim/len(n_selected)    
time_fim = time.time()
time_proc['matrix_sim'] = time_fim-time_ini
#%%
# depois colocar para encontrar o SP

#calculo para fazer um filtro a partir de um label de SP
time_ini=time.time()
centroid_sel=90926    
centroid_row_matrix_sim = matrix_sim[centroid_sel-1,:]

centroid_sel_df = props_df_sel[id_test][['label', 'centroid-0','centroid-1']]
centroid_sel_df['idx'] = centroid_sel_df.index.values

centroid_sel_df_idx= da.from_array(props_df_sel[id_test].index.values, chunks=1000)

centroid_row_matrix_sim_df = dd.concat([dd.from_dask_array(c) for c in [centroid_sel_df_idx,centroid_row_matrix_sim ]], axis=1)
centroid_row_matrix_sim_df.columns = ['idx','sim_value']

dask_centroid_sel_df = dd.from_pandas(centroid_sel_df,npartitions=109)

dask_centroid_sel_df = dask_centroid_sel_df.merge(centroid_row_matrix_sim_df, on='idx')

#dask_centroid_sel_df['sim_value'] = centroid_row_matrix_sim
#dask_centroid_sel_df= dask_centroid_sel_df.assign(sim_value = centroid_row_matrix_sim.compute())

thr_min=0.90
thr_max=1

condition_1 = dask_centroid_sel_df['sim_value']>=thr_min
condition_2 = dask_centroid_sel_df['sim_value']<=thr_max
filter_centroid_sel_df = dask_centroid_sel_df.loc[condition_1 & condition_2]

time_filter=time.time()
time_proc['filter_label_SP'] = time_filter-time_ini
#%%
#timepara plotar

#%%
time_ini = time.time()    
plot_centroids=True
if (plot_centroids):
    # x_centroids=[x for x in filter_centroid_sel_df['centroid-1']]
    # y_centroids=[y for y in filter_centroid_sel_df['centroid-0']]
    x_centroids = filter_centroid_sel_df['centroid-1'].compute().values
    y_centroids = filter_centroid_sel_df['centroid-0'].compute().values
    #print (len(x_centroids), len(y_centroids))
    #plt.scatter(x_centroids, y_centroids,s=1, color=list(filter_centroid_sel_df['cor']))
    plt.scatter(x_centroids, y_centroids,s=1, color='blue')
time_fim = time.time()

#print (time_fim-time_ini)
    #to show image with segmentation
    #plt.imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
plt.imshow(img_sel_norm)
plt.axis('off')
plt.tight_layout()
plt.show()
time_fim2=time.time()
time_proc['plot_centroids'] = time_fim-time_ini
time_proc['plot_centroids_and_image'] = time_fim2 - time_ini

#%%
#### tests with sparky
# img_slic_segment_gen_df(bands_sel, image_band_dic, img_sel='', n_segms=600, sigma=2, \
#                              compactness = 5, mask=None, conectivity=False):

# img_slic_segment_gen_df(bands_sel, \
#                         image_band_dic, img_sel=img_sel_norm,\
#                         n_segms=n_segms, sigma=sigma, \
#                         compactness = compact, mask=mask,\
#                         conectivity=conectivity)
segments_slic_sel2 = slic(img_sel_norm, n_segments=n_segms, compactness=compact, sigma=sigma, \
                             start_label=1,mask=mask, enforce_connectivity=conectivity)
    #print(f'SLIC RGB number of segments : {len(np.unique(segments_slic_sel))}')
    
    # 2. 
    # props_dic = regionprops_table(segments_slic_sel, img_sel, \
    #                             properties=['label','coords', 'centroid','local_centroid'])
   
#26/02/2024: separando 
#%%    
props_dic = regionprops_table(segments_slic_sel, img_sel, \
                                properties=['label','centroid','local_centroid']) 
    
props_dic_SP = regionprops_table(segments_slic_sel, img_sel, \
                                properties=['label','coords'])   


# 3.
time_1 = time.time()
props_df = pd.DataFrame(props_dic)
#%%
from pyspark.sql import SparkSession
os.environ["PYARROW_IGNORE_TIMEZONE"] ='1'
os.environ['SPARK_LOCAL_IP']='127.0.0.1'
spark = SparkSession \
        .builder \
        .appName("DataFrameExample") \
        .getOrCreate()
#%%

id_test=3
time_2 = time.time()
#props_psdf = ps.DataFrame(props_dic)
props_psdf = ps.from_pandas(props_df_sel[id_test])
time_3 = time.time()
print (f'time pd = {time_2-time_1}, time psdf = {time_3-time_2}')
#props_df['num_pixels'] = props_df['coords'].apply(len)
#%%    
# 4. adiciona os valores das bandas do centroids (x,y) como colunas no df, 
#    for each sp (superpixel), adds as column: number of pixels, calculates std and average 
#    of each band and the value of pixels for each band 
for b in image_band_dic.keys():
    props_df[b] = props_df.apply(lambda row: image_band_dic[b][round(row['centroid-0']), round(row['centroid-1'])], axis=1)
    #props_df['desvio_'+b]= props_df['coords'].apply(lambda arr: [image_band_dic[b][elem[0],elem[1]] for elem in arr])
    #pixel_band_value = image_band_dic[b]
    #props_df[['std_'+b, 'mean_'+b, 'seg_'+b ]] = props_df['coords'].apply(calc_std_array, b=b).apply(pd.Series)
time_df = time.time()