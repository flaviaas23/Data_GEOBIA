''''
read file with centroids of SP
read each tif band files for each time and 
gen df :
pixel| label| b1t1| b2t1|...|b6t1|b1t2|...|b6t12|
'''
#%%
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
import psutil
import time
import sys
import os
import glob

from itertools import product

import pickle
import random
from tqdm import tqdm

from sklearn_extra.cluster import CLARA
#%%
from sklearn.metrics import silhouette_samples, silhouette_score
#%%
from tslearn.clustering import TimeSeriesKMeans
import dask.array as da
import dask.dataframe as dd 
import multiprocessing
nproc = multiprocessing.cpu_count()-2

random.seed(999) # para gerar sempre com a mesma seed

def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def gen_files_to_read(name_img, bands, read_dir):
    ''''
    gen array with tif files name to read
    '''
    files_to_read = []
    for b in bands:
        files_to_read.append(read_dir+name_img+b+'.tif')
    
    return files_to_read
#%%
def list_files_to_read(read_dir, padrao):
    ''''
    get a list of files in a eval_dir based on a padrao
    '''
    if not read_dir:
        read_dir='/Users/flaviaschneider/Documents/flavia/data_GEOBIA/data/Cassio/'


    #14/09/2023 padrao_ = os.path.join(eval_dir, '*_'+padrao+'.pkl') #euclidean_Individual.pkl')
    padrao_ = os.path.join(read_dir, '*'+padrao+'*') #euclidean_Individual.pkl')

    print ("list_files_to_read: padrao_: ", padrao_)

    read_files = glob.glob(padrao_)
    print ("list_files_to_read: eval_files", read_files)

    return read_files

def load_image_files2(files_name,pos=-2):
    ''''
    load tiff image bands files of a timestamp Cassios iamges
    this is for 
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

def load_image_files3(files_name,pos=-2):
    ''''
    load tiff image bands files of a timestamp 
    this is for exemple image 
    '''
    #files_name=[file_nbr,file_evi, file_ndvi, file_red,file_green,file_blue]
    image_band_dic={}
    for f in files_name:
        f_name = f.split('_')  
        #print (f_name)
        band = f_name[pos]
        #print (band)
        image_band_dic[band] = imageio.imread(f)

    return image_band_dic

def save_img_png(file_red, file_green, file_blue):
    ''''
    save tif rgb bands to a png file
    '''
    from PIL import Image

    # Open each band file
    r_band = Image.open(file_red)
    g_band = Image.open(file_green)
    b_band = Image.open(file_blue)

    # Convert PIL Images to Numpy arrays
    npRed   = np.array(r_band)
    npGreen = np.array(g_band)
    npBlue  = np.array(b_band)

    npRed[npRed < 0]     = 0
    npBlue[npBlue < 0]   = 0
    npGreen[npGreen < 0] = 0

    max = np.max([npRed,npGreen,npBlue])

    # Scale all channels equally to range 0..255 to fit in a PNG (could use 65,535 and np.uint16 instead)
    R = (npRed * 255/max).astype(np.uint8)
    G = (npGreen * 255/max).astype(np.uint8)
    B = (npBlue * 255/max).astype(np.uint8)

    # Build a PNG
    RGB = np.dstack((R,G,B))

    #Image.fromarray(RGB).save('result.png')
    return RGB
#%%
def get_bandsDates(band_im_files):
    '''
    #From image tif files names return dates and bands 
    '''
    
    pos=-2 #position of the band in name file
    pos_date=-1
    dates=[]
    bands = []
    time1=time.time()
    for f in band_img_files:
        f_name = f.split('_')  
        if f_name[pos] not in bands:
            bands.append(f_name[pos])
        f_name = f_name[pos_date].split('.')
        if f_name[pos] not in dates:
            dates.append(f_name[pos])

    dates = sorted(dates)
    time2=time.time()
    print (time2-time1)
    return bands, dates
#%%
def get_dates_lower(dates, lower=True):
    ''''
    return the earliest date of each month
    '''
    dates_lower = []
    month_dates={}
    for date in dates:
        d = date.split('-')
        month = d[1] 
        day = d[-1]
        if month not in month_dates:
            month_dates[month] = date
        else:
            if lower and date < month_dates[month]:
                month_dates[month] = date
            elif not lower and date > month_dates[month]:
                month_dates[month] = date
    dates = month_dates.values()
    return dates
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
    segments_slic_sel = slic(img_sel, n_segments=n_segms, compactness=compactness, sigma=sigma, \
                             start_label=1,mask=mask, enforce_connectivity=conectivity)
    #print(f'SLIC RGB number of segments : {len(np.unique(segments_slic_sel))}')
    
    # 2. 
    # props_dic = regionprops_table(segments_slic_sel, img_sel, \
    #                             properties=['label','coords', 'centroid','local_centroid'])
   
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

    if len(img_sel):
        return props_df, props_dic_SP, segments_slic_sel
    else:
        return props_df, img_sel, segments_slic_sel

#%%
def gen_mask(image_band_dic, bands_sel):
    ''''
    verifies if there is nan(-9999) in the bands, if there is, generate a mask
    if not set mask to none
    '''
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
    print (time_mask-time_ini)
    return mask

#%%
def gen_param_test_dic(segms = [11000], compactness = [1, 2],\
                       sigmas = [0.1, 0.5],connectivity = [False] ):
    ''''
    generate params for slic sgmentation based on input values 
    '''
    
    # making all possible parameters combinations 
    parameter_combinations = list(product(segms, compactness, sigmas, connectivity))

    params_test_dic = {}
    for id,comb in enumerate(parameter_combinations, start=1):
        if id not in params_test_dic:
            params_test_dic[id] = {}
        params_test_dic[id]['segms'] = comb[0]
        params_test_dic[id]['compactness'] = comb[1]
        params_test_dic[id]['sigma'] = comb[2]
        params_test_dic[id]['connectivity'] = comb[3]

    return params_test_dic
#%%
#save props_df
def save_result_slic(props_df_sel, props_dic_SP, segments_slic_sel, dir_filename):
    ''''
    save the results from slic
    '''
    obj_dic = {}
    obj_dic[id] = {
        "props_df_sel": props_df_sel#, 
        #"segments_slic_sel": segments_slic_sel,
        }
    #save_path+name_img+'_'+dates[-1]
    file_to_save = dir_filename+'_props_df_sel_'+str(id_test)+'.pkl'
    save_to_pickle(obj_dic, file_to_save)
    #save segments
    obj_dic = {}
    obj_dic[id] = {
        #"props_df_sel": props_df_sel#, 
        "segments_slic_sel": segments_slic_sel,
        }
    file_to_save = dir_filename+'_segments_'+str(id_test)+'.pkl'
    save_to_pickle(obj_dic, file_to_save)
    #save label(SP) and coords
    obj_dic = {}
    obj_dic[id] = {
        "props_dic_sel_labels_coords": props_dic_SP#, 
        #"segments_slic_sel": segments_slic_sel,
        }
    file_to_save = dir_filename+'_props_dic_sel_labels_coords_'+str(id_test)+'.pkl'
    save_to_pickle(obj_dic, file_to_save)

#%%
def read_slic_sel(id, open_path, obj_to_read='props_df_sel',output=True):
    ''''
    read props_df_sel and returns them as a dicionary
    '''
    print (open_path)
    dic_df={}
    #for id in tqdm(ids):      #ids #ids_file
    if (obj_to_read == 'props_df_sel'):
        #file_to_open = open_path + '_'+str(id)+'.pkl'
        file_to_open = open_path+obj_to_read+str(id)+'.pkl'
        with open(file_to_open, 'rb') as handle: 
            b = pickle.load(handle)
        dic_df[id] = b[id][obj_to_read][id]
    elif obj_to_read == "cluster":
        obj_to_read = "props_df_sel_" + obj_to_read+str(id)+'.pkl'
        print (obj_to_read)
        dic_df[id] = b[id][obj_to_read][id]
    elif obj_to_read == "segments_slic_sel":
        file_to_open = open_path + 'segments_'+str(id)+'.pkl'
        with open(file_to_open, 'rb') as handle: 
            b = pickle.load(handle)
        dic_df[id] = b[id][obj_to_read]
    elif obj_to_read == "props_dic_sel_labels_coords":
        file_to_open = open_path + obj_to_read+str(id)+'.pkl'
        with open(file_to_open, 'rb') as handle: 
            b = pickle.load(handle)
        dic_df[id] = b[id][obj_to_read]
    elif obj_to_read == "props_df_sel_ts":
        file_to_open = open_path + '_'+obj_to_read+'_'+ str(id)+'.pkl'
        with open(file_to_open, 'rb') as handle: 
            b = pickle.load(handle)
        dic_df[id] = b[id][obj_to_read]
        print (file_to_open) if output else None
             
    return dic_df   
#%%
###### defs for clustering ######
def gen_cluster_ts(n_grupos, X_train, algoritmo, metrica_distancia, seed=997):
    ''' old cria_grupos
    #n_grupos: quantidade de clusters que deve ser identificado
    # X_train: dataset que será classificado em formato de array
    # algoritmo: que será usado para classificar
    # metrica_distancia: que será usada com o algoritmo de classificacao
    # seed: default = 997
    '''
    if algoritmo == "Kmeans":
        if metrica_distancia == "dtw":
            dba_km = TimeSeriesKMeans(n_clusters=n_grupos,
                                      n_init=2,
                                      metric=metrica_distancia,
                                      verbose=True,
                                      dtw_inertia=True,
                                      max_iter_barycenter=10,
                                      random_state=seed,
                                      n_jobs=nproc)

        elif metrica_distancia == "euclidean":
            dba_km = TimeSeriesKMeans(n_clusters=n_grupos,
                                      n_init=2,
                                      metric="euclidean",
                                      verbose=True,
                                      dtw_inertia=True,
                                      max_iter=100,
                                      random_state=seed,
                                      n_jobs=nproc)
        
        dba_km_fit = dba_km.fit(X_train) 
        clusters_centers = dba_km_fit.cluster_centers_
        cluster_labels = dba_km_fit.labels_
        inertia = dba_km_fit.inertia_
        #cluster_labels = dba_km.fit_predict(X_train)

    #print (f'gen cluster lenclusters labels {len(cluster_labels)} labels{cluster_labels}')
    return cluster_labels, clusters_centers, inertia
#%%
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
#get list of files nome_banda_data.tif
#read_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/Cassio/S2-16D_V2_012014_20220728_/'
file_img_dir = '/Library/Frameworks/R.framework/Versions/4.3-x86_64/Resources/library/sitsdata/extdata/Rondonia-20LMR/'
name_img = 'SENTINEL-2_MSI_20LMR'#'S2-16D_V2_012014_20220728_'

band_img_files = list_files_to_read(file_img_dir, name_img)
#%%
bands, dates= get_bandsDates(band_img_files)
#%%
dates_lower = get_dates_lower(dates)
dates_high = get_dates_lower(dates, lower=False)
#%%
#get the most recent tif img band files to slic
day=dates[-1]
#para pegar o tif da data do exemplo '2022-07-16'
day=dates[12]
band_img_file_to_slic=[x for x in band_img_files if day in x]

#%%
current_directory = os.getcwd()
#for new image 23/02/2024
#upper_level_directory = os.path.join(current_directory, '../data/test_segm_results/S2-16D_V2_012014_20220728_/')
#upper_level_directory = os.path.join(current_directory, '../data/test_segm_results/'+name_img+'/')
#para fazer os testes com a imagem do exemplo SENTINEL-2_MSI_20LMR_2022-07-16 :
upper_level_directory = os.path.join(current_directory, 'data/test_segm_results/'+name_img+'/SENTINEL-2_MSI_20LMR_2022-07-16/')
# Specify the filename and path within the upper-level directory
save_path = upper_level_directory #os.path.join(upper_level_directory, name_img)
#%%
#for new image

SAVE_PNG = True
if SAVE_PNG:
    save_path_png = os.path.join(current_directory, './data/img_png/')
    filename = name_img+'_RGB_'+day+'.png'
    RGB = save_img_png(band_img_file_to_slic[1], band_img_file_to_slic[3], band_img_file_to_slic[0])  
    from PIL import Image
    Image.fromarray(RGB).save(save_path_png+filename) 
    del RGB
#%% 
#to do the slic
#load tif bands files
image_band_dic = {}
image_band_dic = load_image_files3(band_img_file_to_slic, pos=-2)
#%%
bands_sel = ['B11', 'B8A', 'B02'] #RGB
img_sel = np.dstack((image_band_dic[bands_sel[0]], image_band_dic[bands_sel[1]], image_band_dic[bands_sel[2]]))
mask = gen_mask(image_band_dic, bands_sel)

#%%
#getting the image with values normalized
time_ini=time.time()
image_band_dic_norm={}
for k in image_band_dic.keys():
    image_band_dic_norm[k]=image_band_dic[k].astype(float)/np.max(image_band_dic[k])
time_norm=time.time()
print (time_norm-time_ini)
img_sel_norm = np.dstack((image_band_dic_norm[bands_sel[0]], image_band_dic_norm[bands_sel[1]], image_band_dic_norm[bands_sel[2]]))
time_norm=time.time()
print (time_norm-time_ini)

#%%
#generate the params for slic
params_test_dic = {}
params_test_dic = gen_param_test_dic()

#%%
stats_df_dic={}
props_df_sel = {}
props_dic_SP={} 
segments_slic_sel ={}
#%%
id_test = 3
ids=[id_test]
time_slic={}
for id in tqdm(ids):
    n_segms = params_test_dic[id]['segms'] #11000
    sigma = params_test_dic[id]['sigma']  #0.1
    compact = params_test_dic[id]['compactness'] #2
    conectivity= params_test_dic[id]['connectivity'] #False
    print ("test and params id: ",id, n_segms, sigma, compact, conectivity)
    time_start = time.time()
    props_df_sel[id], props_dic_SP, \
        segments_slic_sel = img_slic_segment_gen_df(bands_sel, \
                                                image_band_dic_norm, img_sel=img_sel_norm,\
                                                n_segms=n_segms, sigma=sigma, \
                                                compactness = compact, mask=mask,\
                                                conectivity=conectivity)
    time_end = time.time()
    print(f'SLIC RGB number of segments : {props_df_sel[id].shape[0]} Time elapsed: {round(time_end - time_start, 2)}s')

#%%
save_result_slic(props_df_sel, props_dic_SP, segments_slic_sel, save_path+name_img+'_'+day)
#%%
#fazer funcao para fazer o bloco abaixo
#gerar as series temporais 
#gerar df com as series temporais formato:
# label|centroid-0|centroid-1|b1_t1|b2_t1|...|b1t2|...|bn_tn
time1=time.time()
#props_df_sel_ts={}
props_df_sel_ts = props_df_sel[id_test][['label','centroid-0', 'centroid-1']]
for t in dates:
    band_img_file_to_slic=[x for x in band_img_files if t in x]
    image_band_dic = {}
    image_band_dic = load_image_files3(band_img_file_to_slic, pos=-2)
    image_band_dic_norm={}
    
    for b in image_band_dic.keys():
        image_band_dic_norm[b]=image_band_dic[b].astype(float)/np.max(image_band_dic[b])
        
        #gerar o image_band_dic
    #for b in image_band_dic.keys():
        props_df_sel_ts[b+'_'+t] = props_df_sel_ts.apply(lambda row: image_band_dic_norm[b][round(row['centroid-0']), round(row['centroid-1'])], axis=1)
        
time2=time.time()
#%%
obj_dic = {}
obj_dic[id] = {
    "props_df_sel_ts": props_df_sel_ts#, 
    #"segments_slic_sel": segments_slic_sel,
    }
#save_path+name_img+'_'+dates[-1]
file_to_save = save_path+name_img+'_props_df_sel_ts_'+str(id_test)+'.pkl'
save_to_pickle(obj_dic, file_to_save)

time3=time.time()

print (time3-time2)
#%%
# read sp file
props_df_sel_ts2 = read_slic_sel(id_test, save_path+name_img, obj_to_read='props_df_sel_ts',output=True)

#%%
#gerargrafico para avaliacao das series temporais
# selecionar randomicamente os SP que serao plotados
# Fazer um grafico para cada banda para os SPs selecionados
num_SPs=10
df_sp_selected = props_df_sel_ts.sample(num_SPs, random_state=999)

#%%
# Selecione colunas cujos nomes contenham a substring
bands_cluster=[x for x in bands if x not in bands_sel]
#%%
bands_df={}
for b in bands_cluster:
    #b='B02'
    cols_sel = [b+'_'+d for d in dates_lower]
    
    # Crie um novo DataFrame apenas com as colunas selecionadas
    bands_df[b] = df_sp_selected[['label']+cols_sel]
    bands_df[b].set_index('label', inplace=True)
    bands_df[b].columns=dates_high
    bands_df[b] = bands_df[b].T
    bands_df[b].plot()
#%%
x_=dates_high
#%%
import matplotlib.pyplot as plt
#y_centroids=[y for y in filter_centroid_sel_df['centroid-0']]
#plt.scatter(dates_high, ,s=1, color=list(filter_centroid_sel_df['cor']))


# %%
################
#           Cluster
###############
cols_sel=[]
for b in bands_cluster:
    
    for d in dates_lower:
        cols_sel.append(b+'_'+d)

#%%
arraybands_sel = props_df_sel_ts[cols_sel].to_numpy()#.astype('float32')# nao mudou uso de memoria
#arraybands_list_sel = arraybands_sel.tolist()
#%%
#df_cluster_array = props_df_sel_ts.iloc[:,start_cols_dias:num_dias].to_numpy().astype('float32')
#%%
clusters={}
clusters_center={}
# silhouette_avg={}
# sample_silhouette_values={}
sse=[]

n_clusters=20
dist_met = 'dtw' #'euclidean' 'dtw'
seed = 999
time1 =time.time()
for n in tqdm(range(2, n_clusters+1)):
    #print ("n= ", n)
    #cluster_labels[n_cluster],silhouette_avg[n_clusters],sample_silhouette_values[n_clusters]=cria_grupos(n_clusters, df_cluster_array_1k, "Kmeans", "dtw", seed=997)
    #time1 = time.time()
    clusters[n], clusters_center[n], inertia = gen_cluster_ts(n, arraybands_sel, \
                                                "Kmeans", dist_met,\
                                                seed=seed)
    sse.append(inertia)
    '''
    #time2=time.time()
    #print (f"time cluster {dist_met}: {time2-time1}")
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    #silhouette_avg[n] = silhouette_score(arraybands_sel, clusters[n])
    #print("For n_clusters =", n_clusters,
    #       "The average silhouette_score is :",silhouette_avg,
    #    )

    # Compute the silhouette scores for each sample
    #sample_silhouette_values[n] = silhouette_samples(arraybands_sel, clusters[n])
    #time3 = time.time()
    #print (f"time silhouette: {dist_met}: {time3-time2}")
    #print ("silhouette_avg",silhouette_avg)

    # obj_cluster[n_clusters] = {
    #     # "data": df_cluster_sample,# amostra
    #     "data_file_name": file_to_read, # PROCESSED_DIR+data_file+'.pkl',
    #     "seed": seed,
    #     "distance_metric": distance_metric,
    #     "cluster": clusters,     # [] resultado do cluster na amostra
    #     "clusters_centers": clusters_center,
    #     "dias_sample": num_dias,     #dias usados do sample
    #     "silhouette_avg":silhouette_avg, # silhouette_avg
    #     "sample_silhouette_values":sample_silhouette_values,        #[] resultado do silhoute para cada ponto do cluster
    # }
    #'''
time2=time.time()
print (f'{time2-time1}')
# euclideana: 5431.604136943817 s =90.52673561573029 min = 1.508778926928838 h
# dtw: 11562.22420501709 s  = 192 min = 3,212 h 
#%%
#Qdo for salvar o resultado do dtw tirar as colunas 
#de cluster do euclideana q estao no props_df_sel_ts
#drop cols with cluster 
props_df_sel_ts_dtw=props_df_sel_ts.drop(props_df_sel_ts.filter(like='cluster').columns, axis=1)
#%%
for n in tqdm(range(2, n_clusters+1)):
    props_df_sel_ts_dtw['cluster_'+str(n)]=clusters[n]
# %%
n_opt = optimal_number_of_clusters(sse, 2, n_clusters)
n_opt
#%%
obj_dic={}
obj_dic[id_test] = {
    "props_df_sel_cluster_ts": props_df_sel_ts,
    "dic_labels_cluster": clusters,
    "clusters_center": clusters_center,
    "sse": sse,
    "n_opt": n_opt
}
file_to_save = save_path+name_img+'_'+dates[-1]+'_props_df_sel_cluster_ts_'+dist_met+'_'+str(id_test)+'.pkl'
save_to_pickle(obj_dic, file_to_save)
# %%
#fazer a matriz de similaridade 
#e olhar no streamlit
#calculo matrix similaridade
import dask.array as da
import dask.dataframe as dd 

def gen_matrix_sim_da(n_opt,dic_cluster, chunks=1000):
    ''''
    gen matriz similarity dask
    '''
    n_ini=2
    n_selected = list(range(n_ini,ceil(n_opt*1.2)+1))
    #n_selected = [2,3,4]
    time_ini = time.time()
    for i, n in enumerate(n_selected):
        da_arr = da.from_array(np.array(dic_cluster[n]),chunks=chunks)
        if (i==0):
            matrix_sim = (da_arr[:, None] == da_arr[None, :]).astype(int)
            continue
        matrix_sim = (da_arr[:, None] == da_arr[None, :]).astype(int)+matrix_sim
    matrix_sim=matrix_sim/len(n_selected)    
    time_fim = time.time()
    print (time_fim-time_ini)
    
    return matrix_sim
# %%
matrix_sim_dtw = gen_matrix_sim_da(n_opt, clusters)

# %%
# to read the dic cluster and n_opt
id_test=3
dist_met = 'euclidean'
file_to_save = save_path+name_img+ '_props_df_sel_cluster_ts_'+dist_met+'_'+str(id_test)+'.pkl'
file_to_open = file_to_save
with open(file_to_open, 'rb') as handle: 
    b = pickle.load(handle)

dic_cluster={}
dic_cluster = b[id_test]['dic_labels_cluster']

n_opt=b[id_test]['n_opt']

# %%
matrix_sim_euc = gen_matrix_sim_da(n_opt, clusters)

# %%
