''''
programa para plotar os centroides na imagem a partir do label 
de SP informado
'''
#%%
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.colorbar import Colorbar

import numpy as np
from math import ceil
import dask.array as da
import dask.dataframe as dd 

#from skimage.data import astronaut,lily, immunohistochemistry
from skimage.color import rgb2gray
#from skimage.filters import sobel
#from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

import imageio.v2 as imageio
from skimage.measure import regionprops, regionprops_table
import pandas as pd
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

def load_image_norm():
    read_path = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/test_segm_results/S2-16D_V2_012014_20220728_/S2-16D_V2_012014_20220728_'
    file_to_open = read_path + 'img_sel_norm.pkl'
    with open(file_to_open, 'rb') as handle:    
        img_sel_norm = pickle.load(handle)
    return img_sel_norm

def is_multiple(num, multiple):
    return num % multiple == 0

def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(SAVE_DIR+pickle_cluster_file+'.pkl', 'wb') as handle:
    # pickle.dump(obj_cluster, handle, protocol=pickle.HIGHEST_PROTOCOL)

def concat_dfs(dic_df):
    '''
    Concatenates a list of dataframes
    '''
    result_df = pd.DataFrame()
    for i in dic_df.keys():
        if i==0:
            result_df = dic_df[i]
            print (i)
            continue
        result_df = pd.concat([result_df,dic_df[i]], axis=0)

    return result_df

def gen_files_to_read(name_img, bands, read_dir):
    ''''
    gen array with tif files name to read
    '''
    files_to_read = []
    for b in bands:
        files_to_read.append(read_dir+name_img+b)
    
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
        band = f_name[-2].split("_") # para obter onome da banda do arquivo
        #print (band[pos], f_name[-2])
        bands_name.append(band[pos])
        image_band_dic[band[pos]] = imageio.imread(f)

    return image_band_dic

def read_props_df_sel(ids, open_path, obj_to_read='props_df_sel',output=True):
    ''''
    read a list of props_df_sel and returns them as a dicionary
    '''
    print (open_path)
    dic_df={}
    for id in tqdm(ids):      #ids #ids_file
        if (obj_to_read == 'props_df_sel') or (obj_to_read == "cluster"):
            file_to_open = open_path + '_'+str(id)+'.pkl'
            with open(file_to_open, 'rb') as handle: 
                b = pickle.load(handle)
            if obj_to_read == "cluster":
                obj_to_read = "props_df_sel_" + obj_to_read
                print (obj_to_read)
            dic_df[id] = b[id][obj_to_read][id]

        elif obj_to_read == "segments_slic_sel":
            file_to_open = open_path + 'segments_'+str(id)+'.pkl'
            with open(file_to_open, 'rb') as handle: 
                b = pickle.load(handle)
            dic_df[id] = b[id][obj_to_read]
        print (file_to_open) if output else None
        
        
    return dic_df     
def get_df_filter(centroid_sel, props_df_sel, matrix_sim, id_test, thr_min=0.85, thr_max=1):
    #centroid_sel_df = props_df_sel[id_test][['label', 'centroid-0','centroid-1', 'coords']]
    #01/03/2024 removendo o coords daqui para imagens grandes
    #centroid_sel_df = props_df_sel[id_test][['label','std_NBR','std_EVI','std_NDVI' , 'num_pixels','centroid-0','centroid-1', 'coords']]

    centroid_sel_df = props_df_sel[id_test][['label', 'centroid-0','centroid-1']]
    
    centroid_row_matrix_sim = matrix_sim[centroid_sel-1,:]
    
    centroid_sel_df['sim_value'] = centroid_row_matrix_sim

    #filter just the sim values higher than threshold
    #threshold = 0.70
    filter_centroid_sel_df = centroid_sel_df[(centroid_sel_df['sim_value']>=thr_min) &
                                             (centroid_sel_df['sim_value']<=thr_max)]
    filter_centroid_sel_df['cor'] = filter_centroid_sel_df['sim_value'].apply(calc_cor, 'Blues')
    filter_centroid_sel_df.loc[centroid_sel-1,'cor']='red'

    return filter_centroid_sel_df

def calc_cor(valor, c_map='Blues'):
    colormap=plt.get_cmap(c_map)
    #colormap=plt.get_cmap('Blues')
    return colormap(valor) 

def plot_cluster_img_pixel_sel_faster(filter_centroid_sel_df, centroid_sel, \
                                      cl_map='Blues', plot_centroids=False):
    ''''
    plot similares clusters of pixel selected in image
    '''
    filter_centroid_sel_df['cor'] = filter_centroid_sel_df['sim_value'].apply(calc_cor, 'Blues')
    filter_centroid_sel_df.loc[centroid_sel-1,'cor']='red'
    
    time_ini = time.time()    
    
    if (plot_centroids):
        x_centroids=[x for x in filter_centroid_sel_df['centroid-1']]
        y_centroids=[y for y in filter_centroid_sel_df['centroid-0']]
        plt.scatter(x_centroids, y_centroids,s=1, color=list(filter_centroid_sel_df['cor']))

    # lista_original = list(filter_centroid_sel_df['coords'])
    # x_pixels = [p[1] for sublist in lista_original for p in sublist]
    # #x_pixels = #[p[1] for p in filter_centroid_sel_df['coords']]
    # y_pixels = [p[0] for sublist in lista_original for p in sublist]
    # # plt.plot(x_pixels, y_pixels, marker='o',markersize=1, color='blue')#filter_centroid_sel_df['cor'])
    # plt.scatter(x_pixels, y_pixels, s=1, color='blue')
    
    else:
        x_sel= filter_centroid_sel_df.loc[centroid_sel-1, 'centroid-1']
        y_sel= filter_centroid_sel_df.loc[centroid_sel-1, 'centroid-0']
        plt.scatter(x_sel, y_sel,s=1, color='red')
        #plt.plot(x_sel, y_sel,marker='o',markersize=1, color='red')

        df_exploded=filter_centroid_sel_df.explode('coords')
        x_pixels = [p[1] for p in list(df_exploded['coords'])]
        y_pixels = [p[0] for p in list(df_exploded['coords'])]
        plt.scatter(x_pixels, y_pixels, s=1, color=df_exploded['cor'])
    
    time_fim = time.time()
    print (time_fim-time_ini)
    #to show image with segmentation
    #plt.imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
    plt.imshow(img_sel_norm)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

#%%
# getting dir and file name
# Get the current working directory
current_directory = os.getcwd()

#for new image 23/02/2024
upper_level_directory = os.path.join(current_directory, '../data/test_segm_results/S2-16D_V2_012014_20220728_/')

#for new image 23/02/2024
read_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/Cassio/S2-16D_V2_012014_20220728_/'
file_r = read_dir+'S2-16D_V2_012014_20220728_B04.tif'
file_g = read_dir+'S2-16D_V2_012014_20220728_B03.tif'
file_b = read_dir+'S2-16D_V2_012014_20220728_B02.tif'
file_nir = read_dir+'S2-16D_V2_012014_20220728_B08.tif'
file_EVI = read_dir+'S2-16D_V2_012014_20220728_EVI.tif'
file_NDVI = read_dir+'S2-16D_V2_012014_20220728_NDVI.tif'

files_name = [file_r, file_g, file_b, file_nir, file_EVI, file_NDVI]

# Specify the filename and path within the upper-level directory
#filename = 'SENTINEL-2_MSI_20LMR_RGB_2022-07-16'
#for new image
#%%
filename = 'S2-16D_V2_012014_20220728_'
save_path = os.path.join(upper_level_directory, filename)    

#for new image
all_bands = ['B04', 'B03', 'B02', 'B08', 'EVI', 'NDVI']
bands_sel = ['B04', 'B03', 'B02'] # R,G,B bands selection for slic segmentation

#img_sel = np.dstack((image_band_dic[bands_sel[0]], image_band_dic[bands_sel[1]], image_band_dic[bands_sel[2]]))

img_sel_norm = load_image_norm()

segments_slic_sel={}
props_df_sel={}

id_test=3
ids = [id_test]

props_df_sel = read_props_df_sel(ids, save_path+"cluster", obj_to_read="cluster")

# to read the dic cluster and n_opt
file_to_open = save_path+"cluster" + '_'+str(3)+'.pkl'
with open(file_to_open, 'rb') as handle: 
    b = pickle.load(handle)

dic_cluster={}
dic_cluster = b[id_test]['dic_labels_cluster']

n_opt=b[id_test]['n_opt']

# usar das array pq com numpy está crashando na nova imagem
import dask.array as da
from math import ceil

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
print (time_fim-time_ini)

#%%
centroid_sel=90926    
centroid_row_matrix_sim = matrix_sim[centroid_sel-1,:]

centroid_sel_df = props_df_sel[id_test][['label', 'centroid-0','centroid-1']]
centroid_sel_df['idx'] = centroid_sel_df.index.values
#%%
centroid_sel_df_idx= da.from_array(props_df_sel[id_test].index.values, chunks=1000)

centroid_row_matrix_sim_df = dd.concat([dd.from_dask_array(c) for c in [centroid_sel_df_idx,centroid_row_matrix_sim ]], axis=1)
centroid_row_matrix_sim_df.columns = ['idx','sim_value']
#%%
dask_centroid_sel_df = dd.from_pandas(centroid_sel_df,npartitions=109)
#%%
dask_centroid_sel_df = dask_centroid_sel_df.merge(centroid_row_matrix_sim_df, on='idx')
#%%

#%%
#dask_centroid_sel_df['sim_value'] = centroid_row_matrix_sim
#dask_centroid_sel_df= dask_centroid_sel_df.assign(sim_value = centroid_row_matrix_sim.compute())

#%%
thr_min=0.90
thr_max=1

condition_1 = dask_centroid_sel_df['sim_value']>=thr_min
condition_2 = dask_centroid_sel_df['sim_value']<=thr_max
filter_centroid_sel_df = dask_centroid_sel_df.loc[condition_1 & condition_2]

#%%
#filter_centroid_sel_df['cor'] = filter_centroid_sel_df['sim_value'].apply(calc_cor, 'Blues')

#filter_centroid_sel_df['cor'] = filter_centroid_sel_df['sim_value'].apply(calc_cor, meta=('sim_value', 'object'))
#filter_centroid_sel_df.loc[centroid_sel-1,'cor']='red'
    
time_ini = time.time()    
#%%
plot_centroids=True
if (plot_centroids):
    # x_centroids=[x for x in filter_centroid_sel_df['centroid-1']]
    # y_centroids=[y for y in filter_centroid_sel_df['centroid-0']]
    x_centroids = filter_centroid_sel_df['centroid-1'].compute().values
    y_centroids = filter_centroid_sel_df['centroid-0'].compute().values
    print (len(x_centroids), len(y_centroids))
    #plt.scatter(x_centroids, y_centroids,s=1, color=list(filter_centroid_sel_df['cor']))
    plt.scatter(x_centroids, y_centroids,s=1, color='blue')
time_fim = time.time()
print (time_fim-time_ini)
    #to show image with segmentation
    #plt.imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
plt.imshow(img_sel_norm)
plt.axis('off')
plt.tight_layout()
plt.show()

# %%
