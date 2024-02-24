#%%
import pandas as pd
import numpy as np
import os
import sys
import glob
import pickle

import time
from tqdm import tqdm
from itertools import product

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

import imageio.v2 as imageio
from skimage.measure import regionprops, regionprops_table

import matplotlib.pyplot as plt
import plotly.express as px

from f_segmentation import *

seed = random.seed(999)
#%%
def calc_std_array(arr,b=b):
    # Map array elements to dictionary values
    #pixel_values_map = [pixel_band_value[elem[0], elem[1]] for elem in arr]
    #print (arr)
    pixel_values_map = [img_band_dic[b][elem[0], elem[1]] for elem in arr]
    #print (f'{b}\nmapped array: {pixel_values_map}')
    # Calculate standard deviation of mapped values
    return np.std(pixel_values_map), np.mean(pixel_values_map), \
            pixel_values_map,

def img_slic_segment_gen_df(bands_sel, image_band_dic, img_sel='', n_segms=600, sigma=2, \
                             compactness = 5, mask=None, conectivity=False):
    ''''
    if img_sel is empty
    generate image and image bands dictionary
    slic only accepts 3 bands
    generate image with selected bands and returns df of segmented image
    '''
    # 1.
    #criando imagem RGB para segmentar com SLIC 
    
    if not len(img_sel):
        print ("img_sel vazia")
        img_sel = np.dstack((image_band_dic[bands_sel[0]], image_band_dic[bands_sel[1]], image_band_dic[bands_sel[2]]))

    #print ("params for slic: ", n_segms, sigma, compactness, conectivity )
    segments_slic_sel = slic(img_sel, n_segments=n_segms, compactness=compactness, sigma=sigma, \
                             start_label=1,mask=mask, enforce_connectivity=conectivity)
    #print(f'SLIC RGB number of segments : {len(np.unique(segments_slic_sel))}')
    
    # 2. 
    props_dic = regionprops_table(segments_slic_sel, img_sel, \
                                properties=['label','coords', 'centroid','local_centroid'])
   
    # 3.
    props_df = pd.DataFrame(props_dic)
    props_df['num_pixels'] = props_df['coords'].apply(len)

    # 4. adiciona os valores das bandas do centroids (x,y) como colunas no df, 
    #    for each sp (superpixel), adds as column: number of pixels, calculates std and average 
    #    of each band and the value of pixels for each band 
    for b in image_band_dic.keys():
        props_df[b] = props_df.apply(lambda row: image_band_dic[b][round(row['centroid-0']), round(row['centroid-1'])], axis=1)
        #props_df['desvio_'+b]= props_df['coords'].apply(lambda arr: [image_band_dic[b][elem[0],elem[1]] for elem in arr])
        #pixel_band_value = image_band_dic[b]
        #fazer fora
        #props_df[['std_'+b, 'mean_'+b, 'seg_'+b ]] = props_df['coords'].apply(calc_std_array, image_band_dic[b]).apply(pd.Series)

    if len(img_sel):
        return props_df, segments_slic_sel
    else:
        return props_df, img_sel, segments_slic_sel




#%%
cur_dir = os.getcwd()
result_level_directory = os.path.join(cur_dir, '../data/test_segm_results/S2-16D_V2_012014_20220728_/')
read_dir = os.path.join(cur_dir, '../data/Cassio/S2-16D_V2_012014_20220728_/')
filename = 'S2-16D_V2_012014_20220728_'
save_path = os.path.join(result_level_directory, filename)
#%%
files_name = list_files_to_read(read_dir, '012014')
#%% # carregar as bandas dos arquivos
img_band_dic = load_image_files(files_name, pos=-1)
# %%
# gerar a imagem
all_bands = list(img_band_dic.keys())
bands_rgb = ['B04', 'B03', 'B02'] #RGB

# %%
img_rgb = np.dstack((img_band_dic[bands_rgb[0]],img_band_dic[bands_rgb[1]], img_band_dic[bands_rgb[2]]))
#mask = (img_band_dic[bands_rgb[0]] !=-9999)

# %%
for k in bands_rgb:
    print (np.max(img_band_dic[k]), np.min(img_band_dic[k]))
# %%
#getting the image with values normalized
img_band_dic_norm={}
for k in img_band_dic.keys():
    print (k)
    img_band_dic_norm[k]=img_band_dic[k].astype(float)/np.max(img_band_dic[k])

#%%
img_rgb_norm = np.dstack((img_band_dic_norm[bands_rgb[0]], 
                          img_band_dic_norm[bands_rgb[1]], 
                          img_band_dic_norm[bands_rgb[2]]))

#%%
segms = [11000, 110000]
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
# %%
#renomeando para a mesma chave dos testes feitos com a outra combinacao
params_test_dic[314]=params_test_dic.pop(3)
params_test_dic[302]=params_test_dic.pop(1)
params_test_dic[304]=params_test_dic.pop(2)

#%%    
#running tests for all combinations
ids = list(params_test_dic.keys())
stats_df_dic={}
props_df_sel = {} 
segments_slic_sel ={}

#%%
ids=[7]
mask=None
for id in tqdm(ids):
    n_segms = params_test_dic[id]['segms'] #600
    sigma = params_test_dic[id]['sigma']  #0.1
    compact = params_test_dic[id]['compactness']
    conectivity= params_test_dic[id]['connectivity'] #False
    print ("test and params id: ",id, n_segms, sigma, compact, conectivity)
    time_start = time.time()
    props_df_sel[id], segments_slic_sel = img_slic_segment_gen_df(bands_rgb, \
                                                img_band_dic, img_sel=img_rgb_norm,\
                                                n_segms=n_segms, sigma=sigma, \
                                                compactness = compact, mask=mask,\
                                                conectivity=conectivity)
    time_end = time.time()
    print(f'SLIC RGB number of segments : {props_df_sel[id].shape[0]} Time elapsed: {round(time_end - time_start, 2)}s')
    if False: #is_multiple(id, 20): #para salvar cada arquivo de teste
        # Save results to pickle file 
        obj_dic = {}
        obj_dic[id] = {
            "props_df_sel": props_df_sel#, 
            #"segments_slic_sel": segments_slic_sel,
           }
        file_to_save = save_path +str(id)+'.pkl'
        save_to_pickle(obj_dic, file_to_save)

        obj_dic={}
        obj_dic = {
            "segments_slic_sel": segments_slic_sel
        }
        file_to_save = save_path + '_segments_'+str(id)+'.pkl'
        save_to_pickle(obj_dic, file_to_save)
        
        del obj_dic, props_df_sel, segments_slic_sel

        props_df_sel = {}
        segments_slic_sel ={}
# %%
id=314
test_df={}
test_df[314]=props_df_sel[314].head(1)
#%%
for b in img_band_dic.keys():
    
    test_df[id][['std_'+b, 'mean_'+b, 'seg_'+b ]] = test_df[id]['coords'].apply(calc_std_array, b).apply(pd.Series)
# %%

def calculate_stats(coords):
    rows = [coord[0] for coord in coords]
    cols = [coord[1] for coord in coords]
    values = [matrix[row, col] for row, col in zip(rows, cols)]
    return np.mean(values), np.std(values), values
#%%
matrix=img_band_dic['B02']
id=314
# Apply the function to each row of the DataFrame
test_df[id][['mean', 'std', 'seg_']] = test_df[id]['coords'].apply(lambda x: pd.Series(calculate_stats(x)))

# %%
plt.imshow(mark_boundaries(img_rgb_norm[314], segments_slic_sel))