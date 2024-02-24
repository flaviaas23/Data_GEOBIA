#funcoes para segmentacao da imagem
#%%
import pandas as pd
import numpy as np
import os
import sys
import glob
import pickle

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

import imageio.v2 as imageio
from skimage.measure import regionprops, regionprops_table
# import psutil
import time


import random

def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def list_files_to_read(read_dir, padrao):
    ''''
    get a list of files in a eval_dir based on a padrao
    '''
    if not read_dir:
        read_dir='/Users/flaviaschneider/Documents/flavia/data_GEOBIA/data/Cassio/'


    #14/09/2023 padrao_ = os.path.join(eval_dir, '*_'+padrao+'.pkl') #euclidean_Individual.pkl')
    padrao_ = os.path.join(read_dir, '*_'+padrao+'*') #euclidean_Individual.pkl')

    print ("list_files_to_read: padrao_: ", padrao_)

    read_files = glob.glob(padrao_)
    print ("list_files_to_read: eval_files", read_files)

    return read_files



def load_image_files(files_name,pos=-2):
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

#def calc_std_array(arr,b):
#    # Map array elements to dictionary values
#    #pixel_values_map = [pixel_band_value[elem[0], elem[1]] for elem in arr]
#    #print (type(arr))
#    pixel_values_map = [img_band_dic[b][elem[0], elem[1]] for elem in arr]
#    #print (f'{b}\nmapped array: {pixel_values_map}')
#    # Calculate standard deviation of mapped values
#    return np.std(pixel_values_map), np.mean(pixel_values_map), \
#            pixel_values_map

# def img_slic_segment_gen_df(bands_sel, image_band_dic, img_sel='', n_segms=600, sigma=2, \
#                              compactness = 5, mask=None, conectivity=False):
#     ''''
#     if img_sel is empty
#     generate image and image bands dictionary
#     slic only accepts 3 bands
#     generate image with selected bands and returns df of segmented image
#     '''
#     # 1.
#     #criando imagem RGB para segmentar com SLIC 
    
#     if not len(img_sel):
#         print ("img_sel vazia")
#         img_sel = np.dstack((image_band_dic[bands_sel[0]], image_band_dic[bands_sel[1]], image_band_dic[bands_sel[2]]))

#     #print ("params for slic: ", n_segms, sigma, compactness, conectivity )
#     segments_slic_sel = slic(img_sel, n_segments=n_segms, compactness=compactness, sigma=sigma, \
#                              start_label=1,mask=mask, enforce_connectivity=conectivity)
#     #print(f'SLIC RGB number of segments : {len(np.unique(segments_slic_sel))}')
    
#     # 2. 
#     props_dic = regionprops_table(segments_slic_sel, img_sel, \
#                                 properties=['label','coords', 'centroid','local_centroid'])
   
#     # 3.
#     props_df = pd.DataFrame(props_dic)
#     props_df['num_pixels'] = props_df['coords'].apply(len)

#     # 4. adiciona os valores das bandas do centroids (x,y) como colunas no df, 
#     #    for each sp (superpixel), adds as column: number of pixels, calculates std and average 
#     #    of each band and the value of pixels for each band 
#     for b in image_band_dic.keys():
#         props_df[b] = props_df.apply(lambda row: image_band_dic[b][round(row['centroid-0']), round(row['centroid-1'])], axis=1)
#         #props_df['desvio_'+b]= props_df['coords'].apply(lambda arr: [image_band_dic[b][elem[0],elem[1]] for elem in arr])
#         #pixel_band_value = image_band_dic[b]
#         #fazer fora
#         #props_df[['std_'+b, 'mean_'+b, 'seg_'+b ]] = props_df['coords'].apply(calc_std_array, image_band_dic[b]).apply(pd.Series)

#     if len(img_sel):
#         return props_df, segments_slic_sel
#     else:
#         return props_df, img_sel, segments_slic_sel



