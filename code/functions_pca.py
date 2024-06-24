#copied at 17/06/2024
#From: notebooks/Test_PCA_img_full_working-Copy1.ipynb
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

import dask.array as da
import dask.dataframe as dd 
from dask_ml.decomposition import PCA

### Functions to read and save

def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def read_slic_sel(id, open_path, obj_to_read='props_df_sel',output=True):
    ''''
    read props_df_sel and returns them as a dicionary
    '''
    #print (open_path)
    dic_df={}
    #for id in tqdm(ids):      #ids #ids_file
    if (obj_to_read == 'props_df_sel'):
        #file_to_open = open_path + '_'+str(id)+'.pkl'
        file_to_open = open_path+obj_to_read+'_'+str(id)+'.pkl'
        #print (file_to_open)
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
        file_to_open = open_path + obj_to_read+'_'+str(id)+'.pkl'
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

#### Function to list files_to read
def list_files_to_read(read_dir, padrao, sh_print=0):
    ''''
    get a list of files in a eval_dir based on a padrao
    '''
    if not read_dir:
        read_dir='/Users/flaviaschneider/Documents/flavia/data_GEOBIA/data/Cassio/'


    #14/09/2023 padrao_ = os.path.join(eval_dir, '*_'+padrao+'.pkl') #euclidean_Individual.pkl')
    padrao_ = os.path.join(read_dir, '*'+padrao+'*') #euclidean_Individual.pkl')

    #print ("list_files_to_read: padrao_: ", padrao_)

    read_files = glob.glob(padrao_)
    print ("list_files_to_read: eval_files", read_files) if sh_print else None

    return read_files

#### Function to load images bands
def load_image_files3(files_name,pos=-2):
    ''''
    load tiff image bands files of a timestamp 
    this is for example image 
    pos = band pos , -2 for example image, -1 for tile image
    '''
    #files_name=[file_nbr,file_evi, file_ndvi, file_red,file_green,file_blue]
    image_band_dic={}
    for f in files_name:
        f_name = f.split('_')  
        #print (f_name)
        if pos == -1:
            band = f_name[pos].split('.')[0]
        else:
            band = f_name[pos]
        #print (band)
        image_band_dic[band] = imageio.imread(f)

    return image_band_dic

#### Function to load normalized image
def load_image_norm(read_path=''):
    '''
    load img normarlized
    '''
    if not read_path:
        read_path = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/test_segm_results/S2-16D_V2_012014_20220728_/S2-16D_V2_012014_20220728_'
    file_to_open = read_path + 'img_sel_norm.pkl'
    with open(file_to_open, 'rb') as handle:    
        img_sel_norm = pickle.load(handle)
    return img_sel_norm

### Functions to gen df_ts 
#### Functions get_bandsDates and get_dates_lower
##### Functions get_bandsDates
def get_bandsDates(band_img_files, tile=0):
    '''
    #From image tif files names return dates and bands 
    tile = 1 for file images, =0 for example image
    '''
    
    if tile:
        pos=-1 #position of the band in tile name file
        pos_date=-2
    else:
        pos=-2 #position of the band in example imag name file
        pos_date=-1
    dates=[]
    bands = []
    time1=time.time()
    for f in band_img_files:
        f_name = f.split('_')  
        b = f_name[pos].split('.')
        if b[0] not in bands:
            bands.append(b[0])
        f_name = f_name[pos_date].split('.')
        if f_name[pos] not in dates:
            dates.append(f_name[pos])

    dates = sorted(dates)
    time2=time.time()
    print (time2-time1)
    return bands, dates

##### Functions get_dates_lower
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

#### Functions calc_avg_array
def calc_avg_array(arr,b, c, med='med', pca=0):
    # Map array elements to dictionary values if they are not nan (-9999, -32768)
    #pixel_values_map = [pixel_band_value[elem[0], elem[1]] for elem in arr]
    if not pca:
        pixel_values_map = [image_band_dic[b][elem[0], elem[1]] for elem in arr if image_band_dic[b][elem[0], elem[1]] not in [-9999, -32768]]
    else:
        pixel_values_map = [img_comp_dic[c][elem[0], elem[1]] for elem in arr if img_comp_dic[c][elem[0], elem[1]]]
    #print (f'{b}\nmapped array: {pixel_values_map}')
    # Calculate median deviation of mapped values
    #median or mean: average
    
    arr_avg = np.median(pixel_values_map) if pixel_values_map else np.nan
    return arr_avg
    #return np.median(pixel_values_map)#/max_b
    #np.std(pixel_values_map), np.mean(pixel_values_map), \

def calc_avg_array_pca(arr,c, med='med' ):
    # Map array elements to dictionary values if they are not nan (-9999, -32768)
    #pixel_values_map = [pixel_band_value[elem[0], elem[1]] for elem in arr]
    pixel_values_map = [img_comp_dic[c][elem[0], elem[1]] for elem in arr if img_comp_dic[c][elem[0], elem[1]]]
    #print (f'{b}\nmapped array: {pixel_values_map}')
    # Calculate median deviation of mapped values
    #median or mean: average
    
    if med == 'med':
        arr_calc = np.median(pixel_values_map) if len(pixel_values_map) else np.nan
    elif med == 'avg':    
        arr_calc = np.mean(pixel_values_map) if len(pixel_values_map) else np.nan
    elif med == 'std':
        arr_calc = np.std(pixel_values_map) if len(pixel_values_map) else np.nan        
    else:
        arr_calc = pixel_values_map[0]#np.mean(pixel_values_map) if pixel_values_map else np.nan
    return arr_calc
    
    #return np.median(pixel_values_map)#/max_b
    #np.std(pixel_values_map), np.mean(pixel_values_map), \

#### Functions to generate a df to be used in PCA with all pixels of image bands 

##### function to verify the percentage of nans in a band image dictionary

def check_perc_nan(image_band_dic):
    ''''
    function to verify nan image percentage 
    '''
    count_day_nan=0
    num_bands = list(image_band_dic.keys())
    num_elemts = image_band_dic[num_bands[0]].size
    max_b={}
    for b in num_bands:
        count_day_nan+= np.count_nonzero(image_band_dic[b] == -9999)
        count_day_nan+= np.count_nonzero(image_band_dic[b] == -32768)
        # label_neg = set()
        # label_neg_indx=set()
        
        #min_b[b] = np.min(image_band_dic[b], where=~np.isnan(b), initial=10)
        #image_band_dic_norm[b]=image_band_dic[b].astype(float)/max_b

    perc_day_nan = (count_day_nan/(num_elemts*len(num_bands)))*100

    return perc_day_nan

#### Function do generate a dataframe of the all pixels image bands in a specific date<br>
#### format index_orig| coords_0|coords_1|coords| b1 | b2| ...|bn
def gen_df_from_img_band(matrix, bands_name, ar_pos=1, sh_print=0):
    '''
    gen df from img bands matrix
    returns df in format coords|b1|b2|...|bn
    ar_pos = 1 if coords in format [x,y] should be in a column of dft
    '''
    t1=time.time()
    # Get the shape of the matrix
    num_rows, num_cols, cols_df = matrix.shape
    print (f'matrix rows={num_rows}, cols={num_cols}, bands={cols_df}') if sh_print else None
    # Create a list of index positions [i, j]
    positions = [[i, j] for i in range(num_rows) for j in range(num_cols)] 
    pos_0=np.repeat(np.arange(matrix.shape[0]), matrix.shape[1])
    pos_1=np.tile(np.arange(matrix.shape[1]), matrix.shape[0])
    
    # Flatten the matrix values and reshape into a 2D array
    values = matrix.reshape(-1, cols_df)
    
    # Create a DataFrame from the index positions and values
    dft = pd.DataFrame(values, columns=bands_name)#, index=positions)
    
    # Reset the index to create separate columns for 'i' and 'j'
    # dft.reset_index(inplace=True)
    # dft.rename(columns={'index': 'position'}, inplace=True)
    # df['position'] = [f"[{i},{j}]" for i, j in positions]
    #mais rapido abaixo
    dft.insert(0,'coords_0', pos_0)
    dft.insert(1,'coords_1', pos_1)
    if ar_pos:
        dft.insert(2,'coords', positions) 
    
    t2=time.time()
    #0.003547191619873047
    if sh_print: 
        print (t2-t1) 
    
    return dft

##### function gen_dfToPCA_filter, verifies if bands images nan percenteage 
##### is lower than perc
def gen_dfToPCA_filter(dates, band_img_files, pos=-1, perc=15, ar_pos=1, dask=0, sh_print=0):
    # gen df to PCA with original values from bands images files, 
    # including Nans(-9999, -32768), only images with nan percentage 
    # lower than perc will be considered 
    max_bt = {}
    min_bt = {}
    for i,t in tqdm(enumerate(dates)):#['2022-07-16']:#dates_to_pca: ['2022-07-16','2022-08-17']
        band_img_file_to_load=[x for x in band_img_files if t in x]
        image_band_dic = {}
        image_band_dic = load_image_files3(band_img_file_to_load, pos=pos) #pos=-1 for tiles , pos =-2 for ex img
        bands = image_band_dic.keys()
        print ( bands, band_img_file_to_load) if sh_print == 2 else None
        #check if nan percentage is lower than perc
        
        perc_day_nan= check_perc_nan(image_band_dic)
        if perc_day_nan > perc:
            print (f'For {t} perc_day_nan= {perc_day_nan}') if sh_print else None
            continue
        
        #save the max of image
        for b in bands:
            # Crie uma máscara para filtrar os valores Nan
            mask = (image_band_dic[b] != -9999) & (image_band_dic[b] != -32768)

            if b in max_bt:
                max_bt[b].append(np.max(image_band_dic[b]))
                                
                # Aplique a máscara e encontre o valor mínimo
                min_bt[b].append(np.min(image_band_dic[b][mask]))
            else:
                #max_bt[b]=[]
                max_bt[b] = [np.max(image_band_dic[b])]
                min_bt[b] = [np.min(image_band_dic[b][mask])]
        
        img_bands = np.dstack([image_band_dic[x] for x in bands])
        cols_name = [x+'_'+t for x in bands]
        #gen df for images bands of the t day
        dft = pd.DataFrame()
        dft = gen_df_from_img_band(img_bands, cols_name, ar_pos=ar_pos, sh_print=0)
        del img_bands
        #print (f'i= {i} {t}')
        t1=time.time()
        if i==0:
            #print (f'i= {i} {t}')
            dft.insert(0,'orig_index',dft.index)
            df_toPCA= dft.copy()
        else:
            #dft.drop(columns=['coords'], inplace=True)
            if ar_pos:
                dft = dft.drop('coords', axis=1)
            df_toPCA = pd.merge(df_toPCA, dft, on=['coords_0','coords_1'])#left_index=True, right_index=True)# on='coords')
            #df_toPCA = pd.concat([df_toPCA, dft], axis=1)
            
        t2=time.time()
        #print (f'i= {i}, t= {t} t2-t1 ={t2-t1}')
        del dft
    if dask:
        print (df_toPCA.shape[0] , df_toPCA.shape[1])
        # n_part = round(df_toPCA.shape[0] * df_toPCA.shape[1]/1000)
        n_part = round(df_toPCA.shape[0]/1000)# * df_toPCA.shape[1]/1000)
        dd_df_toPCA = dd.from_pandas(df_toPCA, npartitions=n_part)
        #deixar para o dask definiir o tamanho da partiçao
        #no dask tem q passar ou o chunks ou o partitions
        #dd_df_toPCA = dd.from_pandas(df_toPCA)

        return dd_df_toPCA, max_bt, min_bt
        
    #print (f'max of images bands {max_bt}') if print_time else None
    print (f'tempo para gerar df das matrizes de bandas das imagens: {t2-t1}') if sh_print else None
    return df_toPCA, max_bt, min_bt

#### Functions to normalize the columns values
##### Function to norm the columns by its maximun value
#normaliza os valores
def normalize_columns(df, col_ini, norm='max'):
    """
    Normaliza todas as colunas de um DataFrame dividindo cada valor 
    pelo seu máximo.   
    Parâmetros:
        df (DataFrame): O DataFrame a ser normalizado.   
        norm: max, se for para usar a normalizacao x/max(X)
              stand, se for para usar standardization (x-med(X))/std(X)
    Retorna:
        DataFrame: O DataFrame normalizado.
    """
    # Cria uma cópia do DataFrame original para evitar alterações indesejadas
    #df_normalized = df.copy()
    
    # Normaliza cada coluna dividindo pelo seu máximo
    for col in df.columns[col_ini:]:
        if norm=='max':
            df[col] = df[col] / df[col].max()
        elif norm=='stand':
            df[col] = (df[col]- df[col].median()) / df[col].std()
        elif norm =='mean':
            df[col] =  df[col]- df[col].mean()
    
    return df

##### Function to norm the column by its amplitude or reference values of Sentinell satellite
#normaliza os valores
def norm_columns(df, col_ini, min_bt, max_bt, ref=0):
    """
    Normaliza todas as colunas de um DataFrame dividindo cada valor 
    pelo (x - min)/(máximo - mínimo) passado por banda
    
    Parâmetros:
        df (DataFrame): O DataFrame a ser normalizado.   
        col_ini: coluna inicial do df a ser normalizada
        min_b: dicionário com valores min de cada banda
        max_b: dicionário com valores max de cada banda
        ref: se True usar valores de referência de cada banda
             para o satelite sentinel
    Retorna:
        DataFrame: DataFrame normalizado.
    """
    amp_maxB = {}
    min_b = {}
    if ref:
        #mudar isso para qdo tiver os valores de referencia 
       for b in max_bt.keys():
            amp_maxB[b] = max_bt[b] - min_bt[b]
    else:
        for b in max_bt.keys():
            min_b[b] = np.int32(np.min(min_bt[b]))
            amp_maxB[b] = np.int32(np.max(max_bt[b])) - (min_b[b])
            
    
    # Normaliza cada coluna dividindo pelo seu máximo
    for col in df.columns[col_ini:]:
        b=col.split('_')[0]
        df[col] = (df[col]-min_b[b]) / (amp_maxB[b])
        
    
    return df

### Functions for PCA

'''
# A funcao abaixo do Georges retorna exatamente o que a funcao abaixo retorna:
import numpy as np
from sklearn.utils.extmath import randomized_svd
a = np.array([[1, 2, 3, 5],
              [3, 4, 5, 6],
              [7, 8, 9, 10]])
U, s, Vh = randomized_svd(a, n_components=2, random_state=0)
U.shape, s.shape, Vh.shape
#os valores de retorno acima foram iguais ao da funcao do georges abaixo
'''
#%%
# Function from Georges
# preciso entender a saída
def randomized_svd_g(X, K, sparse_random_projection=False, custom_sparsity_param=None, oversampling=10, power_iterations=0):
    # Usa esses parametros:
    # oversampling=5, power_iterations=2
    D = X.shape[1] # number of columns
    if sparse_random_projection:
        if custom_sparsity_param is None:
            S_sparsity = int(math.sqrt(D) // 1) + 1 #Trevor and hasties S parameter for controlling sparsity
        else:
            S_sparsity = custom_sparsity_param
        prob_thres = 1 / (2*S_sparsity)
        prob_mat = np.random.rand(D, K + oversampling)
        proj_matrix = np.zeros((D, K + oversampling))
        proj_matrix[prob_mat < prob_thres] = -1
        proj_matrix[prob_mat > (1 - prob_thres)] = 1
    else:
        proj_matrix = np.random.randn(D, K + oversampling)
    
    Z = X @ proj_matrix
    for _ in range(power_iterations):
        Z = X.T @ Z
        Z = X @ Z
    Q, R = np.linalg.qr(Z)

    Y = Q.T @ X
    #u_y, s_p, vh_p = np.linalg.svd(Y, full_matrices=False)
    u_y, s_p, vh_p = da.linalg.svd(Y)
    u_p = Q @ u_y

    return u_p[:, :K], s_p[:K], vh_p[:K, :]




'''
# %%
from sklearn.utils.extmath import randomized_svd
sklearn.utils.extmath.randomized_svd
from sklearn.decomposition import PC
'''