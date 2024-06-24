
#%%
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.colorbar import Colorbar

import numpy as np
from math import ceil

from skimage.data import astronaut,lily, immunohistochemistry
from skimage.color import rgb2gray
from skimage.filters import sobel
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

from itertools import product

import pickle
import h5py
import random
from tqdm import tqdm

from plotly.subplots import make_subplots

#%%
from pyclustering.cluster.clarans import clarans
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import timedcall;
#%%
seed = random.seed(999) # para gerar sempre com a mesma seed

#for new image 23/02/2024
read_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/Cassio/S2-16D_V2_012014_20220728_/'
file_r = read_dir+'S2-16D_V2_012014_20220728_B04.tif'
file_g = read_dir+'S2-16D_V2_012014_20220728_B03.tif'
file_b = read_dir+'S2-16D_V2_012014_20220728_B02.tif'
file_nir = read_dir+'S2-16D_V2_012014_20220728_B08.tif'

file_EVI = read_dir+'S2-16D_V2_012014_20220728_EVI.tif'
file_NDVI = read_dir+'S2-16D_V2_012014_20220728_NDVI.tif'

files_name = [file_r, file_g, file_b, file_nir, file_EVI, file_NDVI]

#%%
name_img = 'S2-16D_V2_012014_20220728_'
all_bands = ['B04', 'B03', 'B02', 'B08', 'EVI', 'NDVI']
def gen_files_to_read(name_img, bands, read_dir):
    ''''
    gen array with tif files name to read
    '''
    files_to_read = []
    for b in bands:
        files_to_read.append(read_dir+name_img+b+'.tif')
    
    return files_to_read

#%%
tif_names=[]
tif_names=gen_files_to_read(name_img, all_bands, read_dir)


#%%
def load_image_files2(files_name,pos=-2):
    ''''
    load tiff image bands files of a timestamp
    '''
    #files_name=[file_nbr,file_evi, file_ndvi, file_red,file_green,file_blue]
    bands_name =[]
    image_band_dic={}
    for f in files_name:
        f_name = f.split('.')  #para remover o .tif
        band = f_name[0].split("_") # para obter onome da banda do arquivo
        #print (band[pos], f_name[-2])
        bands_name.append(band[pos])
        image_band_dic[band[pos]] = imageio.imread(f)

    return image_band_dic
#%%
def load_image_norm(read_path=''):
    if not read_path:
        read_path = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/test_segm_results/S2-16D_V2_012014_20220728_/S2-16D_V2_012014_20220728_'
    file_to_open = read_path + 'img_sel_norm.pkl'
    with open(file_to_open, 'rb') as handle:    
        img_sel_norm = pickle.load(handle)
    return img_sel_norm

#%%
#####################################
#para criar o dataframe para ser usado ns clusterizacao no formato:
# id superpixel | qtde de pontos| pontos | centroid-0 | centroid-1 |local_centroid-0 | local_centroid-1 | R|G|B|EVI|NDVI|outros
# 
# 0. Ler arquivos tif com as bandas
# 1. faz o slic com o RGB ou com as outras bandas
# 2. usa o regionprops_table para obter um dicionario com label, coords, centroid, local_centroid
# 3. cria dataframe a partir do dicionario: df_props = pd.DataFrame(props)
# 4. Adiciona as informacoes das bandas como colunas no df criado
# 5. Adiciona o desvio padrao para cada segmento em cada banda


#%%
def is_multiple(num, multiple):
    return num % multiple == 0

def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(SAVE_DIR+pickle_cluster_file+'.pkl', 'wb') as handle:
    # pickle.dump(obj_cluster, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
#%%
def get_labels(clusters, len_arraybands_list):
    ''''
    retorna uma lista com os labels de cada elemento baseado no seu indice
    '''
    labels = []
    for elemento_procurado in range(len_arraybands_list): 
        for i, subarray in enumerate(clusters):
            if elemento_procurado in subarray:
                #indice_subarray = i
                labels.append(i)
                break
    
    return np.array(labels)
#%%
def load_image_files(files_name,pos=-2):
    ''''
    load tiff image bands files of a timestamp
    usado com a imagem inicial
    '''
    #files_name=[file_nbr,file_evi, file_ndvi, file_red,file_green,file_blue]
    bands_name =[]
    image_band_dic={}
    for f in files_name:
        band = f.split("_")
        print (f,band[pos])
        bands_name.append(band[pos])
        image_band_dic[band[pos]] = imageio.imread(f)

    return image_band_dic
#%%
# Function to calculate standard deviation of pixels band value of an array 
# of pixels and return the std and pixels band value
def calc_std_array(arr,b):
    # Map array elements to dictionary values
    #pixel_values_map = [pixel_band_value[elem[0], elem[1]] for elem in arr]
    pixel_values_map = [image_band_dic[b][elem[0], elem[1]] for elem in arr]
    #print (f'{b}\nmapped array: {pixel_values_map}')
    # Calculate standard deviation of mapped values
    return np.std(pixel_values_map), np.mean(pixel_values_map), \
            pixel_values_map
           
    
#%%
def gen_stats(df, id_test, conf_test_dic, all_bands):
    '''
    Generates statistics for the segmentation df 
    calculates average, max and min number of pixels in a sp (superpixel) 
    gets the average band value and average std for each band   
    Returns a df with test parameters and its stats 
    '''
    stats_sp_segments={}
    #all_bands = ['B11', 'B8A', 'B02', 'NBR', 'EVI', 'NDVI']
    # for n_segms in [600,1000]:
    #     if n_segms == 600: 
    #         df = props_df_sel
    #     elif n_segms == 1000: 
    #         df = props_df_sel_1k
    #incluir num de sp calculados no teste
    
    if id_test not in stats_sp_segments:
        stats_sp_segments[id_test] = {}
    #print ("1: gen_stats: conf_test_dic:\n", conf_test_dic)
    stats_sp_segments[id_test] = conf_test_dic.copy()
    #print ("2: gen_stats: stats_sp_segments:\n",stats_sp_segments)
    #print ("df.shape[0]:", df.shape[0])
    stats_sp_segments[id_test]['segms_calc'] = df.shape[0]
    stats_sp_segments[id_test]['avg_n_pixels'] = df['num_pixels'].mean()
    stats_sp_segments[id_test]['max_n_pixels'] = df['num_pixels'].max()
    stats_sp_segments[id_test]['min_n_pixels'] = df['num_pixels'].min()
    for b in all_bands:
        stats_sp_segments[id_test]['avg_'+b] = df['mean_'+b].mean()
        stats_sp_segments[id_test]['avg_std_'+b] = df['std_'+b].mean()
    del df
    #print ("3:",stats_sp_segments)
    stats_df = pd.DataFrame(stats_sp_segments).T
    #stats_df = pd.DataFrame(list(stats_sp_segments.items()), columns=['key', 'value'], dtype=object).T

    del stats_sp_segments, conf_test_dic

    #stats_df = stats_df.applymap('{:,.2f}'.format)

    return stats_df

#%%
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
def cria_SimilarityMatrix_freq(dic_cluster, n_clusters=None):
        '''
        Generate a frequency/similarity/Co-association matrix based on 
        frequency of point are together in the clustering of pixels
        '''
        if not n_clusters:
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
                for j in range(i, nrow): #for j in range(0, nrow):
                    #print ("j = ",j , cluster[i], cluster[j], sil[i], sil[j])
                    if cluster[i] == cluster[j]:

                        #freq = (sil[i]+sil[j]+2)/4
                        freq_matrix[i,j] += 1 # freq
                        
                        #freq_matrix[j,i] = freq_matrix[i,j]  # par rebater
                        #print ("j = ",j , cluster[i], cluster[j], sil[i], sil[j], freq)

            #print ("freq_matrix = \n", freq_matrix)
        freq_matrix= freq_matrix/len(n_clusters)
        #para rebater a matrix (18/02/2024)
        freq_matrix=freq_matrix+np.triu(freq_matrix,1).T  #para rebater senao fizer row273
        #
        #print ("freq_matrix = \n", freq_matrix)
        return freq_matrix        

#%%

#%%
def plot_box(props_df_sel, params_test_dic, stdbands=['std_B11','std_B8A','std_B02'],n_cols=2, chart_size=(800, 600)):
    ''''
    funcion to plot box in cols and lines
    '''
    tests = list(props_df_sel.keys())
    #n_cols = 2
    n_rows = np.ceil(len(tests)/n_cols).astype(int)
    sub_titles=[]
    for k in tests:
        segms=f"{params_test_dic[k]['segms']}/{props_df_sel[k].shape[0]}"
        compact=params_test_dic[k]['compactness']
        sigma=params_test_dic[k]['sigma']
        conect=params_test_dic[k]['connectivity']
        subt=f't_{k} - SP={segms} compact={compact} sigma={sigma} conect={conect}'
        sub_titles.append(subt)
    #sub_titles = ['test_'+str(k)+f'segms={params_test_dic['segms']}, compact={}' for k in tests]
     
    fig = make_subplots(rows=int(n_rows), cols=n_cols, subplot_titles=sub_titles)
    r=1 
    c=1
    for i, id_test in enumerate(tests):
        # data1 = props_df_sel[id_test]['std_B11']
        # data2 = props_df_sel[id_test]['std_B8A']
        # data3 = props_df_sel[id_test]['std_B02']
        #fig.add_trace(px.box(y=[data1,data2,data3]).data[0],  row=r, col=c)
        
        df = props_df_sel[id_test][stdbands]
        r = (i//n_cols) + 1
        c = (i%n_cols)+1
        
        box = px.box(df)
        fig.add_trace(box.data[0],  row=r, col=c)
        fig.update_traces(showlegend=True, legendgroup=id_test, name=id_test, row=r, col=c)
    fig.update_layout(showlegend=True, title_font=dict(size=10), width=chart_size[0], height=chart_size[1])

    # Update subtitle font size for all subplots
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=10)


    fig.show()
#%%

def plot_images(img_sel,props_df_sel, segments_slic_sel, params_test_dic, n_cols=3,chart_size=(12, 12)):
    ''''
    funcion to plot images in cols and lines
    '''
    tests = list(props_df_sel.keys())
    #n_cols = 3
    n_rows = np.ceil(len(tests)/n_cols).astype(int)
    sub_titles=[]
    for k in tests:
        segms=f"{params_test_dic[k]['segms']}/{props_df_sel[k].shape[0]}"
        compact=params_test_dic[k]['compactness']
        sigma=params_test_dic[k]['sigma']
        conect=params_test_dic[k]['connectivity']
        subt=f't_{k} - SP={segms} compact={compact} sigma={sigma} conect={conect}'
        sub_titles.append(subt)
    #sub_titles = ['test_'+str(k)+f'segms={params_test_dic['segms']}, compact={}' for k in tests]
     
    fig,axes = plt.subplots(nrows=int(n_rows), ncols=n_cols, sharex=True, sharey=True,figsize=chart_size)
    #axes = axes.flatten()

    for i, id_test in enumerate(tests):
                
        #df = props_df_sel[id_test][['std_B11','std_B8A','std_B02']]
        r = (i//n_cols)# + 1-1
        c = (i%n_cols)#+1-1
        #print (r,c)
        axes[r,c].imshow(mark_boundaries(img_sel, segments_slic_sel[id_test]))
        # Customize subplot title
        axes[r,c].set_title(sub_titles[i], fontsize=7)

        # Hide axis ticks and labels
        axes[r,c].axis('off')
        
        #print (i, id_test. r,c)
        #box = px.box(df)
        #fig.add_trace(box.data[0],  row=r, col=c)
        #fig.update_traces(showlegend=True, legendgroup=id_test, name=id_test, row=r, col=c)
    
    # tem que setar visible to false the subs plots to complement the grid
    # of subplots
    num_subs = n_cols*n_rows
    if len(tests)< num_subs:
        for cc in range(c+1,n_cols):
            print (r,cc)
            axes[r,cc].axis('off')
            axes[r,cc].set_visible(False)

    #fig.update_layout(showlegend=True, title_font=dict(size=10), width=chart_size[0], height=chart_size[1])

    # # Update subtitle font size for all subplots
    # for annotation in fig['layout']['annotations']:
    #     annotation['font'] = dict(size=10)

    # Adjust layout
    plt.tight_layout()
    plt.show()
    #fig.show()

#%%
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
#0.

#%%for new file image 23/02/2024
files_name = [file_r, file_g, file_b, file_nir, file_EVI, file_NDVI]

image_band_dic = {}
image_band_dic = load_image_files2(files_name, pos=-1)
#for new image
#%%
### Automation Tests to identify the optimun parameters for segmentation
# c_segms = [1000, 11000, 2000]  # segms[0]: start, segms[1]: stop, segms[2]: step
# segms = list(range(c_segms[0], c_segms[1] + 1, c_segms[2]))
# compactness = [1, 2,5,10,15]
# sigmas = [0.1, 0.5, 1,2,5,10]
# connectivity = [True, False]
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

# %%
#renomeando para a mesma chave dos testes feitos com a outra combinacao
params_test_dic[314]=params_test_dic.pop(7)
params_test_dic[302]=params_test_dic.pop(5)
params_test_dic[304]=params_test_dic.pop(6)
###
#%%
from pysnic.algorithms.snic import snic, compute_grid
#%%
# getting dir and file name
# Get the current working directory
current_directory = os.getcwd()

##upper_level_directory = os.path.join(current_directory, 'data/test_segm_results/')
#%%
#for new image 23/02/2024
upper_level_directory = os.path.join(current_directory, '../data/test_segm_results/S2-16D_V2_012014_20220728_/')
#%%
# Specify the filename and path within the upper-level directory
#filename = 'SENTINEL-2_MSI_20LMR_RGB_2022-07-16'
#for new image
filename = 'S2-16D_V2_012014_20220728_'
save_path = os.path.join(upper_level_directory, filename)    
#%%    
#running tests for all combinations
ids = list(params_test_dic.keys())
stats_df_dic={}
props_df_sel = {}
 
segments_slic_sel ={}

#%%
#for new image
all_bands = ['B04', 'B03', 'B02', 'B08', 'EVI', 'NDVI']
bands_sel = ['B04', 'B03', 'B02'] # R,G,B bands selection for slic segmentation

img_sel = np.dstack((image_band_dic[bands_sel[0]], image_band_dic[bands_sel[1]], image_band_dic[bands_sel[2]]))
# quando um pixel é -9999 numa banda tmabém é nas outras RGB 
# a mask que deve ser passada para o slic é só um valor True ou false por pixel
#mask = (image_band_dic[bands_sel[0]] !=-9999)
#como nao tem -9999
#%%
mask=None
contains_nan={}
time_ini = time.time()

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
time_fim= time.time()
print (time_fim-time_ini)
del contains_nan


#mais lento que acima
time_ini=time.time()
contains_nan = np.any(img_sel == -9999)
if contains_nan:
    mask=~np.any(img_sel==-9999, axis=-2)
time_fim=time.time()
time_fim-time_ini

#%%
props_dic_SP={}
ids=[3]
for id in tqdm(ids):
    n_segms = params_test_dic[id]['segms'] #600
    sigma = params_test_dic[id]['sigma']  #0.1
    compact = params_test_dic[id]['compactness']
    conectivity= params_test_dic[id]['connectivity'] #False
    print ("test and params id: ",id, n_segms, sigma, compact, conectivity)
    time_start = time.time()
    props_df_sel[id], props_dic_SP, segments_slic_sel = img_slic_segment_gen_df(bands_sel, \
                                                image_band_dic, img_sel=img_sel,\
                                                n_segms=n_segms, sigma=sigma, \
                                                compactness = compact, mask=mask,\
                                                conectivity=conectivity)
    time_end = time.time()
    print(f'SLIC RGB number of segments : {props_df_sel[id].shape[0]} Time elapsed: {round(time_end - time_start, 2)}s')
    if False: #is_multiple(id, 20): #para salvar cada arquivo de teste
    # Save to pickle file if the number is a multiple of 3
        obj_dic = {}
        obj_dic[id] = {
            "props_df_sel": props_df_sel#, 
            #"segments_slic_sel": segments_slic_sel,
           }
        file_to_save = save_path +str(id)+'.pkl'
        save_to_pickle(obj_dic, file_to_save)
        del obj_dic
        obj_dic ={}
        obj_dic[id] = {
            #"props_df_sel": props_df_sel#, 
            "segments_slic_sel": segments_slic_sel,
           }
        file_to_save = save_path+ 'segments_' +str(id)+'.pkl'
        save_to_pickle(obj_dic, file_to_save)
        del obj_dic , props_df_sel, segments_slic_sel

        props_df_sel = {}
        segments_slic_sel ={}
    #stats_df_dic[id] = gen_stats(props_df_sel[id], id, params_test_dic[id], all_bands)

# salve results to pickle 
#%%
SAVE_PNG = False
if SAVE_PNG:
    RGB = save_img_png(file_r, file_g, file_b)  

    from PIL import Image
    Image.fromarray(RGB).save(filename+'.png') 
#%%
#fazer separado para nao dar problemas de mem
#read the file with segms
# comented 25/02/2024
# ids_file = [x for x in ids if is_multiple(x,20)]

# with open(file_to_save, 'rb') as handle: 
#     b = pickle.load(handle)
#%%
stats_df_dic={}

for id in tqdm(ids):      #ids #ids_file
    #file_to_save = save_path + '_'+str(id)+'.pkl'
    file_to_save = save_path +str(id)+'.pkl'  #for new image 24/02/2024
    print (file_to_save)
    with open(file_to_save, 'rb') as handle: 
        b = pickle.load(handle)
    for i in list(b[id]['props_df_sel'].keys()):
        stats_df_dic[i] = gen_stats(b[id]['props_df_sel'][i], i, params_test_dic[i], all_bands)
    del b        
#%%
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
    
#%%    
result_stats_df = concat_dfs(stats_df_dic)
#%%
result_stats_df.insert(0, 'id_test',result_stats_df.index )
#%%
columns =[x for x in list(result_stats_df.columns) if x != 'connectivity' ]
#precisa converter os valores das colunas para numerico/float : nao precisa mais
for c in columns:
    #print (c)    
    result_stats_df[c] = result_stats_df[c].apply(lambda x: float(x.replace(',', ''))).astype(float)
#%% sort result, nao é mais necessario fazer bloco abaixo
stats_df_dic_sorted= {key: stats_df_dic[key] for key in sorted(stats_df_dic.keys())}

result_stats_df_sorted = concat_dfs(stats_df_dic_sorted)
for c in columns:
    #print (c)    
    result_stats_df_sorted[c] = result_stats_df_sorted[c].apply(lambda x: float(x.replace(',', ''))).astype(float)

#%%
#save stats to pickle file
file_to_save = save_path + '_stats_df.pkl'
save_to_pickle(result_stats_df, file_to_save)
#%%
result_stats_df.to_csv(save_path + '_stats_df.csv', index=False)
#%%
result_stats_df.to_excel(save_path + '_stats_df.xlsx', index=False)

#%%
file_to_save = save_path + '_stats_df.pkl'
with open(file_to_save, 'rb') as handle: 
    result_stats_df = pickle.load(handle)

#quit()
#%%
# to get min values of each selected column
min_std_row={}
list_test_ids_min=[]
number_std_min=5
for b in bands_sel:
    column_name = 'avg_std_'+b
    #print (column_name)
    min_std_row[b] = result_stats_df[result_stats_df[column_name] == result_stats_df[column_name].min()]
    list_tmp = result_stats_df.sort_values(by=column_name)[:5].index.tolist()
    print (column_name, list_tmp)
    list_test_ids_min = list(set(list_test_ids_min+list_tmp))
#%%
min_std_row_df = concat_dfs(min_std_row)
#%%
#getting the image with values normalized
time_ini=time.time()
image_band_dic_norm={}
for k in image_band_dic.keys():
    print (k)
    image_band_dic_norm[k]=image_band_dic[k].astype(float)/np.max(image_band_dic[k])
time_fim=time.time()
print (time_fim-time_ini)
img_sel_norm = np.dstack((image_band_dic_norm[bands_sel[0]], image_band_dic_norm[bands_sel[1]], image_band_dic_norm[bands_sel[2]]))

#%%
img_sel_norm = load_image_norm()
#%%
#save img_sel_norm
file_to_save = save_path + 'img_sel_norm.pkl'
save_to_pickle(img_sel_norm, file_to_save)
#%%
segments_slic_sel={}
props_df_sel={}
#%%
for id_test in tqdm([1]): #list_test_ids_min
    #id_test = 242
    n_segms = int(result_stats_df.loc[id_test]['segms']) #600
    sigma = result_stats_df.loc[id_test]['sigma']  #0.1
    compact = result_stats_df.loc[id_test]['compactness']
    conectivity= result_stats_df.loc[id_test]['connectivity'] 
    print ("id_test params: ",id_test, n_segms, sigma, compact, conectivity)
    props_df_sel[id_test], segments_slic_sel[id_test] = img_slic_segment_gen_df(bands_sel, image_band_dic, img_sel=img_sel,\
                                                    n_segms=n_segms, sigma=sigma, compactness = compact, \
                                                    mask=mask, conectivity=conectivity)
#del props_df_sel_
#%%
for i in list_test_ids_min:
    obj_dic={}
    obj_dic = {
        "segments_slic_sel": segments_slic_sel
    }
    file_to_save = save_path + '_segments_'+str(i)+'.pkl'
    save_to_pickle(obj_dic, file_to_save)
    del obj_dic
#%%
mask = (image_band_dic[bands_sel[0]] != -9999)
id_test = 314

props_df_sel_314, segments_slic_sel_314 = img_slic_segment_gen_df(bands_sel, image_band_dic, img_sel=img_sel,\
                                                    n_segms=int(n_segms), sigma=sigma, \
                                                    compactness = compact,mask=mask, conectivity=conectivity)

#%%
plot_box(props_df_sel, params_test_dic, chart_size=(900,900))
#%%
plot_images(img_sel_norm, props_df_sel, segments_slic_sel, params_test_dic, n_cols=3)    

#%%
RGB = save_img_png(file_red, file_green, file_blue)
#plot_images(RGB, props_df_sel, segments_slic_sel, params_test_dic, n_cols=3)    #nao está funcionando
plt.imshow(mark_boundaries(RGB, segments_slic_sel[id_test]))
#%% # filter higher stds in RGB,  box plot filtered and show image
std = float(2000)
id_test = 314
filtered_rows = {}
for id_test in list_test_ids_min:
    filter =(props_df_sel[id_test]['std_B11']< std) & (props_df_sel[id_test]['std_B8A']<std) & (props_df_sel[id_test]['std_B02']<std)    
    filtered_rows[id_test] = props_df_sel[id_test][filter]
#%%
plot_box(filtered_rows, params_test_dic, chart_size=(900,900))

#%%
filtered_rows_314={}
id_test = 314
filter =(props_df_sel[id_test]['std_B11']< std) & (props_df_sel[id_test]['std_B8A']<std) & (props_df_sel[id_test]['std_B02']<std)    
filtered_rows_314[id_test] = props_df_sel[id_test][filter]
##%%
plot_box(filtered_rows_314, params_test_dic, n_cols=1, chart_size=(900,900))
##%%
stdbands=['std_B11','std_B8A','std_B02', 'std_NBR', 'std_EVI', 'std_NDVI']
plot_box(props_df_sel, params_test_dic,stdbands=stdbands, chart_size=(900,900))

##%%
filter2 =(props_df_sel[id_test]['std_B11']>std) & (props_df_sel[id_test]['std_B8A']>std) & (props_df_sel[id_test]['std_B02']>std)    
filtered_rows_314[id_test] = props_df_sel[id_test][filter2]

plt.imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
#%%    # pegar o teste com o maior std 
result_stats_df[result_stats_df[column_name] == result_stats_df[column_name].min()]
id_test=1
plt.imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))

#%%
for i, row in filtered_rows_314[314].iterrows():
    plt.plot(row['centroid-1'], row['centroid-0'], marker='o',markersize=2, color=cl[0])
plt.imshow(mark_boundaries(img_sel_norm, segments_slic_sel[314]))
plt.show()
#%%  # teste de slic com imagem com NAN
#teste de substituição de -9999 nas bandas por NaN, o SLIC nao aceita e dá erro
img_b02=np.where(image_band_dic['B02'] == -9999, np.nan, image_band_dic['B02'])
img_b11=np.where(image_band_dic['B11'] == -9999, np.nan, image_band_dic['B11'])
img_b8a=np.where(image_band_dic['B8A'] == -9999, np.nan, image_band_dic['B8A'])
img_sel_nan = np.dstack((img_b11, img_b8a, img_b02))

props_df_sel1={}
segments_slic_sel1={}
for id_test in tqdm([1]): #list_test_ids_min
    #id_test = 242
    n_segms = int(result_stats_df.loc[id_test]['segms']) #600
    sigma = result_stats_df.loc[id_test]['sigma']  #0.1
    compact = result_stats_df.loc[id_test]['compactness']
    conectivity= result_stats_df.loc[id_test]['connectivity'] 
    print ("id_test params: ",id_test, n_segms, sigma, compact, conectivity)
    props_df_sel1[id_test], segments_slic_sel1[id_test] = img_slic_segment_gen_df(bands_sel, image_band_dic, img_sel=img_sel_nan,\
                                                    n_segms=n_segms, sigma=sigma, compactness = compact, \
                                                    conectivity=conectivity)
#
# se usa mask=mask funciona e retorna o mesmo resultado de usar o valor da 
#imagem com -9999 e mask:
are_equal=props_df_sel1[1].equals(props_df_sel[1])

del  props_df_sel1, segments_slic_sel1, img_b02, img_b11, img_b8a, img_sel_nan

#%% para obter valores minimo e maximo em cada banda desconsiderando -9999:
for k in image_band_dic.keys():
    print (f'{k}: {np.unique(image_band_dic[k].flatten())[1]}, {np.max(image_band_dic[k])}')
# reusltado:
# NBR: -4253, 9898
# EVI: -1934, 9616
# NDVI: -5320, 9226
# B11: 32, 5230
# B8A: 1, 6438
# B02: 99, 4989
#retorna True
#deletar as variaveis de teste acima
#%%
df_melted_props_sel = pd.melt(props_df_sel[id_test], id_vars=['label'], value_vars=['std_B11','std_B8A','std_B02'], var_name='RGB', value_name='Std RGB value')
#%%
fig = px.box(df_melted_props_sel, x='RGB', y='Std RGB value',  hover_data=['label'],
                  title=f'Num segs = {n_segms}, sigma={sigma}, compactness=5, conectivity = {conectivity}')
#%%
# # Exibir o gráfico
fig.show()
#################################
#################################
#%%
# 0. 1. 2. 3. 4.
bands_sel = ['B11', 'B8A', 'B02'] # R=B11, G=B8A, B=B02
#bands_sel = ['NBR','EVI','NDVI']
id=1
n_segms = params_test_dic[id]['segms'] #600
sigma = params_test_dic[id]['sigma']  #0.1
compact = params_test_dic[id]['compactness']
conectivity= params_test_dic[id]['connectivity'] #False
props_df_sel, img_sel, segments_slic_sel = img_slic_segment_gen_df(bands_sel, image_band_dic,\
                                                                    n_segms=n_segms, sigma=sigma, \
                                                                    compactness = compact, conectivity=conectivity)

#%%
#gerar statisticas
stats_df_dic={}
all_bands = ['B11', 'B8A', 'B02', 'NBR', 'EVI', 'NDVI']
# if id not in stats_df_dic:
#     stats_df_dic[id] = {}
stats_df_dic = gen_stats(props_df_sel, id, params_test_dic[id], all_bands)
#%%
df_melted_props_sel = pd.melt(props_df_sel, id_vars=['label'], value_vars=['std_B11','std_B8A','std_B02'], var_name='RGB', value_name='Std RGB value')
#%%
fig_600 = px.box(df_melted_props_sel, x='RGB', y='Std RGB value',  hover_data=['label'],
                  title=f'Num segs = {n_segms}, sigma={sigma}, compactness=5, conectivity = {conectivity}')
#%%
# # Exibir o gráfico
fig_600.show()
#%%
sigma=0.1
n_segms = 1000
conectivity=False
props_df_sel_1k, img_sel_1k, segments_slic_sel_1k = img_slic_segment_gen_df(bands_sel, image_band_dic,\
                                                                    n_segms=n_segms, sigma=sigma, \
                                                                    compactness = 5, conectivity=False)

#%%
df_melted_props_sel_1k = pd.melt(props_df_sel_1k, id_vars=['label'], value_vars=['std_B11','std_B8A','std_B02'], var_name='RGB', value_name='Std RGB value')
#%%
fig1k = px.box(df_melted_props_sel_1k, x='RGB', y='Std RGB value',  hover_data=['label'],
                  title=f'Num segs = {n_segms}, sigma={sigma}, compactness=5,\n conectivity = {conectivity}')

# # Exibir o gráfico
fig1k.show()
#%% # box blot das bandas RGB da imagem
##para plotar o box plot da imagem
array_resultante={}
array_resultante['B11'] = np.array(image_band_dic['B11']).flatten()
array_resultante['B8A'] = np.array(image_band_dic['B8A']).flatten()
array_resultante['B02'] = np.array(image_band_dic['B02']).flatten()

# Criar DataFrame a partir do dicionário
df_RGB = pd.DataFrame([(chave, valor) for chave, valores in array_resultante.items() for valor in valores],
                  columns=['RGB', 'Valor RGB'])

fig = px.box(df_RGB, x='RGB', y='Valor RGB',  
                  title=f'Imagem RGB')
fig.show()
del array_resultante, df_RGB
#%% usando o n_segs =600
plt.imshow(mark_boundaries(img_sel, segments_slic_sel[id_test]))
#%%
########################################################
#       Cluster                                        #
########################################################
# functions for clustering analysis
#%%
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
#%%
def calc_cor(valor, c_map='Blues'):
    colormap=plt.get_cmap(c_map)
    #colormap=plt.get_cmap('Blues')
    return colormap(valor) 

#%%
def plot_cluster_img(img_sel_norm, props_df_sel, segments_slic_sel, id_test, \
                     bands_sel, bands_sel_others, cl, n, limiar=0.7, \
                     chart_size = (12, 12)):
    # Create a figure with 2 rows and 3 columns
    #nrows = len(list_cluster)
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=chart_size)
    #for i,n in enumerate(list_cluster):
    # Original Image
    r = 0
    c = 0
    axes[r, c].imshow(img_sel_norm)
    axes[r, c].set_title(f'Original Image', fontsize=7)
    axes[r, c].axis('off')
    # Image segmented with discrete colormap
    r = 0
    c += 1
    cl = {0: 'red', 1: 'green', 2: 'blue', 3:'white', 4:'orange', 5:'yellow', 6:'magenta', 7:'cyan'}
    for i, row in props_df_sel[id_test].iterrows():
        axes[r, c].plot(row['centroid-1'], row['centroid-0'], marker='o', markersize=2, color=cl[row['cluster_'+str(n)]])
    axes[r, c].imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
    axes[r, c].set_title(f'Image segmented {id_test} discrete colormap', fontsize=7)
    axes[r, c].axis('off')
    # Image segmented with continuous colormap
    var_show=False
    if var_show:
            r = 0
            c += 1
            colormap=plt.get_cmap('viridis')
            for i, row in props_df_sel[id_test].iterrows():
                color = colormap(row['cluster_8'] / 7)
                axes[r, c].plot(row['centroid-1'], row['centroid-0'], marker='o', markersize=2, color=color)
            axes[r, c].imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
            axes[r, c].set_title(f'Image segmented clustered{id_test} continuous colormap', fontsize=7)
            axes[r, c].axis('off')
    else:
        r = 0
        c += 1
        # bands_sel = ['B11','B8A','B02']
        # bands_sel_others = ['NBR','EVI','NDVI']
        maxBand = props_df_sel[id_test][['std_'+band for band in bands_sel]].max().max()
        std_threshold_RGB = limiar*maxBand
        maxBandOther = props_df_sel[id_test][['std_'+band for band in bands_sel_others]].max().max()
        std_threshold_Other = limiar*maxBandOther
        for i, row in props_df_sel[id_test].iterrows():
            #color = colormap(row['cluster_8'] / 7)
            if (row['std_B11'] >=std_threshold_RGB)|(row['std_B8A']>=std_threshold_RGB)|(row['std_B02']>=std_threshold_RGB):
                color=cl[0]
                axes[r, c].plot(row['centroid-1'], row['centroid-0'], marker='o', markersize=2, color=cl[0])
                
            elif (row['std_NBR'] >=std_threshold_Other)|(row['std_EVI']>=std_threshold_Other)|(row['std_NDVI']>=std_threshold_Other):
                color=cl[2]
                axes[r, c].plot(row['centroid-1'], row['centroid-0'], marker='o', markersize=2, color=color)
        axes[r, c].imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
        axes[r, c].set_title(f'Image segmented clustered{id_test} with std>70%*maxStdBand', fontsize=7)
        axes[r, c].axis('off')
    for cc in range(0,3):
        axes[1,cc].axis('off')
        axes[1,cc].set_visible(False)
    plt.tight_layout()
    plt.show()
#%%
def plot_clusters_img(img_sel_norm, props_df_sel, segments_slic_sel, id_test, \
                     bands_sel, bands_sel_others, cl,list_cluster, limiar=0.7, \
                     chart_size = (12, 12)):
    ''''
    function to plot in each line the original image and the clustered one
    '''
    # Create a figure with 2 rows and 3 columns
    
    #fig, axes = plt.subplots(nrows=len(list_cluster), ncols=3, sharex=True, sharey=True, figsize=chart_size)
    fig, axes = plt.subplots(nrows=len(list_cluster), ncols=3, figsize=chart_size)
    cl = {0: 'red', 1: 'green', 2: 'blue', 3:'white', 4:'orange', 5:'yellow', 6:'magenta', 7:'cyan'}
    for i,n in enumerate(list_cluster):
        # Original Image
        r = i
        c = 0
        axes[r, c].imshow(img_sel_norm)
        axes[r, c].set_title(f'Original Image', fontsize=7)
        axes[r, c].axis('off')

        # Image segmented with discrete colormap
        #r = 0
        c += 1
        col_cluster='cluster_'+str(n)
        
        for ii, row in props_df_sel[id_test].iterrows():
            axes[r, c].plot(row['centroid-1'], row['centroid-0'], marker='o', markersize=2, color=cl[row['cluster_'+str(n)]])
        axes[r, c].imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
        axes[r, c].set_title(f'Image segmented {id_test} {n} clusters', fontsize=7)
        axes[r, c].axis('off')

        ##### make boxplot
        #for i, id_test in enumerate(tests):
        # data1 = props_df_sel[id_test]['std_B11']
        # data2 = props_df_sel[id_test]['std_B8A']
        # data3 = props_df_sel[id_test]['std_B02']
        #fig.add_trace(px.box(y=[data1,data2,data3]).data[0],  row=r, col=c)
        
        stdbands = ['std_'+band for band in bands_sel_others]
        df = props_df_sel[id_test][stdbands+[col_cluster]]
        #df_melted = pd.melt(df, id_vars=[col_cluster], value_vars=stdbands, var_name='std_band')
        
        c+=1
        df.boxplot(column=['std_NBR', 'std_NDVI', 'std_EVI'], by=col_cluster, ax=axes[r,c], grid=False)

        #box = px.box(df, x=col_cluster, y=[stdbands], points="all", color=col_cluster, facet_col='variable')
        #box = px.box(df_melted, x='std_band', y='value', color=col_cluster, points="all", facet_col=col_cluster)
        # fig.add_trace(box.data[0],  row=r, col=c)
        # fig.update_traces(showlegend=True, legendgroup=id_test, name=id_test, row=r, col=c)
        # fig.update_layout(showlegend=True, title_font=dict(size=10), width=chart_size[0], height=chart_size[1])
        
    # Update subtitle font size for all subplots
    # for annotation in fig['layout']['annotations']:
    #     annotation['font'] = dict(size=10)

        ######
        # Image segmented with continuous colormap

        ''''
        #var_show=False
        # if var_show:
        #     r = 0
        #     c += 1
        #     colormap=plt.get_cmap('viridis')
        #     for i, row in props_df_sel[id_test].iterrows():
        #         color = colormap(row['cluster_8'] / 7)
        #         axes[r, c].plot(row['centroid-1'], row['centroid-0'], marker='o', markersize=2, color=color)
        #     axes[r, c].imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
        #     axes[r, c].set_title(f'Image segmented clustered{id_test} continuous colormap', fontsize=7)
        #     axes[r, c].axis('off')
        # else:
        #     r = 0
        #     c += 1
        #     # bands_sel = ['B11','B8A','B02']
        #     # bands_sel_others = ['NBR','EVI','NDVI']
        #     maxBand = props_df_sel[id_test][['std_'+band for band in bands_sel]].max().max()
        #     std_threshold_RGB = limiar*maxBand
        #     maxBandOther = props_df_sel[id_test][['std_'+band for band in bands_sel_others]].max().max()
        #     std_threshold_Other = limiar*maxBandOther
        #     for i, row in props_df_sel[id_test].iterrows():
        #         #color = colormap(row['cluster_8'] / 7)
        #         if (row['std_B11'] >=std_threshold_RGB)|(row['std_B8A']>=std_threshold_RGB)|(row['std_B02']>=std_threshold_RGB):
        #             color=cl[0]
        #             axes[r, c].plot(row['centroid-1'], row['centroid-0'], marker='o', markersize=2, color=cl[0])
                    
        #         elif (row['std_NBR'] >=std_threshold_Other)|(row['std_EVI']>=std_threshold_Other)|(row['std_NDVI']>=std_threshold_Other):
        #             color=cl[2]
        #             axes[r, c].plot(row['centroid-1'], row['centroid-0'], marker='o', markersize=2, color=color)
        #     axes[r, c].imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
        #     axes[r, c].set_title(f'Image segmented clustered{id_test} with std>70%*maxStdBand', fontsize=7)
        #     axes[r, c].axis('off')
        '''

    # for cc in range(0,3):
    #     axes[r,cc].axis('off')
    #     axes[r,cc].set_visible(False)
    plt.tight_layout()
    plt.show()

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
plot_clusters_img(img_sel_norm, props_df_sel, segments_slic_sel, id_test, \
                     bands_sel, bands_sel_others, cl,[2,3,4,5,6,7,8], limiar=0.7, \
                     )  

#%%
def plot_images_cluster(img_sel_norm,props_df_sel, segments_slic_sel, id_test,\
                        list_cluster, n_cols=3, chart_size=(12, 12)):
    ''''
    funcion to plot clustered images in cols and rows with the original in 
    first position
    '''
    #tests = list(props_df_sel.keys())
    #n_cols = 3
    num_plots=len(list_cluster)+1
    n_rows = np.ceil(num_plots/n_cols).astype(int)
    sub_titles=[]
    for k in list_cluster:
        # segms=f"{params_test_dic[k]['segms']}/{props_df_sel[k].shape[0]}"
        # compact=params_test_dic[k]['compactness']
        # sigma=params_test_dic[k]['sigma']
        # conect=params_test_dic[k]['connectivity']
        subt=f'cluster_{k}'
        sub_titles.append(subt)
    
    cl = {0: 'red', 1: 'green', 2: 'blue', 3:'white', 4:'orange', \
          5:'yellow', 6:'magenta', 7:'cyan'}
 
    cl = plt.get_cmap('tab20')
    fig,axes = plt.subplots(nrows=int(n_rows), ncols=n_cols, sharex=True, sharey=True,figsize=chart_size)
    #axes = axes.flatten()

    for i, n in enumerate([0]+list_cluster):
                
        #df = props_df_sel[id_test][['std_B11','std_B8A','std_B02']]
        r = (i//n_cols)# + 1-1
        c = (i%n_cols)#+1-1
        #print (r,c)
        if (r==0) & (i==0):
            axes[r, c].imshow(img_sel_norm)
            axes[r, c].set_title(f'Original Image', fontsize=7)
            axes[r, c].axis('off')
            continue

        #axes[r,c].imshow(mark_boundaries(img_sel, segments_slic_sel[id_test]))
        
        for ii, row in props_df_sel[id_test].iterrows():
            axes[r, c].plot(row['centroid-1'], row['centroid-0'], marker='o', markersize=2, color=cl(row['cluster_'+str(n)]))
        axes[r, c].imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
        axes[r, c].set_title(f'Image segmented {id_test} {n} clusters', fontsize=7)
               
        # Customize subplot title
        #axes[r,c].set_title(sub_titles[i], fontsize=7)

        # Hide axis ticks and labels
        axes[r,c].axis('off')
        
        #print (i, id_test. r,c)
        #box = px.box(df)
        #fig.add_trace(box.data[0],  row=r, col=c)
        #fig.update_traces(showlegend=True, legendgroup=id_test, name=id_test, row=r, col=c)
    
    # tem que setar visible to false the subs plots to complement the grid
    # of subplots
    num_subs = n_cols*n_rows
    if (num_plots)< num_subs:
        for cc in range(c+1,n_cols):
            #print (r,cc)
            axes[r,cc].axis('off')
            axes[r,cc].set_visible(False)

    #fig.update_layout(showlegend=True, title_font=dict(size=10), width=chart_size[0], height=chart_size[1])

    # # Update subtitle font size for all subplots
    # for annotation in fig['layout']['annotations']:
    #     annotation['font'] = dict(size=10)

    # Adjust layout
    plt.tight_layout()
    plt.show()
    #fig.show()

#%%
def plot_images_cluster_clara(img_sel_norm,filter_centroid_sel_df,  id_test,\
                        list_clara,label_value, n_cols=3, cl_map='Blues', chart_size=(12, 12)):
    ''''
    funcion to plot clustered images in cols and rows with the original in 
    first position
    '''
    #tests = list(props_df_sel.keys())
    #n_cols = 3
    num_plots=len(list_clara)+1
    n_rows = np.ceil(num_plots/n_cols).astype(int)
    sub_titles=[]
    for k in list_clara:
        # segms=f"{params_test_dic[k]['segms']}/{props_df_sel[k].shape[0]}"
        # compact=params_test_dic[k]['compactness']
        # sigma=params_test_dic[k]['sigma']
        # conect=params_test_dic[k]['connectivity']
        subt=f'{k}'
        sub_titles.append(subt)
    
    #cl = {0: 'red', 1: 'green', 2: 'blue', 3:'white', 4:'orange', \
    #      5:'yellow', 6:'magenta', 7:'cyan'}
 
    cl = plt.get_cmap(cl_map)      # 'tab20'
    fig,axes = plt.subplots(nrows=int(n_rows), ncols=n_cols, sharex=True, sharey=True,figsize=chart_size)
    #axes = axes.flatten()

    for i, n in enumerate(list_clara):
                
        #df = props_df_sel[id_test][['std_B11','std_B8A','std_B02']]
        r = (i//n_cols)# + 1-1
        c = int(i%n_cols)#+1-1
        #print (r,c)
        if (r==0) & (i==0):
            axes[r, c].imshow(img_sel_norm)
            axes[r, c].set_title(f'Original Image', fontsize=7)
            axes[r, c].axis('off')
            continue

        #axes[r,c].imshow(mark_boundaries(img_sel, segments_slic_sel[id_test]))
        
        for ii, row in filter_centroid_sel_df[n].iterrows():
            if ii == label_value-1:
                marker = 's'
                markersz = 2
                cor='red'     
                #print (row['centroid-1'], row['centroid-0'], marker,markersz, cor)           
            else:
                marker = 'o'
                markersz=1
                cor=cl(row['sim_value'])
            axes[r,c].plot(row['centroid-1'], row['centroid-0'], marker=marker,markersize=markersz, color=cor)#row['cor'])
            #axes[r, c].plot(row['centroid-1'], row['centroid-0'], marker='o', markersize=2, color=cl(row['cluster_'+str(n)]))
            for p in row['coords']:
               axes[r,c].plot(p[1], p[0], marker=marker,markersize=markersz, color=cor)#row['cor'])
        axes[r,c].imshow(img_sel_norm)
        #axes[r, c].imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
        axes[r, c].set_title(f'Image segmented {id_test} {k}', fontsize=7)
               
        # Customize subplot title
        #axes[r,c].set_title(sub_titles[i], fontsize=7)

        # Hide axis ticks and labels
        axes[r,c].axis('off')
        
        #print (i, id_test. r,c)
        #box = px.box(df)
        #fig.add_trace(box.data[0],  row=r, col=c)
        #fig.update_traces(showlegend=True, legendgroup=id_test, name=id_test, row=r, col=c)
    
    # tem que setar visible to false the subs plots to complement the grid
    # of subplots
    num_subs = n_cols*n_rows
    if (num_plots)< num_subs:
        for cc in range(c+1,n_cols):
            #print (r,cc)
            axes[r,cc].axis('off')
            axes[r,cc].set_visible(False)

    #fig.update_layout(showlegend=True, title_font=dict(size=10), width=chart_size[0], height=chart_size[1])

    # # Update subtitle font size for all subplots
    # for annotation in fig['layout']['annotations']:
    #     annotation['font'] = dict(size=10)

    # Adjust layout
    plt.tight_layout()
    plt.show()
    #fig.show()

#%%
def plot_clustered_clara(img_sel_norm, n_opt_df, list_cluster_n_opt, \
                         n_cols=3, cl_map='tab20', plot_orig_img=1, chart_size=(12, 12)):
    ''''
    funcion to plot clustered images in cols and rows with the original in 
    first position
    '''
    img_sel_norm = np.clip(img_sel_norm, 0, 1) # fazer isso para remover valores negativos 
    if plot_orig_img:
        elemento='original_image'
        list_cluster_n_opt = np.insert(list_cluster_n_opt, 0, elemento)
    
    num_plots=len(list_cluster_n_opt)
    n_rows = np.ceil(num_plots/n_cols).astype(int)
    
    print (f'num plots = {num_plots} , n_rows = {n_rows}')
    
    sub_titles=[]
    for k in list_cluster_n_opt:
        # segms=f"{params_test_dic[k]['segms']}/{props_df_sel[k].shape[0]}"
        # compact=params_test_dic[k]['compactness']
        # sigma=params_test_dic[k]['sigma']
        # conect=params_test_dic[k]['connectivity']
        subt=f'{k}'
        sub_titles.append(subt)
    
    #cl = {0: 'red', 1: 'green', 2: 'blue', 3:'white', 4:'orange', \
    #      5:'yellow', 6:'magenta', 7:'cyan'}
 
    cl = plt.get_cmap(cl_map)      # 'tab20'
    fig,axes = plt.subplots(nrows=int(n_rows), ncols=n_cols, sharex=True, sharey=True,figsize=chart_size)
    
    #axes = axes.flatten()
  
    x_centroids = n_opt_df['centroid-1']
    y_centroids = n_opt_df['centroid-0']

    for i, n in enumerate(list_cluster_n_opt):
                
        #df = props_df_sel[id_test][['std_B11','std_B8A','std_B02']]
        r = (i//n_cols)# + 1-1
        c = int(i%n_cols)#+1-1

        if n_rows==1:
            ax = axes[c]
        else:
            ax = axes[r,c]
        print (r,c)
        if (r==0) & (i==0):
            ax.imshow(img_sel_norm)
            ax.set_title(f'Original Image', fontsize=7)
            ax.axis('off')
            continue

        colors=[cl(x) for x in n_opt_df[n]]
        ax.scatter(x_centroids, y_centroids, s=1, color=colors)

        #axes[r,c].imshow(mark_boundaries(img_sel, segments_slic_sel[id_test]))
        
        # for ii, row in filter_centroid_sel_df[n].iterrows():
        #     if ii == label_value-1:
        #         marker = 's'
        #         markersz = 2
        #         cor='red'     
        #         #print (row['centroid-1'], row['centroid-0'], marker,markersz, cor)           
        #     else:
        #         marker = 'o'
        #         markersz=1
        #         cor=cl(row['sim_value'])
        #     axes[r,c].plot(row['centroid-1'], row['centroid-0'], marker=marker,markersize=markersz, color=cor)#row['cor'])
        #     #axes[r, c].plot(row['centroid-1'], row['centroid-0'], marker='o', markersize=2, color=cl(row['cluster_'+str(n)]))
        #     for p in row['coords']:
        #        axes[r,c].plot(p[1], p[0], marker=marker,markersize=markersz, color=cor)#row['cor'])
        ax.imshow(img_sel_norm)
        #axes[r, c].imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
        ax.set_title(f'Image clustered n_opt {n}', fontsize=7)
               
        # Customize subplot title
        #axes[r,c].set_title(sub_titles[i], fontsize=7)

        # Hide axis ticks and labels
        ax.axis('off')
        
        #print (i, id_test. r,c)
        #box = px.box(df)
        #fig.add_trace(box.data[0],  row=r, col=c)
        #fig.update_traces(showlegend=True, legendgroup=id_test, name=id_test, row=r, col=c)
    
    # tem que setar visible to false the subs plots to complement the grid
    # of subplots
    num_subs = n_cols*n_rows
    if (num_plots)< num_subs:
        for cc in range(c+1,n_cols):
            #print (r,cc)
            ax.axis('off')
            ax.set_visible(False)

    #fig.update_layout(showlegend=True, title_font=dict(size=10), width=chart_size[0], height=chart_size[1])

    # # Update subtitle font size for all subplots
    # for annotation in fig['layout']['annotations']:
    #     annotation['font'] = dict(size=10)

    # Adjust layout
    plt.tight_layout()
    plt.show()
    #fig.show()


#%%
def plot_box_clusters(props_df_sel, id_test, list_cluster, stdbands):
    ''''
    plot box plot of bands per subgroup in cluster
    '''
    # Create grouped box plots for each group
    for group in list_cluster:
        col_cluster='cluster_'+str(group)
        # Filter data for the current group
        df = props_df_sel[id_test][stdbands+[col_cluster]]
        df_melted = pd.melt(df, id_vars=[col_cluster], value_vars=stdbands, var_name='std_band')

        filtered_data = df_melted[df_melted[col_cluster].isin(range(group))]

        # Create a grouped box plot
        fig = px.box(filtered_data, x='std_band', y='value', color=col_cluster, points="all", facet_col=col_cluster, facet_col_wrap=3)
        
        # Update layout
        fig.update_layout(title=f'Group {group} - Boxplot das bandas por clusters', showlegend=False)
        del df, df_melted, filtered_data
        # Show the figure
        fig.show()
#%%
plot_box_clusters(props_df_sel, id_test, list_cluster, stdbands)
######
#%%
list_cluster=[2,3,4,5,6,7,8]
list_cluster2=[x for x in range(2,41)]
plot_images_cluster(img_sel_norm,props_df_sel, segments_slic_sel, id_test,\
                        list_cluster2, n_cols=3, chart_size=(12, 12))
#%% # ler arquivos props selecionados
id_test=3
ids = [id_test]
#%%
props_df_sel=read_props_df_sel(ids, save_path+"props_df_sel")
#%%
segments_slic_sel=read_props_df_sel(ids,save_path, obj_to_read='segments_slic_sel')
#%%
# 5. Fazer cluster da imagem
# seleciona as bandas que Vão ser usadas na clusterizacao e converte para numpy 
# e depois para list
 # 304, 302
#bands_to_cluster = ['NBR','EVI','NDVI']
bands_to_cluster = ['B08','EVI','NDVI'] #new image
#arraybands_sel = props_df_sel[['NBR','EVI','NDVI']].to_numpy()
arraybands_sel = props_df_sel[id_test][bands_to_cluster].to_numpy()
arraybands_list_sel = arraybands_sel.tolist()
#%%
import random
from tqdm import tqdm
random.seed(999) # para gerar sempre com a mesma seed

#%%
dic_cluster = {}
#%% #CLARA

from sklearn_extra.cluster import CLARA
#from sklearn.datasets import make_blobs

#for n in range (2, n_clusters+1):
    
#X, _ = make_blobs(centers=[[0,0],[1,1]], n_features=2,random_state=0)
#%%
n_clusters=30 
dic_cluster={}
sse=[]
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
#%%
n_opt = optimal_number_of_clusters(sse, 2, n_clusters)
n_opt
#%%
plt.plot(range(2,n_clusters+1),sse)

#plt.plot([2, sse[0]], [n_clusters, sse[-1]], color='red', linestyle='--')
plt.plot([2, n_clusters], [sse[0], sse[-1]], 'r--', label='Linha de Referência')
#plt.xsticks(range(2,n_clusters+1))
plt.show()
#%% #CLARANS
#CLARANS
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

obj_dic={}
obj_dic[id_test] = {
    "props_df_sel_cluster": props_df_sel,
    "dic_labels_cluster": dic_cluster,
    "sse": sse,
    "n_opt": n_opt
}
file_to_save = save_path + 'cluster_'+str(id_test)+'.pkl'
save_to_pickle(obj_dic, file_to_save)
#%%
# ler os arquivos para criar 
# com "props_df_sel" tem o props_df_sel sem cluster
# com "cluster_" tem o props_df_sel com cluster

ids=[3]
#%%
props_df_sel = read_props_df_sel(ids, save_path+"props_df_sel")
#%%
props_df_sel = read_props_df_sel(ids, save_path+"cluster", obj_to_read="cluster")
#%%
# to read the dic cluster and n_opt
id_test=3
file_to_open = save_path+"cluster" + '_'+str(3)+'.pkl'
with open(file_to_open, 'rb') as handle: 
    b = pickle.load(handle)

dic_cluster={}
dic_cluster = b[id_test]['dic_labels_cluster']

n_opt=b[id_test]['n_opt']
#%%
#read the label and coords
label_coords={}
file_to_open = save_path+"props_dic_sel_labels_coords" + '_'+str(3)+'.pkl'
with open(file_to_open, 'rb') as handle: 
    b = pickle.load(handle)
#%%f
coords = b[3]['props_df_sel_label_pixels']['coords']
#labels = b[3]['props_df_sel_label_pixels']['label']

#%%
matrix_sim={}
#%%
n_clusters= list(range(2,ceil(n_opt*1.2)+1))
time_ini = time.time()
matrix_sim[id_test] = cria_SimilarityMatrix_freq(dic_cluster, n_clusters)
time_fim = time.time()
print (time_fim-time_ini)

#%%
# usar dask array pq com numpy está crashando na nova imagem
import dask.array as da
from math import ceil

n_ini=2
n_selected = list(range(n_ini,ceil(n_opt*1.2)+1))
#n_selected = [2,3,4]
time_ini = time.time()
for i, n in enumerate(n_selected):
    da_arr = da.from_array(np.array(dic_cluster[n]),chunks=10000)
    if (i==0):
        matrix_sim = (da_arr[:, None] == da_arr[None, :]).astype(int)
        continue
    matrix_sim = (da_arr[:, None] == da_arr[None, :]).astype(int)+matrix_sim
matrix_sim=matrix_sim/len(n_selected)    
time_fim = time.time()
print (time_fim-time_ini)

#%%
centroid_sel=5205
filter_centroid_sel_df = get_df_filter(centroid_sel, props_df_sel,\
                                        matrix_sim,id_test, thr_min=0.85, thr_max=1)

#%%
centroid_sel_df = props_df_sel[id_test][['label', 'centroid-0','centroid-1']]
#%%
import dask.dataframe as dd 
dask_centroid_sel_df = dd.from_pandas(centroid_sel_df,npartitions=11)

#%%
centroid_sel=7605    
centroid_row_matrix_sim = matrix_sim[centroid_sel-1,:]#.compute()
#%%    
dask_centroid_sel_df['sim_value'] = centroid_row_matrix_sim

#%%
#filter just the sim values higher than threshold
#threshold = 0.70
thr_min=0.85
thr_max=1
filter_centroid_sel_df = dask_centroid_sel_df[(dask_centroid_sel_df['sim_value']>=thr_min) & 
                                              (dask_centroid_sel_df['sim_value']<=thr_max)]

#%%
filter_centroid_sel_df['cor'] = filter_centroid_sel_df['sim_value'].apply(calc_cor, 'Blues')
#%%
filter_centroid_sel_df.loc[centroid_sel-1,'cor']='red'

#%%
plot_cluster_img_pixel_sel_faster(filter_centroid_sel_df, centroid_sel,plot_centroids=True)

#%%
#teste de achar pixel no coords e depois o correspondente label
pos_to_find = [0,1]
time_ini=time.time()
pos_to_find=[641,450]
for i, sub in enumerate(coords):

    is_found=any(np.array_equal(pos_to_find, c) for c in sub)
    if is_found:
        label_sp = i+1
        break
time_fim=time.time()
print (time_fim-time_ini, label_sp)

#%%
centroid_row_matrix_sim = matrix_sim[label_sp-1,:]

#%%
#save with hdf5 Ainda nao está funcionando ...
hdf5_path='/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/test_segm_results/S2-16D_V2_012014_20220728_/matrix_sim/'
#%%
#matrix_sim.to_hdf5(hdf5_path+'matrix_sim.hdf5','/matrix_sim') nao funcionou
#da.to_npy_stack(hdf5_path, matrix_sim) nao funcionou
# Salvando cada bloco individualmente
import h5py
#%%
with h5py.File("mytestfile.hdf5", "w") as f:
    dset = f.create_dataset("mydataset", matrix_sim, dtype='float64')

#%%
obj_dic={}
obj_dic[id_test] = {
    "props_df_sel_cluster": props_df_sel,
    "dic_labels_cluster": dic_cluster,
    "sse": sse,
    "n_opt": n_opt,
    "matrix_sim":matrix_sim
}
file_to_save = save_path + '_cluster_clara_20_5iter_'+str(id_test)+'.pkl'
save_to_pickle(obj_dic, file_to_save)
#%%
#atribui a cor nao precisa fazer isso
#props_df_sel['color'] = props_df_sel.apply(lambda row: cl[row['cluster_3']], axis=1)
#%% # read file of id_test with cluster info
file_to_open = save_path + '_cluster_clara_20_5iter_'+str(id_test)+'.pkl'
with open(file_to_open, 'rb') as handle: 
    b_props_cluster = pickle.load(handle)


props_df_sel[id_test]=b_props_cluster['props_df_sel_cluster'][id_test]
matrix_sim={}
matrix_sim[id_test]=b_props_cluster['matrix_sim'][id_test]
#%%
file_to_open = save_path + '_segments_'+str(id_test)+'.pkl'
with open(file_to_open, 'rb') as handle: 
    b = pickle.load(handle)
segments_slic_sel[id_test]= b['segments_slic_sel'][id_test]

#%%
cl = {0: 'red', 1: 'green', 2: 'blue', 3:'white', 4:'orange', 5:'yellow', 6:'magenta', 7:'cyan'}
#%% # show original image, segmented image , segmented image with std >limiar clustered
bands_sel_others = ['NBR','EVI','NDVI']
plot_cluster_img(img_sel_norm, props_df_sel, segments_slic_sel, id_test,\
                  bands_sel, bands_sel_others, cl, 8)

#%%  #code for plot the same as it is in plot_cluster_img()
# Create a figure with 2 rows and 3 columns
chart_size = (12, 12)
fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=chart_size)

# Original Image
r = 0
c = 0
axes[r, c].imshow(img_sel_norm)
axes[r, c].set_title(f'Original Image', fontsize=7)
axes[r, c].axis('off')

# Image segmented with discrete colormap
r = 0
c += 1
cl = {0: 'red', 1: 'green', 2: 'blue', 3:'white', 4:'orange', 5:'yellow', 6:'magenta', 7:'cyan'}
for i, row in props_df_sel[id_test].iterrows():
    axes[r, c].plot(row['centroid-1'], row['centroid-0'], marker='o', markersize=2, color=cl[row['cluster_8']])
axes[r, c].imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
axes[r, c].set_title(f'Image segmented {id_test} discrete colormap', fontsize=7)
axes[r, c].axis('off')

# Image segmented with continuous colormap

var_show=False
if var_show:
    r = 0
    c += 1
    colormap=plt.get_cmap('viridis')
    for i, row in props_df_sel[id_test].iterrows():
        color = colormap(row['cluster_8'] / 7)
        axes[r, c].plot(row['centroid-1'], row['centroid-0'], marker='o', markersize=2, color=color)
    axes[r, c].imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
    axes[r, c].set_title(f'Image segmented clustered{id_test} continuous colormap', fontsize=7)
    axes[r, c].axis('off')
else:
    r = 0
    c += 1
    #bands_sel = ['B11','B8A','B02']
    
    maxBand = props_df_sel[id_test][['std_'+band for band in bands_sel]].max().max()
    std_threshold_RGB = 0.7*maxBand
    maxBandOther = props_df_sel[id_test][['std_'+band for band in bands_sel_others]].max().max()
    std_threshold_Other = 0.7*maxBandOther
    for i, row in props_df_sel[id_test].iterrows():
        #color = colormap(row['cluster_8'] / 7)
        if (row['std_B11'] >=std_threshold_RGB)|(row['std_B8A']>=std_threshold_RGB)|(row['std_B02']>=std_threshold_RGB):
            color=cl[0]
            axes[r, c].plot(row['centroid-1'], row['centroid-0'], marker='o', markersize=2, color=cl[0])
            
        elif (row['std_NBR'] >=std_threshold_Other)|(row['std_EVI']>=std_threshold_Other)|(row['std_NDVI']>=std_threshold_Other):
            color=cl[2]
            axes[r, c].plot(row['centroid-1'], row['centroid-0'], marker='o', markersize=2, color=color)
    axes[r, c].imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
    axes[r, c].set_title(f'Image segmented clustered{id_test} with std>70%*maxStdBand', fontsize=7)
    axes[r, c].axis('off')

for cc in range(0,3):
    axes[1,cc].axis('off')
    axes[1,cc].set_visible(False)
plt.tight_layout()
plt.show()


#%% #show segmented image using a discrete colormap
for i, row in props_df_sel[id_test].iterrows():
    plt.plot(row['centroid-1'], row['centroid-0'], marker='o',markersize=2, color=cl[row['cluster_8']])
plt.imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
plt.show()

#%% #show segmented image using a continuos colormap
colormap=plt.get_cmap('viridis')
for i, row in props_df_sel[id_test].iterrows():
    c=colormap(row['cluster_8']/7)
    plt.plot(row['centroid-1'], row['centroid-0'], marker='o',markersize=2, color=c)
plt.imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))

#%% #show segmented image of std of one band using a continuos colormap
stdBand = 'std_B11'
maxBand = props_df_sel[id_test][stdBand].max()
minBand = props_df_sel[id_test][stdBand].min()
for i, row in props_df_sel[id_test].iterrows():
    c=colormap(row[stdBand]/maxBand)
    plt.plot(row['centroid-1'], row['centroid-0'], marker='o',markersize=2, color=c)

img_plot = plt.imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
##%% #add colorbar for reference
# Add colorbar for reference
# Create a colorbar axis
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.1)

norm = plt.Normalize(minBand, maxBand)  # Adjust the range based on your 'cluster_8' values
sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])  # Dummy array for the ScalarMappable
cbar = plt.colorbar(sm,cax=cax, orientation='vertical', label=stdBand)
# Set colorbar ticks and format
cbar.locator = MaxNLocator(nbins=5)  # Adjust the number of ticks as needed
cbar.formatter = FormatStrFormatter('%d')  # Format ticks as integers

# Update ticks
cbar.update_ticks()
plt.show()
#%%
def plot_imgSeg_std(img_sel_norm, props_df_sel,segments_slic_sel, id_test,\
                    bands_sel, colorm='YlOrRd'):
    #plot in 3 columns img std RGB
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=chart_size)

    # RB11 band
    r = 0
    #c = 0
    #bands_sel = ['B11', 'B8A', 'B02']
    maxBand = props_df_sel[id_test][['std_'+band for band in bands_sel]].max().max()
    minBand = props_df_sel[id_test][['std_'+band for band in bands_sel]].min().min()    
    colormap=plt.get_cmap(colorm)#'viridis', 'Wistia', 'cool', 'bwr','YlOrRd'
    # colormaps in https://matplotlib.org/stable/users/explain/colors/colormaps.html
    for c, band in enumerate(bands_sel):
        stdBand = 'std_'+band
        # maxBand = props_df_sel[id_test][stdBand].max()
        # minBand = props_df_sel[id_test][stdBand].min()    
        
        for i, row in props_df_sel[id_test].iterrows():
            color=colormap(row[stdBand]/maxBand)
            axes[r,c].plot(row['centroid-1'], row['centroid-0'], marker='o',markersize=2, color=color)
            
        img_plot=axes[r,c].imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
        axes[r,c].set_title(f'Image segmented {id_test} {stdBand}', fontsize=7)
        axes[r,c].axis('off')
        

    for cc in range(0,3):
        axes[1,cc].axis('off')
        axes[1,cc].set_visible(False)


    plt.tight_layout()
    plt.show()
#%%
plot_imgSeg_std(img_sel_norm, props_df_sel,segments_slic_sel, id_test,\
                    bands_sel, colorm='hot')

#%%
#plot img, img with cluster, box plot of band_sel_other for each group of cluster

#########
#%%
# Create a grid for the subplot with two rows and one column
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05], hspace=0.05)

# Set the colorbar in the second row
cbar = plt.colorbar(img_plot, cax=plt.subplot(gs[1]), orientation='horizontal', label='Cluster')

# Show the plot
plt.show()
#%%
plt.imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
#plt.imshow(img_sel_norm)
#%%
#criar uma visualizacao onde se passa um centroid e todos os centroids que tem um valor de similaridademaior que 0,75
#sao colocados na mesma cor na imagem
#criar um dataframe para o ponto a ser mostrado
centroid_sel = 5728#40 #14 #16, 11
#%%


#%%
#nao preciso fazer isso:
def set_value(row):
    if row['sim_value'] >= 0.85:
        return 'orange'
    else:
        return 'white'
#%%
# Apply the custom function to create a new column 'NewColumn'
centroid_sel_df['cor'] = centroid_sel_df.apply(set_value, axis=1)
centroid_sel_df.loc[centroid_sel,'cor']='red'
#%%
def plot_cluster_img_pixel_sel(filter_centroid_sel_df, centroid_sel, cl_map='Blues'):
    ''''
    plot similares clusters of pixel selected in image using iterrows
    é mais lento
    '''
    colormap=plt.get_cmap(cl_map)
    time_ini = time.time()
    for i, row in filter_centroid_sel_df.iterrows():
        if i == centroid_sel-1:
            #print ("centroid=",i, row['label'], row['cor'])
            marker = 's'
            markersz = 2
            c='red'
        else:
            marker = 'o'
            markersz=1
            c=colormap(row['sim_value'])
        plt.plot(row['centroid-1'], row['centroid-0'], marker=marker,markersize=markersz, color=c)#row['cor'])
        x_pixels = [p[1] for p in row['coords']]
        y_pixels = [p[0] for p in row['coords']]
        plt.plot(x_pixels, y_pixels, marker=marker,markersize=markersz, color=c)#row['cor'])
        # for p in row['coords']:
        #     plt.plot(p[1], p[0], marker=marker,markersize=markersz, color=c)#row['cor'])
    time_fim = time.time()
    print (time_fim-time_ini)
    #to show image with segmentation
    #plt.imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
    plt.imshow(img_sel_norm)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

#%%
def plot_cluster_img_pixel_sel_faster(img_sel_norm, filter_centroid_sel_df, \
                                      centroid_sel, \
                                      cl_map='Blues',simple=True, 
                                      plot_centroids=True):
    ''''
    plot similares clusters of pixel selected in image
    '''
    # simple = true plot only cluster, else plot according to sim matrix
    if simple:
        filter_centroid_sel_df['cor'] ='blue'
    else:
        filter_centroid_sel_df['cor'] = filter_centroid_sel_df['sim_value'].apply(calc_cor, 'Blues')
    filter_centroid_sel_df.loc[centroid_sel-1,'cor']='red'
    
    time_ini = time.time()    
    
    if (plot_centroids):
        x_centroids=[x for x in filter_centroid_sel_df['centroid-1']]
        y_centroids=[y for y in filter_centroid_sel_df['centroid-0']]
        plt.scatter(x_centroids, y_centroids,s=1, color=list(filter_centroid_sel_df['cor']))
        
        #para plotar o ponto selecionado como vermelho
        c1=filter_centroid_sel_df['label']== centroid_sel  #label_value-1
        x_label = filter_centroid_sel_df.loc[c1,'centroid-1']
        y_label = filter_centroid_sel_df.loc[c1,'centroid-0']
        
        plt.scatter([x_label], [y_label],s=10, color='red')#list(filter_centroid_sel_df['cor']))

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
colormap=plt.get_cmap('Blues')
filter_centroid_sel_df = get_df_filter(centroid_sel, props_df_sel, id_test, thr_min=0.85)
#%%
plot_cluster_img_pixel_sel(filter_centroid_sel_df, centroid_sel)
#%%
#plot clara filtered pixel selected

#%% # read file of id_test with cluster info
props_df_sel_clara={}
matrix_sim_clara={}
#%%
#clara_test = '_cluster_clara_200_'
#clara_tests = ['_cluster_clara_','_cluster_clara_200_', '_cluster_clara_400_5iter_']
#for new image 27/02/2024
clara_tests = ['_cluster_clara_','_cluster_clara_matrixsim_nopt_']
clara_tests =['_clara_40_5iter_']
for cla in clara_tests:

    file_to_open = save_path + cla +str(id_test)+'.pkl'
    print (file_to_open)
    with open(file_to_open, 'rb') as handle: 
        b_props_cluster = pickle.load(handle)

    if cla not in props_df_sel_clara:
        props_df_sel_clara[cla]={}
    props_df_sel_clara[cla][id_test]=b_props_cluster['props_df_sel_cluster'][id_test]
    
    if cla not in matrix_sim_clara:
        matrix_sim_clara[cla]={}
    matrix_sim_clara[cla][id_test]=b_props_cluster['matrix_sim'][id_test]
    del b_props_cluster
#%%
filter_centroid_sel_df_clara={}
#%%
centroid_sel=5205
for cla in clara_tests:
    filter_centroid_sel_df_clara[cla] = get_df_filter(centroid_sel, props_df_sel_clara[cla],\
                                            matrix_sim_clara[cla],id_test, threshold=0.70)
#%%
cla = '_cluster_clara_'
for cla in [cla]:#clara_tests:
    print (cla)
    plot_cluster_img_pixel_sel_faster(filter_centroid_sel_df_clara[cla], centroid_sel,plot_centroids=False)
#%% #plot images clara tests for pixel selected
plot_images_cluster_clara(img_sel_norm,filter_centroid_sel_df_clara,  id_test,\
                        clara_tests,centroid_sel, n_cols=3, cl_map='Blues', chart_size=(12, 12))


#%%
# show the similarity/co-association matrix using a heatmap
import plotly.express as px
#%%
fig = px.imshow(matrix_sim, color_continuous_scale='purples')
fig.show()

#%%
#to show matrix coassociation as graph
#trabalhar melhor esta visualizacao.
import plotly.graph_objects as go
import networkx as nx
#%%
# Create a graph from the adjacency matrix
# Create a graph from the adjacency matrix
G = nx.Graph(matrix_sim)
#%%
# Get positions for plotting (you can use different layout algorithms)
pos = nx.spring_layout(G)
#%%
nx.set_node_attributes(G,pos,'pos')

#%%
node_x = []   # store x coordinates
node_y = []   # store y coordinates
node_text = [] # store text when mouse hovers over the node
for node,node_attr_dict in G.nodes(data=True):  # recall anatomy 
    x,y = node_attr_dict['pos']
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)
node_trace = go.Scatter(name='nodes',x=node_x,y=node_y,mode='markers',hoverinfo='text',text=node_text,marker={'color':'green','size':5})

#%%
edge_x = []
edge_y = []
for edge_end1,edge_end2,edge_attr_dict in G.edges(data=True):
    x0,y0 = G.nodes[edge_end1]['pos']
    x1,y1 = G.nodes[edge_end2]['pos']
    x2,y2 = None,None
    for x,y in zip([x0,x1,x2],[y0,y1,y2]):
        edge_x.append(x)
        edge_y.append(y)
edge_trace = go.Scatter(name='lines',x=edge_x,y=edge_y,mode='lines',line=go.scatter.Line(color='black',width=2))
#%%
fig_layout = go.Layout(showlegend=True,title='network',xaxis=dict(title_text='coordinate x'))
# Combine nodes and edges in a Plotly Figure
fig = go.Figure(data=[edge_trace, node_trace], layout=fig_layout)

# Show the Plotly figure
fig.show()

#####################################################################
#####################################################################
#clustering example
#%%
from pyclustering.cluster.clarans import clarans;
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import timedcall;
from sklearn import datasets
#%%
#import iris dataset from sklearn library
iris =  datasets.load_iris();

#get the iris data. It has 4 features, 3 classes and 150 data points.
data = iris.data
#%%
"""!
The pyclustering library clarans implementation requires
list of lists as its input dataset.
Thus we convert the data from numpy array to list.
"""
data = data.tolist()

#get a glimpse of dataset
print("A peek into the dataset : ",data[:4])

#%%
"""!
@brief Constructor of clustering algorithm CLARANS.
@details The higher the value of maxneighbor, the closer is CLARANS to K-Medoids, and the longer is each search of a local minima.
@param[in] data: Input data that is presented as list of points (objects), each point should be represented by list or tuple.
@param[in] number_clusters: amount of clusters that should be allocated.
@param[in] numlocal: the number of local minima obtained (amount of iterations for solving the problem).
@param[in] maxneighbor: the maximum number of neighbors examined.        
"""
clarans_instance = clarans(data, 3, 6, 4)

#calls the clarans method 'process' to implement the algortihm
(ticks, result) = timedcall(clarans_instance.process)
print("Execution time : ", ticks, "\n")
#%%
#returns the clusters 
clusters = clarans_instance.get_clusters()

#returns the mediods 
medoids = clarans_instance.get_medoids()


print("Index of the points that are in a cluster : ",clusters)
print("The target class of each datapoint : ",iris.target)
print("The index of medoids that algorithm found to be best : ",medoids)


##############################################

#%%

segments_slic_rgb_1 = slic(img_rgb, n_segments=100, compactness=10, sigma=1, start_label=1)
print(f'SLIC RGB number of segments : {len(np.unique(segments_slic_rgb_1))}')

#%%
segments_slic_bands_1 = slic(img_bands, n_segments=250, compactness=10, sigma=1, start_label=1)
print(f'SLIC other bands number of segments : {len(np.unique(segments_slic_bands_1))}')
#%%
plt.imshow(mark_boundaries(img_rgb, segments_slic_rgb_1))
plt.show()

plt.imshow(mark_boundaries(img_rgb, segments_slic_bands_1))
plt.show()

#%%

segments_slic_bands_false = slic(img_bands, compactness=10, sigma=1, start_label=1, \
                                 enforce_connectivity=False)
print(f'SLIC RGB number of segments : {len(np.unique(segments_slic_rgb_1))}')
segments_slic_rgb_false = slic(img_rgb, compactness=10, sigma=1, start_label=1, \
                               enforce_connectivity=False)
print(f'SLIC other bands number of segments : {len(np.unique(segments_slic_rgb_1))}')
plt.imshow(mark_boundaries(img_rgb, segments_slic_rgb_false))
plt.show()

###################

# %%
#exemplo de cluster com kmeans
from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs
# Criar dados de exemplo
X, _ = make_blobs(n_samples=100, centers=4, cluster_std=1.0, random_state=42)

# Aplicar o algoritmo K-means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# Obter os rótulos dos clusters e os centróides
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualizar os resultados
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='k')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, linewidths=2, color='red')
plt.title('Resultados do K-means')
plt.show()
# %%
