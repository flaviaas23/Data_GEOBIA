#copied at 17/06/2024
#From: notebooks/ComparaSLIC_SNIC.ipynb
# 20241001: baixado do exacta

import numpy as np
from math import ceil

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
#from dask_ml.decomposition import PCA

from pysnic.algorithms.snic import snic
#For snic SP
from pysnic.algorithms.snic import compute_grid
from pysnic.ndim.operations_collections import nd_computations
from itertools import chain
from pysnic.metric.snic import create_augmented_snic_distance
from timeit import default_timer as timer

#for slic
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

import imageio.v2 as imageio
from skimage.measure import regionprops, regionprops_table

from functions.functions_pca import save_to_pickle, calc_avg_array, \
                                    calc_avg_array_pca

# 20241124: function to get coords positons to get the image qudrant
def get_quadrant_coords(q, imgSize):
    '''
    q = number of quadrant
    '''
    Q_imgSize = imgSize//2
    if q == 1:
        rowi, rowf = 0, Q_imgSize
        coli, colf = 0, Q_imgSize
    elif q == 2:
        rowi, rowf = 0, Q_imgSize
        coli, colf = Q_imgSize, imgSize
    elif q == 3:
        rowi, rowf = Q_imgSize, imgSize
        coli, colf = 0, Q_imgSize
    elif q == 4:
        rowi, rowf,  = Q_imgSize, imgSize
        coli, colf = Q_imgSize, imgSize
    return rowi, rowf, coli, colf

#Functions for snic
##### Function to gen snic coords df
def gen_coords_snic_df(segments_snic_sel_sp, sh_print=False):
    #create dictionaries for coords and num pixels
    t1=time.time()
    coords_snic_dic={}
    for i in range(len(segments_snic_sel_sp)):
        for j in range(len(segments_snic_sel_sp[i])):
            current = segments_snic_sel_sp[i][j]
            if current in coords_snic_dic:
                coords_snic_dic[current].append([i , j ])
            else:
                coords_snic_dic[current] = [[i , j ]]
    t2=time.time()    
    #count number of pixels if each group
    num_pixels_dic={}
    for l in coords_snic_dic.keys():
        num_pixels_dic[l] = len(coords_snic_dic[l])
    t3=time.time()    
    print (f'{t2-t1:.2f}, {t3-t2:.2f}') if sh_print else None
    # create a df using the dictionaries coords_snic_dic and num_pixels_dic
    t4=time.time()
    coords_snic_df =pd.DataFrame({'label':coords_snic_dic.keys(), \
                                  'coords':coords_snic_dic.values(), \
                                  'num_pixels': num_pixels_dic.values()},
                                index=coords_snic_dic.keys())
    coords_snic_df.sort_values('label', inplace=True)
    #coords_snic_df['label'] = coords_snic_dic.keys()
    #coords_snic_df=coords_snic_df.set_index('label')
    t5=time.time()
    print (f'Tempo total para gerar coords df: {t5-t1:.2f}, tempo para gerar só df {t5-t4:.2f}') if sh_print else None
    # n_part = round(coords_snic_df.shape[0] * coords_snic_df.shape[1]/1000)    
    # dd_coords_snic_df = dd.from_pandas(coords_snic_df,npartitions=n_part)    
    return coords_snic_df

##### Function to gen snic df , 
# 20240823: antiga gen_centroid_snic_df, gera em dask os centroids e coords df
def gen_centroid_snic_ddf(image_band_dic, centroids_snic_sp, coords_snic_df, \
                         bands_sel, ski=True, stats=True, sh_print=False):
    '''
    Function to gen a df from snic centroids
    Ski: 
        True: considere the avg values of snic centroid 
              and do avg only of bands not used in snic
        False: do avg of all bands 
    '''   
    # gen a dictionary with num_pixels, centroids and bands/pca 
    # values and do a df with results of snic segmentation    
    t1=time.time()
    snic_n_segms = len(centroids_snic_sp)
    #centroids_snic_only = [subar[0] for subar in centroids_snic_sp]
    centroids_snic_sp_dic={}
    centroids_snic_sp_dic['label'] = [i for i in range(snic_n_segms)]
    centroids_snic_sp_dic['num_pixels'] = [subar[2] for subar in centroids_snic_sp]
    centroids_snic_sp_dic['centroid-0'] = [subar[0][0] for subar in centroids_snic_sp]
    centroids_snic_sp_dic['centroid-1'] = [subar[0][1] for subar in centroids_snic_sp]
    if ski:
        for i,b in enumerate(bands_sel):
            centroids_snic_sp_dic['avg_'+b] = [subar[1][i] for subar in centroids_snic_sp]
    t2=time.time()
    print (t2-t1) if sh_print else None
    snic_centroid_df = pd.DataFrame(centroids_snic_sp_dic)
    del centroids_snic_sp_dic     
    # n_part = round(snic_centroid_df.shape[0] * snic_centroid_df.shape[1]/1000)    
    # dd_snic_centroid_df = dd.from_pandas(snic_centroid_df,npartitions=n_part)        
    # dd_coords_snic_df =  dd.from_pandas(coords_snic_df,npartitions=n_part)
    n_part = round(snic_centroid_df.shape[0] * snic_centroid_df.shape[1]/10000)        
    dd_snic_centroid_df = dd.from_pandas(snic_centroid_df,npartitions=n_part)  
    dd_snic_centroid_df = dd_snic_centroid_df.repartition(partition_size="100MB")
    dd_coords_snic_df =  dd.from_pandas(coords_snic_df,npartitions=n_part)
    dd_coords_snic_df = dd_coords_snic_df.repartition(partition_size="100MB")
    del coords_snic_df, snic_centroid_df
    #criar as stats das bandas e SPs
    if stats:                
        for c in image_band_dic.keys():            
            # no nome da banda vai ser a mediana do sp 
            dd_snic_centroid_df['med_'+c] = dd_coords_snic_df['coords'].apply(calc_avg_array, img_dic=image_band_dic[c],\
                                                              c=c, med='med', meta=dd_coords_snic_df['coords'])
            dd_snic_centroid_df['std_'+c] = dd_coords_snic_df['coords'].apply(calc_avg_array, img_dic=image_band_dic[c],\
                                                              c=c, med='std', meta=dd_coords_snic_df['coords'])
            # Não precisa fazer a média pq o snic retorna as medias das bandas de cada sp (grupo)
            # valores incluidos acima no if do ski
            # testei e vi que os valores sao iguais
            # testar se o avg da banda já nao existe e incluir
            if not ski:
                print (c, ski, "do avg") if sh_print else None
                dd_snic_centroid_df['avg_'+c] = dd_coords_snic_df['coords'].apply(calc_avg_array, img_dic=image_band_dic[c],\
                                                                   c=c, med='avg', meta=dd_coords_snic_df['coords'])
            elif (c not in bands_sel): 
                print (c, "do avg") if sh_print else None
                dd_snic_centroid_df['avg_'+c] = dd_coords_snic_df['coords'].apply(calc_avg_array, img_dic=image_band_dic[c],\
                                                                   c=c, med='avg', meta=dd_coords_snic_df['coords'])
                
                #dd_centroid_df[[c, 'avg_'+c, 'std_'+c]] = dd_coords_SP_df['coords'].compute().apply(calc_avg_array, img_dic=image_band_dic[c],\
                                                           #   c=c).apply(pd.Series)
    return dd_snic_centroid_df, dd_coords_snic_df    

#20240823: Usei como base a funcao gen_centroid_snic_df que virou gen_centroid_snic_ddf,
#          principal diferença no apply,  meta= foi removido e nao gera mais coords em dask
#          testada no jupyter notebooks/spk_gen_df_toPCA_exemplo.ipynb 
def gen_centroid_snic_df(image_band_dic, centroids_snic_sp, coords_snic_df, \
                         bands_sel, ski=True, stats=True, dask=False, sh_print=True):
    '''
    Function to gen a df from snic centroids em pandas
    Ski: 
        True: considere the avg values of snic centroid 
              and do avg only of bands not used in snic
        False: do avg of all bands 
    '''   
    # gen a dictionary with num_pixels, centroids and bands/pca 
    # values and do a df with results of snic segmentation    
    t1=time.time()
    snic_n_segms = len(centroids_snic_sp)
    #centroids_snic_only = [subar[0] for subar in centroids_snic_sp]
    centroids_snic_sp_dic={}
    centroids_snic_sp_dic['label'] = [i for i in range(snic_n_segms)]
    centroids_snic_sp_dic['num_pixels'] = [subar[2] for subar in centroids_snic_sp]
    centroids_snic_sp_dic['centroid-0'] = [subar[0][0] for subar in centroids_snic_sp]
    centroids_snic_sp_dic['centroid-1'] = [subar[0][1] for subar in centroids_snic_sp]
    if ski:
        for i,b in enumerate(bands_sel):
            centroids_snic_sp_dic['avg_'+b] = [subar[1][i] for subar in centroids_snic_sp]
    t2=time.time()
    print (f'time to gen centroids dictionary: {t2-t1:.2f}') if sh_print else None
    snic_centroid_df = pd.DataFrame(centroids_snic_sp_dic)
    del centroids_snic_sp_dic  
    if dask:
        n_part = round(snic_centroid_df.shape[0] * snic_centroid_df.shape[1]/10000)        
        dd_snic_centroid_df = dd.from_pandas(snic_centroid_df,npartitions=n_part)  
        dd_snic_centroid_df = dd_snic_centroid_df.repartition(partition_size="100MB")
        dd_coords_snic_df =  dd.from_pandas(coords_snic_df,npartitions=n_part)
        dd_coords_snic_df = dd_coords_snic_df.repartition(partition_size="100MB")
        del coords_snic_df, snic_centroid_df
    else:
        dd_snic_centroid_df = snic_centroid_df
        dd_coords_snic_df = coords_snic_df
    #criar as stats das bandas e SPs
    if stats:                
        for c in image_band_dic.keys():            
            # no nome da banda vai ser a mediana do sp 
            t1=time.time()
            dd_snic_centroid_df['med_'+c] = dd_coords_snic_df['coords'].apply(calc_avg_array, img_dic=image_band_dic[c],\
                                                              c=c, med='med', pca=1) #, meta=dd_coords_snic_df['coords']) meta usado para dask
            t2=time.time()
            dd_snic_centroid_df['std_'+c] = dd_coords_snic_df['coords'].apply(calc_avg_array, img_dic=image_band_dic[c],\
                                                              c=c, med='std',pca=1)#, meta=dd_coords_snic_df['coords'])meta usado para dask
            t3=time.time()
            print (f'{c} tempo calc mediana: {t2-t1:.2f}s, std: {t3-t2:.2f}s ') if sh_print else None
            # Não precisa fazer a média pq o snic retorna as medias das bandas de cada sp (grupo)
            # valores incluidos acima no if do ski
            # testei e vi que os valores sao iguais
            # testar se o avg da banda já nao existe e incluir
            if not ski:
                print (c, ski, "do avg") if sh_print else None
                dd_snic_centroid_df['avg_'+c] = dd_coords_snic_df['coords'].apply(calc_avg_array, img_dic=image_band_dic[c],\
                                                                   c=c, med='avg', pca=1)#, meta=dd_coords_snic_df['coords'])meta usado para dask
            elif (c not in bands_sel): 
                print (c, "do avg") if sh_print else None
                dd_snic_centroid_df['avg_'+c] = dd_coords_snic_df['coords'].apply(calc_avg_array, img_dic=image_band_dic[c],\
                                                                   c=c, med='avg', pca=1) # meta=dd_coords_snic_df['coords']) meta usado para dask
                
                #dd_centroid_df[[c, 'avg_'+c, 'std_'+c]] = dd_coords_SP_df['coords'].compute().apply(calc_avg_array, img_dic=image_band_dic[c],\
                                                           #   c=c).apply(pd.Series)
    return dd_snic_centroid_df#, dd_coords_snic_df  
#20240730: versao ok
#veio do arquivo gen_matrix_dist.py
def gen_centroid_snic_dask(image_band_dic, centroids_snic_sp, coords_snic_df, \
                         bands_sel, ski=True, stats=True, sh_print=True):
    '''
    Function to gen a df from snic centroids
    Ski: 
        True: considere the avg values of snic centroid 
              and do avg only of bands not used in snic
        False: do avg of all bands 
    '''   
    # gen a dictionary with num_pixels, centroids and bands/pca 
    # values and do a df with results of snic segmentation    
    t1=time.time()
    snic_n_segms = len(centroids_snic_sp)
    #centroids_snic_only = [subar[0] for subar in centroids_snic_sp]
    centroids_snic_sp_dic={}
    centroids_snic_sp_dic['label'] = [i for i in range(snic_n_segms)]
    centroids_snic_sp_dic['num_pixels'] = [subar[2] for subar in centroids_snic_sp]
    centroids_snic_sp_dic['centroid-0'] = [subar[0][0] for subar in centroids_snic_sp]
    centroids_snic_sp_dic['centroid-1'] = [subar[0][1] for subar in centroids_snic_sp]
    if ski:
        for i,b in enumerate(bands_sel):
            centroids_snic_sp_dic['avg_'+b] = [subar[1][i] for subar in centroids_snic_sp]
    t2=time.time()
    print (f'time to gen centroids dictionary: {t2-t1}') if sh_print else None
    snic_centroid_df = pd.DataFrame(centroids_snic_sp_dic)
    del centroids_snic_sp_dic     

    n_part = round(snic_centroid_df.shape[0] * snic_centroid_df.shape[1]/10000)        
    dd_snic_centroid_df = dd.from_pandas(snic_centroid_df,npartitions=n_part)  
    dd_snic_centroid_df = dd_snic_centroid_df.repartition(partition_size="100MB")
    dd_coords_snic_df =  dd.from_pandas(coords_snic_df,npartitions=n_part)
    dd_coords_snic_df = dd_coords_snic_df.repartition(partition_size="100MB")
    del coords_snic_df, snic_centroid_df
    
    #criar as stats das bandas e SPs
    if stats:                
        for c in image_band_dic.keys():            
            # no nome da banda vai ser a mediana do sp 
            t1=time.time()
            dd_snic_centroid_df['med_'+c] = dd_coords_snic_df['coords'].apply(calc_avg_array_pca, img_dic=image_band_dic[c],\
                                                              c=c, med='med', meta=dd_coords_snic_df['coords'])
            t2=time.time()
            dd_snic_centroid_df['std_'+c] = dd_coords_snic_df['coords'].apply(calc_avg_array_pca, img_dic=image_band_dic[c],\
                                                              c=c, med='std', meta=dd_coords_snic_df['coords'])
            t3=time.time()
            print (f'{c} tempo calc mediana: {t2-t1}s, std: {t3-t2}s ') if sh_print else None
            # Não precisa fazer a média pq o snic retorna as medias das bandas de cada sp (grupo)
            # valores incluidos acima no if do ski
            # testei e vi que os valores sao iguais
            # testar se o avg da banda já nao existe e incluir
            if not ski:
                print (c, ski, "do avg") if sh_print else None
                dd_snic_centroid_df['avg_'+c] = dd_coords_snic_df['coords'].apply(calc_avg_array_pca, img_dic=image_band_dic[c],\
                                                                   c=c, med='avg', meta=dd_coords_snic_df['coords'])
            elif (c not in bands_sel): 
                print (c, "do avg") if sh_print else None
                dd_snic_centroid_df['avg_'+c] = dd_coords_snic_df['coords'].apply(calc_avg_array_pca, img_dic=image_band_dic[c],\
                                                                   c=c, med='avg', meta=dd_coords_snic_df['coords'])
                
                #dd_centroid_df[[c, 'avg_'+c, 'std_'+c]] = dd_coords_SP_df['coords'].compute().apply(calc_avg_array, img_dic=image_band_dic[c],\
                                                           #   c=c).apply(pd.Series)
    return dd_snic_centroid_df, dd_coords_snic_df  
##### Function run_snic_gen_dask
def run_snic_gen_dask(id,img_sel_norm_shape, ski_img_sel_norm_list, n_segms, \
             compact, f_name, img_band_dic_norm, bands_sel, save_snic=False, sh_print=True):
    '''
    Function to run snic
    '''    
    t1=time.time()
    # compute grid
    grid = compute_grid(img_sel_norm_shape, n_segms)
    seeds = list(chain.from_iterable(grid))
    seed_len = len(seeds)    
    # choose a distance metric #se nao fornecido faz exatamente isso
    distance_metric = create_augmented_snic_distance(img_sel_norm_shape, seed_len, compact)    
    #start = timer()
    t2=time.time()
    segments_snic_sel_ski_sp, dist_snic_sp, centroids_snic_ski_sp = snic(
                            ski_img_sel_norm_list,
                            #img_sel_norm.tolist(),
                            seeds,
                            compact, nd_computations["3"], distance_metric)#,
                            #update_func=lambda num_pixels: print("processed %05.2f%%" % (num_pixels * 100 / number_of_pixels)))    
    t3=time.time()
    print(f"id: {id}, snic took: {t3 - t2:.2f}") if sh_print  else None
    ## gen coords_df
    #for image with ski
    coords_snic_ski_df = gen_coords_snic_df(segments_snic_sel_ski_sp)
    #separar a linha com label -1
    coords_snic_ski_df_nan = coords_snic_ski_df[coords_snic_ski_df['label'] == -1]
    coords_snic_ski_df = coords_snic_ski_df[coords_snic_ski_df['label'] != -1]
    #gen centroid and coords em dask
    dd_snic_centroid_ski_df, dd_coords_snic_ski_df = gen_centroid_snic_df(img_band_dic_norm,\
                                                                centroids_snic_ski_sp, 
                                                                coords_snic_ski_df, bands_sel,
                                                                ski=True, stats=True)
        
    #t_snic_sp=t3-t2
    t4 = time.time()
    print (f'id: {id}, gen centroids and coords df toof {t4-t3:.2f}') if sh_print else None
    
    if save_snic:
        t5 = time.time()
        obj_dic = {}
        obj_dic[id] = {
            "centroid": dd_snic_centroid_ski_df.compute(),#, 
            "params_test": [n_segms, compact, True] #skimage used = True
            
           }
        file_to_save = f_name +'_snic_centroid_'+str(id)+'.pkl'
        save_to_pickle(obj_dic, file_to_save)
        t6 = time.time()
        #print (f'centroid file: {file_to_save}')
        print (f'id: {id}, save centroids df took {t6-t5:.2f}') if sh_print else None
        del obj_dic
        t5 = time.time()
        obj_dic = {}
        obj_dic[id] = {
            "segments": segments_snic_sel_ski_sp
        }
        file_to_save = f_name +'_snic_segments_'+str(id)+'.pkl'
        save_to_pickle(obj_dic, file_to_save)
        t6 = time.time()
        #print (f'centroid file: {file_to_save}')
        print (f'id: {id}, save segments took {t6-t5:.2f}') if sh_print else None
        del obj_dic
        t5 = time.time()
        obj_dic ={}
        obj_dic[id] = {
            #"props_df_sel": props_df_sel#, 
            "coords": coords_snic_ski_df,
            "coords_nan": coords_snic_ski_df_nan            
           }
        file_to_save = f_name + '_snic_coords_' +str(id)+'.pkl'
        save_to_pickle(obj_dic, file_to_save)
        t6 = time.time()
        #print (f'coords file: {file_to_save}')
        print (f'id: {id}, save coords df took {t6-t5:.2f}') if sh_print else None
        del obj_dic  
    return  dd_snic_centroid_ski_df, dd_coords_snic_ski_df, segments_snic_sel_ski_sp    
## Function to run snic and gen dfs from snic output
# 20240823 : testada no jupyter notebooks/spk_gen_df_toPCA_exemplo.ipynb 
def run_snic_gen_dfs(id,img_sel_norm_shape, ski_img_sel_norm_list, n_segms, \
             compact, f_name, img_band_dic_norm, bands_sel, save_snic=False, sh_print=True):
    '''
    Function to run snic and return snic_centroid_ski_df, coords_snic_ski_df, segments_snic_sel_ski_sp
    ski = true para usar o avg que é retornado do snic na gen do df
    '''
    print(f"run_snic 0. id: {id}, {n_segms} {compact}") if sh_print  else None
    
    t1=time.time()
    # compute grid
    grid = compute_grid(img_sel_norm_shape, n_segms)
    seeds = list(chain.from_iterable(grid))
    seed_len = len(seeds)
    t2=time.time()
    print(f"run_snic 1. id: {id}, compute_grid took: {t2 - t1:.2f}") if sh_print  else None
    
    t1=time.time()

    # choose a distance metric #se nao fornecido faz exatamente isso
    distance_metric = create_augmented_snic_distance(img_sel_norm_shape, seed_len, compact)
    
    t2=time.time()
    print(f"run_snic 1. id: {id}, create_augmented_snic_distance took: {t2 - t1:.2f}") if sh_print  else None

    #start = timer()
    t2=time.time()
    segments_snic_sel_ski_sp, dist_snic_sp, centroids_snic_ski_sp = snic(
                            ski_img_sel_norm_list,
                            #img_sel_norm.tolist(),
                            seeds,
                            compact, nd_computations["3"], distance_metric)#,
                            #update_func=lambda num_pixels: print("processed %05.2f%%" % (num_pixels * 100 / number_of_pixels)))
    
    t3=time.time()
    print(f"run snic 2. id: {id}, snic took: {t3 - t2:.2f}") if sh_print  else None
    ## gen coords_df
    #for image with ski
    coords_snic_ski_df = gen_coords_snic_df(segments_snic_sel_ski_sp)
    #separar a linha com label -1
    coords_snic_ski_df_nan = coords_snic_ski_df[coords_snic_ski_df['label'] == -1]
    coords_snic_ski_df = coords_snic_ski_df[coords_snic_ski_df['label'] != -1]

    #gen centroid and coords em dask
    dd_snic_centroid_ski_df = gen_centroid_snic_df(img_band_dic_norm,\
                                                    centroids_snic_ski_sp, 
                                                    coords_snic_ski_df, bands_sel,
                                                    ski=True, stats=True, dask=False) #20240823 mudei skipara True para nao 
                                                                                     #fazer o avg q já vem do snic

    #t_snic_sp=t3-t2
    t4 = time.time()
    print (f'run snic 3. id: {id}, gen centroids and coords df toof {t4-t3:.2f}') if sh_print else None
    
    if save_snic:
        t5 = time.time()
        obj_dic = {}
        obj_dic[id] = {
            "centroid_df": dd_snic_centroid_ski_df, #.compute(),#, 
            "params_test": [n_segms, compact, True] #skimage used = True
            
           }
        file_to_save = f_name +'_snic_centroid_df_'+str(id)+'.pkl'
        save_to_pickle(obj_dic, file_to_save)
        t6 = time.time()
        #print (f'centroid file: {file_to_save}')
        print (f'id: {id}, save centroids df took {t6-t5:.2f}') if sh_print else None
        del obj_dic

        t5 = time.time()
        obj_dic = {}
        obj_dic[id] = {
            "segments": segments_snic_sel_ski_sp
        }
        file_to_save = f_name +'_snic_segments_'+str(id)+'.pkl'
        save_to_pickle(obj_dic, file_to_save)
        t6 = time.time()
        #print (f'centroid file: {file_to_save}')
        print (f'id: {id}, save segments took {t6-t5:.2f}') if sh_print else None
        del obj_dic
        
        t5 = time.time()
        obj_dic ={}
        obj_dic[id] = {
            #"props_df_sel": props_df_sel#, 
            "coords": coords_snic_ski_df,
            "coords_nan": coords_snic_ski_df_nan            
           }
        file_to_save = f_name + '_snic_coords_' +str(id)+'.pkl'
        save_to_pickle(obj_dic, file_to_save)
        t6 = time.time()
        #print (f'coords file: {file_to_save}')
        print (f'id: {id}, save coords df took {t6-t5:.2f}') if sh_print else None
        del obj_dic  
        

    return  dd_snic_centroid_ski_df, segments_snic_sel_ski_sp   
#### 
##### Function gen_snic_dask_df : read snic files and gen dfs em dask
def gen_snic_dask_df(id,img_sel_norm_shape, ski_img_sel_norm_list, n_segms, \
             compact, f_name, img_band_dic_norm, bands_sel, save_snic=False, sh_print=True):
    '''
    Function to gen snic dfs reading  segments and centroids from pkl files
    '''    
    # t1=time.time()
    # # compute grid
    # grid = compute_grid(img_sel_norm_shape, n_segms)
    # seeds = list(chain.from_iterable(grid))
    # seed_len = len(seeds)    
    # # choose a distance metric #se nao fornecido faz exatamente isso
    # distance_metric = create_augmented_snic_distance(img_sel_norm_shape, seed_len, compact)    
    # #start = timer()
    # t2=time.time()
    # segments_snic_sel_ski_sp, dist_snic_sp, centroids_snic_ski_sp = snic(
    #                         ski_img_sel_norm_list,
    #                         #img_sel_norm.tolist(),
    #                         seeds,
    #                         compact, nd_computations["3"], distance_metric)#,
    #                         #update_func=lambda num_pixels: print("processed %05.2f%%" % (num_pixels * 100 / number_of_pixels)))    
    # t3=time.time()
    # print(f"id: {id}, snic took: {t3 - t2:.2f}") if sh_print  else None
    
    ## gen coords_df
    #for image with ski
    #read segments_snic_sel_ski_sp and centroids_snic_ski_sp
    #n110k_c2_s1_con0_centroids_snic_sel_pca_sp.pkl
    f_read = f_name+'segments_snic_sel_pca_sp.pkl'
    with open(f_read, 'rb') as handle: 
        segments_snic_sel_ski_sp = pickle.load(handle)

    f_read = f_name+'centroids_snic_sp.pkl'
    with open(f_read, 'rb') as handle: 
        centroids_snic_ski_sp = pickle.load(handle)

    coords_snic_ski_df = gen_coords_snic_df(segments_snic_sel_ski_sp)
    #separar a linha com label -1
    coords_snic_ski_df_nan = coords_snic_ski_df[coords_snic_ski_df['label'] == -1]
    coords_snic_ski_df = coords_snic_ski_df[coords_snic_ski_df['label'] != -1]
    #gen centroid and coords em dask
    t3 = time.time()
    dd_snic_centroid_df, dd_coords_snic_df = gen_centroid_snic_df(img_band_dic_norm,\
                                                                centroids_snic_ski_sp, 
                                                                coords_snic_ski_df, bands_sel,
                                                                ski=False, stats=True)
        
    #t_snic_sp=t3-t2
    t4 = time.time()
    print (f'id: {id}, gen centroids and coords df took {t4-t3:.2f}') if sh_print else None
    
    if save_snic: #precisa acerta este if 20240729
        t5 = time.time()
        obj_dic = {}
        obj_dic[id] = {
            "centroid": dd_snic_centroid_df.compute(),#, 
            "params_test": [n_segms, compact, True] #skimage used = True
            
           }
        file_to_save = f_name +'_snic_centroid_'+str(id)+'.pkl'
        save_to_pickle(obj_dic, file_to_save)
        t6 = time.time()
        #print (f'centroid file: {file_to_save}')
        print (f'id: {id}, save centroids df took {t6-t5:.2f}') if sh_print else None
        del obj_dic
        t5 = time.time()
        obj_dic = {}
        obj_dic[id] = {
            "segments": segments_snic_sel_ski_sp
        }
        file_to_save = f_name +'_snic_segments_'+str(id)+'.pkl'
        save_to_pickle(obj_dic, file_to_save)
        t6 = time.time()
        #print (f'centroid file: {file_to_save}')
        print (f'id: {id}, save segments took {t6-t5:.2f}') if sh_print else None
        del obj_dic
        t5 = time.time()
        obj_dic ={}
        obj_dic[id] = {
            #"props_df_sel": props_df_sel#, 
            "coords": coords_snic_ski_df,
            "coords_nan": coords_snic_ski_df_nan            
           }
        file_to_save = f_name + '_snic_coords_' +str(id)+'.pkl'
        save_to_pickle(obj_dic, file_to_save)
        t6 = time.time()
        #print (f'coords file: {file_to_save}')
        print (f'id: {id}, save coords df took {t6-t5:.2f}') if sh_print else None
        del obj_dic  
    return  dd_snic_centroid_df, dd_coords_snic_df, segments_snic_sel_ski_sp    

###Function to save to pickle results of segmentation 
# 20240826: 
def save_segm(id, snic_centroid_df, segments_snic_sel_sp, 
              f_name, params_test, str_fn='', sh_print=False ):
    '''
    Function to save results of segmentation
    params_test = {n_segms:, compact:, ski_img:}
    '''

    t5 = time.time()
    obj_dic = {}
    obj_dic[id] = {
        "centroid_df": snic_centroid_df, #.compute(),#, 
        "params_test": params_test #skimage used = True
            
           }
    test = str_fn+'_n_'+str(params_test['segms'])+'_comp_'+str(params_test['compactness'])
    f_name = f_name+test
    file_to_save = f_name +'_snic_centroid_df_'+str(id)+'.pkl'
    save_to_pickle(obj_dic, file_to_save)

    t6 = time.time()
    print (f'centroid file: {file_to_save}') if sh_print else None
    print (f'id: {id}, save centroids df took {t6-t5:.2f}') if sh_print else None
    del obj_dic

    t5 = time.time()
    obj_dic = {}
    obj_dic[id] = {
        "segments": segments_snic_sel_sp,
        "params_test": params_test #skimage used = True
        }
    file_to_save = f_name +'_snic_segments_'+str(id)+'.pkl'
    save_to_pickle(obj_dic, file_to_save)

    t6 = time.time()
    print (f'segments file: {file_to_save}') if sh_print else None
    print (f'id: {id}, save segments took {t6-t5:.2f}') if sh_print else None
    del obj_dic
        
        # t5 = time.time()
        # obj_dic ={}
        # obj_dic[id] = {
        #     #"props_df_sel": props_df_sel#, 
        #     "coords": coords_snic_ski_df,
        #     "coords_nan": coords_snic_ski_df_nan            
        #    }
        # file_to_save = f_name + '_snic_coords_' +str(id)+'.pkl'
        # save_to_pickle(obj_dic, file_to_save)
        # t6 = time.time()
        # #print (f'coords file: {file_to_save}')
        # print (f'id: {id}, save coords df took {t6-t5:.2f}') if sh_print else None
        #del obj_dic  


##### Function to gen statistics of bands(cols) of slic sp df
def gen_stats(df_dd, id_test, conf_test_dic, bands, segm_type='slic', mean=True, dask=True):
    '''
    Generates statistics for the segmentation df 
    calculates average, max and min number of pixels in a sp (superpixel) 
    gets the average band value and average std for each band   
    Returns a df with test parameters and its stats 
    '''
    stats_sp_segments={}
    
    if id_test not in stats_sp_segments:
        stats_sp_segments[id_test] = {}

    stats_sp_segments[id_test] = conf_test_dic[id_test].copy()
    #print ("2: gen_stats: stats_sp_segments:\n",stats_sp_segments)
    #print ("df.shape[0]:", df.shape[0])
    #print ("3:",stats_sp_segments)

    cols = ['num_pixels']
    for b in bands:
        cols.append('avg_'+b)
        cols.append('std_'+b)
    #print (cols)
    t1=time.time()
    if dask:
        df=df_dd[cols].compute()
    else:
        df=df_dd[cols]
    # stats_sp_segments[id_test]['segms_calc'] = df.shape[0].compute()
    # stats_sp_segments[id_test]['avg_n_pixels'] = df['num_pixels'].mean().compute()
    # stats_sp_segments[id_test]['median_n_pixels'] = df['num_pixels'].compute().median()
    # stats_sp_segments[id_test]['max_n_pixels'] = df['num_pixels'].max().compute()
    # stats_sp_segments[id_test]['min_n_pixels'] = df['num_pixels'].min().compute()
    # for b in bands:
    #     stats_sp_segments[id_test]['avg_'+b] = df['avg_'+b].compute().mean()
    #     stats_sp_segments[id_test]['avg_std_'+b] = df['std_'+b].compute().mean()
    stats_sp_segments[id_test]['id_test'] = id_test
    stats_sp_segments[id_test]['segm_type'] = segm_type
    stats_sp_segments[id_test]['segms_calc'] = df.shape[0]
    stats_sp_segments[id_test]['avg_n_pixels'] = df['num_pixels'].mean()
    stats_sp_segments[id_test]['median_n_pixels'] = df['num_pixels'].median()
    stats_sp_segments[id_test]['max_n_pixels'] = df['num_pixels'].max()
    stats_sp_segments[id_test]['min_n_pixels'] = df['num_pixels'].min()
    for b in bands:
        if mean:
            stats_sp_segments[id_test]['avg_'+b] = df['avg_'+b].mean()
            stats_sp_segments[id_test]['avg_std_'+b] = df['std_'+b].mean()
        else:
            stats_sp_segments[id_test][b] = df[b].median()
            stats_sp_segments[id_test]['avg_std_'+b] = df['std_'+b].median()
    del df
    t2=time.time()
    print (t2-t1)
    
    stats_df = pd.DataFrame(stats_sp_segments).T
    #stats_df = pd.DataFrame(list(stats_sp_segments.items()), columns=['key', 'value'], dtype=object).T

    del stats_sp_segments, conf_test_dic
    stats_df.insert(0, 'id_test', stats_df.pop('id_test'))
    stats_df.insert(1, 'segm_type', stats_df.pop('segm_type'))
    #stats_df = stats_df.applymap('{:,.2f}'.format)

    return stats_df
    #return stats_sp_segments

##### Function to make all possible combinations of the parameters to slic
def gen_params_test(segms = [11000],compactness = [1, 2], \
                    sigmas = [0.1, 0.5], connectivity = [False]):
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

##### Function to normalize the values of image band dictionary and replace -9999 amd -32768 to np.nan
def norm_img_dic(image_band_dic, replace_nans=False, show_print=True):
    # making the image dictionary with values normalized
    # Replace (-9999 and -32768) to np.nan if requested  
    time_ini=time.time()
    image_band_dic_norm={}   
    for k in image_band_dic.keys():        
        if replace_nans:
            image_band_dic_norm[k]=np.where((image_band_dic[k] == -9999) |(image_band_dic[k] == -32768), np.nan, image_band_dic[k])
        print (k, np.max(image_band_dic_norm[k]), np.max(image_band_dic[k]) ) if show_print else None
        image_band_dic_norm[k]=image_band_dic_norm[k].astype(float)/np.max(image_band_dic[k])
    time_fim=time.time()
    print (time_fim-time_ini) if show_print else None    
    return image_band_dic_norm

##### Function to normalize the values of dask image band dictionary and replace -9999 amd -32768 to np.nan
def norm_img_dic_da(image_band_dic, replace_nans=False, show_print=True):
    # making the image dictionary with values normalized
    # Replace (-9999 and -32768) to np.nan if requested  
    time_ini=time.time()
    image_band_dic_norm={}   
    for k in image_band_dic.keys(): 
        max_k = da.nanmax(image_band_dic[k])       
        if replace_nans:
            image_band_dic_norm[k]=np.where((image_band_dic[k] == -9999) |(image_band_dic[k] == -32768), np.nan, image_band_dic[k])
            image_band_dic_norm[k]=image_band_dic_norm[k]/max_k
        else:
            image_band_dic_norm[k]=image_band_dic[k]/max_k
        print (k, max_k), da.nanmin(image_band_dic[k])  if show_print else None        
        
    time_fim=time.time()
    print (time_fim-time_ini) if show_print else None    
    return image_band_dic_norm

##### function gen_mask 0f -9999 an -32768
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

def gen_mask_np(img_comp_dic, comps_sel):
    # Check if the matrix contains the element Nan and gen mask
    # como para o PCA nao podem ser passados nans, algumas coordenadas nao sao passadas e portanto nao existem componentes
    # pca associados nestas corordenadas, assim é necessário passar uma mascara parar este Nans
    time_ini = time.time()
    contains_nan={}
    for i,c  in enumerate(comps_sel):
        # Check if the matrix contains the element -9999
        print (c)
        contains_nan[i] = np.isnan(img_comp_dic[c]).any()
        if contains_nan[i]:
            #print (f"{c} true")
            if i==0:
                #mask = (img_comp_dic[c] != np.nan) 
                mask = ~np.isnan(img_comp_dic[c])
            else:
                mask = ~np.isnan(img_comp_dic[c]) & mask
    time_fim= time.time()
    print (time_fim-time_ini)
    del contains_nan
    return mask

#function para gerar mask for the dask array
def gen_mask_da(img_comp_dic, comps_sel):
    c='c1'
    contains_nan={}
    time_ini=time.time()
    for i,c  in enumerate(comps_sel):
        # Check if the matrix contains the element -9999
        print (c)
        contains_nan[i] = da.isnan(img_comp_dic[c]).any()
        if contains_nan[i]:
            #print (f"{c} true")
            if i==0:
                #mask = (img_comp_dic[c] != np.nan) 
                mask = ~da.isnan(img_comp_dic[c])
            else:
                mask = ~da.isnan(img_comp_dic[c]) & mask
    time_fim= time.time()
    print (time_fim-time_ini)
    del contains_nan
    return mask

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



