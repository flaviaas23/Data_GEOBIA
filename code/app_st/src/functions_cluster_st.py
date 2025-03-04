
import numpy as np
from math import ceil

import pandas as pd
import psutil
import time
import datetime
import sys
import os
import gc
import glob
from pathlib import Path

# import copy

# from itertools import product

import pickle
import random
# from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.colorbar import Colorbar

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from joblib import Parallel, delayed  # For parallel processing

# Add the parent directory (code/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from functions_pca.py
from functions.functions_pca import get_bandsDates,load_image_files3, list_files_to_read, save_to_pickle
from functions.functions_segmentation import get_quadrant_coords

# 20250211: Save filtered df to csv
def save_filter_thr(df, d_name, f_name, threshold, label_sel, coord_sel,\
                    quad, img_sz, save_filter=1, sh_print=0):
    '''
    save columns coords, threshold and sim value to a file
    save_filter=1 ou 2: save centroids
    save_filter=3 ou 2: save pixels
    '''
    thr_min = threshold[0]/100
    thr_max = threshold[1]/100

    file_path = d_name + f_name
    diretorio = Path(d_name)
    diretorio.mkdir(parents=True, exist_ok=True)
    #get the row of pixel selected
    row_pixel_df =  df[df['label']==label_sel]
    #position it as first row
    result_df = pd.concat([row_pixel_df, df.drop(row_pixel_df.index)], ignore_index=True)

    #save centroids:
    if (save_filter == 1) or (save_filter==2):
        save_df = result_df.loc[:,('centroid-0','centroid-1', 'sim_value')]
        save_df['thr_min'] = thr_min
        save_df['thr_max'] = thr_max
        st.write(save_df.head(2))
        
        save_df.to_csv(file_path+'.csv', index=False)

    #save pixels:
    if (save_filter == 2) or (save_filter==3):
        rowi, rowf, coli, colf = get_quadrant_coords(quad, img_sz)
        st.write (f'rowi = {rowi}, coli = {coli}')
        save_df = result_df.loc[:,('centroid-0','centroid-1', 'coords','sim_value')]
        save_df = save_df.explode('coords')
        # colocar a linha do pixel selecionado primeiro leva mais tempo que fazer depois do zip
        # t1=time.time()
        # row_pixel_df =  save_df[save_df['coords'].apply(lambda x: x == coord_sel)]
        # t2=time.time()
        # st.write (f'tempo para fazer row_pixel_df com apply :{t2-t1}')
        # st.write(row_pixel_df)
        
        st.write(save_df.head(2))
        save_df["coord_0"], save_df["coord_1"] = zip(*save_df["coords"])
        save_df.drop(columns=["coords"], inplace=True) 

        t1=time.time()
        row_pixel_df =  save_df[(save_df['coord_0'] == coord_sel[0]) & (save_df['coord_1']== coord_sel[1])]
        t2=time.time()
        st.write (f'tempo para fazer row_pixel_df depois do zip :{t2-t1}')
        st.write(row_pixel_df.head(2))

        t2=time.time()
        save_df = pd.concat([row_pixel_df, save_df.drop(row_pixel_df.index)], ignore_index=True)
        t3=time.time()
        st.write (f'tempo para concat row_pixel_df :{t2-t3}')

        #colocar coordenada referenciada ao tile original
        save_df["tile_coord_0"] = save_df["coord_0"] + rowi
        save_df["tile_coord_1"] = save_df["coord_1"] + coli

        st.write(save_df.shape)
        st.write(save_df.head(2))
        save_df.to_csv(file_path+'.csv', index=False)

        del save_df
        gc.collect()

    st.write (f'filtered df save to {file_path}')
    return


#Function to plot imagem with clara cluster
#20240904: copied from Test_PCA_img_full_working-Copy notebook
def plot_clustered_clara_st(img_sel_norm, n_opt_df, list_cluster_n_opt, \
                         n_cols=3, cl_map='tab20', plot_orig_img=1, plot_centroids=1, chart_size=(12, 12)):
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
    
    #print (f'num plots = {num_plots} , n_rows = {n_rows}')
    
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

    if (plot_centroids):
        x_centroids = n_opt_df['centroid-0']
        y_centroids = n_opt_df['centroid-1']

    else:
        
        df_exploded = n_opt_df.explode('coords')
        x_pixels = [p[1] for p in list(df_exploded['coords'])]
        y_pixels = [p[0] for p in list(df_exploded['coords'])]
        
    for i, n in enumerate(list_cluster_n_opt):
                
        #df = props_df_sel[id_test][['std_B11','std_B8A','std_B02']]
        r = (i//n_cols)# + 1-1
        c = int(i%n_cols)#+1-1

        if n_rows==1:
            ax = axes[c]
        else:
            ax = axes[r,c]
        #print (r,c)
        if (r==0) & (i==0):
            ax.imshow(img_sel_norm)
            ax.set_title(f'Original Image', fontsize=7)
            ax.axis('off')
            continue

        #colors=[cl(x) for x in n_opt_df[n]]
        #n_opt_df['cor'] = n_opt_df[n].apply(lambda row: cl(row))
        
        if (plot_centroids):
            colors=[cl(x) for x in n_opt_df[n]]
            ax.scatter(x_centroids, y_centroids, s=1, color=colors)
            
            #n_opt_df['cor'] = n_opt_df[n].apply(lambda row: cl(row))
            #ax.scatter(x_centroids, y_centroids, s=1, color=n_opt_df['cor'])
        else:
            
            #df_exploded = n_opt_df.explode('coords')

            #estava usando esta opcao
            # colors=[cl(x) for x in df_exploded[n]] #tempo similar a criar uma coluna
            # ax.scatter(x_pixels, y_pixels, s=1, color=colors)
            
            # x_pixels = [p[1] for p in list(df_exploded['coords'])]
            # y_pixels = [p[0] for p in list(df_exploded['coords'])]
                
            #df_exploded['cor'] = df_exploded[n].apply(lambda row: cl(row))                        
            #ax.scatter(x_pixels, y_pixels, s=1, color=df_exploded['cor'])
            
            n_opt_df['cor'] = n_opt_df[n].apply(lambda row: cl(row))
            df_exploded = n_opt_df.explode('coords')
            ax.scatter(x_pixels, y_pixels, s=1, color=df_exploded['cor'])
            del df_exploded            
        
        ax.imshow(img_sel_norm)
        #axes[r, c].imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
        ax.set_title(f'Image clustered n_opt {n}', fontsize=7)
               
        # Customize subplot title
        #axes[r,c].set_title(sub_titles[i], fontsize=7)

        # Hide axis ticks and labels
        ax.axis('off')
        
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
    # plt.tight_layout()
    # plt.show()
    #fig.show()
    # Display the plot in Streamlit
    st.pyplot(fig)

def plot_images_cluster_st(img_sel_norm, filter_centroid_sel_df_orig,  id_test,\
                        list_clara,label_value, n_cols=3, plot_orig_img=1, cl_map='Blues', chart_size=(12, 12)):
    ''''
    funcion to plot images with cluster group based on pixel selection in cols and rows with the original in 
    first position
    '''
    filter_centroid_sel_df = filter_centroid_sel_df_orig.copy()
    img_sel_norm = np.clip(img_sel_norm, 0, 1) # fazer isso para remover valores negativos 
    if plot_orig_img:
        elemento='original_image'
        list_clara = np.insert(list_clara, 0, elemento)
    num_plots=len(list_clara)
    n_rows = np.ceil(num_plots/n_cols).astype(int)
    #print (f'nrows = {n_rows}, cols = {n_cols}, num_plots = {num_plots}')
    sub_titles=[]
    for k in list_clara:
        subt=f'{k}'
        sub_titles.append(subt)

    #print (sub_titles)
    
    fig,axes = plt.subplots(nrows=int(n_rows), ncols=n_cols, sharex=True, sharey=True,figsize=chart_size)
    #axes = axes.flatten()

    
    for i, n in enumerate(list_clara):
                
        #df = props_df_sel[id_test][['std_B11','std_B8A','std_B02']]
        r = (i//n_cols)# + 1-1
        c = int(i%n_cols)#+1-1
        #print (r,c, n)
        if n_rows==1:
            ax = axes[c]
        else:
            ax = axes[r,c]
            
        if (r==0) & (i==0) & plot_orig_img:
            ax.imshow(img_sel_norm)
            ax.set_title(f'Original Image', fontsize=7)
            ax.axis('off')
            continue
        
        filter_centroid_sel_df[n] = filter_centroid_sel_df_orig[n].copy()
        # print(filter_centroid_sel_df[n].dtypes)
        filter_centroid_sel_df[n].loc[:,'cor'] ='blue'
        #axes[r,c].imshow(mark_boundaries(img_sel, segments_slic_sel[id_test]))
        x_centroids=[x for x in filter_centroid_sel_df[n]['centroid-0']]
        y_centroids=[y for y in filter_centroid_sel_df[n]['centroid-1']]
        ax.scatter(x_centroids, y_centroids,s=1, color=list(filter_centroid_sel_df[n]['cor']))

        #para plotar o ponto selecionado como vermelho
        c1 = filter_centroid_sel_df[n]['label'] == label_value #centroid_sel  #label_value-1
        x_label = filter_centroid_sel_df[n].loc[c1,'centroid-0']
        y_label = filter_centroid_sel_df[n].loc[c1,'centroid-1']

        del filter_centroid_sel_df[n]
        gc.collect()

        ax.scatter([x_label], [y_label],s=10, color='red')#list(filter_centroid_sel_df['cor']))
        
        ax.imshow(img_sel_norm)
        #axes[r, c].imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
        ax.set_title(f'Image segmented {n}', fontsize=7)
               
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
            if n_rows==1:
                ax = axes[cc]
            else:
                ax = axes[r,cc]
            #print (r,cc)
            ax.axis('off')
            ax.set_visible(False)

    #fig.update_layout(showlegend=True, title_font=dict(size=10), width=chart_size[0], height=chart_size[1])

    # # Update subtitle font size for all subplots
    # for annotation in fig['layout']['annotations']:
    #     annotation['font'] = dict(size=10)

    # Adjust layout
    # plt.tight_layout()
    # plt.show()
    #fig.show()
    # Display the plot in Streamlit
    st.pyplot(fig)

#plot the nans of image
def plot_images_nans_st(img_sel_norm, n_opt_df, n_opt_key_ms, dic_cluster_ski_ms,\
                        test, coords_snic_ski_df_nan, list_clara, \
                        coord_sel, plot_centroids=1, n_cols=3, plot_orig_img=1,\
                        cl_map='Blues', chart_size=(12, 12)): #cl_map=Blues
    ''''
    funcion to plot the nans of the images with the original in 
    first position
    '''
    # coords_snic_ski_df_nan = coords_snic_ski_df_nan.copy()
    st.write(f'1 coords_snic_ski_df_nan shape {coords_snic_ski_df_nan.shape}')
    coords_snic_ski_df_nan_exploded = coords_snic_ski_df_nan.explode('coords')
    st.write(f'2 coords_snic_ski_df_nan exploded shape {coords_snic_ski_df_nan_exploded.shape}')

    img_sel_norm = np.clip(img_sel_norm, 0, 1) # fazer isso para remover valores negativos 
    if plot_orig_img:
        elemento='original_image'
        list_clara = np.insert(list_clara, 0, elemento)
    num_plots=len(list_clara)
    n_rows = np.ceil(num_plots/n_cols).astype(int)
    #print (f'nrows = {n_rows}, cols = {n_cols}, num_plots = {num_plots}')
    sub_titles=[]
    for k in list_clara:
        subt=f'{k}'
        sub_titles.append(subt)

    #print (sub_titles)
    
    fig,axes = plt.subplots(nrows=int(n_rows), ncols=n_cols, sharex=True, sharey=True,figsize=chart_size)
    #axes = axes.flatten()

    
    for i, n in enumerate(list_clara):
                
        #df = props_df_sel[id_test][['std_B11','std_B8A','std_B02']]
        r = (i//n_cols)# + 1-1
        c = int(i%n_cols)#+1-1
        #print (r,c, n)
        if n_rows==1:
            ax = axes[c]
        else:
            ax = axes[r,c]
            
        if (r==0) & (i==0) & plot_orig_img:
            ax.imshow(img_sel_norm)
            ax.set_title(f'Original Image', fontsize=7)
            ax.axis('off')
            continue
        
        # filter_centroid_sel_df[n] = filter_centroid_sel_df_orig[n].copy()
        # print(filter_centroid_sel_df[n].dtypes)
        # filter_centroid_sel_df[n].loc[:,'cor'] ='blue'
        coords_snic_ski_df_nan_exploded.loc[:,'cor'] ='white'
        #axes[r,c].imshow(mark_boundaries(img_sel, segments_slic_sel[id_test]))
        # x_centroids=[x for x in filter_centroid_sel_df[n]['centroid-0']]
        # y_centroids=[y for y in filter_centroid_sel_df[n]['centroid-1']]
        #axes[r,c].imshow(mark_boundaries(img_sel, segments_slic_sel[id_test]))
        x_pixels=[x[1] for x in coords_snic_ski_df_nan_exploded['coords']]
        y_pixels=[y[0] for y in coords_snic_ski_df_nan_exploded['coords']]
        ax.scatter(x_pixels, y_pixels,s=1, color=list(coords_snic_ski_df_nan_exploded['cor']))
        # ax.scatter(x_centroids, y_centroids,s=1, color=list(filter_centroid_sel_df[n]['cor']))
        del coords_snic_ski_df_nan_exploded
        gc.collect()

        #para plotar o ponto selecionado como black
        # c1 = filter_centroid_sel_df[n]['label'] == label_value #centroid_sel  #label_value-1
        x_label = coord_sel[1] #filter_centroid_sel_df[n].loc[c1,'centroid-0']
        y_label = coord_sel[0] # filter_centroid_sel_df[n].loc[c1,'centroid-1']

        ax.scatter([x_label], [y_label],s=10, color='black')#list(filter_centroid_sel_df['cor']))
        
        #### plotar os clusters junto com os nans 
        cl = plt.get_cmap(cl_map)
        col_name = test+'_'+str(n_opt_key_ms)
        n_opt_df[col_name] = dic_cluster_ski_ms[n_opt_key_ms]
        if plot_centroids:
            x_centroids = n_opt_df['centroid-0']
            y_centroids = n_opt_df['centroid-1']  
            colors=[cl(x) for x in n_opt_df[col_name]] 
            ax.scatter(x_centroids, y_centroids, s=1, color=colors)
            ax.set_title(f'Image with Nans pixels and clusters centroids', fontsize=7)
            
        else:
            #plot pixels

            n_opt_df_exploded = n_opt_df.explode('coords')
            x_pixels=[x[1] for x in n_opt_df_exploded['coords']]
            y_pixels=[y[0] for y in n_opt_df_exploded['coords']]
            colors=[cl(x) for x in n_opt_df_exploded[col_name]] 
            ax.scatter(x_pixels, y_pixels, s=1, color=colors)
            ax.set_title(f'Image with Nans pixels and clusters pixels', fontsize=7)
            
            del n_opt_df_exploded
            gc.collect()

        ###
        ax.imshow(img_sel_norm)
        #axes[r, c].imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test]))
        # ax.set_title(f'Image with Nans and clusters', fontsize=7)
               
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
            if n_rows==1:
                ax = axes[cc]
            else:
                ax = axes[r,cc]
            #print (r,cc)
            ax.axis('off')
            ax.set_visible(False)

    #fig.update_layout(showlegend=True, title_font=dict(size=10), width=chart_size[0], height=chart_size[1])

    # # Update subtitle font size for all subplots
    # for annotation in fig['layout']['annotations']:
    #     annotation['font'] = dict(size=10)

    # Adjust layout
    # plt.tight_layout()
    # plt.show()
    #fig.show()
    # Display the plot in Streamlit
    st.pyplot(fig)


###
def gen_filter_centroid_df(test, df, n_opt, centroid_sel, cluster='' ):
    '''filter in df SPs of selected centroid'''
    
    if not cluster:
        filter_centroid_sel_df = df.loc[:,('label', 'centroid-0','centroid-1', 'cluster_'+str(n_opt))]
        cl_val_centroid_sel = filter_centroid_sel_df.loc[filter_centroid_sel_df['label'] == centroid_sel, 'cluster_'+str(n_opt)].iloc[0]
        filter_centroid_sel_df = filter_centroid_sel_df.loc[filter_centroid_sel_df['cluster_'+str(n_opt)] == cl_val_centroid_sel]
    else:
        filter_centroid_sel_df = df.loc[:,('label', 'centroid-0','centroid-1')]
        filter_centroid_sel_df[test+'_'+str(n_opt)] = cluster[n_opt]
        cl_val_centroid_sel = filter_centroid_sel_df.loc[filter_centroid_sel_df['label'] == centroid_sel, test+'_'+str(n_opt)].iloc[0]
        filter_centroid_sel_df = filter_centroid_sel_df.loc[filter_centroid_sel_df[test+'_'+str(n_opt)] == cl_val_centroid_sel]
    print (cl_val_centroid_sel)
    

    return filter_centroid_sel_df

def gen_filter_coord_df(test, df, n_opt, coord_sel, cluster='' ):
    '''filter in df SPs of selected pixel'''
    ti = time.time()
    
    if not cluster:
        #passando - n_opt_df com a colunas do cluster dos n_opt (test+'_'+str(n_opt))
        t1=time.time()
        df_exploded = df.explode('coords')
        t2=time.time()
        print (f'tempo explode {t2-t1}')
        #obter valor do grupo do cluster do pixel selecionado
        res = df_exploded.loc[df_exploded["coords"].apply(lambda x: x == coord_sel), ["label",test+'_'+str(n_opt)]]#.iloc[0]
        cl_val_pixel_sel = res[test+'_'+str(n_opt)].iloc[0]
        label = res["label"].iloc[0]
        filter_centroid_sel_df = df.loc[df[test+'_'+str(n_opt)] == cl_val_pixel_sel]
    
    
    else:
        #passando o df com coords e o cluster para gerar o df e fazer o filtro
        filter_centroid_sel_df = df.loc[:,('label', 'centroid-0','centroid-1', 'coords')]
        filter_centroid_sel_df[test+'_'+str(n_opt)] = cluster[n_opt]
        t11 = time.time()
        filter_centroid_sel_df_exploded = filter_centroid_sel_df.explode('coords')
        t22 = time.time()
        st.write (f'tempo explode {t22-t11}')
        #obter valor do grupo do cluster e do label do pixel selecionado
        res = filter_centroid_sel_df_exploded.loc[filter_centroid_sel_df_exploded["coords"].apply(lambda x: x == coord_sel), ["label",test+'_'+str(n_opt)]]#.iloc[0]
        # st.write(f'res = {res}')
        if res.empty:
            cl_val_pixel_sel = None
            label = None
            filter_centroid_sel_df = None
            st.write(f"coords sel {coord_sel} is none")
        else:
            cl_val_pixel_sel = res[test+'_'+str(n_opt)].iloc[0]
            label = res["label"].iloc[0]
            filter_centroid_sel_df = filter_centroid_sel_df.loc[filter_centroid_sel_df[test+'_'+str(n_opt)] == cl_val_pixel_sel]

    tf = time.time()
    print (label,cl_val_pixel_sel)
    st.write(f'Tempo gen_filter_coord_df = {tf-ti}')
    return label, filter_centroid_sel_df

# 20250130: gen filter cluster df based on label of the coords sel
def gen_filter_centroid_sel_df(test, filter_centroid_sel_df, n_opt_key, \
                               label, dic_cluster_ski, sh_print=0 ):
    '''filter df based on cluster of the label
    '''
    st.write(f'label = {label} n_opt_key = {n_opt_key}')
    filter_centroid_sel_df[test+'_'+str(n_opt_key)] = dic_cluster_ski[n_opt_key]
    # res = filter_centroid_sel_df.loc[filter_centroid_sel_df["label"].apply(lambda x: x == coord_sel), ["label",test+'_'+str(n_opt_key)]]#.iloc[0]
    res = filter_centroid_sel_df.loc[filter_centroid_sel_df['label'] == label]
    if sh_print:
        st.write(f'res cluster ={res}')
    if not res.empty:
        cl_val_pixel_sel = res[test+'_'+str(n_opt_key)].iloc[0]
        filter_centroid_sel_df = filter_centroid_sel_df.loc[filter_centroid_sel_df[test+'_'+str(n_opt_key)] == cl_val_pixel_sel]

    return filter_centroid_sel_df

#
def gen_filter_coord_thres_df(test, df, n_opt, coord_sel, matrix_sim, threshold, cluster='', sh_print =0 ):
    '''filter in df SPs of selected pixel with sim > threshold
       threshold: similarity threshold min and max
    '''
    
    thr_min = threshold[0]/100
    thr_max = threshold[1]/100
    
    if not cluster:
        #passando - n_opt_df com a colunas do cluster dos n_opt
        t1=time.time()
        df_exploded = df.explode('coords')
        t2=time.time()
        print (f'tempo explode {t2-t1}')
        #obter valor do grupo do cluster do pixel selecionado
        res = df_exploded.loc[df_exploded["coords"].apply(lambda x: x == coord_sel), ["label",test+'_'+str(n_opt)]]#.iloc[0]
        cl_val_pixel_sel = res[test+'_'+str(n_opt)].iloc[0]
        label = res["label"].iloc[0]
        
        centroid_row_matrix_sim = matrix_sim[label-1,:]
        st.write(matrix_sim[label-1,label-1], matrix_sim[label,label])
        df['sim_value'] = centroid_row_matrix_sim

        # filter_centroid_sel_df = df.loc[df[test+'_'+str(n_opt)] == cl_val_pixel_sel]
        # filter_centroid_sel_df = df.loc[df[test+'_'+str(n_opt)] == cl_val_pixel_sel]
    
        filter_centroid_sel_df = df[(df['sim_value']>=thr_min) &
                                    (df['sim_value']<=thr_max)]

    else:
        #passando o df com coords e o cluster para gerar o df e fazer o filtro
        filter_centroid_sel_df = df.loc[:,('label', 'centroid-0','centroid-1', 'coords')]
        filter_centroid_sel_df[test+'_'+str(n_opt)] = cluster[n_opt]
        t1=time.time()
        filter_centroid_sel_df_exploded = filter_centroid_sel_df.explode('coords')
        t2=time.time()
        print (f'tempo explode {t2-t1}')
        #obter valor do grupo do cluster e do label do pixel selecionado
        res = filter_centroid_sel_df_exploded.loc[filter_centroid_sel_df_exploded["coords"].apply(lambda x: x == coord_sel), ["label",test+'_'+str(n_opt)]]#.iloc[0]
        if res.empty:
            label = None
            filter_centroid_sel_df = None
        else:
            cl_val_pixel_sel = res[test+'_'+str(n_opt)].iloc[0]
            label = res["label"].iloc[0]
            ind_label_sel = filter_centroid_sel_df.index[filter_centroid_sel_df['label'] == label]
            if sh_print:
                st.write(f'indice do label sel: {ind_label_sel} matrix_sim shape {matrix_sim.shape}')
            
            #Get the similarity of the centroid selected with others and adds it as column to df
            # centroid_row_matrix_sim = matrix_sim[label-1,:]
            # centroid_row_matrix_sim = matrix_sim[label,:] #comentei e coloquei pelo indice do label
            # st.write(f'1 len centroid_row_matrix_sim do label {label}: {len(centroid_row_matrix_sim)}')
            centroid_row_matrix_sim = matrix_sim[ind_label_sel[0],:]
            # st.write(f'2 len centroid_row_matrix_sim do ind label {ind_label_sel}: {len(centroid_row_matrix_sim)}')
            # df['sim_value'] = centroid_row_matrix_sim #20250130
            df['sim_value'] = centroid_row_matrix_sim
            # filter_centroid_sel_df = filter_centroid_sel_df.loc[filter_centroid_sel_df[test+'_'+str(n_opt)] == cl_val_pixel_sel]
            filter_centroid_sel_df = df[(df['sim_value']>=thr_min) &
                                        (df['sim_value']<=thr_max)]

            print (label,cl_val_pixel_sel)

    return label, filter_centroid_sel_df

# 20253001 gen df with similarity info of pixel selected with threshold
def gen_filter_thres_df(test, df, n_opt, label, ind_label_sel, matrix_sim, threshold, cluster='', sh_print =0 ):
    ''' filter df based on threshold range of similaity values
    '''
    thr_min = threshold[0]/100
    thr_max = threshold[1]/100

    df[test+'_'+str(n_opt)] = cluster[n_opt]
    centroid_row_matrix_sim = matrix_sim[ind_label_sel[0],:]
    df['sim_value'] = centroid_row_matrix_sim

    filter_centroid_sel_df = df[(df['sim_value']>=thr_min) &
                                (df['sim_value']<=thr_max) ]

    return filter_centroid_sel_df


#get the info from coord 
def get_coord_label(df, coord_sel, cluster=1, sh_print =0 ):
    '''filter in df SPs of selected pixel with sim > threshold
       threshold: similarity threshold min and max
    '''
    
   
    if not cluster:
        #passando - n_opt_df com a colunas do cluster dos n_opt
        t1=time.time()
        df_exploded = df.explode('coords')
        t2=time.time()
        print (f'tempo explode {t2-t1}')
        #obter valor do grupo do cluster do pixel selecionado
        res = df_exploded.loc[df_exploded["coords"].apply(lambda x: x == coord_sel), ["label",test+'_'+str(n_opt)]]#.iloc[0]
        cl_val_pixel_sel = res[test+'_'+str(n_opt)].iloc[0]
        label = res["label"].iloc[0]
        
        centroid_row_matrix_sim = matrix_sim[label-1,:]
        st.write(matrix_sim[label-1,label-1], matrix_sim[label,label])
        df['sim_value'] = centroid_row_matrix_sim

        # filter_centroid_sel_df = df.loc[df[test+'_'+str(n_opt)] == cl_val_pixel_sel]
        # filter_centroid_sel_df = df.loc[df[test+'_'+str(n_opt)] == cl_val_pixel_sel]
    
        filter_centroid_sel_df = df[(df['sim_value']>=thr_min) &
                                    (df['sim_value']<=thr_max)]

    else:
        #passando o df com coords e o cluster para gerar o df e fazer o filtro
        filter_centroid_sel_df = df.loc[:,('label', 'centroid-0','centroid-1', 'coords')] #comentei 20250210
        # filter_centroid_sel_df[test+'_'+str(n_opt)] = cluster[n_opt]
        t1=time.time()
        filter_centroid_sel_df_exploded = filter_centroid_sel_df.explode('coords')
        t2=time.time()
        #obter valor do grupo do cluster e do label do pixel selecionado
        res = filter_centroid_sel_df_exploded.loc[filter_centroid_sel_df_exploded["coords"].apply(lambda x: x == coord_sel), ["label"]]#.iloc[0]
        if sh_print:
            st.write(f'tempo explode {t2-t1}, coord_sel = {coord_sel}')
            st.write(f'get_coord_label res = {res}')
        del filter_centroid_sel_df_exploded
        gc.collect()
        if res.empty:
            label = None
            ind_label_sel = None
            filter_centroid_sel_df = None
        else:
            # cl_val_pixel_sel = res[test+'_'+str(n_opt)].iloc[0]
            label = res["label"].iloc[0]
            ind_label_sel = filter_centroid_sel_df.index[filter_centroid_sel_df['label'] == label]
            if sh_print:
                st.write(f'label ={label} indice do label sel: {ind_label_sel} ')
            
            # #Get the similarity of the centroid selected with others and adds it as column to df
            # # centroid_row_matrix_sim = matrix_sim[label-1,:]
            # # centroid_row_matrix_sim = matrix_sim[label,:] #comentei e coloquei pelo indice do label
            # # st.write(f'1 len centroid_row_matrix_sim do label {label}: {len(centroid_row_matrix_sim)}')
            # centroid_row_matrix_sim = matrix_sim[ind_label_sel[0],:]
            # # st.write(f'2 len centroid_row_matrix_sim do ind label {ind_label_sel}: {len(centroid_row_matrix_sim)}')
            # # df['sim_value'] = centroid_row_matrix_sim #20250130
            # df['sim_value'] = centroid_row_matrix_sim
            # # filter_centroid_sel_df = filter_centroid_sel_df.loc[filter_centroid_sel_df[test+'_'+str(n_opt)] == cl_val_pixel_sel]
            # filter_centroid_sel_df = df[(df['sim_value']>=thr_min) &
            #                             (df['sim_value']<=thr_max)]

            st.write(f'label= {label}, ind_label_sel = {ind_label_sel[0]}')

    return label, ind_label_sel # , filter_centroid_sel_df 20250210: como nao estou filtrando nao devolvo mais 
##

#20250115: copiado do code/app_st/geo_vis.py
def calc_cor(valor,c_map='Blues'):
    colormap=plt.get_cmap(c_map)
    #colormap=plt.get_cmap('Blues')
    return colormap(valor)
    
#20250115: copiado do code/app_st/geo_vis.py e modificado
#funcao para plotar o df filtrado 
def plot_img_pixel_sel(img_sel_norm, filter_centroid_sel_df_orig, label_value, #threshold, \
                       plot_centroids=True,\
                       id_test=0, sh_print=0 ):
    ''''
    plot image with groups of pixel selected
    '''
    filter_centroid_sel_df = filter_centroid_sel_df_orig.copy()
    fig,ax = plt.subplots()
    # thr_min = threshold[0]/100
    # thr_max = threshold[1]/100
    # centroid_row_matrix_sim = matrix_sim[id_test][label_value-1,:]
    
    # centroid_sel_df['sim_value'] = centroid_row_matrix_sim                        
     
    # filter_centroid_sel_df = centroid_sel_df[(centroid_sel_df['sim_value']>=thr_min) &
    #                                          (centroid_sel_df['sim_value']<=thr_max)]
    
    filter_centroid_sel_df.loc[:,'cor'] = filter_centroid_sel_df['sim_value'].astype(object).apply(calc_cor, args=('Blues',))
    filter_centroid_sel_df.loc[label_value,'cor']='red'
    if sh_print:
        st.write(filter_centroid_sel_df.shape)
        st.write(filter_centroid_sel_df.head(2))
        st.write(filter_centroid_sel_df.tail(2))
    #colormap=plt.get_cmap('Blues')
    
    time_ini = time.time()
    if (plot_centroids):
        x_centroids=[x for x in filter_centroid_sel_df['centroid-0']]
        y_centroids=[y for y in filter_centroid_sel_df['centroid-1']]
        plt.scatter(x_centroids, y_centroids,s=1, color=list(filter_centroid_sel_df['cor']))
    else:
        x_sel= filter_centroid_sel_df.loc[label_value-1, 'centroid-0']
        y_sel= filter_centroid_sel_df.loc[label_value-1, 'centroid-1']
        plt.scatter(x_sel, y_sel,s=1, color='red')
        #plt.plot(x_sel, y_sel,marker='o',markersize=1, color='red')

        df_exploded = filter_centroid_sel_df.explode('coords')
        x_pixels = [p[1] for p in list(df_exploded['coords'])]
        y_pixels = [p[0] for p in list(df_exploded['coords'])]
        plt.scatter(x_pixels, y_pixels, s=1, color=df_exploded['cor'])
    
    del filter_centroid_sel_df
    gc.collect()

    time_fim = time.time()
    #pintar o ponto escolhido obrigatoriamente
    #plt.plot(row['centroid-1'], row['centroid-0'], marker=marker,markersize=markersz, color=row['cor'])
    #ax.imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test], color=(128,128,128)))
    ax.imshow(img_sel_norm)
    ax.axis('off')
    plt.tight_layout()
    #plt.show()
    st.pyplot(fig)
    if sh_print:
        st.write(time_fim-time_ini)
    return #filter_centroid_sel_df

#20250126: function to load quadrant to be evaluated
def load_img_quadrant(tif_dir, name_img, bands_sel, t_day, q, img_full=0, sh_print=0):
    "load quadrant of image"
    st.write(f'load_img_quadrant q= {q}')
    band_tile_img_files = list_files_to_read(tif_dir, name_img, sh_print=0)
    # bands, dates = get_bandsDates(band_tile_img_files, tile=0)
    
    image_band_dic = {}
    pos = -2#-1
    band_img_file_to_load = [x for x in band_tile_img_files if t_day in x.split('/')[-1]]
    image_band_dic = load_image_files3(band_img_file_to_load, pos=pos)

    # load full tif RGB image to show
    if img_full:
        t_img_band_dic_norm = {}
        for b in bands_sel:
            max_b = np.max(image_band_dic[b])
            t_img_band_dic_norm[b]=image_band_dic[b].astype(float)/max_b

        img_sel_norm = np.dstack((t_img_band_dic_norm[bands_sel[0]],\
                            t_img_band_dic_norm[bands_sel[1]],\
                            t_img_band_dic_norm[bands_sel[2]]))

        img_sel_norm = np.clip(img_sel_norm, 0, 1)
        img_width = img_sel_norm.shape[0]
        img_sel_uint8 = (img_sel_norm * 255).astype(np.uint8)  # Scale and convert to uint8

    # Get the image quadrant
    img_width = image_band_dic[bands_sel[0]].shape[0]
    rowi, rowf, coli, colf = get_quadrant_coords(q, img_width)
    # rowi, rowf, coli, colf
    image_band_dic_q={}
    for x in bands_sel:
        image_band_dic_q[x] = image_band_dic[x][rowi:rowf, coli:colf]

    t_img_band_dic_q_norm={}
    for b in bands_sel:
        max_b = np.max(image_band_dic_q[b])
        t_img_band_dic_q_norm[b]=image_band_dic_q[b].astype(float)/max_b

    t_img_sel_norm_q = np.dstack((t_img_band_dic_q_norm[bands_sel[0]],\
                        t_img_band_dic_q_norm[bands_sel[1]],\
                        t_img_band_dic_q_norm[bands_sel[2]]))

    minValue = np.amin(t_img_sel_norm_q)
    if minValue < 0:
        print (minValue)
        t_img_sel_norm_q = np.clip(t_img_sel_norm_q, 0, 1)
    t_img_sel_uint8_q = (t_img_sel_norm_q * 255).astype(np.uint8)  # Scale and convert to uint8

    if img_full:
        return img_sel_norm, img_width, img_sel_uint8, t_img_sel_norm_q, t_img_sel_uint8_q
    else:
        return t_img_sel_norm_q, t_img_sel_uint8_q, img_width


def gen_sub_matrix_sup(df, mm_matrix_sim_sel, cluster_all=0, sh_print=0):
    # recebe df só com um grupo de cluster
    # e retorna a matriz superior sem a diagonal

    # row_indices =  df.index.tolist()
    row_indices = list(map(int, df.index.tolist() ))
    print (f'len indices df {len(row_indices)}, shape df {df.shape}') if sh_print else None

    if cluster_all:
        #gera sub_matrix com os pontos do cluster em relacao a todos 
        sub_matrix = mm_matrix_sim_sel[row_indices, :]#[:, row_indices]
        # Convert dtype to avoid ValueError
        sub_matrix = sub_matrix.astype(np.float32, copy=False)
    else:
        # gera a submatrix de similaridade dentro do cluster 
        # sub_matrix = mm_matrix_sim_sel[np.ix_(row_indices, row_indices)]  #nao sei qual opcao mais rapida
        t1=time.time()
        sub_matrix = mm_matrix_sim_sel[row_indices, :][:, row_indices]
        # Convert dtype to avoid ValueError
        sub_matrix = sub_matrix.astype(np.float32, copy=False)
        t2=time.time()
        print (f'tempo para obter submatriz com [row_indices, :][:, row_indices] = {t2-t1}')
    
    # print (sub_matrix) if sh_print else None
    # pega apenas os valores da matriz superior excluindo a diagonal
    
    sub_matrix_superior = sub_matrix[np.triu_indices_from(sub_matrix, k=1)]
    
    print (sub_matrix.shape, len(sub_matrix_superior)) if sh_print else None
    
    return sub_matrix_superior

# Cache expensive computations
# @st.cache_resource
# @st.cache_data(hash_funcs={zarr.core.Array: lambda _: None})
@st.cache_data
# Function to compute histogram data for a cluster
def compute_histogram_data(cluster_df, _mm_matrix_sim_sel):
    # sub_matrix = gen_sub_matrix_sup(cluster_df, mm_matrix_sim_sel, cluster_all=0, sh_print=0)
    # return np.array(sub_matrix)
    return gen_sub_matrix_sup(cluster_df, _mm_matrix_sim_sel, cluster_all=0, sh_print=0)

def plot_hist_clusters(n_key_ms_sel, list_clusters, n_cols, n_opt_df, mm_matrix_sim_sel ):
    "plot historam of similarity values of each cluster of n"

    # n_key_ms_sel = list(dic_cluster_ski_ms.keys())[n_opt_ind_ms+1]
    col_name = 'cluster_ms_' + n_key_ms_sel

    # list_clusters = n_opt_df[col_name].unique()
    # list_clusters.sort()


    # Calculate the number of rows needed for subplots
    n_rows = ceil(int(n_key_ms_sel.split('_')[0])/n_cols) 

    fig = make_subplots(rows=int(n_rows), 
                        cols=int(n_cols), 
                        subplot_titles=[f"Cluster {n}" for n in list_clusters],
                        horizontal_spacing=0.1, vertical_spacing=0.1
                        )

    # Precompute filtered DataFrames for each cluster
    # cluster_data = {n: n_opt_df.loc[n_opt_df[col_name] == n] for n in list_clusters}

    # Compute histogram data in parallel
    # histogram_data = Parallel(n_jobs=-1)(
    #     delayed(compute_histogram_data)(cluster_df, mm_matrix_sim_sel) for cluster_df in cluster_data.values()
    # )
   
    for i, n in enumerate(list_clusters):
    # Add histogram traces to the figure
    # for i, (n, data) in enumerate(zip(list_clusters, histogram_data)):
        r = (i // n_cols) + 1  # Plotly is 1-based index for subplots
        c = (i % n_cols) + 1

        filter_cluster_df = n_opt_df.loc[n_opt_df[col_name] == n]
        # sub_matrix_superior1 = gen_sub_matrix_sup(filter_cluster_df, mm_matrix_sim_sel, cluster_all=0, sh_print=0)
        # Generate histogram data **without keeping it in memory**
        data = compute_histogram_data(filter_cluster_df, mm_matrix_sim_sel)
        fig.add_trace(
            go.Histogram(x=data, nbinsx=100, name=f"Cluster {n}"),
            row=r, col=c
        )
        del data, filter_cluster_df
        gc.collect()

    n_matrix = 2 # por enquanto só olho a matriz de sim gerada da clusterizacao com matriz de dist
                 # se resolver olhas a 1a matriz de sim passar como parametro
    fig.update_xaxes(range=[0, 1])
    fig.update_layout(title=f"Histograms of Clusters for n_opt {n_key_ms_sel} for Matrix sim {n_matrix}", 
                      showlegend=True
                      )
    
    st.plotly_chart(fig)

    # Clean up memory
    # del cluster_data, histogram_data
    # gc.collect()

def plot_hist_clusters_old(n_key_ms_sel, n_cols, n_opt_df, mm_matrix_sim_sel ):
    "plot historam of similarity values of each cluster of n"

    # n_key_ms_sel = list(dic_cluster_ski_ms.keys())[n_opt_ind_ms+1]
    col_name = 'cluster_ms_' + n_key_ms_sel

    list_clusters = n_opt_df[col_name].unique()
    list_clusters.sort()

    n_rows = ceil(int(n_key_ms_sel.split('_')[0])/n_cols) 
    fig = make_subplots(rows=int(n_rows), 
                        cols=int(n_cols), 
                        subplot_titles=[f"Cluster {n}" for n in list_clusters]
                        )

   
    for i, n in enumerate(list_clusters):
        r = (i // n_cols) + 1  # Plotly is 1-based index for subplots
        c = (i % n_cols) + 1

        filter_cluster_df = n_opt_df.loc[n_opt_df[col_name] == n]
        sub_matrix_superior1 = gen_sub_matrix_sup(filter_cluster_df, mm_matrix_sim_sel, cluster_all=0, sh_print=0)

        fig.add_trace(
            go.Histogram(x=sub_matrix_superior1, nbinsx=100, name=f"Cluster {n}"),
            row=r, col=c
        )
        del sub_matrix_superior1, filter_cluster_df
        gc.collect()

    n_matrix = 2 # por enquanto só olho a matriz de sim gerada da clusterizacao com matriz de dist
                 # se resolver olhas a 1a matriz de sim passar como parametro
    fig.update_xaxes(range=[0, 1])
    fig.update_layout(title=f"Histograms of Clusters for n_opt {n_key_ms_sel} for Matrix sim {n_matrix}", showlegend=True)
    
    st.plotly_chart(fig)

from scipy.sparse import csr_matrix
# Function to compute submatrix for a cluster
def compute_submatrix_sparse(indices, mm_matrix_sim_sel, sh_print=0):
    
    if sh_print:
        # Debugging prints
        print("Matrix dtype:", mm_matrix_sim_sel.dtype)
        print("Contains NaNs:", np.isnan(mm_matrix_sim_sel).any())

    # Ensure mm_matrix_sim_sel is a NumPy array before indexing
    if not isinstance(mm_matrix_sim_sel, np.ndarray):
        mm_matrix_sim_sel = np.array(mm_matrix_sim_sel) # np.asarray(mm_matrix_sim_sel)
        
    # Extract submatrix
    submatrix = mm_matrix_sim_sel[np.ix_(indices, indices)]  # Safe slicing
    if sh_print:
        # Debugging prints
        print("Submatrix dtype:", submatrix.dtype)
        print("Contains NaNs:", np.isnan(submatrix).any())

    # Convert dtype to avoid ValueError
    submatrix = submatrix.astype(np.float32, copy=False)
    
    # # Replace NaNs with 0 (optional: can replace with np.nan_to_num(submatrix))
    # if np.isnan(submatrix).any():
    #     submatrix = np.nan_to_num(submatrix, nan=0.0)
    # submatrix = mm_matrix_sim_sel[indices, :][:, indices]

    if sh_print:
        print("Submatrix shape:", submatrix.shape)
        print("Submatrix dtype:", submatrix.dtype)
        print("Submatrix min/max:", np.nanmin(submatrix), np.nanmax(submatrix))
        print("Contains NaNs:", np.isnan(submatrix).any())
        print("Contains Infs:", np.isinf(submatrix).any())
        print("Unique dtypes:", {type(x) for x in submatrix.ravel()})  # Detect mixed types
    # Ensure contiguous memory for better performance
    submatrix = np.ascontiguousarray(submatrix, dtype=np.float32)

    # return csr_matrix(submatrix)
    return submatrix

def plot_heatmap_clusters(n_rows, list_cluster_n, n_cols,n_key_ms_sel, df, mm_matrix_sim_sel ):
    #plot the heatmap of submatrix of clusters of a n
    # n_key_ms_sel = n_opt_key_ms
    col_name= 'cluster_ms_' + n_key_ms_sel
    
    
    # list_cluster_n = df[col_name].unique()
    # list_cluster_n.sort()
    
    # Precompute row indices for each cluster
    cluster_indices = {n: df.loc[df[col_name] == n].index.tolist() for n in list_cluster_n}

    fig = make_subplots(rows=int(n_rows), cols=int(n_cols), 
                        subplot_titles=[f"Cluster {n}" for n in list_cluster_n],
                        horizontal_spacing=0.1, vertical_spacing=0.1
                       )
    # Define a shared colorbar
    shared_colorbar = dict(
        len=0.8,  # Length of the colorbar
        y=0.5,    # Vertical position of the colorbar (centered)
        x=1.02    # Horizontal position of the colorbar (outside the subplots)
    )
    # Function to compute submatrix for a cluster
    # def compute_submatrix(indices):
    #     return mm_matrix_sim_sel[indices, :][:, indices]

    # # Compute submatrices in parallel
    # submatrices = Parallel(n_jobs=-1)(
    #     delayed(compute_submatrix)(indices, mm_matrix_sim_sel) for indices in cluster_indices.values()
    # )

    # Add heatmap traces to the figure
    for i, n in enumerate(list_cluster_n):
    # for i, (n, sub_matrix) in enumerate(zip(list_cluster_n, submatrices)):
        r = (i // n_cols) + 1  # Plotly is 1-based index for subplots
        c = (i % n_cols) + 1
    
        # filter_cluster_df = df.loc[df[col_name] == n]
        ## sub_matrix_superior1 = gen_sub_matrix_sup(filter_cluster_df, mm_matrix_sim_sel2, cluster_all=0, sh_print=0)
        # print (f'n = {n}')
        sub_matrix = compute_submatrix_sparse(cluster_indices[n], mm_matrix_sim_sel)

        # row_indices =  filter_cluster_df.index.tolist()
        # sub_matrix = mm_matrix_sim_sel[row_indices, :][:, row_indices]
            
        fig.add_trace(
            go.Heatmap(z=sub_matrix,  # The submatrix values
                       colorscale='Viridis',  # Use a colormap like Viridis
                       zmin=0, zmax=1,  # Set the range for the color scale
                       # colorbar=dict(len=0.8/n_rows, y=1 - (r-1)/n_rows)  # Adjust colorbar position
                       showscale=(i == 0)
                      ),
            row=r, col=c
        )
        # **Delete submatrix immediately after use**
        del sub_matrix #, filter_cluster_df
        gc.collect()
        

    # fig.update_xaxes(range=[0, 1])
    # fig.update_layout(title=f"Heatmap of Clusters for n_opt {n_key_ms_sel} foor Matrix sim 2", showlegend=True)
    # Update layout
    fig.update_layout(
        title=f"Heatmaps of Clusters for n= {n_key_ms_sel} for Matrix sim 2",
        showlegend=False,
        width=1000, height=800,  # Adjust the figure size as needed
        # coloraxis=dict(colorscale='Viridis', colorbar=shared_colorbar)  # Add shared colorbar
    )
    # **Free large variables before plotting**
    del cluster_indices
    gc.collect()

    st.plotly_chart(fig)

def plot_heatmap_clusters_old(n_rows, n_cols,n_key_ms_sel, df, mm_matrix_sim_sel, sh_print=1 ):
    #plot the heatmap of submatrix of clusters of a n
    # n_key_ms_sel = n_opt_key_ms
    col_name= 'cluster_ms_' + n_key_ms_sel
    
    
    list_cluster_n = df[col_name].unique()
    list_cluster_n.sort()

    # Ensure `mm_matrix_sim_sel2` is a NumPy array
    if isinstance(mm_matrix_sim_sel2, zarr.core.Array):
        mm_matrix_sim_sel = mm_matrix_sim_sel[:]
        
    fig = make_subplots(rows=int(n_rows), cols=int(n_cols), 
                        subplot_titles=[f"Cluster {n}" for n in list_cluster_n],
                        horizontal_spacing=0.1, vertical_spacing=0.1
                       )
    # Define a shared colorbar
    shared_colorbar = dict(
        len=0.8,  # Length of the colorbar
        y=0.5,    # Vertical position of the colorbar (centered)
        x=1.02    # Horizontal position of the colorbar (outside the subplots)
    )
    
    for i, n in enumerate(list_cluster_n):
        r = (i // n_cols) + 1  # Plotly is 1-based index for subplots
        c = (i % n_cols) + 1
    
        filter_cluster_df = df.loc[df[col_name] == n]
        # sub_matrix_superior1 = gen_sub_matrix_sup(filter_cluster_df, mm_matrix_sim_sel2, cluster_all=0, sh_print=0)
    
        # Extract valid indices
        # row_indices =  filter_cluster_df.index.tolist()
        row_indices = list(map(int, filter_cluster_df.index))

        # Use `np.ix_()` for efficient slicing
        # sub_matrix = mm_matrix_sim_sel[row_indices, :][:, row_indices]
        sub_matrix = mm_matrix_sim_sel2[np.ix_(row_indices, row_indices)]

        # Debugging info
        if sh_print:
            print("Submatrix shape:", sub_matrix.shape)
            print("Submatrix dtype:", sub_matrix.dtype)
            print("Contains NaNs:", np.isnan(sub_matrix).any())

        fig.add_trace(
            go.Heatmap(z=sub_matrix,  # The submatrix values
                       colorscale='Viridis',  # Use a colormap like Viridis
                       zmin=0, zmax=1,  # Set the range for the color scale
                       # colorbar=dict(len=0.8/n_rows, y=1 - (r-1)/n_rows)  # Adjust colorbar position
                       showscale=(i == 0)
                      ),
            row=r, col=c
        )
        del sub_matrix, filter_cluster_df
        gc.collect()

    # fig.update_xaxes(range=[0, 1])
    # fig.update_layout(title=f"Heatmap of Clusters for n_opt {n_key_ms_sel} foor Matrix sim 2", showlegend=True)
    # Update layout
    fig.update_layout(
        title=f"Heatmaps of Clusters for n_opt {n_key_ms_sel} for Matrix sim 2",
        showlegend=False,
        width=1000, height=800,  # Adjust the figure size as needed
        coloraxis=dict(colorscale='Viridis', colorbar=shared_colorbar)  # Add shared colorbar
    )
    st.plotly_chart(fig)
    return


def plot_heatmap_cluster_n(n, df, col_name, mm_matrix_sim_sel2 ):
    "plot heatmap of a group n in cluster selected"
            
    row_indices = list(map(int, df.loc[df[col_name] == n].index.tolist() ))
    t1=time.time()
    sub_matrix = mm_matrix_sim_sel2[np.ix_(row_indices, row_indices)]
    # Convert dtype to avoid ValueError
    sub_matrix = sub_matrix.astype(np.float32, copy=False)
    # Convert to sparse matrix plotly doesn't accpet sparse matrix
    # sparse_sub_matrix = csr_matrix(sub_matrix)
    t2=time.time()
    print (f'tempo para obter submatriz com [np.ix_(row_indices, row_indices)] = {t2-t1}')

    fig = go.Figure(data=go.Heatmap(
        z=sub_matrix, #sub_matrix, 
        colorscale='Viridis', 
        zmin=0, 
        zmax=1
    ))
    del sub_matrix
    gc.collect()

    fig.update_layout(title=f"Heatmap of Submatrix {n} of {col_name}")
    st.plotly_chart(fig)

def plot_hist_cluster_n(n, df, col_name, mm_matrix_sim_sel ):
    "plot heatmap of a group n in cluster selected"
            
    # row_indices = list(map(int, df.loc[df[col_name] == n].index.tolist() ))
    # sub_matrix = mm_matrix_sim_sel[np.ix_(row_indices, row_indices)]

    filter_cluster_df = df.loc[df[col_name] == n]
    # sub_matrix_superior1 = gen_sub_matrix_sup(filter_cluster_df, mm_matrix_sim_sel, cluster_all=0, sh_print=0)
    # Generate histogram data **without keeping it in memory**
    data_h = compute_histogram_data(filter_cluster_df, mm_matrix_sim_sel)
    fig = go.Figure(data =[go.Histogram(
                                        x=data_h, nbinsx=100, name=f"Cluster {n}"
                                        )
                          ]
                    )
    del data_h, filter_cluster_df
    gc.collect()
        
    fig.update_layout(title=f"Heatmap of Submatrix {n} of {col_name}",    
                  xaxis_title="Similarity Values",
                  yaxis_title="Frequency")
    st.plotly_chart(fig)