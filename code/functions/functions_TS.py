# 20250422: funcoes usadas sem uso do spark

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
import copy
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from functions.functions_pca import load_image_files3, list_files_to_read,\
                                    get_bandsDates, check_perc_nan 
from functions.functions_segmentation import get_quadrant_coords                                    

# 20250422 funcao copiada do functions_pca.py para tirar a aprte do spark q nao está sendo usada
### Function gen_arr_from_img_band_wCoords
# 20240912: created based on gen_sdf_from_img_band
def gen_arr_from_img_band_wCoords(matrix, bands_name, ar_pos=0, sp=0, sh_print=0, logger=''):
    
    # gen arr from img bands matrix including values for coords of the matrtix/img
    # to be transformed in a df or psdf
    # returns df in format [[[coords_0,coords_1,b1,b2,...,bn
    # ar_pos = 1 if coords in format [x,y] should be in a column of dft
    
    logger.info(f'Function gen_arr_from_img_band_wCoords begin') if logger else  None
    # é usado no pandas-on-Spark (pyspark.pandas, agora chamado pyspark.pandas ou apenas ps), 
    # e serve para permitir operações entre DataFrames de diferentes origens (ou seja, 
    # "frames diferentes"): 
    # ps.options.compute.ops_on_diff_frames = True

    t1=time.time()
    # Get the shape of the matrix
    num_rows, num_cols, cols_df = matrix.shape
    print (f'matrix rows={num_rows}, cols={num_cols}, bands={cols_df}') if sh_print else None
    logger.info(f'matrix rows={num_rows}, cols={num_cols}, bands={cols_df}') if logger else None
    # Create a list of index positions [i, j]
    #positions = [[i, j] for i in range(num_rows) for j in range(num_cols)] 
    t1=time.time()
    shape_0 = matrix.shape[0]
    shape_1 = matrix.shape[1]
    pos_0 = np.repeat(np.arange(shape_0), shape_1).astype(np.int16)
    pos_1 = np.tile(np.arange(shape_1), shape_0).astype(np.int16)
    t2=time.time()
    
    logger.info(f'Tempo para gerar 2 listas com as coordenadas {t2-t1:.2f}') if logger else None
    logger.info(f'pos_0 dtype {pos_0.dtype} , type pos_0 0 {type(pos_0[0])}') if logger else None
    
    # #chage the type for int16
    # t1=time.time()
    # pos_0 = pos_0.astype(np.int16)
    # pos_1 = pos_1.astype(np.int16)
    # t2=time.time()
    # logger.info(f'Tempo para mudar tipo para int16 {type(pos_0[0])} {t2-t1:.2f}')

    # Reshape pos_0 and pos_1 to match matrix's first two dimensions
    m_shape = matrix.shape[:2]
    pos_0 = pos_0.reshape(m_shape)
    pos_1 = pos_1.reshape(matrix.shape[:2])

    # Stack pos_0 and pos_1 along the last axis
    # pos_combined = np.stack((pos_0_reshaped, pos_1_reshaped), axis=-1)
    pos_combined = np.stack((pos_0, pos_1), axis=-1)
    del pos_0, pos_1
    gc.collect()

    t1 = time.time()
    # Concatenate pos_combined to the original matrix along the last axis
    matrix = np.concatenate((pos_combined, matrix), axis=-1)
    t2 = time.time()
    logger.info(f'Tempo para concatenar coords com bands {t2-t1:.2f}s') if logger else None
    del pos_combined
    gc.collect()

    # Flatten the matrix values and reshape into a 2D array   
    num_rows, num_cols, cols_df = matrix.shape
    logger.info(f'matrix shape: {num_rows, num_cols, cols_df}')  if logger else None
    t1=time.time()
    values = matrix.reshape(-1, cols_df)
    t2=time.time()
    logger.info(f'Tempo para fazer o flatten da matriz {t2-t1:.2f}') if logger else None
    del matrix 
    gc.collect()
    logger.info(f'types values {values.dtype} and values elements {type(values[0,0])}') if logger else None
    
    return_array=1
    if return_array:
        logger.info(f'Returning values, Function gen_arr_from_img_band_wCoords END') if logger else  None
        return values
    
    # region Spark code (comentado em 20250422 para não usar Spark)
    # else: # 20250422: comentado para nao usar o spark 
    #     # Create a DataFrame from the index positions and values
    #     #dft = pd.DataFrame(values, columns=bands_name)#, index=positions)
    #     t1 = time.time()
    #     sdft = ps.DataFrame(values, columns=['coords_0','coords_1']+bands_name)
    #     t2 = time.time()
    #     info = f'Tempo para fazer o sdft {t2-t1:.2f}s, {(t2-t1)/60:.2f}m'
    #     logger.info(info) if logger else None
    #     del values
    #     gc.collect()
    #     # Reset the index to create separate columns for 'i' and 'j'
    #     # dft.reset_index(inplace=True)
    #     # dft.rename(columns={'index': 'position'}, inplace=True)
    #     # df['position'] = [f"[{i},{j}]" for i, j in positions]
    #     #mais rapido abaixo
    #     #pyspark pandas dataframe doesn't support ndarray
    #     t1=time.time()
    #     # pos_0 = pos_0.tolist()
    #     # # pos_1 = pos_1.tolist()
    #     # t2=time.time()
    #     # info = f'Tempo para gerar lista da coords_0 {t2-t1:.2f}'
    #     # logger.info(info)
    #     # t3=time.time()

    #     #como estou inserindo na matriz nao preciso mais fazer assim
    #     insert_coords=0
    #     if insert_coords:
    #         pos_0 = ps.Series(pos_0)
    #         sdft.insert(0,'coords_0', pos_0)
            
    #         #sdft['coords_0'] = pos_0
    #         t3_1=time.time()
    #         info = f'Tempo para inserir coords_0 {t3_1-t1:.2f}s, {(t3_1-t1)/60:.2f}m'
    #         logger.info(info)  if logger else None
    #         del pos_0
    #         gc.collect()

    #         sdft = sdft.sort_index()
    #         t3_2 = time.time()
    #         info = f'Tempo para sort coords_0  {t3_2-t3_1:.2f}'
    #         logger.info(info)  if logger else None
            
    #         #como estou inserindo na matriz nao preciso mais fazer assim
    #         pos_1 = ps.Series(pos_1)
    #         # pos_1 = pos_1.tolist()
    #         sdft.insert(1,'coords_1', pos_1)
    #         # sdft['coords_1'] = pos_1
    #         t3_3= time.time()
    #         del pos_1
    #         gc.collect()
    #         info = f'Tempo para inserir coords_1 {t3_3-t3_2:.2f}s, {(t3_3-t3_2)/60:.2f}m'
    #         logger.info(info)  if logger else None

    #         t3_4= time.time()
    #         sdft = sdft.sort_index()
    #         info = f'Tempo parafazer sort das coords_1 {t3_4-t3_3:.2f}'
    #         logger.info(info)  if logger else None

    #         t4=time.time()
    #         info = f'Tempo para gerar inserir e fazer sort das coordenadas {t4-t1:.2f}s, {(t4-t1)/60:.2f}m'
    #         logger.info(info)  if logger else None
    
    #         t1=time.time()
    #         if sp == 2:
    #             #cria pelo dataframe spark o spark dataframe 
    #             sp_sdft = sdft.to_spark()
    #             t2=time.time()
    #             print (f'Time to gen sdft: {t2-t1:.2f}')
    #             logger.info(f'Time to gen sdft: {t2-t1:.2f}') if logger else None
    #             del sdft
    #             gc.collect()
    #             return sp_sdft

    #         logger.info(f'Function gen_sdf_from_img_band END') if logger else  None
            
    #         return sdft
    # t1=time.time()
    # if ar_pos:
    #     # Create a list of index positions [i, j]
    #     positions = [[i, j] for i in range(num_rows) for j in range(num_cols)] 
    #     sdft.insert(2,'coords', positions)     
    # t2=time.time()
    #0.003547191619873047
    # if sh_print: 
    #     print (t2-t1)  
    # endregion

       

def gen_TSdF(band_img_file_to_load, quadrants, dates, check_nan=False, band='NDVI', sh_print =False):
    #gen TSs for a quadrant of a band
    # Para o quadrante escolhido gerar o df da banda selecionada com todas as datas

    #carregar por data
    image_band_dic = {}
    # pos = -2 # position of band in name file
    pos = -1 # position of date in name file
    t1=time.time()
    image_band_dic = load_image_files3(band_img_file_to_load, pos=pos)
    t2=time.time()
    print (f'Tempo de load dos tiffs: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
    del band_img_file_to_load
    gc.collect()
    imgSize = image_band_dic[dates[0]].shape[0]
    df = {}
    for q in quadrants:
        rowi, rowf, coli, colf = get_quadrant_coords(q, imgSize)
        a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        print (f'{a}: Quadrante Q{q} {rowi}:{rowf} {coli}:{colf}') if sh_print else None
        #gerar a imagem tif do quadrante
        t1=time.time()
        image_band_dic_q={}
        for x in dates:
            image_band_dic_q[x] = image_band_dic[x][rowi:rowf, coli:colf]
        t2 = time.time()
        img_band_ts = []
        img_band_ts = np.dstack([image_band_dic_q[x]for x in dates]) 
        t3 = time.time()

        if check_nan:
            t1=time.time()
            perc_day_nan_q= check_perc_nan(image_band_dic_q, sh_print=1)
            t2=time.time()
            print (f'Percentual de nan no quadrante {q} tiffs {band}, imgSize= {image_band_dic_q[dates[0]].shape[0]}: {perc_day_nan_q}%') #if sh_print else None
    
        del image_band_dic_q
        gc.collect()
    
        cols_name = [x for x in dates]
        
        t1 = time.time()
        values  = gen_arr_from_img_band_wCoords(img_band_ts, cols_name, ar_pos=0, sp=0, sh_print=0)
        t2 = time.time()
    
        del img_band_ts
        gc.collect()

        columns = ['coords_0','coords_1']+cols_name
        df[q] = pd.DataFrame(values, columns=columns)
    return df

def gen_groupClusterTS_df(cluster_df, df_tif, col_name, k ):
    '''
    Gen df of a group (k) of cluster (col_name) with its pixels time series 
    cluster_df: n_opt_df
    col_name: cluster_column name
    k: group of cluster to filter
    '''
    # k=52
    filtered_df = cluster_df[(cluster_df[col_name] == k) ] #| (n_opt_df.index == 5781)
    # print (filtered_df.shape)
    
    # filtered_df.head(2)

    # Explode the 'coords' column (each list of coordinates becomes its own row)
    filtered_df_exploded = filtered_df.explode("coords")
    
    # Create 'coords_0' and 'coords_1' by splitting the list into separate columns
    filtered_df_exploded[["coords_0", "coords_1"]] = pd.DataFrame(filtered_df_exploded["coords"].tolist(), index=filtered_df_exploded.index)
    
    # Drop the original 'coords' column (optional)
    filtered_df_exploded = filtered_df_exploded.drop(columns=["coords"])

    
    # Merge on coords_0 and coords_1, keeping only matching rows
    merged_df = filtered_df_exploded.merge(df_tif, on=["coords_0", "coords_1"], how="inner")

    del cluster_df, filtered_df, filtered_df_exploded
    gc.collect()
    
    return merged_df
    merged_df.head()

def gen_df_percentils(noptdf, df, col_name, percentiles=''):
    # 1. Gera um df com os percentil's para todos os k's 
    t1 = time.time()
    clusters = sorted(noptdf[col_name].unique())
    t2 = time.time()
    # print (t2-t1, clusters)
    percentiles = percentiles if percentiles else [2.5, 5, 10,25,50,75,90,95,97.5] 
    print (percentiles)
    result =[]
    for k in clusters:
        cluster_df = gen_groupClusterTS_df(noptdf, df, col_name, k)
       
        # Count unique labels and sum pixels for the cluster
        num_labels = cluster_df["label"].nunique()  # Number of unique labels
        # num_pixels = cluster_df["num_pixels"].sum() # Sum num_pixels for the cluster soma labels duplicados
        num_pixels = cluster_df.drop_duplicates(subset="label")['num_pixels'].sum() # Sum num_pixels of unique labels for the cluster
        # Calculate percentiles
        for p in percentiles:
            row = {
                    col_name: k,
                    "num_labels": num_labels,
                    "num_pixels": num_pixels,
                    "percentil": p
                    }
        
            # Percentiles for each date column
            for col in date_columns:
                row[col] = np.percentile(cluster_df[col], p)
                    
            result.append(row)
    df_percentils = pd.DataFrame(result)
    del result
    gc.collect()
    t3 = time.time()
    print (t3-t2)
    return df_percentils

def gen_groupClusterTS_df(cluster_df, df_tif, col_name, k ):
    '''
    Gen df of a group (k) of cluster (col_name) with its pixels time series 
    cluster_df: n_opt_df
    col_name: cluster_column name
    k: group of cluster to filter
    '''
    # k=52
    filtered_df = cluster_df[(cluster_df[col_name] == k) ] #| (n_opt_df.index == 5781)
    # print (filtered_df.shape)
    
    # filtered_df.head(2)

    # Explode the 'coords' column (each list of coordinates becomes its own row)
    filtered_df_exploded = filtered_df.explode("coords")
    
    # Create 'coords_0' and 'coords_1' by splitting the list into separate columns
    filtered_df_exploded[["coords_0", "coords_1"]] = pd.DataFrame(filtered_df_exploded["coords"].tolist(), index=filtered_df_exploded.index)
    
    # Drop the original 'coords' column (optional)
    filtered_df_exploded = filtered_df_exploded.drop(columns=["coords"])

    
    # Merge on coords_0 and coords_1, keeping only matching rows
    merged_df = filtered_df_exploded.merge(df_tif, on=["coords_0", "coords_1"], how="inner")

    del cluster_df, filtered_df, filtered_df_exploded
    gc.collect()
    
    return merged_df
    

# 20250423: copied from notebook Test_PCA_aval_STs     
def plot_TS_perc_clusters(df, list_clusters=[], percentiles=[], col_name='', n_key_sel='', n_cols=4):
    ''' 
    Plot TSs percentils for each group n of a selected cluster
    copied from plot_hist_clusters
    '''

    col_name= col_name if col_name else 'cluster_ms_' + n_key_sel

    # list_clusters = df[col_name].unique()
    # list_clusters.sort()
    list_clusters = list_clusters if list_clusters else sorted(df[col_name].unique()) 
    
    num_clusters = len(list_clusters)
    
    n_cols = n_cols if num_clusters > n_cols else num_clusters
    n_rows = ceil(int(n_key_sel.split('_')[0])/n_cols) if n_key_sel else ceil(num_clusters/n_cols)

    percentiles = percentiles if percentiles else sorted(df['percentil'].unique()) 
    # Define a color map for percentiles using Plotly colors
    percentile_colors = {p: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, p in enumerate(percentiles)}


    date_columns = [col for col in df.columns if col.startswith("20")]  # Select date columns 
    date_columns.sort()

    # Calculate global min and max for consistent y-axis
    # global_min = df[date_columns].min().min()
    # global_max = df[date_columns].max().max()
    
    # generating subtitles for subplots and get min and max for subplots
    t1 = time.time()
    subtitles = []
    # Initialize global_min and global_max to handle the first iteration
    global_min = float('inf')  # Start with a very high value
    global_max = float('-inf') # Start with a very low value
    for i, n in enumerate(list_clusters):
        filter_cluster_df = df.loc[df[col_name] == n]

        # Further filter by the specified percentiles
        filter_cluster_df = filter_cluster_df[filter_cluster_df['percentil'].isin(percentiles)]

        # Fetch num_labels and num_pixels for the cluster to add to subtitle
        cluster_info = filter_cluster_df.iloc[0]
        num_labels = int(cluster_info["num_labels"])
        num_pixels = int(cluster_info["num_pixels"])
        subt = f"Cl {n} | n.SPs {num_labels}, n.pixels {num_pixels}"
        subtitles.append(subt)

        # Calculate min and max
        # min_y = filter_cluster_df[date_columns].min().min()
        # max_y = filter_cluster_df[date_columns].max().max()
        # code below is faster than above
        min_y = filter_cluster_df[date_columns].to_numpy().min()
        max_y = filter_cluster_df[date_columns].to_numpy().max()
        
        # Update global min and max
        # global_min = min_y if (min_y < global_min) or (i == 0) else global_min
        # global_max = max_y if (max_y > global_max) or (i == 0) else global_max
        # code below is faster than above
        global_min = min(min_y, global_min)
        global_max = max(max_y, global_max)
        
        del filter_cluster_df
        gc.collect()
    print (f'global_min={global_min} , global_max={global_max}')
    t2 = time.time()
    print (f'tempo para gerar subtitles {(t2-t1)/60:.2f}')
    
    fig = make_subplots(rows=int(n_rows), cols=int(n_cols), 
                        shared_xaxes=True,
                        subplot_titles=subtitles, #[f"Cluster {n}" for n in list_clusters],
                        vertical_spacing=0.009)
    t1 = time.time()
    for i, n in enumerate(list_clusters):
        r = (i // n_cols) + 1  # Plotly is 1-based index for subplots
        c = (i % n_cols) + 1

        # print (i, r, c)
        
        # merged_df = gen_groupClusterTS_df(cl_df, ts_df, col_name, n)

        
        filter_cluster_df = df.loc[df[col_name] == n]
       
        # sub_matrix_superior1 = gen_sub_matrix_sup(filter_cluster_df, mm_matrix_sim_sel, cluster_all=0, sh_print=0)
    
        # fig.add_trace(
        #     go.Histogram(x=sub_matrix_superior1, nbinsx=nbins, name=f"Cluster {n}"),
        #     row=r, col=c
        # )
        # fig.update_xaxes(range=[0, 1], row=r, col=c)
        show_legend = (i == 0)
        for p in percentiles:
            df_percentile = filter_cluster_df[filter_cluster_df["percentil"] == p]
            
            fig.add_trace(
                go.Scatter(
                    x=date_columns,
                    y=df_percentile[date_columns].values.flatten(),
                    mode='lines+markers',
                    name=f"Perc {p}",
                    showlegend = show_legend,
                    line=dict(color=percentile_colors[p])  # Consistent color
                ),
                row=r,
                col=c,
            )

        # Set consistent y-axis range for each subplot
        fig.update_yaxes(range=[global_min, global_max], row=r, col=c)
        
        del filter_cluster_df
        gc.collect()
        

    t2 = time.time()
    print (f'Tempo para gerar subplots {(t2-t1)/60}')
    # Layout adjustments
    

    # fig.update_xaxes(title_text="Date", title_font=dict(size=12))
    # fig.update_yaxes(title_text="NDVI",title_font=dict(size=12))
    fig.update_xaxes(tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9))
    # Add annotations to layout
    
    # Single y-axis label
    fig.update_yaxes(matches = 'y',
                     range=[global_min, global_max],
                     title_text="NDVI", 
                     title_font=dict(size=11),
                     tickfont=dict(size=9),
                     row= round( n_rows / 2) if n_rows > 1 else 1, 
                     col= 1 )#round(n_cols/2) )  # Centered label
    # Adjusting title font size
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=9)  # Adjust title font size

    # Single x-axis label
    fig.update_xaxes(title_text="Date", 
                     title_font=dict(size=11),
                     tickfont=dict(size=9),
                     row=n_rows,  # Centered label
                     col= 2 )#round(n_cols/2) )  # Centered label

    title=f"Cluster Percentile TS Analysis for {col_name}"
    # Calculate dynamic height
    max_height = 600
    plot_height = max(200 * n_rows, max_height) if n_rows > 1 else 300
    print (f'plot_height = {plot_height}')
    fig.update_layout(
        width = 300*n_cols,
        # height= max(200 * n_rows, 600), #300*n_rows,
        height = plot_height, #300*n_rows,
        title_text = title,
        showlegend = True,
    )
        
    # fig.update_xaxes(range=[0, 1])
    # fig.update_layout(width=1000, height=800)
    # fig.update_layout(title=f"Percentiles of Clusters for {col_name}", showlegend=True)
    # fig.show()
    return fig

# function to plot TSs of n_rows of a df and group percentiles
def plot_TS_sel(sampled_df, title, xaxis, yaxis='NDVI', df_percentile='', percentiles=[], median_values='' ):
    '''
    plot of TSs of df 
    '''
    fig = go.Figure()
 
    # Add each selected row as a separate line in the plot
    cols = sampled_df.columns

    date_columns = [col for col in cols if col.startswith("20")]  # Select date columns 
    for index, row in sampled_df.iterrows():
        row_name = f'Perc {row["percentil"]}' if "percentil" in cols else f'Row {index}' 
        fig.add_trace(go.Scatter(
            x=xaxis, 
            y=row[date_columns], 
            mode='lines+markers',
            # name=f'Row {index}'
            name=row_name
            
        ))
    
    # # Compute the median across selected rows
    # median_values = sampled_df[date_columns].median()

    if len(median_values):
        # Add the median line to the plot
        fig.add_trace(go.Scatter(
            x=xaxis,
            y=median_values,
            mode='lines',
            line=dict(color='black', dash='dash', width=4),  # Dashed black line
            name='Mean'
        ))

    # Add the percentils 
    if percentiles:
        percentiles = percentiles if percentiles else sorted(df_percentile['percentil'].unique()) 
        # Define a color map for percentiles using Plotly colors
        percentile_colors = {p: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, p in enumerate(percentiles)}

        show_legend = True #(i == 0)
        for p in percentiles:
            df_perc_p = df_percentile[df_percentile["percentil"] == p]
            
            fig.add_trace(
                go.Scatter(
                    x=date_columns,
                    y=df_perc_p[date_columns].values.flatten(),
                    mode='lines+markers',
                    name=f"Perc {p}",
                    showlegend = show_legend,
                    # line=dict(color=percentile_colors[p], dash='dash', width=2)  # Consistent color
                    line=dict(color='black', dash='dash', width=2)  # Consistent color
                ),
                
            )
    
    # Customize layout
    fig.update_layout(
        title = title,
        xaxis_title = 'Date',
        yaxis_title = yaxis,
        xaxis = dict(tickangle=-45),  # Rotate x-axis labels
        template='plotly_white',  # Clean background style
    )
    
    # Show the plot
    # fig.show()
    return fig