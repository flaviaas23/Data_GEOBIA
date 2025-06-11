# 20250103: New streamlit app to analyse the results with pca, segmentation,
#           clara cluster, clara cluster using similarity matrix 
import os
import gc
import time
import datetime
import pandas as pd
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import pickle
import time
from pathlib import Path
import imageio.v2 as imageio
import requests, os, sys
import zarr
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
import streamlit.components.v1 as components
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image

from streamlit_js_eval import streamlit_js_eval
from streamlit_javascript import st_javascript

from src.functions_cluster_st import plot_clustered_clara_st, gen_filter_centroid_df,\
                                     gen_filter_coord_df, gen_filter_coord_thres_df,\
                                     plot_images_cluster_st, plot_images_nans_st,\
                                     plot_img_pixel_sel, load_img_quadrant,\
                                     get_coord_label, gen_filter_centroid_sel_df,\
                                     gen_filter_thres_df, save_filter_thr,\
                                     plot_hist_clusters, plot_heatmap_clusters,\
                                     plot_heatmap_cluster_n, plot_hist_cluster_n

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)                        

# Add the parent directory (code/) to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from functions.functions_segmentation import gen_coords_snic_df
from functions.functions_pca import get_bandsDates, list_files_to_read
from functions.functions_cluster import gen_df_percentils
from functions.functions_TS import gen_TSdF#, plot_TS_perc_clusters,\
#                                  gen_groupClusterTS_df, plot_TS_sel

#bloco usado para recarregar as funcoes sem precisar parar o programa
import importlib
import functions.functions_TS
import functions.functions_cluster

importlib.reload(functions.functions_TS)
importlib.reload(functions.functions_cluster)

gen_TSdF = functions.functions_TS.gen_TSdF
gen_df_percentils = functions.functions_cluster.gen_df_percentils
gen_groupClusterTS_df = functions.functions_TS.gen_groupClusterTS_df
plot_TS_perc_clusters= functions.functions_TS.plot_TS_perc_clusters # updated dynamically
plot_TS_sel = functions.functions_TS.plot_TS_sel

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# -- Set page config
apptitle = 'PCA_Seg_Cluster_Eval'

#call back function when quadrants changes
def on_quad_change():
    st.session_state["LOAD_CLUSTER"] = True
    LOAD_CLUSTER = st.session_state["LOAD_CLUSTER"]
    st.session_state["QUAD_CHANGE"] = True
    QUAD_CHANGE = st.session_state["QUAD_CHANGE"]
    st.write(f"call back quad change = {QUAD_CHANGE} ")

#call back function when day changes
def on_day_change():
    st.session_state["DAY_CHANGE"] = True
    DAY_CHANGE = st.session_state["DAY_CHANGE"]
    st.write(f"call back day change = {DAY_CHANGE} ")

# @st.cache_data
# def get_hist_data(n_key_ms_sel, list_clusters, ncols, n_opt_df, _matrix_sim_sel_2 ):
#     return plot_hist_clusters(n_key_ms_sel, list_clusters, ncols, n_opt_df, _matrix_sim_sel_2 )

# @st.cache_data
# def get_heat_data(n_rows, list_clusters, ncols,n_key_ms_sel, n_opt_df, _matrix_sim_sel_2 ):
#     return plot_heatmap_clusters(n_rows, list_clusters, ncols,n_key_ms_sel, n_opt_df, _matrix_sim_sel_2 )

def main():

    st.set_page_config(
        page_title="PCA_Seg_Cluster_Eval",
        page_icon="🎯"#,
        # layout="wide",
    )
    local_css("code/app_st/styles.css")
    st.session_state.update(st.session_state)      

    if "conf" in st.session_state:
        base_dir = st.session_state["conf"][0]
        # base_dir = st.session_state["base_dir"]
        # q = st.session_state["conf"][3]
        q = st.session_state["quad"]
        quad = st.session_state["quad"]
    else:
        base_dir = os.getcwd()
    
    sh_print = st.sidebar.checkbox('Show print cluster')
    
    base_dir = st.sidebar.text_input('base Diretório:', value=base_dir, key='base_dir')  # Valor padrão é o diretório atual
    tif_dir = base_dir+ st.sidebar.text_input('Tif Diretório:',value='/tif/reg_30d/' )  # Valor padrão é o diretório atual

    name_img = st.sidebar.text_input('Image Name:', value='031027')
    save_dir = st.sidebar.text_input('PCA Diretório:', value= f'data/img_'+name_img+'/')  # Valor padrão é o diretório atual
    save_etapas_dir = base_dir + '/' + save_dir
    
    st.session_state.save_etapas_dir = save_etapas_dir

    #select image tile quadrante
    if 'quad' not in st.session_state:
        st.session_state.quad = 1  # Default value
        QUAD_CHANGE = False

    if 'QUAD_CHANGE' not in st.session_state:
        st.write("quad change not in st session state")
        st.session_state["QUAD_CHANGE"] = False
        QUAD_CHANGE = st.session_state["QUAD_CHANGE"]

    if 'DAY_CHANGE' not in st.session_state:
        st.write("quad change not in st session state")
        st.session_state["DAY_CHANGE"] = True
        DAY_CHANGE = st.session_state["DAY_CHANGE"]

    
    q = st.sidebar.selectbox(
            "Select Image Quadrant",
            [1,2,3,4], 
            index=[1, 2, 3, 4].index(st.session_state.quad),
            key='quad',
            on_change=on_quad_change,  # Call the function when the value changes
            )
    quad = st.session_state["quad"]
    pca_snic_dir = save_etapas_dir +'spark_pca_snic/Quad_'+str(quad)+'/'       

    #select day of image
    #get the tif files name and its dates and bands
    if "dates" not in st.session_state:
            band_tile_img_files = list_files_to_read(tif_dir, name_img, sh_print=0)
            bands, dates = get_bandsDates(band_tile_img_files, tile=0)
            st.session_state['dates'] = dates
            st.session_state['band_tile_img_files'] = band_tile_img_files
            
    else:
        dates = st.session_state['dates']
        band_tile_img_files = st.session_state['band_tile_img_files']
        
    t_day = st.sidebar.selectbox(
            "Select Image Day",
            dates,
            key='t_day',
            on_change=on_day_change,  # Call the function when the value changes
            )

    # st.write(f'inicio cluster page\n {base_dir}, q= {q}, quad={quad}')

    # if st.session_state:
    #     st.write("Keys in session_state:", list(st.session_state.keys()))
    # else:
    #     st.write("No keys in session_state.")

    if "LOAD_CLUSTER" not in st.session_state:
        st.session_state["LOAD_CLUSTER"] = True
        st.write("if load cluster: ",st.session_state["LOAD_CLUSTER"])
 
    LOAD_CLUSTER = st.session_state["LOAD_CLUSTER"]

    # st.write(f'LOAD_CLUSTER = {LOAD_CLUSTER}')
    window_width = st_javascript("window.innerWidth")  # Fetch browser width dynamically
    # st.write(f"Window width: {window_width}px")  # Debug info

    if LOAD_CLUSTER: 
        
        save_etapas_dir = st.session_state.save_etapas_dir # Default value
        pca_snic_dir = save_etapas_dir +'spark_pca_snic/Quad_'+str(q)+'/'

        #read snic segments to make df (segments df)        
        file_to_open = f'{pca_snic_dir}quad{str(q)}_n_30000_comp_2_snic_segments_0.pkl'
        with open(file_to_open, 'rb') as handle:    
            b = pickle.load(handle)

        id = list(b.keys())[0]
        segments_snic_sel_ski_sp = b[id]['segments']
        if sh_print:
            st.write(id, b[id].keys(), segments_snic_sel_ski_sp.shape)

        del b
        gc.collect()
        
        ## gen coords_df
        #for image with ski
        coords_snic_ski_df = gen_coords_snic_df(segments_snic_sel_ski_sp)
        #separar a linha com label -1
        coords_snic_ski_df_nan = coords_snic_ski_df[coords_snic_ski_df['label'] == -1]
        coords_snic_ski_df = coords_snic_ski_df[coords_snic_ski_df['label'] != -1]
        st.session_state["coords_snic_ski_df_nan"] = coords_snic_ski_df_nan
        st.session_state["coords_snic_ski_df"] = coords_snic_ski_df
        
        del segments_snic_sel_ski_sp
        gc.collect()

        #read snic_centroid_df (centroids df)
        file_to_open = f'{pca_snic_dir}quad{str(q)}_n_30000_comp_2_snic_centroid_df_0.pkl'

        with open(file_to_open, 'rb') as handle:    
            b = pickle.load(handle)

        id = list(b.keys())[0]
        snic_centroid_df = b[id]['centroid_df']
        if sh_print:
            st.write(id, b[id].keys(), snic_centroid_df.shape)

        del b
        gc.collect()

        snic_centroid_df = snic_centroid_df.dropna()
        st.session_state["snic_centroid_df"] = snic_centroid_df
        
        ### read dic_cluster simple

        read_quad = q
        d_name = save_etapas_dir + 'pca_snic_cluster/'
        file_to_open = d_name+'clara_'+str(id)+'_quad_'+str(read_quad)+'.pkl'
        with open(file_to_open, 'rb') as handle:    
            b = pickle.load(handle)

        # print (b.keys())
        dic_cluster_ski = b['dic_cluster_ski']
        sse_ski = b['sse_ski']
        n_clusters = len(sse_ski)

        st.session_state["id"] = id
        st.session_state["dic_cluster_ski"] = dic_cluster_ski
        st.session_state["sse_ski"] = sse_ski
        st.session_state["n_clusters"] = n_clusters

        del b
        gc.collect()
        dic_cluster_ski_keys = list(dic_cluster_ski.keys())

        # Read n_opt, inter and intra cluster simple
        file_to_open = save_etapas_dir + 'pca_snic_cluster/clara_'+str(id)+'_InterIntra_quad_'+str(read_quad)+'.pkl'

        with open(file_to_open, 'rb') as handle:    
            b = pickle.load(handle)

        df_ski = b["df_inter_intra"]
        result = b["stats_inter_intra_per_cluster"]
        stats_IntraInter_df = b["stats_inter_intra"]
        n_opt_ind = b["n_opt_ind"]
        n_opt_key = b["n_opt_key"]
    
        dunn_index={} 
        dunn_index['clara_snic'] = df_ski['inter'].min()/df_ski['intra'].max()

        del b
        gc.collect()

        if sh_print:
            st.write(f'n_opt_ind = {n_opt_ind}, n_opt_key = {n_opt_key}')

        st.session_state["df_ski"] = df_ski
        st.session_state["n_opt_ind"] = n_opt_ind
        st.session_state["n_opt_key"] = n_opt_key
        
        ### Read dic cluster similarity matrix and its stats
        # matriz generated with clusters with higher distances
        # file_to_open = save_etapas_dir + 'pca_snic_cluster/clara_ms_'+str(id)+'_quad_'+str(read_quad)+'.pkl'
        # matriz generated with clusters from n_opt to the end (n=30)
        # file_to_open = save_etapas_dir + 'pca_snic_cluster/clara_ms_'+str(id)+'_30_quad_'+str(read_quad)+'.pkl'
        # matriz generated with clusters with higher distances of ms1
        file_to_open = save_etapas_dir + 'pca_snic_cluster/clara_ms2_'+str(id)+'_quad_'+str(read_quad)+'.pkl'


        with open(file_to_open, 'rb') as handle:    
            b_ms = pickle.load(handle)

        sse_ski_ms = b_ms['sse_ski']
        n_clusters_ms = len(sse_ski_ms)
        # dic_cluster_rd_ms = b_ms['dic_cluster_rd']
        dic_cluster_ski_ms = b_ms['dic_cluster_ski']
        sse_ski_ms = b_ms['sse_ski']

        # print (b_ms.keys())
        st.session_state["dic_cluster_ski_ms"] = dic_cluster_ski_ms
        st.session_state["sse_ski_ms"] = sse_ski_ms
        
        del b_ms
        gc.collect()

        dic_cluster_ski_ms_keys = list(dic_cluster_ski_ms.keys())

        # sim matriz generated with clusters with higher distances for clustering
        # file_to_open = save_etapas_dir + 'pca_snic_cluster/clara_ms_'+str(id)+'_InterIntra_quad_'+str(read_quad)+'.pkl'
        # matriz generated with clusters from n_opt to the end (n=30) for clustering
        # file_to_open = save_etapas_dir + 'pca_snic_cluster/clara_ms_'+str(id)+'_30_InterIntra_quad_'+str(read_quad)+'.pkl'
        # cluster with sim matrix2 considering the max distances 
        file_to_open = save_etapas_dir + 'pca_snic_cluster/clara_ms2_'+str(id)+'_InterIntra_quad_'+str(read_quad)+'.pkl'
        
        with open(file_to_open, 'rb') as handle:    
            b_ms = pickle.load(handle)

        if sh_print:
            st.write(b_ms.keys())
        
        st.write(f'b_ms.keys()= {b_ms.keys()}')

        df_ski_ms = b_ms["df_inter_intra"]   
        n_opt_ind_ms = b_ms["n_opt_ind"]
        n_opt_key_ms = b_ms["n_opt_key"]

        dunn_index['clara_ms_snic'] = df_ski_ms['inter'].min()/df_ski_ms['intra'].max()
        df_dunn_index = pd.DataFrame(dunn_index.items(),columns=['MethodCluster', 'DunnIndex'])        

        st.session_state["df_ski_ms"] = df_ski_ms
        st.session_state["n_opt_ind_ms"] = n_opt_ind_ms
        st.session_state["n_opt_key_ms"] = n_opt_key_ms
        st.session_state["df_dunn_index"] = df_dunn_index

        st.write(f'dentro do LOAD_CLUSTER n_opt_key_ms = {n_opt_key_ms}')
        if sh_print:
            st.write(f'n_opt_ind_ms = {n_opt_ind_ms}, n_opt_key_ms = {n_opt_key_ms}')
            st.write(df_dunn_index)
        
        del b_ms
        gc.collect()

        #read similarity matrix of cluster simple
        # considering max distances to gen sim
        matrix_path = save_etapas_dir + 'spark_pca_matrix_sim/matrix_similarity_npmem_job_Quad_'+str(q)
        # considering n_opt to n=30
        # matrix_path = save_etapas_dir + 'spark_pca_matrix_sim/matrix_similarity_npmem_job_30_Quad_'+str(q)
        t1 = time.time()
        zarr_group = zarr.open(matrix_path, mode='a')
        t2 = time.time()
        matrix_sim_sel = zarr_group['arr_0']
        st.session_state["matrix_sim"] = matrix_sim_sel
        if sh_print:
            st.write (f'1.5 matrix_sim_sel.shape: {matrix_sim_sel.shape}')

        #read similarity matrix of cluster with distance sim matrix
        matrix_path = save_etapas_dir + 'spark_pca_matrix_sim/matrix_similarity_npmem_job_ms_Quad_'+str(q)
        # sim matriz of cluster using dist matriz of higer distances
        # matrix_path = save_etapas_dir + 'spark_pca_matrix_sim/matrix_similarity_npmem_job_ms_3_Quad_'+str(q)
        # sim matriz of cluster using dist matriz of n_opt to n=30
        # matrix_path = save_etapas_dir + 'spark_pca_matrix_sim/matrix_similarity_npmem_job_ms_30_Quad_'+str(q)
        t1 = time.time()
        zarr_group = zarr.open(matrix_path, mode='r')
        t2 = time.time()
        matrix_sim_sel_2 = zarr_group['arr_0']
        st.session_state["matrix_sim_2"] = matrix_sim_sel_2
        if sh_print:
            st.write (f'1.5 matrix_sim_sel_2.shape: {matrix_sim_sel_2.shape}')

        st.session_state["LOAD_CLUSTER"] = False

    tabs = ["Cluster Metrics Evaluation", "Cluster Results", "Cluster Comparation", "Pixel Selection","Clusters Analysis"]
    tab_eval, tab_result, tab_comp, tab_filter, tab_ana = st.tabs(tabs)

    #load the values of st.session_state
    # avaliar se é melhor carregar só o que será usado em cada tab
    # ou carregar tudo aqui caso estejam no st.session_state
    
    with tab_eval:
        if st.checkbox("Dunn Index"):
            df_dunn_index = st.session_state["df_dunn_index"]

            # Plot Dunn index of cluster simple and cluster with similarity matrix - pca
            figD = px.bar(df_dunn_index, x='MethodCluster', y='DunnIndex') #scatter
            # Add title
            figD.update_layout(
                title="Dunn Index for Simple Cluster x Distance Similarity Matrix",
                title_x=0.3  # Center-align the title (optional)
            )
            st.plotly_chart(figD)

        if st.checkbox("Inter and Intra Cluster simple and with distance matrix comparation"):
            df_ski = st.session_state["df_ski"]
            df_ski_ms = st.session_state["df_ski_ms"]
            figC = go.Figure()
            figC.add_trace(go.Box(y=df_ski["intra"],name='intra'))
            figC.add_trace(go.Box(y=df_ski_ms["intra"], name = 'intra_ms'))

            # Add title
            figC.update_layout(
                title="Intra Values Simples Cluster x Distance Similarity Matrix",
                title_x=0.5  # Center-align the title (optional)
            )

            st.plotly_chart(figC)

            figCInter = go.Figure()
            figCInter.add_trace(go.Box(y=df_ski["inter"],name='inter'))
            figCInter.add_trace(go.Box(y=df_ski_ms["inter"], name = 'inter_ms'))

            # Add title
            figCInter.update_layout(
                title="Inter Values Simples Cluster x Distance Similarity Matrix",
                title_x=0.5  # Center-align the title (optional)
            )
            st.plotly_chart(figCInter)

        if st.checkbox("Inter and Intra Cluster"):
            n_cols = 2
            col1,col2 = st.columns(n_cols)
            df_ski = st.session_state["df_ski"]
            df_ski_ms = st.session_state["df_ski_ms"]
            with col1:
                figSCIntra = px.box(df_ski, y="intra")
                figSCIntra.update_layout(
                    title="Intra of Simple Cluster",
                    title_x=0.3  # Center-align the title (optional)
                )
                st.plotly_chart(figSCIntra)
                
                figMSCIntra = px.box(df_ski_ms, y="intra")
                # Add title
                figMSCIntra.update_layout(
                    title="Intra of Cluster with Distance Similarity",
                    title_x=0.3  # Center-align the title (optional)
                )
                st.plotly_chart(figMSCIntra)
            with col2:
                figSCInter = px.box(df_ski, y="inter")
                figSCInter.update_layout(
                    title="Inter of Simple Cluster",
                    title_x=0.3  # Center-align the title (optional)
                )
                st.plotly_chart(figSCInter)
                               
                figMSCInter = px.box(df_ski_ms, y="inter")
                figMSCInter.update_layout(
                    title="Inter of Cluster with Distance Similarity",
                    title_x=0.3  # Center-align the title (optional)
                )
                st.plotly_chart(figMSCInter)

    with tab_result:

        # t_img_sel_norm_q = st.session_state["t_img_sel_norm_q"]
        if st.session_state['QUAD_CHANGE'] or st.session_state['DAY_CHANGE'] or ("t_img_sel_norm_q" not in st.session_state):
            QUAD_CHANGE = st.session_state['QUAD_CHANGE']
            st.write(f"if 1 quad change = {QUAD_CHANGE} quad = {quad} q= {q}")
            bands_sel = ['B11', 'B8A', 'B02'] # R,G,B bands
            t_img_sel_norm_q, t_img_sel_uint8_q, img_sz = load_img_quadrant(tif_dir, name_img, bands_sel, t_day, q, img_full=0)
            st.session_state["t_img_sel_norm_q"] = t_img_sel_norm_q
            st.session_state["t_img_sel_uint8_q"] = t_img_sel_uint8_q
            st.session_state['QUAD_CHANGE'] = False
            st.session_state['DAY_CHANGE'] = False
            QUAD_CHANGE = False        

        snic_centroid_df = st.session_state["snic_centroid_df"]
        if st.checkbox("Cluster Simple"):
            #fazer um df só e adicionar as colunas do cluster de cada n_opt respectivamente
            dic_cluster_ski = st.session_state["dic_cluster_ski"]
            cluster_df = snic_centroid_df.loc[:,('label', 'centroid-0','centroid-1')]
            list_cluster=[]
            #adiciona o valor do cluster do n_opt
            dic_cluster_ski_keys = list(dic_cluster_ski.keys())
            for x in list(dic_cluster_ski_keys):
                cluster_df['cluster_'+x] = dic_cluster_ski[x]
                list_cluster.append('cluster_'+x)
            
            if sh_print:
                st.write(f'list_cluster = {list_cluster}')
                st.table(cluster_df.head(3))

            time_i = time.time()
            plot_clustered_clara_st(t_img_sel_norm_q, cluster_df , list_cluster, \
                                    n_cols=5, plot_centroids=1, cl_map='tab20')
            time_f=time.time()
            st.write(f'{time_f-time_i:.2f}')

        if st.checkbox("Cluster with Matrix Similarity Distance 2"):
            dic_cluster_ski_ms = st.session_state["dic_cluster_ski_ms"]
            cluster_ms_df = snic_centroid_df.loc[:,('label', 'centroid-0','centroid-1')]
            list_cluster_ms=[]
            #adiciona o valor do cluster do n_opt
            dic_cluster_ski_ms_keys = list(dic_cluster_ski_ms.keys())
            for x in list(dic_cluster_ski_ms_keys):
                cluster_ms_df['cluster_ms_'+x] = dic_cluster_ski_ms[x]
                list_cluster_ms.append('cluster_ms_'+x)
            
            #plot centroids only of each cluster
            time_i = time.time()
            plot_clustered_clara_st(t_img_sel_norm_q, cluster_ms_df , list_cluster_ms, \
                                    n_cols=5, plot_centroids=1, cl_map='tab20')
            time_f=time.time()
            st.write(f'{time_f-time_i:.2f} {(time_f-time_i)/60:.2f}')
            
    with tab_comp:
        #fazer um df só e adicionar as colunas do cluster de cada n_opt respectivamente
        snic_centroid_df = st.session_state["snic_centroid_df"]
        n_opt_df = snic_centroid_df.loc[:,('label', 'centroid-0','centroid-1')]

        dic_cluster_ski = st.session_state["dic_cluster_ski"]
        dic_cluster_ski_ms = st.session_state["dic_cluster_ski_ms"]
        n_opt_key = st.session_state["n_opt_key"]
        n_opt_key_ms = st.session_state["n_opt_key_ms"]
         
        #adiciona o valor do cluster do n_opt
        test1 ='cluster'
        n_opt_df[test1+'_'+str(n_opt_key)] = dic_cluster_ski[n_opt_key]

        test2='cluster_ms'
        n_opt_df[test2+'_'+str(n_opt_key_ms)] = dic_cluster_ski_ms[n_opt_key_ms]

        list_cluster_n_opt = [x for x in n_opt_df.columns if 'cluster'in x]

        time_i = time.time()
        t_img_sel_norm_q = st.session_state["t_img_sel_norm_q"]
        plot_clustered_clara_st(t_img_sel_norm_q, n_opt_df , list_cluster_n_opt, \
                                n_cols=3, plot_centroids=1, cl_map='tab20')
        time_f=time.time()
        st.write(f'{time_f-time_i:.2f} {(time_f-time_i)/60:.2f}')

    with tab_filter:
        sh_print_f = st.checkbox('Show prints cluster')

        # load vars
        dic_cluster_ski = st.session_state["dic_cluster_ski"]
        dic_cluster_ski_ms = st.session_state["dic_cluster_ski_ms"]
        snic_centroid_df = st.session_state["snic_centroid_df"]
        coords_snic_ski_df = st.session_state["coords_snic_ski_df"]
        coords_snic_ski_df_nan = st.session_state["coords_snic_ski_df_nan"]
        n_opt_key = st.session_state["n_opt_key"]
        n_opt_key_ms = st.session_state["n_opt_key_ms"]
        matrix_sim_sel = st.session_state["matrix_sim"] 
        matrix_sim_sel_2 = st.session_state["matrix_sim_2"] 
        id = st.session_state["id"]

        #select day of image
        #get the tif files name and its dates and bands
        if "dates" not in st.session_state:
            band_tile_img_files = list_files_to_read(tif_dir, name_img, sh_print=0)
            bands, dates = get_bandsDates(band_tile_img_files, tile=0)
            st.session_state['dates'] = dates
        else:
            dates = st.session_state['dates']
        # t_day = st.sidebar.selectbox(
        #         "Select Image Day",
        #         dates,
        #         key='t_day',
        #         on_change=on_day_change,  # Call the function when the value changes
        #         )

        #select the pixel in image and show all pixels of the same group of pixel selected
        # st.write(f"antes if quad change = {QUAD_CHANGE} ")

        if st.session_state['QUAD_CHANGE'] or st.session_state['DAY_CHANGE'] or ("t_img_sel_norm_q" not in st.session_state):
            QUAD_CHANGE = st.session_state['QUAD_CHANGE']
            st.write(f"if 1 quad change = {QUAD_CHANGE} quad = {quad} q= {q}")
            bands_sel = ['B11', 'B8A', 'B02'] # R,G,B bands
            t_img_sel_norm_q, t_img_sel_uint8_q, img_sz = load_img_quadrant(tif_dir, name_img, bands_sel, t_day, q, img_full=0)
            st.session_state["t_img_sel_norm_q"] = t_img_sel_norm_q
            st.session_state["t_img_sel_uint8_q"] = t_img_sel_uint8_q
            st.session_state['QUAD_CHANGE'] = False
            st.session_state['DAY_CHANGE'] = False
            QUAD_CHANGE = False        
            # st.write(f"if 2 quad change = {QUAD_CHANGE} ")
        elif st.session_state["t_img_sel_uint8_q"] is not None:
            t_img_sel_uint8_q = st.session_state["t_img_sel_uint8_q"]  # Scale and convert to uint8
            img_sz = t_img_sel_uint8_q.shape[0]
            # st.write(f"elif  quad change = {QUAD_CHANGE} ")
            QUAD_CHANGE = False
        
        n_cols = 1 
        # col1, col2 = st.columns(n_cols)
        # window_width = st_javascript("window.innerWidth")  # Fetch browser width dynamically
        col_width = round(window_width/n_cols)

        if sh_print | sh_print_f:
            st.write(f"Window width: {window_width}px, col_width = {col_width}")  # Debug info
        
            st.write(f'matrix_sim shape = {matrix_sim_sel.shape}')
            st.write(f'matrix_sim_2 shape = {matrix_sim_sel_2.shape}')
            st.write(f"Window width: {window_width}px, col_width = {col_width}")  # Debug info

        st.subheader(f"Select a pixel in Quadrant {quad} of image {name_img}", divider="gray")
        value = streamlit_image_coordinates(t_img_sel_uint8_q,
                                                height=col_width,
                                                width=col_width,
                                                key="click_image"
                                        )
        #threshold slider estaav aqui

        if value:
            y = round(value['x']*img_sz/col_width)
            x = round(value['y']*img_sz/col_width)
            coord_sel = [x, y]
            if sh_print_f:
                st.write(f"value={value}, {x}, {y}, {t_img_sel_uint8_q.shape[0]} , {coord_sel}") 

        dic_filter_centroid_sel_pca_df2={}
        tests = ['cluster_simple', 'cluster_ms']#, 'cluster_ms_f']
        # label_sel = 9175 #249# 5975 #3148#5975

        n_opt_df = snic_centroid_df.loc[:,('label', 'centroid-0','centroid-1')]
        t1 = time.time()
        n_opt_df = pd.merge(n_opt_df, coords_snic_ski_df, on='label')
        t2 = time.time()
        # st.write(f'tempo merge nopt = {(t2-t1)/60:.2f}')
        # dic_filter_centroid_sel_pca_df2[tests[0]] = gen_filter_centroid_df(tests[0], snic_centroid_df, n_opt_key, label_sel, cluster=dic_cluster_ski )
        # dic_filter_centroid_sel_pca_df2[tests[1]] = gen_filter_centroid_df(tests[1], snic_centroid_df, n_opt_key_ms, label_sel, cluster=dic_cluster_ski_ms )
        # dic_filter_centroid_sel_pca_df2[tests[2]] = gen_filter_centroid_df(tests[2], slic_pca_df_sel, n_opt_pca_ms_f, centroid_sel, cluster=dic_cluster_pca_ms_f)
        
        if value:
            # t1 = time.time()
            # n_opt_df = pd.merge(n_opt_df, coords_snic_ski_df, on='label')
            # t2 = time.time()
            # plot all centroids of the cluster of the pixel selected
            plot_cluster_coord_sel = 1
            if plot_cluster_coord_sel:
                st.subheader(f"Cluster of selected pixel", divider="gray")
                
                t1=time.time()
                # label_sel, dic_filter_centroid_sel_pca_df2[tests[0]] = gen_filter_coord_df(test, n_opt_df, n_opt_key, coord_sel, cluster=dic_cluster_ski  )
                # 20250130: dividi a funcao acima em 2 partes para aproveitar com o cluster ms e com a matriz de sim e 
                # melhorar o tempo quando mostrando o cluster e cluster_ms
                # label_sel, ind_label_sel, filter_centroid_sel_df = get_coord_label(n_opt_df, coord_sel, sh_print=0) #20250210: como nao altero o df no preciso mais retorna-lo
                label_sel, ind_label_sel = get_coord_label(n_opt_df, coord_sel, sh_print=0)
                test = 'cluster'
                if label_sel:
                    dic_filter_centroid_sel_pca_df2[tests[0]] = gen_filter_centroid_sel_df(test,n_opt_df, n_opt_key,label_sel, dic_cluster_ski)
                    # st.write(f'df filter centroid \n{dic_filter_centroid_sel_pca_df2[tests[0]].head(2)}')
                t2=time.time()    
                # st.write(f'cluster label  {label_sel}')
                
                test = 'cluster_ms'
                # label_sel, dic_filter_centroid_sel_pca_df2[tests[1]] = gen_filter_coord_df(test, n_opt_df, n_opt_key_ms, coord_sel, cluster=dic_cluster_ski_ms )
                if label_sel:
                    dic_filter_centroid_sel_pca_df2[tests[1]] = gen_filter_centroid_sel_df(test,n_opt_df, n_opt_key_ms,label_sel, dic_cluster_ski_ms)
                    # st.write(f'df filter centroid \n{dic_filter_centroid_sel_pca_df2[tests[0]].head(2)}')
                
                # ind_label_sel = n_opt_df.index[n_opt_df['label'] == label_sel]
                t3=time.time()
                # st.write(f'cluster_ms label_sel {label_sel}, indice label_sel {ind_label_sel}')
                # st.write(f'tempo filtro cluster simples {t2-t1} ms {t3-t2} os dois {t3-t1}')

                if label_sel is not None:
                    plot_images_cluster_st(t_img_sel_norm_q, dic_filter_centroid_sel_pca_df2, id,\
                                           list(dic_filter_centroid_sel_pca_df2.keys()),\
                                           label_sel, n_cols=3,plot_orig_img=1, cl_map='Blues',\
                                           chart_size=(12, 12))
                else:
                    # plot the nans and clusters over the image
                    st.write(f'coords_snic_ski_df_nan shape {coords_snic_ski_df_nan.shape}\n coords_snic_ski_df_nan {coords_snic_ski_df_nan.head(2)}')
                    plot_images_nans_st(t_img_sel_norm_q, n_opt_df, n_opt_key_ms, dic_cluster_ski_ms,\
                                       test, coords_snic_ski_df_nan, ['Nans'],\
                                       coord_sel, plot_centroids=0, n_cols=2, plot_orig_img=1, cl_map='tab20')

            #plot the centroids with similarity of pixel selected in threshold range
            st.subheader(f"Pixels with similarity with selected one", divider="gray")
            threshold = st.slider(
                    'Select a range for similarity',
                    0, 100, (80, 100))
            # st.write(f'threshold = {threshold}')
            sh_sim_dist = st.checkbox('Show similarity distribution')

            if sh_sim_dist:
                # st.write(st.config.get_option("server.maxMessageSize"))
                # matrix_sim_sel_upper = matrix_sim_sel[np.triu_indices(matrix_sim_sel.shape[0], k=1)]
                # df_upper = pd.DataFrame({'Values': matrix_sim_sel_upper, 'Triangle': 'Upper'})
                # del matrix_sim_sel_upper
                # gc.collect()
                # spark._jvm.System.gc() #20241210
                # Plot histogram using Plotly Express
                # fig_h = px.histogram(matrix_sim_sel_upper, nbins=100,  # Adjust the number of bins
                sh_hist_sim = st.checkbox('Show histogram of Similarity Matrix1')
                centroid_row_matrix_sim = matrix_sim_sel[ind_label_sel[0],:]
                centroid_row_matrix_sim2 = matrix_sim_sel_2[ind_label_sel[0],:]
                # st.write(len(centroid_row_matrix_sim))
                if sh_hist_sim:
                    fig_h = px.histogram(centroid_row_matrix_sim, nbins=100,  # Adjust the number of bins
                                        title="Distribution of Similarity Matrix Values - pixel sel",
                                        labels={"x": "Similarity Values", "y": "Count"})
                    # fig_h.update_layout(bargap=0.1)
                    # fig_h.show()
                    st.plotly_chart(fig_h, use_container_width=True)
                    
                    # Optional: Density plot nao está plotando este aqui
                    fig_density = px.violin(centroid_row_matrix_sim,   points="all", 
                                            title="Density of Sim Matrix")
                    # fig_density.show()
                    st.plotly_chart(fig_density)

                sh_hist_sim2 = st.checkbox('Show histogram of Similarity Matrix2')
                # st.write(len(centroid_row_matrix_sim))
                if sh_hist_sim2:
                    fig_h2 = px.histogram(centroid_row_matrix_sim2, nbins=100,  # Adjust the number of bins
                                        title="Distribution of Similarity Matrix Values 2 - pixel sel",
                                        labels={"x": "Similarity Values", "y": "Count"})
                    # fig_h.update_layout(bargap=0.1)
                    # fig_h.show()
                    st.plotly_chart(fig_h2, use_container_width=True)

                # Create the histogram traces
                trace1 = go.Histogram(
                    x=centroid_row_matrix_sim, 
                    nbinsx=100, 
                    name="Similarity Matrix Values - Pixel Sel",
                    marker=dict(opacity=0.5)  # Transparency to distinguish overlaps
                )
                trace2 = go.Histogram(
                    x=centroid_row_matrix_sim2, 
                    nbinsx=100, 
                    name="Similarity Matrix Values 2 - Pixel Sel",
                    marker=dict(opacity=0.5)  # Transparency to distinguish overlaps
                )

                # Create the figure and layout
                fig_dh = go.Figure(data=[trace1, trace2])

                # Add layout options
                fig_dh.update_layout(
                    title="Distribution of Similarity Matrix Values",
                    xaxis_title="Similarity Values",
                    yaxis_title="Count",
                    barmode='overlay'  # 'stack' or 'overlay' for visibility
                )
                st.plotly_chart(fig_dh, use_container_width=True)

                df_row_matrix = pd.DataFrame({
                                    "Sim Values": np.concatenate([centroid_row_matrix_sim, centroid_row_matrix_sim2]),
                                    "Group": ["Row Matrix 1"] * len(centroid_row_matrix_sim) + ["Row Matrix 2"] * len(centroid_row_matrix_sim2)
                                    })
                # Create a grouped violin plot
                fig_v = px.violin(df_row_matrix, x="Group", y="Sim Values", points="all", box=True, title="Grouped Violin Plot")
                st.plotly_chart(fig_v, use_container_width=True)
            test = 'cluster_ms'
            t1 = time.time()
            # label_sel, filter_centroid_sel_df = gen_filter_coord_thres_df(test, n_opt_df, n_opt_key_ms, coord_sel, matrix_sim_sel, threshold, cluster=dic_cluster_ski_ms )
            if label_sel is not None:
                n_cols = 2
                col1, col2 = st.columns(n_cols)
                with col1:
                    # ind_label_sel = n_opt_df.index[n_opt_df['label'] == label_sel]
                    # filter_centroid_sel_df1 = gen_filter_thres_df(test, filter_centroid_sel_df, n_opt_key_ms, \
                    filter_centroid_sel_df1 = gen_filter_thres_df(test, n_opt_df, n_opt_key_ms, \
                                                                    label_sel,ind_label_sel, matrix_sim_sel, \
                                                                    threshold, cluster=dic_cluster_ski_ms )
                    
                    t2 = time.time()
                    if sh_print:
                        st.write(f' label:{label_sel}, ind_label_sel:{ind_label_sel} sim filter shape: {filter_centroid_sel_df1.shape}')
                        st.write(f'threshold = {threshold}')
                    
                    plot_img_pixel_sel(t_img_sel_norm_q, filter_centroid_sel_df1, ind_label_sel)
                    # plot_images_cluster_st(t_img_sel_norm, dic_filter_centroid_sel_pca_df2,  id,\
                    #             list(dic_filter_centroid_sel_pca_df2.keys()), label_sel, n_cols=3,plot_orig_img=0, cl_map='Blues', chart_size=(12, 12))

                with col2:
                    # ind_label_sel = n_opt_df.index[n_opt_df['label'] == label_sel]
                    # filter_centroid_sel_df2 = gen_filter_thres_df(test, filter_centroid_sel_df, n_opt_key_ms, \
                    filter_centroid_sel_df2 = gen_filter_thres_df(test, n_opt_df, n_opt_key_ms, \
                                                                    label_sel, ind_label_sel, matrix_sim_sel_2, \
                                                                    threshold, cluster=dic_cluster_ski_ms )
                    
                    t2 = time.time()
                    if sh_print:
                        st.write(f' label:{label_sel}, ind_label_sel:{ind_label_sel} sim filter shape: {filter_centroid_sel_df2.shape}')
                        st.write(f'threshold = {threshold}')

                    plot_img_pixel_sel(t_img_sel_norm_q, filter_centroid_sel_df2, ind_label_sel)
                #save pixels/centroids to a file
                save_filter = st.checkbox('Save filtered threshold')

                if save_filter:
                    st.write(filter_centroid_sel_df2.head(2))
                    st.write('{save_etapas_dir}')
                    d_name = save_etapas_dir +'Filter_threshold/Quad_' + str(quad)+'/'
                    f_name = f'img_{name_img}_quad_{quad}_pixel_{coord_sel[0]}_{coord_sel[1]}_thr_{threshold[0]}_{threshold[1]}_pixels'
                    # st.write(f'{d_name}{f_name}')
                    save_filter_thr(filter_centroid_sel_df2, d_name, f_name,\
                                    threshold, label_sel, coord_sel, quad, img_sz*2, save_filter=3, sh_print=0)

            else: 
                st.write(f'Pixel selected {coord_sel} is None')
            st.write(f'tempo para filtro da matrix de sim {t2-t1}')

    with tab_ana:
        # load vars
        dic_cluster_ski = st.session_state["dic_cluster_ski"]
        dic_cluster_ski_ms = st.session_state["dic_cluster_ski_ms"]
        snic_centroid_df = st.session_state["snic_centroid_df"]
        coords_snic_ski_df = st.session_state["coords_snic_ski_df"]
        coords_snic_ski_df_nan = st.session_state["coords_snic_ski_df_nan"]
        n_opt_key = st.session_state["n_opt_key"]
        n_opt_key_ms = st.session_state["n_opt_key_ms"]
        matrix_sim_sel = st.session_state["matrix_sim"] 
        matrix_sim_sel_2 = st.session_state["matrix_sim_2"] 
        dates = st.session_state['dates']
        
        id = st.session_state["id"]

        cluster_keys = list(dic_cluster_ski_ms.keys())
        
        n_key_ms_sel = n_opt_key_ms
        n_key_ms_sel = st.selectbox(
            "Select number of Clusters",
            cluster_keys, 
            index=cluster_keys.index(n_opt_key_ms)#,
            #key='quad' 
            #on_change=on_quad_change,  # Call the function when the value changes
            )
        # col_name = 'cluster_ms_' + n_key_ms_sel
        col_name = 'cluster_ms_' + n_key_ms_sel + '_ClaraMS2'
        ncols=4
        # st.write(f'ncols = {ncols}, {n_key_ms_sel}')

        sh_sim_clusters = st.checkbox(f'Show Similarity histogram/heatmap for each cluster of {n_key_ms_sel}')
        if sh_sim_clusters:
            if 'n_opt_df' not in locals():     
                st.write('n_opt_df nao existe')       
                n_opt_df = snic_centroid_df.loc[:,('label', 'centroid-0','centroid-1')]
            n_opt_df_columns = n_opt_df.columns
            if 'coords' in n_opt_df_columns:
                n_opt_df = n_opt_df.drop('coords', axis=1)
            if col_name not in n_opt_df.columns:
                n_opt_df[col_name] = dic_cluster_ski_ms[n_key_ms_sel]

            st.write(n_opt_df.head(2))    

            # col_name = 'cluster_ms_' + n_key_ms_sel
            col_name = 'cluster_ms_' + n_key_ms_sel + '_ClaraMS2'

            list_clusters = n_opt_df[col_name].unique()
            list_clusters.sort()

            sh_sim_hist_clusters = st.checkbox(f'Show Similarity Histogram for each cluster of {n_key_ms_sel}')
            if sh_sim_hist_clusters:
                plot_hist_clusters(n_key_ms_sel, list_clusters, ncols, n_opt_df, matrix_sim_sel_2 )
                # get_hist_data(n_key_ms_sel, list_clusters, ncols, n_opt_df, matrix_sim_sel_2 )
            
            sh_sim_heat_clusters = st.checkbox(f'Show Similarity Heatmap for selected cluster of {n_key_ms_sel}')            
            if sh_sim_heat_clusters:
                n_rows = ceil(int(n_key_ms_sel.split('_')[0])/ncols) 
                plot_heatmap_clusters(n_rows, list_clusters, ncols,n_key_ms_sel, n_opt_df, matrix_sim_sel_2 )
                # get_heat_data(n_rows, list_clusters, ncols,n_key_ms_sel, n_opt_df, matrix_sim_sel_2 )


            sh_sim_heat_clustern = st.checkbox(f'Show Similarity Histogram/Heatmap for cluster  of {n_key_ms_sel}')
            if sh_sim_heat_clustern:
                clustern = st.selectbox(
                                        f"Select group of Clusters of {n_key_ms_sel}",
                                        list_clusters#, 
                                        # index=cluster_keys.index(n_opt_key_ms)#,
                                        #key='quad' 
                                        #on_change=on_quad_change,  # Call the function when the value changes
                                        )
                
                #plot heat of n in col_name
                plot_hist_cluster_n(clustern, n_opt_df, col_name, matrix_sim_sel_2 )
                plot_heatmap_cluster_n(clustern, n_opt_df, col_name, matrix_sim_sel_2 )
                
        sh_perc_clusters = st.checkbox(f'Show percentils for cluster {n_key_ms_sel} of {col_name}')
        if sh_perc_clusters:
            intervalo_perc = {}
            percentils = [2.5,5,10,25,50,75,90,95,97.5]
            intervalo_perc[95] = [2.5,50,97.5]
            intervalo_perc[90] = [5,50,95]
            intervalo_perc[80] = [10,50,90]
            

            if 'n_opt_df' not in locals():     
                st.write('n_opt_df nao existe')       
                n_opt_df = snic_centroid_df.loc[:,('label', 'centroid-0','centroid-1', 'coords')]
            n_opt_df_columns = n_opt_df.columns

            # st.write(n_opt_df_columns) 
            # st.write(n_opt_df.head(2))    
            if col_name not in n_opt_df.columns:
                n_opt_df[col_name] = dic_cluster_ski_ms[n_key_ms_sel]
                        
            with st.expander("Settings for percentils"):
                col1, col2 = st.columns(2)
                with col1:
                    intPerc = st.selectbox(
                        "Select percentils interval",
                        [80,90,95], 
                        index=1
                        # cluster_keys, 
                        # index=cluster_keys.index(n_opt_key_ms)#,
                        #key='quad' 
                        #on_change=on_quad_change,  # Call the function when the value changes
                        )
                
                sh_all_perc = st.checkbox(f'Show all percentils for cluster {n_key_ms_sel} of {col_name}')
                sh_int_perc = st.checkbox(f'Show percentils interval for cluster {n_key_ms_sel} of {col_name}')
                sh_samp_perc = st.checkbox(f'Show sample TS and percentils for cluster {n_key_ms_sel} of {col_name}')
            
            # st.write(n_opt_df.head(2))    

            # gen df with TSs
            if "dic_df_ts" not in st.session_state:
                dic_df_ts = {}
                               
                band = 'NDVI' # depois colocar para ser selecionado
                band_img_file_to_load = [x for x in band_tile_img_files if band in x.split('/')[-1]]
                dic_df_ts = gen_TSdF(band_img_file_to_load, [quad], dates, band='NDVI')

                st.session_state["dic_df_ts"] = dic_df_ts
                # st.write(f'not in session_state:{dic_df_ts.keys()}')
            else:
                dic_df_ts = st.session_state["dic_df_ts"]
                # st.write(dic_df_ts.keys())#s,dic_df_ts[quad].head(2))
                # del st.session_state["dic_df_ts"]
               
            # st.write(dic_df_ts[quad].head(2))

            #gen df percentils for n_opt
            if "dic_df_percentils" not in st.session_state:
                dic_df_percentils = {}
                t1 = time.time()                               
                dic_df_percentils[n_key_ms_sel] = gen_df_percentils(n_opt_df, dic_df_ts[quad], col_name, percentils)
                t2 = time.time()
                st.session_state["dic_df_percentils"] = dic_df_percentils
                # st.write(f'df_percentils not in session_state:{dic_df_percentils.keys()}, t2-t1={t2-t1:.2f}')
            else:
                dic_df_percentils = st.session_state["dic_df_percentils"]

                if n_key_ms_sel not in dic_df_percentils:
                    t1 = time.time()                               
                    dic_df_percentils[n_key_ms_sel] = gen_df_percentils(n_opt_df, dic_df_ts[quad], col_name, percentils)
                    t2 = time.time()                               
                    st.session_state["dic_df_percentils"] = dic_df_percentils
                
                # st.write(f't2-t1={t2-t1:.2f} {dic_df_percentils.keys()}')#s,dic_df_ts[quad].head(2))
                # del st.session_state["dic_df_ts"]
                # st.write(f't2-t1={t2-t1:.2f}')#s,dic_df_ts[quad].head(2))
            # st.write(dic_df_percentils[n_key_ms_sel].head(2))

            # sh_all_perc = st.checkbox(f'Show all percentils for cluster {n_key_ms_sel} of {col_name}')
            if sh_all_perc:
                fig0 = plot_TS_perc_clusters(dic_df_percentils[n_key_ms_sel], col_name=col_name )
                st.plotly_chart(fig0)
            
            # sh_int_perc = st.checkbox(f'Show percentils interval for cluster {n_key_ms_sel} of {col_name}')
            if sh_int_perc:
                figInt = plot_TS_perc_clusters(dic_df_percentils[n_key_ms_sel], percentiles=intervalo_perc[intPerc], col_name=col_name )
                st.plotly_chart(figInt)

            # sh_samp_perc = st.checkbox(f'Show sample TS and percentils for cluster {n_key_ms_sel} of {col_name}')
            if sh_samp_perc:
                n = int(n_key_ms_sel.split('_')[0])
                # st.write(f'n = {n}')
                k_list = list(range(n))
                grouped_df = n_opt_df.groupby(col_name).agg({
                                                            "num_pixels": "sum",   # Sum num_pixels
                                                            "label": list          # Collect label values in a list
                                                            }).reset_index()
                k_min = grouped_df.loc[grouped_df['num_pixels'].idxmin()][col_name]
                k_max = grouped_df.loc[grouped_df['num_pixels'].idxmax()][col_name]
                
                # st.write(grouped_df.loc[grouped_df['num_pixels'].idxmin()])
                # st.write (f'k_min = {k_min}, k_max = {k_max}, col_name = {col_name}')
                with st.expander(f"Settings for sample of {n_key_ms_sel}"):
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        k = st.selectbox(
                            f"Select one group of {n_key_ms_sel}",
                            k_list, 
                            index = k_list.index(k_min),
                            key=f"select_k_{n_key_ms_sel}"  # Unique key to avoid conflicts
                            # cluster_keys, 
                            # index=cluster_keys.index(n_opt_key_ms)#,
                            #key='quad' 
                            #on_change=on_quad_change,  # Call the function when the value changes
                            )
                    # with c2:
                        n_samples = st.number_input("Num of samples", min_value=0, max_value=100, value=40, step=1, key=f"samples_{n_key_ms_sel}")
                        rd_state = st.number_input("Random state", min_value=0,value=42, step=1, key=f"rdstate_{n_key_ms_sel}")
                # k=8
                merged_df = gen_groupClusterTS_df(n_opt_df, dic_df_ts[quad], col_name, k)
                # num_samples = 40  # Adjust the number of samples
                sampled_df = merged_df.sample(n=min(n_samples, len(merged_df)), random_state=rd_state)
                filter_perc_df = dic_df_percentils[n_key_ms_sel].loc[dic_df_percentils[n_key_ms_sel][col_name] == k]
                # filter_perc_df.head(2)
                title = f'Randomly selected rows and Percentil for k={k} of {col_name}'
                
                figSample = plot_TS_sel(sampled_df, title, dates, yaxis='NDVI', df_percentile=filter_perc_df, percentiles=intervalo_perc[intPerc] )
                st.plotly_chart(figSample)
            
            
if __name__ == "__main__":
    main()