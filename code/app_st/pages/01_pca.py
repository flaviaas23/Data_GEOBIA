#20250106: visualizations related to the pca process
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import imageio.v2 as imageio
import requests, os, sys

import streamlit as st
import streamlit.components.v1 as components
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image

import zarr
import datetime

# Add the parent directory (code/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from functions_pca.py
from functions.functions_pca import get_bandsDates,load_image_files3, list_files_to_read, save_to_pickle
from functions.functions_segmentation import get_quadrant_coords

from src.functions_cluster_st import load_img_quadrant

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# -- Set page config
apptitle = 'PCA_Seg_Cluster_Eval'

#call back function when quadrants changes
def on_quad_change():
    # st.session_state["LOAD_CLUSTER"] = True
    # LOAD_CLUSTER = st.session_state["LOAD_CLUSTER"]
    st.session_state["QUAD_CHANGE"] = True
    QUAD_CHANGE = st.session_state["QUAD_CHANGE"]
    st.write(f"call back quad change = {QUAD_CHANGE} ")

#call back function when day changes
def on_day_change():
    st.session_state["DAY_CHANGE"] = True
    DAY_CHANGE = st.session_state["DAY_CHANGE"]
    st.write(f"call back day change = {DAY_CHANGE} ")
    
def main():

    st.set_page_config(
        page_title="PCA_Seg_Cluster_Eval",
        page_icon="🎯"#,
        # layout="wide",
    )
    local_css("code/app_st/styles.css")
    st.session_state.update(st.session_state)
    # Entrada para o diretório
    if "conf" in st.session_state:
        base_dir = st.session_state["conf"][0]
        # base_dir = st.session_state["base_dir"]
        # q = st.session_state["conf"][3]
        q = st.session_state["quad"]
        quad = st.session_state["quad"]
    else:
        base_dir = os.getcwd()
    
    tif_dir = base_dir+ '/tif/reg_30d/' # Valor padrão é o diretório atual
    name_img = st.sidebar.text_input('Image Name:', value='031027')

    # base_dir = st.session_state["conf"][0]
    # base_dir = st.session_state["base_dir"]
    # q = st.session_state["conf"][3]
    # q = st.session_state["quad"]
    # quad = st.session_state["quad"]

    sh_print = st.sidebar.checkbox('Show prints')
    
    if sh_print:
        st.write (f'pca page q = {q}, sessio stat quad={quad}')
        st.write(f'pca page base dir = {base_dir}')

    save_dir = st.sidebar.text_input('PCA Diretório:', value= 'data/img_'+name_img+'/')  # Valor padrão é o diretório atual
    save_etapas_dir = base_dir + '/' + save_dir 
    img_pca_dir = save_etapas_dir + st.sidebar.text_input('PCA images Directory:', value='spark_pca_images/')  # Valor padrão é o diretório atual
    
    if 'save_etapas_dir' not in st.session_state:
        st.session_state.save_etapas_dir = save_etapas_dir  # Default value
        st.write(f'quad not in st session state {st.session_state.save_etapas_dir}')

    st.session_state.save_etapas_dir = save_etapas_dir  # Default value

    if sh_print:
        st.write(f'pca page img_pca_dir = {img_pca_dir}')
    
    padrao = st.sidebar.text_input('Filtro:', value='*')  # Valor padrão é o diretório atual

    #select image tile quadrante
    if 'quad' not in st.session_state:
        st.session_state.quad = 1  # Default value
        st.write(f'quad not in st session state {st.session_state.quad}')

    if 'QUAD_CHANGE' not in st.session_state:
        st.write("quad change not in st session state")
        st.session_state["QUAD_CHANGE"] = False
        QUAD_CHANGE = st.session_state["QUAD_CHANGE"]
    
    if 'DAY_CHANGE' not in st.session_state:
        st.write("day change not in st session state")
        st.session_state["DAY_CHANGE"] = False
        DAY_CHANGE = st.session_state["QUAD_CHANGE"]

    if sh_print:
        st.write(f'quad in st session state {st.session_state.quad} ')
    
    q = st.sidebar.selectbox(
            "Select Image Quadrant",
            [1,2,3,4], 
            index=[1, 2, 3, 4].index(st.session_state.quad),
            key='quad',
            on_change=on_quad_change,  # Call the function when the value changes
            )
    quad = st.session_state["quad"]

    #select day of image
    #get the tif files name and its dates and bands
    if "dates" not in st.session_state:
        band_tile_img_files = list_files_to_read(tif_dir, name_img, sh_print=0)
        bands, dates = get_bandsDates(band_tile_img_files, tile=0)
        st.session_state['dates'] = dates
    else:
        dates = st.session_state['dates']
    
    t_day = st.sidebar.selectbox(
            "Select Image Day",
            dates,
            key='t_day',
            on_change=on_day_change,  # Call the function when the value changes
            )

    if sh_print:
        st.write(f'1 st.session state quad = {quad} {q}')

    if "LOAD_EXPLAIN" not in st.session_state:
        st.session_state["LOAD_EXPLAIN"] = False
    
    sh_explain = st.sidebar.checkbox('Show explanability', key='sh_explain')
    
    st.write(f'sh_explain={sh_explain}')
    if sh_explain:
        # if "LOAD_TIF" not in st.session_state:
        st.session_state["LOAD_EXPLAIN"]=True
    
    LOAD_EXPLAIN = st.session_state["LOAD_EXPLAIN"]

    #ler as componentes pca do dir usando o padrao informado
    img_files = list_files_to_read(img_pca_dir, padrao, sh_print=0)
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    # print (f'{a}: 1.1 PCA images files: {img_files}')
    t1 = time.time()
    pca_cols = []
    img_pca_dic = {}
    for f in img_files:
        c = f.split('_')[-1]
        # pca_cols.append(c)
        #print (f'{a}: {c}, {f}')
        if 'Quad_'+str(q) in f:
            # print (f'loading {f}, {c}') #if sh_print else None
            pca_cols.append(c)
            img_pca_dic[c] = zarr.load(f)      
    #seleciona as components que vao gerar a imagem pca
    cols_sel = pca_cols[:3]
    pca_cols.sort()

    if sh_print:
        st.write(pca_cols, cols_sel, pca_cols.sort()) 

    img_pca = np.dstack((img_pca_dic[pca_cols[0]], img_pca_dic[pca_cols[1]], img_pca_dic[pca_cols[2]]))

    fig,axes = plt.subplots(nrows=int(1), ncols=4, sharex=True)#, sharey=True)#,figsize=chart_size)
    axes[0].imshow(np.clip(img_pca_dic[pca_cols[0]], 0, 1))
    axes[0].set_title(f"PCA {pca_cols[0]}")
    
    axes[1].imshow(np.clip(img_pca_dic[pca_cols[1]], 0, 1))
    axes[1].set_title(f"PCA {pca_cols[1]}")
    
    axes[2].imshow(np.clip(img_pca_dic[pca_cols[2]], 0, 1))
    axes[2].set_title(f"PCA {pca_cols[2]}")
    
    axes[3].imshow(np.clip(img_pca, 0, 1))
    axes[3].set_title(f"PCA img_pca")

    # Remove axes for cleaner display
    for ax in axes:
        ax.axis('off')

    # Display the plot in Streamlit
    st.pyplot(fig)

    if st.session_state['QUAD_CHANGE'] or st.session_state['DAY_CHANGE'] or ("t_img_sel_uint8_q" not in st.session_state):    
        bands_sel = ['B11', 'B8A', 'B02'] # R,G,B bands
        t_img_sel_norm_q, t_img_sel_uint8_q, img_sz = load_img_quadrant(tif_dir, name_img, bands_sel, t_day, q, img_full=0)
        st.session_state["t_img_sel_norm_q"] = t_img_sel_norm_q
        st.session_state["t_img_sel_uint8_q"] = t_img_sel_uint8_q
        st.session_state['QUAD_CHANGE'] = False
        st.session_state['DAY_CHANGE'] = False
        QUAD_CHANGE = False        
    elif "t_img_sel_norm_q" in st.session_state:
        t_img_sel_norm_q = st.session_state["t_img_sel_norm_q"]
        t_img_sel_uint8_q = st.session_state["t_img_sel_uint8_q"]
        st.session_state['QUAD_CHANGE'] = False
        QUAD_CHANGE = False        
        st.write(f"if 2 quad change = {QUAD_CHANGE} ")
    fig,axes = plt.subplots(nrows=int(1), ncols=2, sharex=True)#, sharey=True)#,figsize=chart_size)
    axes[0].imshow(t_img_sel_norm_q)
    axes[0].set_title(f"Original image - Q{q}")
    axes[1].imshow(np.clip(img_pca, 0, 1))
    axes[1].set_title(f"PCA img_pca")

    # Remove axes for cleaner display
    for ax in axes:
        ax.axis('off')

    # Display the plot in Streamlit
    st.pyplot(fig)

    if LOAD_EXPLAIN:
        st.session_state["LOAD_EXPLAIN"] = False
        LOAD_EXPLAIN = False

        if q==0: # qdo gerar o explain num arquivo para o q=1 tirar o if-else
            import findspark
            findspark.init()
            # Set the environment variable for spark
            os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
            from pyspark.sql import SparkSession
            from pyspark import StorageLevel

            from pyspark.sql import functions as F
            from pyspark.sql.functions import when, col, array, udf
            from pyspark.ml.feature import VectorAssembler
            from pyspark.ml.linalg import Vectors
            from pyspark.ml.feature import StandardScaler

            from pyspark.ml.feature import PCA
            from pyspark.ml.feature import PCAModel


            # Step 1: Start a Spark session
            # senao inicializar uma sessao é inicializada qdo chama o sparky
            spark = SparkSession.builder \
                .master('local[*]') \
                .appName("PySpark to PCA") \
                .config("spark.local.dir", base_dir+"data/tmp_spark") \
                .config("spark.executorEnv.PYARROW_IGNORE_TIMEZONE", "1") \
                .config("spark.driverEnv.PYARROW_IGNORE_TIMEZONE", "1") \
                .config("spark.driver.memory", "400g") \
                .config("spark.executor.memory", "400g") \
                .config("spark.driver.maxResultSize", "400g")\
                .config("spark.sql.shuffle.spill.compress", "true") \
                .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
                .getOrCreate()

            # Set log level to ERROR to suppress INFO and WARN messages
            spark.sparkContext.setLogLevel("ERROR") 

            modelPath =  save_etapas_dir + "spark_pca_scaled-model_Quad_" + str(q)
            # Load the PCA model from the specified path
            pca_model_scaled = PCAModel.load(modelPath)

            
            pca_expl = pca_model_scaled.explainedVariance
            df_expl = pd.DataFrame([pca_expl], columns=['c'+str(i) for i in range(len(pca_expl))])
            df_expl['Explain Total'] = df_expl.sum(axis=1)  # Sum along rows (axis=1)
            st.table(df_expl)
            st.write(f'Explainability of 3 main pca components = {sum(pca_expl[:3])}')
            
            spark.stop()
        else: # qdo gerar o explain num arquivo para o q=1 tirar o if-else
            explPath = save_etapas_dir + "spark_pca_expl_Quad_" +str(q)+".pkl"

            with open(explPath, 'rb') as handle:    
                df_expl = pickle.load(handle)
                
            st.table(df_expl)
            st.write(f'Explainability of 3 main pca components = {df_expl[["c0", "c1", "c2"]].sum(axis=1)[0]}')
            

        
if __name__ == "__main__":
    main()