# 20250103: New streamlit app to analyse the results with pca, segmentation,
#           clara cluster, clara cluster using similarity matrix 
# python -m streamlit run code/app_st/00_geo_image.py
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

from streamlit_js_eval import streamlit_js_eval
from streamlit_javascript import st_javascript

# Add the parent directory (code/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from functions_pca.py
from functions.functions_pca import get_bandsDates,load_image_files3, list_files_to_read, save_to_pickle
from functions.functions_segmentation import get_quadrant_coords

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# -- Set page config
apptitle = 'PCA_Seg_Cluster_Eval'

def main():

    st.set_page_config(
        page_title="PCA_Seg_Cluster_Eval",
        page_icon="🎯"#,
        # layout="wide",
    )
    local_css("code/app_st/styles.css")
    st.session_state.update(st.session_state)      
    # if st.session_state:
    #     st.write("Keys in session_state:", list(st.session_state.keys()))
    # else:
    #     st.write("No keys in session_state.")
    # st.write(f'session_state keys {st.session_state.keys()}')
    if "LOAD_CONF" not in st.session_state:
        st.session_state["LOAD_CONF"] = True
        st.write("if load conf",st.session_state["LOAD_CONF"])
               
    LOAD_CONF = st.session_state["LOAD_CONF"]

    if "LOAD_TIF" not in st.session_state:
        st.session_state["LOAD_TIF"] = False
        st.write("if load tif",st.session_state["LOAD_TIF"])
    

    sh_tif = st.sidebar.checkbox('Show Tif image', key='sh_tif')
    sh_print = st.sidebar.checkbox('Show prints')
    if sh_print:
        st.write(f'sh_tif={sh_tif}')
    if sh_tif:
        # if "LOAD_TIF" not in st.session_state:
        st.session_state["LOAD_TIF"]=True

    LOAD_TIF = st.session_state["LOAD_TIF"]
    
    st.sidebar.header('Configurações')
     
    # 
    # Entrada para o diretório
    base_dir = st.sidebar.text_input('base Diretório:', value=os.getcwd(), key='base_dir')  # Valor padrão é o diretório atual
    # tif_dir = base_dir+ st.sidebar.text_input('Tif Diretório:',value='/data/Cassio/S2-16D_V2_012014_20220728_/' )  # Valor padrão é o diretório atual
    tif_dir = base_dir+ st.sidebar.text_input('Tif Diretório:',value='/tif/reg_30d/' )  # Valor padrão é o diretório atual

    # Entrada para string
    # name_img = st.sidebar.text_input('Image Name:', value='S2-16D_V2_012014')
    name_img = st.sidebar.text_input('Image Name:', value='031027')

    #select image tile quadrante
    if 'quad' not in st.session_state:
        st.session_state.quad = 1  # Default value
        if sh_print:
            st.write(f'quad not in st session state {st.session_state.quad}')
    
    if sh_print:
        st.write(f'quad in st session state {st.session_state.quad} ')
    
    q = st.sidebar.selectbox(
            "Select Image Quadrant",
            [1,2,3,4], 
            index=[1, 2, 3, 4].index(st.session_state.quad),
            key='quad')
    quad = st.session_state["quad"]
    q = quad
    if sh_print: 
        st.write(f'1 st.session state quad = {quad} {q}')

    if LOAD_CONF:    
        st.session_state["LOAD_CONF"]=False
        st.session_state["conf"] = [base_dir, tif_dir, name_img]

        if sh_print:
            quad2 = st.session_state["quad"]
            st.write ('updating LOAD_CONF to false e vars of conf')
            st.write(f'st.session state quad2 = {quad2}')

    base_dir, tif_dir, name_img = st.session_state["conf"]
    if sh_print: 
        st.write(f'2 st.session state quad = {quad} {q}')

    # id any conf value should be updated
    upd_conf = st.sidebar.button('Update')
    if upd_conf:
        st.session_state["LOAD_CONF"] = True
        st.write ('update button pressed: updating LOAD_CONF to true')
     

    # st.write(f' b_conf = {b_conf}')
    # b_conf=True
    # Botão para exibir resultados
    
    # if b_conf:
    

    #get the tif files name and its dates and bands
    band_tile_img_files = list_files_to_read(tif_dir, name_img, sh_print=0)
    bands, dates = get_bandsDates(band_tile_img_files, tile=0)
    
    if sh_print:
        st.write(f'Diretório informado: {tif_dir} ')
        st.write(f'Nome imagem informada: {name_img}')
        st.write(f'bands: {bands}\n {dates}')

    # Multiselect widget for selecting bands
    bands_sel = st.sidebar.multiselect(
        "Choose 3 Bands",  # Label
        bands,             # Options (from get_bandsDates)
        # default=['B04', 'B03', 'B02'] # R,G,B bands default selection, # Default selection
        default=['B11', 'B8A', 'B02'] # R,G,B bands default selection, # Default selection
    )

    # Validate Selection
    if len(bands_sel) != 3:
        st.sidebar.error("Please select exactly 3 bands!")

    #select day of image
    if "dates" not in st.session_state:
        band_tile_img_files = list_files_to_read(tif_dir, name_img, sh_print=0)
        bands, dates = get_bandsDates(band_tile_img_files, tile=0)
        st.session_state['dates'] = dates
    else:
        dates = st.session_state['dates']
    t_day = st.sidebar.selectbox(
            "Select Image Day",
            dates,
            key='t_day')
    

    # sh_tif = st.sidebar.checkbox('Show Tif image')
    # st.write(f'sh_tif={sh_tif}')
    # if sh_tif:
    #     if "LOAD_TIF" not in st.session_state:
    #         st.session_state["LOAD_TIF"]=True

    # LOAD_TIF = st.session_state["LOAD_TIF"]
    # Display selected bands
    if sh_print:
        st.write(f"**Sel Bands:*** {bands_sel}, Sel day: {t_day},SelQuadrant: {q}, sh_tif={sh_tif}")
        
    if LOAD_TIF:
        #carregar tif bands files
        image_band_dic = {}
        pos = -2#-1
        band_img_file_to_load = [x for x in band_tile_img_files if t_day in x.split('/')[-1]]
        image_band_dic = load_image_files3(band_img_file_to_load, pos=pos)
        # bands = list(image_band_dic.keys())

        # load tif RGB image to show
        if sh_print:
            st.write(f"**Selected Bands:** {bands_sel}, Selected day: {t_day}, sh_tif={sh_tif}")
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

        st.session_state["LOAD_TIF"]=False
        if sh_print: 
            st.write("if load tif",st.session_state["LOAD_TIF"], LOAD_TIF)

        st.session_state["images"] = [img_sel_uint8, img_width]
        img_sel_uint8, img_width = st.session_state["images"]

        n_cols = 2
        col1, col2 = st.columns(n_cols)
        # test=streamlit_js_eval(js_expressions='window.innerWidth', key = 'WIDTH', want_output = True,)
        # print (f'js_eval = {test}')


        window_width = st_javascript("window.innerWidth")  # Fetch browser width dynamically
        if sh_print:
            st.write(f"Window width: {window_width}px")  # Debug info

        # st.write(streamlit_js_eval(js_expressions='window.innerWidth', key = 'SCR'))
        # window_width = int(streamlit_js_eval(js_expressions='window.innerWidth', key = 'WIDTH'))
        type_wind = type(window_width)
        col_width = round(window_width/n_cols)
        
        if sh_print:
            st.write(f"Screen width is {window_width} n_cols {n_cols} type_wind{type_wind}, col_width {col_width}")
           
        with col1:
            print (f'col1 = {col1}')

            # img_sel_norm=load_image_files()
            # img = Image.open(tif_rgb)
            # st.write(img.size)
            # scaled_width = col_width    # 300
            # img_sel_uint8 = (img_sel_norm * 255).astype(np.uint8)  # Scale and convert to uint8

            # value = streamlit_image_coordinates(img_sel_uint8,#img_sel_norm,
            #                                     #Image.open(tif_rgb),
            #                                     # width=scaled_width,
            #                                     height=col_width,
            #                                     width=col_width,
            #                                     # use_column_width = "auto",
            #                                     # key="click_image"# "png",utf-8
            #                                     key="click_image"
                    
            #                             )

            # st.write(f"value={value}")

            # img_display = (img_sel_norm * 255).astype(np.uint8)

            # Display using st.image
            # st.write(f'Image day: {t_day}')
            st.image(img_sel_uint8, caption=f"Original Image day: {t_day} bands{bands_sel}", use_container_width=True)
            

            # Display using pyplot
            # fig,ax = plt.subplots()

            # ax.imshow(img_sel_norm)
            # ax.axis('off')
            # plt.tight_layout()
            # st.pyplot(fig)
        
        with col2:
            rowi, rowf, coli, colf = get_quadrant_coords(q, img_width)
            img_dic_width = image_band_dic['B11'].shape[0]
            # st.write(f'img_width = {img_width}, img dic width = {img_dic_width}')
            # rowi, rowf, coli, colf
            image_band_dic_q={}
            for x in bands:
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

            # Display using st.image of quadrant
            st.image(t_img_sel_uint8_q, caption=f"Quadrant {q}", use_container_width=True)
            

            #show quadrant to get the pixel selected
            # # st.write(f'col_width = {col_width}')
            # value = streamlit_image_coordinates(t_img_sel_uint8,
            #                                     height=col_width,
            #                                     width=col_width,
            #                                     key="click_image"
            #                             )
            # if value:
            #     st.write(f"value={value} {type(value)}")
            #     x= value['x']
            #     st.write(f'x= {x}')

            # st.session_state["images"] = [img_sel_uint8, img_width, t_img_sel_uint8]
            st.session_state["t_img_sel_norm_q"] = t_img_sel_norm_q
            # st.session_state["images"].append(t_img_sel_uint8_q)
            st.session_state["t_img_sel_uint8_q"] = t_img_sel_uint8_q

            t_img_sel_norm_q = st.session_state["t_img_sel_norm_q"]
            t_img_sel_uint8_q = st.session_state["t_img_sel_uint8_q"]

if __name__ == "__main__":
    main()