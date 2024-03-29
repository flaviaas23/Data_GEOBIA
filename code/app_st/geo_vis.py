import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import pickle
import time
import imageio.v2 as imageio
from skimage.segmentation import mark_boundaries
#import tqdm 

import streamlit as st
import streamlit.components.v1 as components
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image

from streamlit_js_eval import streamlit_js_eval
#st.set_page_config(layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def read_props_df_sel(ids, open_path, obj_to_read='props_df_sel',output=True):
    ''''
    read a list of props_df_sel and returns them as a dicionary
    '''
    
    dic_df={}
    for id in ids:      #ids #ids_file
        if obj_to_read == 'props_df_sel':
            file_to_open = open_path + '_'+str(id)+'.pkl'
            with open(file_to_open, 'rb') as handle: 
                b = pickle.load(handle)
            dic_df[id] = b[id][obj_to_read][id]
        elif obj_to_read == "segments_slic_sel":
            file_to_open = open_path + '_segments_'+str(id)+'.pkl'
            with open(file_to_open, 'rb') as handle: 
                b = pickle.load(handle)
            dic_df[id] = b[obj_to_read][id]
        #print (file_to_open) if output else None
        
        
    return dic_df     

def load_image_files():
    file_nbr = '/Library/Frameworks/R.framework/Versions/4.3-x86_64/Resources/library/sitsdata/extdata/Rondonia-20LMR/SENTINEL-2_MSI_20LMR_NBR_2022-07-16.tif'
    file_evi = '/Library/Frameworks/R.framework/Versions/4.3-x86_64/Resources/library/sitsdata/extdata/Rondonia-20LMR/SENTINEL-2_MSI_20LMR_EVI_2022-07-16.tif'
    file_ndvi = '/Library/Frameworks/R.framework/Versions/4.3-x86_64/Resources/library/sitsdata/extdata/Rondonia-20LMR/SENTINEL-2_MSI_20LMR_NDVI_2022-07-16.tif'
    file_blue = '/Library/Frameworks/R.framework/Versions/4.3-x86_64/Resources/library/sitsdata/extdata/Rondonia-20LMR/SENTINEL-2_MSI_20LMR_B02_2022-07-16.tif'
    file_red = '/Library/Frameworks/R.framework/Versions/4.3-x86_64/Resources/library/sitsdata/extdata/Rondonia-20LMR/SENTINEL-2_MSI_20LMR_B11_2022-07-16.tif'
    file_green = '/Library/Frameworks/R.framework/Versions/4.3-x86_64/Resources/library/sitsdata/extdata/Rondonia-20LMR/SENTINEL-2_MSI_20LMR_B8A_2022-07-16.tif'

    files_name=[file_nbr,file_evi, file_ndvi, file_red,file_green,file_blue]
    bands_name =[]
    image_band_dic={}
    for f in files_name:
        band = f.split("_")
        bands_name.append(band[-2])
        image_band_dic[band[-2]] = imageio.imread(f)

    #all_bands = ['B11', 'B8A', 'B02', 'NBR', 'EVI', 'NDVI']
    bands_sel = ['B11', 'B8A', 'B02'] # R,G,B bands selection for slic segmentation
    #img_sel = np.dstack((image_band_dic[bands_sel[0]], image_band_dic[bands_sel[1]], image_band_dic[bands_sel[2]]))

    image_band_dic_norm={}
    for k in image_band_dic.keys():
        #print (k)
        image_band_dic_norm[k]=image_band_dic[k].astype(float)/np.max(image_band_dic[k])

    img_sel_norm = np.dstack((image_band_dic_norm[bands_sel[0]], image_band_dic_norm[bands_sel[1]], image_band_dic_norm[bands_sel[2]]))

    
    # id_test=314
    # ids = [id_test]

    # save_path = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/git/SENTINEL-2_MSI_20LMR_RGB_2022-07-16'
    # #props_df_sel=read_props_df_sel(ids, save_path)
    
    # segments_slic_sel = read_props_df_sel(ids,save_path, obj_to_read='segments_slic_sel')

    # fig,ax = plt.subplots()
    # ax.imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test] ))
    # plt.tight_layout()

    # def onclick(event):
    #     if event.inaxes is not None:
    #         x = int(event.xdata)
    #         y = int(event.ydata)
    #         st.write(f"clicked at ({x}, {y})")
    #         #print (f'x {x}, y {y}')

    # cid = fig.canvas.mpl_connect('button_press_event', onclick)
    # st.write(f"clicked at ({cid})")
    
    #st.pyplot(fig)
    # st.write(f"2 clicked at ({cid})")
    #
    return img_sel_norm

def load_image_norm():
    read_path = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/test_segm_results/S2-16D_V2_012014_20220728_/S2-16D_V2_012014_20220728_'
    file_to_open = read_path + 'img_sel_norm.pkl'
    with open(file_to_open, 'rb') as handle:    
        img_sel_norm = pickle.load(handle)
    return img_sel_norm

def original_pos(scaled_x,scaled_y, original_width=1200, scaled_width =600):

    # Dimensions of the original image
    
    original_height = original_width

    # Dimensions of the scaled image
    scaled_height = scaled_width  # height of the scaled image

    # Scaling factors
    scale_factor_width = original_width / scaled_width
    scale_factor_height = original_height / scaled_height

    
    # Map to original pixel position
    #st.write("scaled_x,scaled_y",scaled_x,scaled_y)
    original_x = int(scaled_x * scale_factor_width)
    original_y = int(scaled_y * scale_factor_height)

    return [ original_y, original_x]

def load_cluster(id_test=314,read_segms=True, tipo_cluster='clara_'):
    '''
    load dic with cluster infos
    '''
    ### organizar
    save_path = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/test_segm_results/SENTINEL-2_MSI_20LMR_RGB_2022-07-16'
    #id_test=314
    #props_df_sel=read_props_df_sel([id_test],save_path, obj_to_read='props_df_sel')
    #label_file= '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/test_segm_results/SENTINEL-2_MSI_20LMR_RGB_2022-07-16_label_dict_314.pkl'
    file_to_open = save_path + '_cluster_'+tipo_cluster+str(id_test)+'.pkl'
    with open(file_to_open, 'rb') as handle: 
        b_props_cluster = pickle.load(handle)
    props_df_sel={}
    props_df_sel[id_test]=b_props_cluster['props_df_sel_cluster'][id_test]
    matrix_sim={}
    matrix_sim[id_test]=b_props_cluster['matrix_sim'][id_test]
    del b_props_cluster
    centroid_sel_df = props_df_sel[id_test][['label','std_NBR','std_EVI','std_NDVI' , 'num_pixels','centroid-0','centroid-1', 'coords']]
    
    if read_segms:
        ids = [id_test]
        segments_slic_sel=read_props_df_sel(ids,save_path, obj_to_read='segments_slic_sel')
        cl = {0: 'red', 1: 'green', 2: 'blue', 3:'white', 4:'orange', 5:'yellow', 6:'magenta', 7:'cyan'}

        return props_df_sel, matrix_sim, centroid_sel_df, segments_slic_sel, cl
    else:
        return props_df_sel, matrix_sim, centroid_sel_df

def calc_cor(valor,c_map='Blues'):
    colormap=plt.get_cmap(c_map)
    #colormap=plt.get_cmap('Blues')
    return colormap(valor)
#19/02/2024 nao preciso passar o props_df_sel por isso retirei do input
#def plot_img_pixel_sel(img_sel_norm, df_sel, label_value, threshold, \
def plot_img_pixel_sel(img_sel_norm, label_value, threshold, \
                       matrix_sim,centroid_sel_df, plot_centroids=False,\
                       id_test=314 ):
    ''''
    plot image with groups of pixel selected
    '''
    
    fig,ax = plt.subplots()
    thr_min = threshold[0]/100
    thr_max = threshold[1]/100
    centroid_row_matrix_sim = matrix_sim[id_test][label_value-1,:]
    
    centroid_sel_df['sim_value'] = centroid_row_matrix_sim                        
     
    filter_centroid_sel_df = centroid_sel_df[(centroid_sel_df['sim_value']>=thr_min) &
                                             (centroid_sel_df['sim_value']<=thr_max)]
    
    filter_centroid_sel_df['cor'] = filter_centroid_sel_df['sim_value'].apply(calc_cor, 'Blues')
    filter_centroid_sel_df.loc[label_value-1,'cor']='red'
    
    #colormap=plt.get_cmap('Blues')
    
    time_ini = time.time()
    if (plot_centroids):
        x_centroids=[x for x in filter_centroid_sel_df['centroid-1']]
        y_centroids=[y for y in filter_centroid_sel_df['centroid-0']]
        plt.scatter(x_centroids, y_centroids,s=1, color=list(filter_centroid_sel_df['cor']))
    else:
        x_sel= filter_centroid_sel_df.loc[label_value-1, 'centroid-1']
        y_sel= filter_centroid_sel_df.loc[label_value-1, 'centroid-0']
        plt.scatter(x_sel, y_sel,s=1, color='red')
        #plt.plot(x_sel, y_sel,marker='o',markersize=1, color='red')

        df_exploded=filter_centroid_sel_df.explode('coords')
        x_pixels = [p[1] for p in list(df_exploded['coords'])]
        y_pixels = [p[0] for p in list(df_exploded['coords'])]
        plt.scatter(x_pixels, y_pixels, s=1, color=df_exploded['cor'])
    
    time_fim = time.time()
    #pintar o ponto escolhido obrigatoriamente
    #plt.plot(row['centroid-1'], row['centroid-0'], marker=marker,markersize=markersz, color=row['cor'])
    #ax.imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test], color=(128,128,128)))
    ax.imshow(img_sel_norm)
    ax.axis('off')
    plt.tight_layout()
    #plt.show()
    st.pyplot(fig)
    st.write(time_fim-time_ini)
    return filter_centroid_sel_df

#nao precisa mais ler o dicionario 
#with open(label_file, 'rb') as handle: 
#     label_dict = pickle.load(handle)
#st.write(label_dict)
#st.write(props_df_sel[id_test]) 

def main():

    st.set_page_config(
        page_title="Pixels Cluster Identification",
        page_icon="🎯",
        layout="wide",
    )
    local_css("code/app_st/styles.css")

    if "LOAD_IMG" not in st.session_state:
        st.session_state["LOAD_IMG"]=True
        #st.write("load image",st.session_state["LOAD_IMG"])

    if "LOAD_CLUSTER" not in st.session_state:
        st.session_state["LOAD_CLUSTER"]=True
        #st.write("load cluster",st.session_state["LOAD_CLUSTER"])
    
    #LOAD_IMG=st.session_state["LOAD_CLUSTER"]  #24/02/2024 nao melembro pq LOAD_CLUSTER...
    LOAD_IMG=st.session_state["LOAD_IMG"]

    LOAD_CLUSTER = st.session_state["LOAD_CLUSTER"]

    if LOAD_IMG:
        #mg_sel_norm=load_image_files()
        img_sel_norm = load_image_norm()
        
        #png_rgb = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/code/result.png'
        png_rgb = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/code/S2-16D_V2_012014_20220728_.png'
        img = Image.open(png_rgb)
       
        st.session_state["LOAD_IMG"]=False
        #st.write("if load img",st.session_state["LOAD_IMG"])

        st.session_state["images"] = [img, img_sel_norm]

    img, img_sel_norm = st.session_state["images"]

    id_test=314
    if LOAD_CLUSTER:
        st.session_state["LOAD_CLUSTER"]=False
        #st.write("if load df",st.session_state["LOAD_CLUSTER"])

        #clara default
        props_df_sel, matrix_sim, centroid_sel_df, segments_slic_sel, cl =load_cluster(id_test)
        # #clara 200 5 iteracoes
        props_df_sel1, matrix_sim1, centroid_sel_df1 =load_cluster(id_test, read_segms=False, tipo_cluster='clara_200_')
        # #clara 20 5 iteracoes
        # props_df_sel5, matrix_sim5, centroid_sel_df5 =load_cluster(id_test, read_segms=False, tipo_cluster='clara_20_5iter_')
        # #clara 200 10 iteracoes
        # props_df_sel3, matrix_sim3, centroid_sel_df3 =load_cluster(id_test, read_segms=False, tipo_cluster='clara_200_10iter_')
        # #clara 400 5 iteracoes
        props_df_sel4, matrix_sim4, centroid_sel_df4 =load_cluster(id_test, read_segms=False, tipo_cluster='clara_400_5iter_')

        #clarans
        props_df_sel2, matrix_sim2, centroid_sel_df2 =load_cluster(id_test, read_segms=False, tipo_cluster='')
        
        st.session_state["cluster"] = [props_df_sel, matrix_sim, centroid_sel_df, segments_slic_sel, cl,
                                        matrix_sim1, centroid_sel_df1, 
                                        # matrix_sim3, centroid_sel_df3, 
                                        matrix_sim4, centroid_sel_df4,
                                        # matrix_sim5, centroid_sel_df5,
                                        matrix_sim2, centroid_sel_df2]

    props_df_sel, matrix_sim, centroid_sel_df, segments_slic_sel, cl,\
            matrix_sim1, centroid_sel_df1, \
            matrix_sim4, centroid_sel_df4,\
            matrix_sim2, centroid_sel_df2  = st.session_state["cluster"]
        #  matrix_sim3, centroid_sel_df3, 
         
        #  matrix_sim5, centroid_sel_df5,
        

    n_cols = 2
    col1, col2 = st.columns(n_cols)
    col_width = int(streamlit_js_eval(js_expressions='window.innerWidth', key = 'SCR')/n_cols)
    
    #st.write(f"Screen width is {col_width}")
   
    #props_df_sel, matrix_sim, centroid_sel_df, segments_slic_sel, cl =load_cluster(id_test)
    
    LOAD_TABLE = False

    threshold = st.slider(
        'Select a range for similarity',
        0, 100, (85, 100))
    
    #st.write('Values:', threshold[0], threshold[1])
    ###

    with col1:
        # img_sel_norm=load_image_files()
        # img = Image.open(tif_rgb)
        #st.write(img.size)
        scaled_width = col_width    # 300
        
        value = streamlit_image_coordinates(img,
                                            #Image.open(tif_rgb),
                                            width=scaled_width,
                                            key="png",
                                            )
        #st.write("if load img",st.session_state["LOAD_IMG"])

    #st.write("value:",value)
    col2_1, col2_2 = st.columns(n_cols)       
    with col2_1:
        if value:
            #st.write("if load df",st.session_state["LOAD_CLUSTER"])
            pos_to_find = original_pos(value['x'],value['y'],scaled_width =scaled_width)
            
            # st.write("SP:",label_found)
            #st.write(props_df_sel[id_test].head(2))
            for i, row in props_df_sel[id_test].iterrows():
                
                coords=np.array(row['coords'])
                
                is_found = any(np.array_equal(pos_to_find, c) for c in coords)
                
                if is_found:                    
                    label_value = int(row['label'])   
                    #st.write(label_value, row['label'])                 
                    break

            st.write("CLARA parametros default (40 + 2 * k)")
            filter_centroid_sel_df=plot_img_pixel_sel(img_sel_norm,  label_value, threshold, \
                   matrix_sim,centroid_sel_df, id_test=314 )
            #st.write(filter_centroid_sel_df.shape)
            # # st.write("CLARA 20 +1*k, 5 iteracoes ")
            # # filter_centroid_sel_df5=plot_img_pixel_sel(img_sel_norm,  label_value, threshold, \
            #                                            matrix_sim5,centroid_sel_df5, id_test=314 )                          
            st.write("CLARA 200 +20*k, 5 iteracoes ")
            filter_centroid_sel_df1=plot_img_pixel_sel(img_sel_norm,  label_value, threshold, \
                   matrix_sim1,centroid_sel_df1, id_test=314 )
            # st.write("CLARA parametros 200 +20*k, 10 iteracoes ")
            # filter_centroid_sel_df3=plot_img_pixel_sel(img_sel_norm,  label_value, threshold, \
            #        matrix_sim3,centroid_sel_df3, id_test=314 )
            # st.write("CLARA parametros 400 + 40*k , 5 iteracoes")
            # filter_centroid_sel_df4=plot_img_pixel_sel(img_sel_norm,  label_value, threshold, \
            #        matrix_sim4,centroid_sel_df4, id_test=314 )
            #fig,ax = plt.subplots()
            #thr_min = threshold[0]/100
            #thr_max = threshold[1]/100
            #centroid_row_matrix_sim = matrix_sim[id_test][label_value-1,:]
            #
            #centroid_sel_df['sim_value'] = centroid_row_matrix_sim                        
            # 
            #filter_centroid_sel_df = centroid_sel_df[(centroid_sel_df['sim_value']>=thr_min) &
            #                                         (centroid_sel_df['sim_value']<=thr_max)]
            #filter_centroid_sel_df['cor'] = 'blue'  #colocar degrade em funcao da similaridade
            #filter_centroid_sel_df.loc[label_value-1,'cor']='red'
            ##st.write(filter_centroid_sel_df.loc[label_value-1,'cor'])
            #time_ini = time.time()
            #for i, row in filter_centroid_sel_df.iterrows():
            #    if i == label_value-1:
            #        #st.write ("centroid=",i, row['label'], row['cor'], label_value )
            #        marker = 's'
            #        markersz = 2
            #    else:
            #        marker = 'o'
            #        markersz=1
            #    plt.plot(row['centroid-1'], row['centroid-0'], marker=marker,markersize=markersz, color=row['cor'])
            #    for p in row['coords']:
            #        plt.plot(p[1], p[0], marker=marker,markersize=markersz, color=row['cor'])
            #time_fim = time.time()
            ##pintar o ponto escolhido obrigatoriamente
            ##plt.plot(row['centroid-1'], row['centroid-0'], marker=marker,markersize=markersz, color=row['cor'])
            ##ax.imshow(mark_boundaries(img_sel_norm, segments_slic_sel[id_test], color=(128,128,128)))
            #ax.imshow(img_sel_norm)
            #ax.axis('off')
            #plt.tight_layout()
            ##plt.show()
            #st.pyplot(fig)
            ##st.write(time_fim-time_ini)
        
    with col2_2:
        if value:
            
            st.write("CLARA parametros 400 + 40*k , 5 iteracoes")
            filter_centroid_sel_df4=plot_img_pixel_sel(img_sel_norm,  label_value, threshold, \
                   matrix_sim4,centroid_sel_df4, id_test=314 )
            ############################ Para Clarans
            st.write("CLARANS")
            filter_centroid_sel_df2=plot_img_pixel_sel(img_sel_norm,  label_value, threshold, \
                   matrix_sim2,centroid_sel_df2, id_test=314 )
            
                        # fig,ax = plt.subplots()
            # centroid_row_matrix_sim2 = matrix_sim2[id_test][label_value-1,:]
            # 
            # centroid_sel_df2['sim_value'] = centroid_row_matrix_sim2                       
            #  
            # filter_centroid_sel_df2 = centroid_sel_df2[(centroid_sel_df2['sim_value']>=thr_min) &
                                                    #  (centroid_sel_df2['sim_value']<=thr_max)]
            # filter_centroid_sel_df2['cor'] = 'blue'  #colocar degrade em funcao da similaridade
            # filter_centroid_sel_df2.loc[label_value-1,'cor']='red'
            ##st.write(filter_centroid_sel_df.loc[label_value-1,'cor'])
            # time_ini = time.time()
            # for i, row in filter_centroid_sel_df2.iterrows():
                # if i == label_value-1:
                    ##st.write ("centroid=",i, row['label'], row['cor'], label_value )
                    # marker = 's'
                    # markersz = 2
                # else:
                    # marker = 'o'
                    # markersz=1
                # plt.plot(row['centroid-1'], row['centroid-0'], marker=marker,markersize=markersz, color=row['cor'])
                ##for p in row['coords']:
                ##    plt.plot(p[1], p[0], marker=marker,markersize=markersz, color=row['cor'])
            # time_fim = time.time()
            ##pintar o ponto escolhido obrigatoriamente
            # #plt.plot(row['centroid-1'], row['centroid-0'], marker=marker,markersize=markersz, color=row['cor'])
            # #ax.imshow(mark_boundaries(img_sel_norm, segments_slic_sel2[id_test], color=(128,128,128)))
            # ax.imshow(img_sel_norm)
            # ax.axis('off')
            # plt.tight_layout()
            #plt.show()
            # st.pyplot(fig)
            ##### fim clarans

            LOAD_TABLE=True

    # if st.checkbox("show Clara clustering"):
    #     if value:
    #         # # st.write("CLARA 20 +1*k, 5 iteracoes ")
    #         # # filter_centroid_sel_df5=plot_img_pixel_sel(img_sel_norm,  label_value, threshold, \
    #         #                                            matrix_sim5,centroid_sel_df5, id_test=314 )                          
    #         st.write("CLARA 200 +20*k, 5 iteracoes ")
    #         filter_centroid_sel_df1=plot_img_pixel_sel(img_sel_norm,  label_value, threshold, \
    #                matrix_sim1,centroid_sel_df1, id_test=314 )
    #         # st.write("CLARA parametros 200 +20*k, 10 iteracoes ")
    #         # filter_centroid_sel_df3=plot_img_pixel_sel(img_sel_norm,  label_value, threshold, \
    #         #        matrix_sim3,centroid_sel_df3, id_test=314 )
    #         st.write("CLARA parametros 400 + 40*k , 5 iteracoes")
    #         filter_centroid_sel_df4=plot_img_pixel_sel(img_sel_norm,  label_value, threshold, \
    #                matrix_sim4,centroid_sel_df4, id_test=314 )
    if LOAD_TABLE:
        if st.checkbox("Show selected SPs"):
            st.dataframe(filter_centroid_sel_df[['label','sim_value','std_NBR','std_EVI','std_NDVI' , 'num_pixels', 'cor']])
            #st.write(time.time()-time_fim)

if __name__ == "__main__":
    main()