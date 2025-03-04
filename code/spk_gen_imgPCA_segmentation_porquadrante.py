# Programa para gerar o img pca from spark df pca em parquet
# 20241001: Programa baixado do exacta spk_gen_imgPCA_segmentation
# 20241001: Programa alterado para funcionar no laptop
# 20241014: Programa alterado  para fazer segmentacao por quadrante, dividindo
#           a imagem 
# 20241124: inclusão da funcao get_quadrant_coords 
import os
import gc
import time
import datetime
import pandas as pd
import numpy as np
import argparse

import zarr

#for terminal
#from code.functions.functions_pca import list_files_to_read, get_bandsDates, gen_dfToPCA_filter

# Set the environment variable for spark
# os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

#from functions_pca import list_files_to_read, get_bandsDates, gen_dfToPCA_filter, gen_sdfToPCA_filter
#imports for spark
# import pyspark.pandas as ps
# from pyspark.sql import SparkSession

# from pyspark.sql import functions as F
# from pyspark.sql.functions import when, col, array, udf
# from pyspark.ml.feature import VectorAssembler
# from pyspark.ml.linalg import Vectors
# from pyspark.ml.feature import StandardScaler

# from pyspark.ml.functions import vector_to_array


# # Step 1: Start a Spark session
# # senao inicializar uma sessao é inicializada qdo chama o sparky
# spark = SparkSession.builder \
#     .appName("PySpark to PCA") \
#     .config("spark.executorEnv.PYARROW_IGNORE_TIMEZONE", "1") \
#     .config("spark.driverEnv.PYARROW_IGNORE_TIMEZONE", "1") \
#     .config("spark.driver.memory", "90g") \
#     .config("spark.ui.showConsoleProgress", "false") \
#     .getOrCreate()

# # Set log level to ERROR to suppress INFO and WARN messages
# spark.sparkContext.setLogLevel("ERROR") 

# # Disable progress bar
# # spark.conf.set("spark.ui.showConsoleProgress", "false")

# base_dir = '/scratch/flavia/'

# # read spark df_toPCA orig_index, coords_0 and coords_1
# sdfPath = base_dir + "pca/spark_sdf_toPCA"
# spark_sdf_toPCA= spark.read.parquet(sdfPath).select("orig_index", "coords_0", "coords_1")
# print (f'1.0 spark_sdf_toPCA read: {spark_sdf_toPCA.show(n=10)}')

# #sort it spark df_toPCA by coords_0 and coords_1
# spark_sdf_toPCA = spark_sdf_toPCA.orderBy(spark_sdf_toPCA["coords_0"].asc(), spark_sdf_toPCA["coords_1"].asc())
# print (f'1.1 spark_sdf_toPCA_read and sort: {spark_sdf_toPCA.show(n=10)}')

# # read spark_sdf_with_pca_scaled
# sdfPath = base_dir + "pca/spark_sdf_with_pca_scaled"
# spark_sdf_with_pca_scaled= spark.read.parquet(sdfPath).select("orig_index", "coords_0", "coords_1","pca_features_scaled")
# print (f'1.2 spark_sdf_with_pca_scaled: {spark_sdf_with_pca_scaled.show(n=10)}')

# spark_sdf_with_pca_scaled = spark_sdf_with_pca_scaled.orderBy(spark_sdf_with_pca_scaled["coords_0"].asc(), spark_sdf_with_pca_scaled["coords_1"].asc())

# #para transformar cada component do pca_features_scaled em colunas, precisa converter o vector(FDesnse Vector)
# # em array:
# n_comp = len(spark_sdf_with_pca_scaled.select("pca_features_scaled").first()[0])
# print (f'2.1 numero de componentes pca: {n_comp}')
# spark_sdf_with_pca_scaled= spark_sdf_with_pca_scaled.withColumn("pca_array", vector_to_array("pca_features_scaled"))
# #pode fazer o drop da column pca_features_scaled

# # Criar novas colunas para cada elemento do array, precisa fazer antes pq as colunas com null/nan vao dar erro
# comps=[]
# for i in range(n_comp):
#     comps.append('c'+str(i))
#     spark_sdf_with_pca_scaled = spark_sdf_with_pca_scaled.withColumn(f"c{i}", col("pca_array").getItem(i))

# print (f'2.2 spark_sdf_with_pca_scaled: {spark_sdf_with_pca_scaled.show(n=5)}')

# # Merge the sdfs to have all coords of the image
# merged_df = spark_sdf_toPCA.join(spark_sdf_with_pca_scaled, 
#                                       on=["orig_index", "coords_0", "coords_1"], 
#                                       how="outer")
# #sort by coords_0 and coords_1
# merged_df = merged_df.orderBy(merged_df["coords_0"].asc(), merged_df["coords_1"].asc())
# t1=time.time()
# merged_df=merged_df.fillna(value=np.nan)
# t2=time.time()
# print (f'Tempo para replace None to np.nan: {t2-t1}s')
# print (merged_df.show(n=5))
# print (f'3.3 merged_df schema: {merged_df.printSchema()}')

# #generate the image pca for each pca component
# cols_sel = ['c0', 'c1', 'c2']
# img_sz = 10560
# img_pca_dic={}
# t=0
# for c in comps:
#     t1 = time.time()
#     col_c = merged_df.select(c).rdd.flatMap(lambda x: x).collect()
#     col_c = np.array(col_c)
#     img_pca_dic[c] = col_c.reshape(img_sz,img_sz)
#     #outra forma de trocar o none por nan nas imagens:
#     #img_pca[c][np.equal(img_pca[c], None)] = np.nan
#     t2=time.time()
#     t+=t2-t1
#     print (f'4.1 Tempo para gerar imagem {c}: {t2-t1} s {(t2-t1)/60} m, {t}')
#     #save images
#     t3=time.time()
#     img_path=base_dir+'/pca/spark_pca_images/img_pca_scaled_'+c
#     zarr.save(img_path, img_pca_dic[c])
#     t4=time.time()
#     print(f'4.2 Tempo para salvar imagem {c}: {t4-t3} s {(t4-t3)/60} m')

# del merged_df, img_pca_dic
# gc.collect()

# #close sparky session
# spark.stop()

# quit()

########### fazer a segmentacao
###########depois separar em outro programa
from pathlib import Path
from pysnic.algorithms.snic import snic
#For snic SP
from pysnic.algorithms.snic import compute_grid
from pysnic.ndim.operations_collections import nd_computations
from itertools import chain
from pysnic.metric.snic import create_augmented_snic_distance
from functions.functions_pca import calc_avg_array, list_files_to_read, update_procTime_file
from functions.functions_segmentation import run_snic_gen_dfs, gen_coords_snic_df, \
                                             gen_centroid_snic_df, save_segm, get_quadrant_coords

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
#b = datetime.datetime.now().strftime('%H:%M:%S')
nome_prog = os.path.basename(__file__)
print (f'{a}: 0. INICIO {nome_prog}')
time_i = time.time()

ri=0        #indice do dicionario com os tempos de cada subetapa
proc_dic = {}
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'Segment image pca per quadrant'
#read the sim matrix s

# Inicializa o parser
parser = argparse.ArgumentParser(description="Program segment image")

# Define os argumentos
parser.add_argument("-dsi", '--image_dir', type=str, help="Diretorio da imagem", default='data/tmp/spark_pca_images/')
parser.add_argument("-q", '--quadrante', type=int, help="Numero do quadrante da imagem [0-all,1,2,3,4]", default=1)
parser.add_argument("-p", '--padrao', type=str, help="Filtro para o diretorio ", default='*')
parser.add_argument("-pd", '--process_time_dir', type=str, help="Dir para df com os tempos de processamento", default='data/tmp/')
args = parser.parse_args()
# print (args)
read_quad = args.quadrante 
img_dir = args.image_dir
padrao = args.padrao
process_time_dir = args.process_time_dir

print (read_quad, img_dir, padrao, process_time_dir)

#For exacta
# base_dir = '/scratch/flavia/'
# img_dir = base_dir + 'pca/spark_pca_images/'
#for laptop
base_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/'
# img_dir = base_dir + 'data/tmp/spark_pca_images/'
img_dir = base_dir + img_dir if base_dir else img_dir 

padrao =  '*' #'pca_scaled'

if not padrao:
    #read images usando a lista de components e o diretorio a ser lido
    pca_cols = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5']
    cols_sel = pca_cols[:3]
    print (f'pca_cols={pca_cols}, cols_sel={cols_sel}')
    img_pca_dic = {}
    t1 = time.time()
    print (f'Reading the PCA images')
    for c in pca_cols:
        img_path=base_dir+'/pca/spark_pca_images/img_pca_scaled_'+c
        img_pca_dic[c] = zarr.load(img_path)

    t2 = time.time()
    
else:
    #ler as componentes do dir usando o padrao informado
    img_files = list_files_to_read(img_dir,padrao,sh_print=0)
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a}: 1.1 PCA images files: {img_files}')
    t1 = time.time()
    pca_cols = []
    img_pca_dic = {}
    for f in img_files:
        c = f.split('_')[-1]
        pca_cols.append(c)
        #print (f'{a}: {c}, {f}')
        img_pca_dic[c] = zarr.load(f)


    t2 = time.time()
    #seleciona as components que vao gerar a imagem pca
    pca_cols.sort()
    cols_sel = pca_cols[:3]
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')

print (f'{a}: 1.2 PCA components and components selected: {pca_cols}, {cols_sel}')
print (f'{a}: 1.3 Tempo para ler as componentes das imagens: {t2-t1:.2f} s {(t2-t1)/60:.2f} m ')

proc_dic[ri]['subetapa'] = 'read components images'
proc_dic[ri]['tempo'] = t2-t1

pca_cols.sort()
q_number =  4 #number of quadrants of image
imgSize = img_pca_dic[pca_cols[0]].shape[0]
# Q_imgSize = imgSize//2
# se quadrante da imagem for definido fazer o snic neste quadrante, senao fazer nos 4
quadrants = [read_quad] if read_quad else range(1,q_number+1)

print (f'{a}: Qdrantes to be segmented {quadrants}')

for q in quadrants:

    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f"{a}: 1.4.1 Quadrant image size{imgSize//2}")
    i=0
    # 20241124 bloco abaixo substituido pela funcao get_quadrant_coords
    '''
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
        rowi, rowf = Q_imgSize, imgSize
        coli, colf = Q_imgSize, imgSize
    '''

    rowi, rowf, coli, colf = get_quadrant_coords(q, imgSize)
    
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a}: 1.4.2 Quadrante Q{q} {rowi}:{rowf} {coli}:{colf}')
    #gerar a imagem pca do quadrante
    t1=time.time()
    img_pca_dic_q={}
    for c in pca_cols:
        img_pca_dic_q[c] = img_pca_dic[c][rowi:rowf, coli:colf]
    t2 = time.time()
    img_pca = np.dstack((img_pca_dic_q[pca_cols[0]], 
                        img_pca_dic_q[pca_cols[1]],
                        img_pca_dic_q[pca_cols[2]]))
    # img_pca = np.dstack((img_pca_dic[pca_cols[0]][rowi:rowf, coli:colf], 
    #                      img_pca_dic[pca_cols[1]][rowi:rowf, coli:colf],
    #                      img_pca_dic[pca_cols[2]][rowi:rowf, coli:colf]))
    
    # #gerar a imagem pca

    # img_pca = np.dstack((img_pca_dic[pca_cols[0]], img_pca_dic[pca_cols[1]], img_pca_dic[pca_cols[2]]))
    t3 = time.time()
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a}: 1.4.3 Tempo gerar a quad dic image pca Q{q} com as componentes {pca_cols}: {t2-t1:.2f} s {(t2-t1)/60:.2f} m ')
    print (f'{a}: 1.4.4 Tempo gerar a quad image pca Q{q} com as componentes {pca_cols}: {t3-t2:.2f} s {(t3-t2)/60:.2f} m ')
    print (f'{a}: 1.5 type img {type(img_pca)}, shape: {img_pca.shape}')

    ri+=1
    proc_dic[ri]={} if ri not in proc_dic else None
    proc_dic[ri]['etapa'] = 'Segment image pca per quadrant'
    proc_dic[ri]['subetapa'] = f'Time to gen quadrant {q} dic of components'
    proc_dic[ri]['tempo'] = t2-t1
    proc_dic[ri]['size'] = img_pca.shape

    # img_pca_shape = img_pca.shape
    # grid = compute_grid(img_pca.shape, 2)
    # seeds = list(chain.from_iterable(grid))
    # seed_len = len(seeds)
    # compact=0.2
    # # choose a distance metric #se nao fornecido faz exatamente isso
    # distance_metric = create_augmented_snic_distance(img_pca_shape, seed_len, compact)
        
    # #start = timer()
    # t2=time.time()
    # segments_snic_sel_sp, dist_snic_sp, centroids_snic_sp = snic(
    #                             img_pca.tolist(),
    #                             #img_sel_norm.tolist(),
    #                             seeds,
    #                             compact, nd_computations["3"], distance_metric)#,
    #                             #update_func=lambda num_pixels: print("processed %05.2f%%" % (num_pixels * 100 / number_of_pixels)))
        
    # t3=time.time()
    print (f'{a}: 2. Segment image, run_snic_gen_dfs')
    id=0
    img_pca_shape = img_pca.shape
    n_segms= 30000 #110000
    #grid = compute_grid(img_pca.shape, n_segms)
    #seeds = list(chain.from_iterable(grid))
    #seed_len = len(seeds)
    compact = 2
    bands_sel = cols_sel
    t1=time.time()
    snic_centroid_df,  segments_snic_sel_sp = run_snic_gen_dfs(id, img_pca_shape, 
                                                            img_pca.tolist(), 
                                                            n_segms, \
                                                            compact, \
                                                            '', \
                                                            #img_pca_dic, 
                                                            img_pca_dic_q, 
                                                            bands_sel, 
                                                            save_snic=False, 
                                                            sh_print=True)

    t2=time.time()
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a}: 2.1 Tempo para segmentar a imagem : {t2-t1:.2f} s {(t2-t1)/60:.2f} m')

    print (f'{a}: 2.2 snic_centroid_df: \n{snic_centroid_df.head()}')
    #print (f'{a}: 2.3 segments:\n{segments_snic_sel_sp[1]}' )
    del img_pca, img_pca_dic_q
    gc.collect()
    
    ri+=1
    proc_dic[ri]={} if ri not in proc_dic else None
    proc_dic[ri]['etapa'] = 'Segment image pca per quadrant'
    proc_dic[ri]['subetapa'] = f'Time to segment quadrant {q}'
    proc_dic[ri]['tempo'] = t2-t1
    
    ### Save the results of snic
    print (f'{a}: 3. Saving results snic')
    params_test = {"segms":n_segms, "compactness":compact, "ski_img": True}
    # f_name = base_dir +'pca/pca_snic/'
    f_name = base_dir +'data/tmp/spark_pca_snic2/'+'Quad_'+str(q)+'/'
    # Verificar se driretorio existe, caso nao exista, criar
    diretorio = Path(f_name)
    diretorio.mkdir(parents=True, exist_ok=True)
    t1=time.time()
    save_segm(id, snic_centroid_df, segments_snic_sel_sp, 
                f_name, params_test, str_fn='quad'+str(q)+'_', sh_print=True )
    t2=time.time()              
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a}: 3.1 Tempo para salvar em pkl o resultado da segmentacao imagem Quad{q}: {t2-t1:.2f} s {(t2-t1)/60:.2f} m ')

    ri+=1
    proc_dic[ri]={} if ri not in proc_dic else None
    proc_dic[ri]['etapa'] = 'Segment image pca per quadrant'
    proc_dic[ri]['subetapa'] = f'Time to save segmented quadrant {q}'
    proc_dic[ri]['tempo'] = t2-t1
    proc_dic[ri]['size'] = snic_centroid_df.shape

    del snic_centroid_df, segments_snic_sel_sp
    print (f'{a}: 4. Fim do {nome_prog}')

del img_pca_dic
time_f = time.time()

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'Segment image pca per quadrant'
proc_dic[ri]['subetapa'] = f'{nome_prog} time execution total'
proc_dic[ri]['tempo'] = time_f-time_i

# time_file = base_dir + 'data/tmp/process_times.pkl'
time_file = base_dir + process_time_dir

update_procTime_file(proc_dic, time_file)
print ("FIM")
# Fazer teste de salvar em spark para ver tempo e espacó usado

#close sparky session
#spark.stop()


