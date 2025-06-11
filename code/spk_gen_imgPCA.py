# Programa para gerar o img pca from spark df pca em parquet
# 20241001: programa baixado do exacta
# 20241001: alterado para rodar no laptop, funcionou
# 20241211: alterado para ler 1/4 de tile do que gera o PCA da imagem
#           incluido passagem de argumentos
#           incluido os tempos de processamento no df
# 20250527: Alterado para gerar as imagens PCA de cada componente do PCA 
#           da imagem full, pode gerar apenas de  uma quadrantes ou as 4
#%%
import os
import gc
import time
import datetime
import pandas as pd
import logging
import numpy as np
import argparse

import findspark
findspark.init()

import zarr
#for terminal
#from code.functions.functions_pca import list_files_to_read, get_bandsDates, gen_dfToPCA_filter
#%%
# Set the environment variable for spark
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
base_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/'
ti=time.time()
#%%
#from functions_pca import list_files_to_read, get_bandsDates, gen_dfToPCA_filter, gen_sdfToPCA_filter
#imports for spark
# import pyspark.pandas as ps
from pyspark.sql import SparkSession

from pyspark.sql import functions as F
from pyspark.sql.functions import when, col, array, udf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StandardScaler

from pyspark.ml.functions import vector_to_array
#%%

# Step 1: Start a Spark session
# senao inicializar uma sessao é inicializada qdo chama o sparky
spark = SparkSession.builder \
    .appName("PySpark to PCA") \
    .config("spark.local.dir", base_dir+"data/tmp_spark") \
    .config("spark.executorEnv.PYARROW_IGNORE_TIMEZONE", "1") \
    .config("spark.driverEnv.PYARROW_IGNORE_TIMEZONE", "1") \
    .config("spark.driver.memory", "90g") \
    .config("spark.ui.showConsoleProgress", "false") \
    .config("spark.driver.extraJavaOptions", "-Djava.io.tmpdir=" + base_dir + "data/tmp_java")\
    .config("spark.executor.extraJavaOptions", "-Djava.io.tmpdir=" + base_dir + "data/tmp_java")\
    .getOrCreate()
#%%
# Set log level to ERROR to suppress INFO and WARN messages
spark.sparkContext.setLogLevel("ERROR") 

# Disable progress bar
# spark.conf.set("spark.ui.showConsoleProgress", "false")

from functions.functions_pca import cria_logger, update_procTime_file
from functions.functions_segmentation import get_quadrant_coords

# 20250528: Function to generate and save PCA images from the merged DataFrame
def genSavePCA_images(merged_df_q, comps, read_quad, save_etapas_dir, proc_dic,\
                        img_sz=5280, pca_fullImg=1, sh_print=0):
    
    """
    Generate and save PCA images from the merged DataFrame.
    Parameters: 
    """
    ri=list(proc_dic)[-1]
    img_pca_dic = {}
    t = 0
    proc_dic = {}
    ri = 0

    for c in comps:
        print(f'Geracao imagem pca {c} de {comps} para salvar') if sh_print else None
        t1 = time.time()
        col_c = merged_df_q.select(c).rdd.flatMap(lambda x: x).collect()
        col_c = np.array(col_c)
        img_pca_dic[c] = col_c.reshape(img_sz, img_sz)

        t2 = time.time()
        t += t2 - t1
        print(f'4.1 Tempo para gerar imagem {c}: {(t2-t1):.2f} s {(t2-t1)/60:.2f} m, {t/60:.2f}') if sh_print else None

        # Save images
        t3 = time.time()
        img_path = save_etapas_dir + f'spark_pca_images/img_pca_scaled_Quad_{read_quad}_{c}' + ('_fromFull' if pca_fullImg == 1 else '')
        zarr.save(img_path, img_pca_dic[c])

        t4 = time.time()
        del img_pca_dic[c]
        gc.collect()
        print(f'4.2 Tempo para salvar imagem {c}: {(t4-t3):.2f} s {(t4-t3)/60:.2f} m') if sh_print else None
        
        # Update processing dictionary
        ri+=1        #indice do dicionario com os tempos de cada subetapa
        proc_dic[ri]={} if ri not in proc_dic else None
        proc_dic[ri]['etapa'] = 'Gen PCA images'
        proc_dic[ri]['subetapa'] = f'Save img_pca_scaled_Quad_{read_quad}_{c} {sdfPath}'
        proc_dic[ri]['tempo'] = t2-t1

    del merged_df_q  
    gc.collect()
    return proc_dic

def getQuadrant_df(merged_df, quad, dic_row_col, sh_print=0):
    """
    Get the DataFrame for the specified quadrant.
    """
    # dic_row_col = get_quadrant_coords(quad, imgSize)
    rowi, rowf, coli, colf = dic_row_col[quad]
    print(f'GetQuadrant_df:quad\nrowi, rowf, coli, colf: {rowi}, {rowf}, {coli}, {colf}') if sh_print else None

    # Select the DataFrame for the specified quadrant
    if quad == 1:  # Quadrant 1: Superior Esquerdo
        return merged_df.filter((col('coords_0') < rowf) & (col('coords_1') < colf))
    elif quad == 2:  # Quadrant 2: Superior Direito
        return merged_df.filter((col('coords_0') < rowf) & (col('coords_1') >= coli))
    elif quad == 3:  # Quadrant 3: Inferior Esquerdo
        return merged_df.filter((col('coords_0') >= rowi) & (col('coords_1') < colf))
    elif quad == 4:  # Quadrant 4: Inferior Direito
        return merged_df.filter((col('coords_0') >= rowi) & (col('coords_1') >= coli))

# 20241211 bloco de parse incluido
# Inicializa o parser
parser = argparse.ArgumentParser(description="Program segment image")

# Define os argumentos
parser.add_argument("-bd", '--base_dir', type=str, help="Diretorio base", default='')
parser.add_argument("-sd", '--save_dir', type=str, help="Dir base para salvar saidas de cada etapa", default='data/tmp2/')
# parser.add_argument("-td", '--tif_dir', type=str, help="Dir dos tiffs", default='data/Cassio/S2-16D_V2_012014_20220728_/')
parser.add_argument("-q", '--quadrante', type=int, help="Numero do quadrante da imagem [0-all,1,2,3,4,9]", default=1)
parser.add_argument("-ld", '--log_dir', type=str, help="Dir do log", default='code/logs/')
parser.add_argument("-i", '--name_img', type=str, help="Nome da imagem", default='S2-16D_V2_012014')
parser.add_argument("-sp", '--sh_print', type=int, help="Show prints", default=0)
parser.add_argument("-pd", '--process_time_dir', type=str, help="Dir para df com os tempos de processamento", default='data/tmp2/')
# parser.add_argument("-rf", '--READ_df_features', type=int, help="Read or create df with features", default=0 )
parser.add_argument("-nc", '--num_components', type=int, help="number of PCA components", default=4 )
parser.add_argument("-pfi", '--pca_fullImg', type=int, help="usar pca da iamgem full", default=1 )
args = parser.parse_args()
 
# base_dir = args.base_dir
save_etapas_dir = base_dir + args.save_dir if base_dir else args.save_dir + args.name_img +'/'
# tif_dir = base_dir + args.tif_dir if base_dir else args.tif_dir
read_quad = args.quadrante 
log_dir = base_dir + args.log_dir if base_dir else args.log_dir
name_img = args.name_img
process_time_dir = base_dir + args.process_time_dir
sh_print = args.sh_print
n_components = args.num_components
pca_fullImg = args.pca_fullImg

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
t = datetime.datetime.now().strftime('%Y%m%d_%H%M_')

nome_prog = os.path.basename(__file__) #"spk_gen_df_ToPCA_nb_term_from_sdft" #os.path.basename(__file__)
print (f'{a}: ######## INICIO {nome_prog} {os.path.basename(__file__)} ##########')

#cria logger
nome_log = t+nome_prog.split('.')[0]+'.log'
# log_dir = base_dir + 'code/logs/'

logger = cria_logger(log_dir, nome_log)
logger.info(f'######## INICIO {nome_prog} ##########')

# List all configurations
conf_list = spark.sparkContext.getConf().getAll()

# Print all configurations
logger.info(f'SPARK configurarion')
logger.info(f'defaultParallelism: {spark.sparkContext.defaultParallelism}')
logger.info(f'spark.sql.files.openCostInBytes: {spark.conf.get("spark.sql.files.openCostInBytes")}')
for conf in conf_list:
    logger.info(f"{conf[0]}: {conf[1]}")
    # print (f"{conf[0]}: {conf[1]}")

ri=0        #indice do dicionario com os tempos de cada subetapa
proc_dic = {}
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'Gen PCA images'

# base_dir = '/scratch/flavia/'

# read spark df_toPCA orig_index, coords_0 and coords_1 with all coords
# sdfPath = base_dir + "pca/spark_sdf_toPCA"
# sdfPath = base_dir + "data/tmp/spark_sdf_toPCA2_6d" #20241211: commented
if pca_fullImg==1:  # read of full image
    sdfPath = save_etapas_dir + "spark_df_toPCA_Full"
else:
    sdfPath = save_etapas_dir + "spark_df_toPCA_Quad_" + str(read_quad)


t1 = time.time() 
# spark_sdf_toPCA= spark.read.parquet(sdfPath).select("orig_index", "coords_0", "coords_1")
spark_sdf_toPCA= spark.read.parquet(sdfPath).select( "coords_0", "coords_1")
t2 = time.time() 
print (f'1.0 spark_sdf_toPCA read: {spark_sdf_toPCA.show(n=10)}') if sh_print else None
logger.info(f'1.0 spark_sdf_toPCA quad = {read_quad} read: {spark_sdf_toPCA.show(n=10)}')
logger.info(f'Tempo para read spark_sdf_toPCA quad {read_quad} {(t2-t1):.2f}s {(t2-t1)/60:.2f}')

proc_dic[ri]['subetapa'] = f'Read spark_sdf_toPCA quad {read_quad} {sdfPath}'
proc_dic[ri]['tempo'] = t2-t1

#sort it spark df_toPCA by coords_0 and coords_1
t1 = time.time() 
spark_sdf_toPCA = spark_sdf_toPCA.orderBy(spark_sdf_toPCA["coords_0"].asc(), spark_sdf_toPCA["coords_1"].asc())
t2 = time.time() 
print (f'1.1 spark_sdf_toPCAQuad_{read_quad} sort: {spark_sdf_toPCA.show(n=10)}')
logger.info(f'1.1 spark_sdf_toPCAQuad_{read_quad}  sort: {spark_sdf_toPCA.show(n=10)}')

ri+=1        #indice do dicionario com os tempos de cada subetapa
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'Gen PCA images'
proc_dic[ri]['subetapa'] = f'Sort spark_df_toPCA_Quad_{read_quad} {sdfPath}'
proc_dic[ri]['tempo'] = t2-t1

# read spark_sdf_with_pca_scaled
# sdfPath = base_dir + "pca/spark_sdf_with_pca_scaled"
# sdfPath = base_dir + "data/tmp/df_with_pca_scaled"      #20241211 
if pca_fullImg==1:    # read of full image
    sdfPath = save_etapas_dir + "df_pca_scaled_Full"
else:
    sdfPath = save_etapas_dir + "df_pca_scaled_Quad_" +  str(read_quad)

t1 = time.time() 
# spark_sdf_with_pca_scaled= spark.read.parquet(sdfPath).select("orig_index", "coords_0", "coords_1","pca_features_scaled")
spark_sdf_with_pca_scaled= spark.read.parquet(sdfPath).select("coords_0", "coords_1","pca_features_scaled")
# print (f'1.2 spark_sdf_with_pca_scaled: {spark_sdf_with_pca_scaled.show(n=10)}')
t2 = time.time() 

ri+=1        #indice do dicionario com os tempos de cada subetapa
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'Gen PCA images'
proc_dic[ri]['subetapa'] = f'Read df_pca_scaled_Quad_{read_quad} {sdfPath}'
proc_dic[ri]['tempo'] = t2-t1

t1 = time.time() 
spark_sdf_with_pca_scaled = spark_sdf_with_pca_scaled.orderBy(spark_sdf_with_pca_scaled["coords_0"].asc(), spark_sdf_with_pca_scaled["coords_1"].asc())
t2 = time.time() 

ri+=1        #indice do dicionario com os tempos de cada subetapa
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'Gen PCA images'
proc_dic[ri]['subetapa'] = f'Sort df_pca_scaled_Quad_{read_quad} {sdfPath}'
proc_dic[ri]['tempo'] = t2-t1

#para transformar cada component do pca_features_scaled em colunas, precisa converter o vector(FDesnse Vector)
# em array:
n_comp = len(spark_sdf_with_pca_scaled.select("pca_features_scaled").first()[0])
print (f'2.1 numero de componentes pca: {n_comp}') if sh_print else None
logger.info(f'2.1 numero de componentes pca: {n_comp}')
spark_sdf_with_pca_scaled = spark_sdf_with_pca_scaled.withColumn("pca_array", vector_to_array("pca_features_scaled"))
#pode fazer o drop/select da column pca_features_scaled

# Criar novas colunas para cada elemento do array, precisa fazer antes pq as colunas com null/nan vao dar erro
comps=[]
for i in range(n_comp):
    comps.append('c'+str(i))
    spark_sdf_with_pca_scaled = spark_sdf_with_pca_scaled.withColumn(f"c{i}", col("pca_array").getItem(i))

print (f'2.2 spark_sdf_with_pca_scaled: {spark_sdf_with_pca_scaled.show(n=5)}') if sh_print else None
print (f'2.3 spark_sdf_with_pca_scaled columns: {spark_sdf_with_pca_scaled.columns}') if sh_print else None

logger.info(f'2.3 spark_sdf_with_pca_scaled columns: {spark_sdf_with_pca_scaled.columns}')
# 20250528 nao seria melhor fazer o select das colunas coords_0, coords_1 e c0, c1, c2, c3 antes do merge?
#          testar os tempos de merge com e sem o select
# spark_sdf_with_pca_scaled = spark_sdf_with_pca_scaled.select("coords_0", "coords_1", *comps)
t1 = time.time()
# Merge the sdfs to have all coords of the image
merged_df = spark_sdf_toPCA.join(spark_sdf_with_pca_scaled, 
                                    #   on=["orig_index", "coords_0", "coords_1"], 
                                      on=["coords_0", "coords_1"], 
                                      how="outer")
t2 = time.time()
print (f'Tempo para fazer merge(join) spark_sdf_toPCA spark_sdf_with_pca_scaled on coords {(t2-t1):.2f}') if sh_print else None
print (f'2.4 merged_df.columns: {merged_df.columns}') if sh_print else None
logger.info(f'Tempo para fazer merge(join) spark_sdf_toPCA spark_sdf_with_pca_scaled on coords {(t2-t1):.2f}')
logger.info(f'2.4 merged_df.columns: {merged_df.columns}')
#select only the columns needed
merged_df = merged_df.select("coords_0", "coords_1", *comps)  
# fazer como abaixo se quiser selecionar as colunas
# merged_df = merged_df.select( "coords_0", "coords_1", 'c0','c1','c2','c3')
#sort by coords_0 and coords_1

t1 = time.time()
merged_df = merged_df.orderBy(merged_df["coords_0"].asc(), merged_df["coords_1"].asc())

t2 = time.time()
merged_df=merged_df.fillna(value=np.nan)
t3 = time.time()
print (f'Tempo para sort merged_df: {t2-t1:.2f}s') if sh_print else None
print (f'Tempo para replace None to np.nan: {t3-t2:.2f}s') if sh_print else None
print (merged_df.show(n=5)) if sh_print else None
# print (f'3.3 merged_df schema: {merged_df.printSchema()}')
logger.info(f'Tempo para sort merged_df: {t2-t1:.2f}s')
logger.info(f'Tempo para replace None to np.nan: {t3-t2:.2f}s')
logger.info(f'3.3 merged_df schema: {merged_df.printSchema()}')
img_sz = 5280 #10560/2

if pca_fullImg==1:  # read of full image
    imgSize = 10560
    if read_quad == 9:  # gen pca images for all 4 quadrants
        # 
        print (f'read_quad = {read_quad}, imgSize = {imgSize}') if sh_print else None
        # ver como fazer codigo para obter o merged_df_q para cada quadrante
        dic_row_col = {}
        # dic_row_col[1] = (0, imgSize//2, 0, imgSize//2)  # Quadrant 1
        dic_row_col[1] = get_quadrant_coords(1, imgSize)
        # dic_row_col[2] = (0, imgSize//2, imgSize//2, imgSize)  # Quadrant 2
        dic_row_col[2] = get_quadrant_coords(2, imgSize)
        # dic_row_col[3] = (imgSize//2, imgSize, 0, imgSize//2)  # Quadrant 3   
        dic_row_col[3] = get_quadrant_coords(3, imgSize)
        # dic_row_col[4] = (imgSize//2, imgSize, imgSize//2, imgSize)  # Quadrant 4
        dic_row_col[4] = get_quadrant_coords(4, imgSize)
    
    else:   # save only quadrant specified by read_quad
        # Define the rowf and colf for the full image
        dic_row_col = {}
        # rowi, rowf, coli, colf = get_quadrant_coords(read_quad, imgSize)
        dic_row_col[read_quad] = get_quadrant_coords(read_quad, imgSize)
        # rowi, rowf, coli, colf = dic_row_col[read_quad]
        # print (f'4.1 rowi, rowf, coli, colf: {rowi}, {rowf}, {coli}, {colf}') if sh_print else None
        #seleciona no df a parte referente ao quadrante
        # if read_quad == 1: # Quadrante 1: Superior Esquerdo
        #     merged_df_q = merged_df.filter((col('coords_0') < rowf) & (col('coords_1') < colf))
        # elif read_quad == 2: # Quadrante 2: Superior Direito
        #     merged_df_q = merged_df.filter((col('coords_0') < rowf) & (col('coords_1') >= coli))
        # elif read_quad == 3: # Quadrante 3: Inferior Esquerdo
        #     merged_df_q = merged_df.filter((col('coords_0') >= rowi) & (col('coords_1') < colf))
        # elif read_quad == 4: # Quadrante 4: Inferior Direito
        #     merged_df_q = merged_df.filter((col('coords_0') >= rowi) & (col('coords_1') >= coli))

    # generate the image pca for each quadrant
    quads = dic_row_col.keys()
    for q in quads:
        rowi, rowf, coli, colf = dic_row_col[q]
        print (f'4.1 rowi, rowf, coli, colf: {rowi}, {rowf}, {coli}, {colf}') if sh_print else None

        # #seleciona no df a parte referente ao quadrante
        # if q == 1: # Quadrante 1: Superior Esquerdo
        #     merged_df_q = merged_df.filter((col('coords_0') < rowf) & (col('coords_1') < colf))
        # elif q == 2: # Quadrante 2: Superior Direito
        #     merged_df_q = merged_df.filter((col('coords_0') < rowf) & (col('coords_1') >= coli))
        # elif q == 3: # Quadrante 3: Inferior Esquerdo
        #     merged_df_q = merged_df.filter((col('coords_0') >= rowi) & (col('coords_1') < colf))
        # elif q == 4: # Quadrante 4: Inferior Direito
        #     merged_df_q = merged_df.filter((col('coords_0') >= rowi) & (col('coords_1') >= coli))

        merged_df_q = getQuadrant_df(merged_df, q, dic_row_col, sh_print=sh_print)
        proc_dic = genSavePCA_images(merged_df_q, comps, q, save_etapas_dir, proc_dic,\
                        img_sz=img_sz, pca_fullImg=pca_fullImg, sh_print=1)
        del merged_df_q #, img_pca_dic
        gc.collect()

else:   # read only one quadrant

    merged_df_q = merged_df
    proc_dic = genSavePCA_images(merged_df_q, comps, read_quad, save_etapas_dir, proc_dic,\
                        img_sz=img_sz, pca_fullImg=pca_fullImg, sh_print=1)
    del merged_df_q #, img_pca_dic
    gc.collect()
    
#generate the image pca for each pca component
#fazer uma funcao para gerar e salvar as imagens pca
# cols_sel = ['c0', 'c1', 'c2']
# 20250528: bloco abaixo comentado pq fiz uma funcao para gerar e salvar as imagens pca
# img_sz = 5280 #10560/2
# img_pca_dic={}
# t=0
# for c in comps:
#     print (f'Geracao imagem pca {c} de {comps} para salvar') if sh_print else None
#     t1 = time.time()
#     col_c = merged_df_q.select(c).rdd.flatMap(lambda x: x).collect()
#     col_c = np.array(col_c)
#     img_pca_dic[c] = col_c.reshape(img_sz,img_sz)
#     #outra forma de trocar o none por nan nas imagens:
#     #img_pca[c][np.equal(img_pca[c], None)] = np.nan
#     t2=time.time()
#     t+=t2-t1
#     print (f'4.1 Tempo para gerar imagem {c}: {(t2-t1):.2f} s {(t2-t1)/60:.2f} m, {t/60:.2f}')  if sh_print else None
#     #save images
#     t3=time.time()
#     # img_path=base_dir+'/pca/spark_pca_images/img_pca_scaled_'+c
#     # img_path = base_dir+'/data/tmp/spark_pca_images/img_pca_scaled_'+c #20241211 commented
#     if pca_fullImg==1:  # save from full image
#         img_path = save_etapas_dir + 'spark_pca_images/img_pca_scaled_Quad_'+ str(read_quad)+'_'+c+'_fromFull'
#     else: # save from quadrant
#         img_path = save_etapas_dir + 'spark_pca_images/img_pca_scaled_Quad_'+ str(read_quad)+'_'+c
#     zarr.save(img_path, img_pca_dic[c])
#     t4=time.time()
#     del img_pca_dic[c]
#     gc.collect()
#     print(f'4.2 Tempo para salvar imagem {c}: {(t4-t3):.2f} s {(t4-t3)/60:.2f} m')  if sh_print else None
    
#     ri+=1        #indice do dicionario com os tempos de cada subetapa
#     proc_dic[ri]={} if ri not in proc_dic else None
#     proc_dic[ri]['etapa'] = 'Gen PCA images'
#     proc_dic[ri]['subetapa'] = f'Save img_pca_scaled_Quad_{read_quad}_{c} {sdfPath}'
#     proc_dic[ri]['tempo'] = t2-t1


del merged_df#, img_pca_dic
gc.collect()

tf=time.time()
print (f'Tempo total do {nome_prog}: {tf-ti:.2f}s, {(tf-ti)/60:.2f}m')

#close sparky session
spark.stop()
ri=list(proc_dic)[-1]
ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'Gen PCA images'
proc_dic[ri]['subetapa'] = f'{nome_prog} time execution total'
proc_dic[ri]['tempo'] = tf-ti

time_file = process_time_dir + "process_times.pkl"
update_procTime_file(proc_dic, time_file)

print (f'process_time file: {time_file}')
print ("FIM")

