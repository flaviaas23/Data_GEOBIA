# 20240924: programa foi criado a partir do spk_gen_df_ToPCA_nb_term, que foi criado a partir
#           do spk_gen_df_ToPCA baixado do exacta
#           lê sdft por data com todas as bandas e inseri no df_toPCA
# 20241202: ajustado para incluir a criação do df com os tempos de processamento, 
#           passagem dos parametros de configuração por argumentos 
# 20241210: Ler o sdft da data e do quadrante especificado
import os
import gc
import time
import datetime
import pandas as pd
import logging
import numpy as np
import argparse
from pathlib import Path

import findspark
findspark.init()

# Set the environment variable for spark
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
base_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/'

ti=time.time()
from pyspark.sql import SparkSession
from pyspark import StorageLevel
#.config("spark.shuffle.spill.compress", "false") \
# .config("spark.memory.offHeap.enabled","false") \
spark = SparkSession.builder \
    .master('local[*]') \
    .appName("PySpark to PCA") \
    .config("spark.local.dir", base_dir+"data/tmp_spark") \
    .config("spark.executorEnv.PYARROW_IGNORE_TIMEZONE", "1") \
    .config("spark.driverEnv.PYARROW_IGNORE_TIMEZONE", "1") \
    .config("spark.driver.memory", "300g") \
    .config("spark.executor.memory", "300g") \
    .config("spark.driver.maxResultSize", "300g")\
    .config("spark.sql.shuffle.spill.compress", "true") \
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
    .getOrCreate()

# Set log level to ERROR to suppress INFO and WARN messages
spark.sparkContext.setLogLevel("ERROR") 

from functions.functions_pca import cria_logger, list_files_to_read,\
                                    get_bandsDates, gen_dfToPCA_filter,\
                                    gen_sdfToPCA_filter, gen_arr_from_img_band_wCoords,\
                                    update_procTime_file, get_Dates

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')

# nome_prog = "spk_gen_df_ToPCA_nb_term_from_sdft" #os.path.basename(__file__)
nome_prog = os.path.basename(__file__)

# Inicializa o parser
parser = argparse.ArgumentParser(description="Program segment image")

# Define os argumentos
parser.add_argument("-bd", '--base_dir', type=str, help="Diretorio base", default='')
parser.add_argument("-sd", '--save_dir', type=str, help="Diretorio base para salvar saidas de cada etapa", default='data/tmp2/')
# parser.add_argument("-td", '--tif_dir', type=str, help="Dir dos tiffs", default='data/Cassio/S2-16D_V2_012014_20220728_/')
parser.add_argument("-q", '--quadrante', type=int, help="Numero do quadrante da imagem [0-all,1,2,3,4]", default=1)
parser.add_argument("-ld", '--log_dir', type=str, help="Dir do log", default='code/logs/')
parser.add_argument("-i", '--name_img', type=str, help="Nome da imagem", default='S2-16D_V2_012014')
parser.add_argument("-sp", '--sh_print', type=int, help="Show prints", default=1)
parser.add_argument("-pd", '--process_time_dir', type=str, help="Dir para df com os tempos de processamento", default='data/tmp2/')
args = parser.parse_args()

# base_dir = args.base_dir
save_etapas_dir = base_dir + args.save_dir if base_dir else args.save_dir + args.name_img+'/'
# tif_dir = base_dir + args.tif_dir if base_dir else args.tif_dir
read_quad = args.quadrante 
log_dir = base_dir + args.log_dir if base_dir else args.log_dir
name_img = args.name_img
process_time_dir = base_dir + args.process_time_dir
sh_print = args.sh_print

print (f'{a}: ######## INICIO {nome_prog} ##########') if sh_print else None

#cria logger
t = datetime.datetime.now().strftime('%Y%m%d_%H%M_')
nome_log = t + nome_prog.split('.')[0]+'.log'
# log_dir = base_dir + 'code/logs/'

logger = cria_logger(log_dir, nome_log)
logger.info(f'######## INICIO {nome_prog} ##########')
logger.info(f'args: sd={save_etapas_dir}\nread_quad={read_quad}\nld={log_dir} i={name_img}\npd={process_time_dir}\nsp={args.sh_print}') if sh_print else None

spark_local_dir = spark.conf.get("spark.local.dir")
logger.info(f"spark.local.dir: {spark_local_dir}")
print (spark_local_dir) if sh_print else None

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
proc_dic[ri]['etapa'] = 'Read sdft per day and Gen df_toPCA'
# Get all bands
# all_bands = ['B04', 'B03', 'B02', 'B08', 'EVI', 'NDVI']

# read_dir = base_dir + 'data/Cassio/S2-16D_V2_012014_20220728_/'
#read_dir = '/scratch/flavia/S2-16D_V2_012014_20220728_/'
# name_img = 'S2-16D_V2_012014'
# band_tile_img_files = list_files_to_read(read_dir, name_img)
# bands, dates= get_bandsDates(band_tile_img_files, tile=1)

# a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
# info = f'1. bands: {bands},\ndates: {dates}'
# print (f'{a}: {info}')
# logger.info(info)

padrao = 'spark_sdft_'
spark_sdft_dir = list_files_to_read(save_etapas_dir, padrao)
dates = get_Dates(spark_sdft_dir, pos_date=-1)
# temp_path= base_dir + '/data/tmp'
# temp_path= save_etapas_dir  #20250120 comentei
# dates = ['20220728', '20220829', '20220830', '20220901', '20220902', '20220903', '20220904'] #20250120 comentei
# dates = ['20220728']

print (f'dates= {dates}')

from functions.functions_pca import load_image_files3, check_perc_nan, gen_sdf_from_img_band,gen_arr_from_img_band_wCoords
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

i=0 
q=read_quad
for t in dates:#[:6]:
    print (f'******************** {i}: {t} *******************')
    # band_img_files = band_tile_img_files
    # band_img_file_to_load=[x for x in band_img_files if t in x.split('/')[-1]]
    # band_img_file_to_load
    
    # image_band_dic = {}
    # pos=-1
    # image_band_dic = load_image_files3(band_img_file_to_load, pos=pos)
    
    # bands = image_band_dic.keys()
    # print ( bands)#, band_img_file_to_load) 
    # logger.info(f'Generating columns of image for {t}')
    # t1=time.time()
    # perc_day_nan= check_perc_nan(image_band_dic, logger)
    # t2=time.time()
    # logger.info(f'Tempo para verificar percentual de nan nos tiffs: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
    
    # t1=time.time()
    # img_bands = []
    # img_bands = np.dstack([image_band_dic[x] for x in bands])
    # del image_band_dic
    # gc.collect()
    
    # cols_name = [x+'_'+t for x in bands]
    # t2=time.time()
    # logger.info(f'Columns names: {cols_name}')
    # print (f'Tempo para gerar img_bands of {t}: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
    # logger.info(f'Tempo para gerar img_bands of {t}: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
    
    # t1 = time.time()
    # values  = gen_arr_from_img_band_wCoords(img_bands, cols_name, ar_pos=0, sp=0, sh_print=0, logger=logger)
    # t2 = time.time()
    # logger.info(f'Tempo para gerar values with coords and bands: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
    # print (f'Tempo para gerar values with coords and bands: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
    
    # print (f'values: {values.shape}, {type(values)}, {type(values[0,0])}')

    # del img_bands
    # gc.collect()

    # print (cols_name)
    
    # schema = StructType([StructField(x, IntegerType(), True) for x in ['coords_0','coords_1']+cols_name])
    # print (f'schema: {schema}')
    
    # t1=time.time()
    # sdft = spark.createDataFrame(values, schema=schema)
    # sdft.persist(StorageLevel.DISK_ONLY)
    # t2=time.time()
    # logger.info(f'Tempo para criar spark sdft {t2-t1}s, {(t2-t1)/60:2f}m')
    # t2=time.time()
    # sdft=sdft.repartition(500)
    # t4=time.time()
    
    # logger.info(f'Tempo para repartition spark sdft {t4-t2}s, {(t4-t2)/60:2f}m')
    # sdft.unpersist()
    # print(f'Tempo para criar criar sdft {t2-t1}s, {(t2-t1)/60:2f}m')
    # print(f'Tempo para criar repartition sdft {t4-t2}s, {(t4-t2)/60:2f}m')
    
    # del values
    # gc.collect()

    # merge sdft with df_toPCA
    ar_pos=0
    t1=time.time()
    # sdfPath = temp_path + "/spark_sdft_"+t
    # sdfPath = save_etapas_dir + "spark_sdft_"+t  #commented 20241210
    sdfPath = save_etapas_dir + "spark_sdft_"+ t + '/Quad_'+str(read_quad)
    if i==0:   
        # window_spec = Window.orderBy('coords_0','coords_1')    
        # # Add a sequential index column
        # sdft = sdft.withColumn("orig_index", row_number().over(window_spec))
        tr1 = time.time()
        df_toPCA = spark.read.parquet(sdfPath) #sdft #.copy()
        tr2 = time.time()
        # df_toPCA.persist(StorageLevel.MEMORY_AND_DISK)
        # df_toPCA.persist(StorageLevel.DISK_ONLY)
        i+=1
        t2=time.time()
        proc_dic[ri]['subetapa'] = f'read first sdft of {t}'
        proc_dic[ri]['tempo'] = tr2-tr1
              
    else:
        if ar_pos:
            sdft = sdft.drop('coords', axis=1)
        tr1 = time.time()
        sdft = spark.read.parquet(sdfPath)
        tr2 = time.time()
        logger.info(f'Tempo para ler sdft parquet de {t}: {tr2-tr1:.2f}s, {(tr2-tr1)/60:.2f}m')
        sdft_columns = sdft.columns
        # print (f'sdft.columns {sdft_columns}') if sh_print else None
        
        sdft = sdft.select([c for c in sdft.columns if c != 'orig_index'])
        
        # print (f'sdft.columns after drop {sdft.columns}') if sh_print else None
        t1 = time.time()
        df_toPCA = df_toPCA.join(sdft, on=['coords_0', 'coords_1'], how='inner')
        i+=1
        t2=time.time()

        del sdft
        gc.collect()
        spark._jvm.System.gc() #20241210

        ri+=1
        proc_dic[ri]={} if ri not in proc_dic else None
        proc_dic[ri]['etapa'] = 'Read sdft per day and Gen df_toPCA'
        proc_dic[ri]['subetapa'] = f'read sdft of {t} for quad = {q}'
        proc_dic[ri]['tempo'] = tr2-tr1

        ri+=1
        proc_dic[ri]={} if ri not in proc_dic else None
        proc_dic[ri]['etapa'] = 'Read sdft per day and Gen df_toPCA'
        proc_dic[ri]['subetapa'] = f'sdft of {t} for quad = {q} inserido no df_toPCA'
        proc_dic[ri]['tempo'] = t2-t1
        
    info = f'tempo para adicionar sdft da matriz de bandas da imagen {t} : {t2-t1:.2f}'
    print (info) if sh_print else None 
    logger.info(info)
    logger.info(f'sdft of tiffs of {t} for quad = {q} inseridos no df_toPCA')
    logger.info(f'Tempo do join of {t} for quad = {q} {(t2-t1)/60:.2f}')
    print(f'****** Tempo parcial do total do inseridos até {t} for quad = {q} {(t2-t1)/60:.2f}') if sh_print else None
           
    print (df_toPCA.columns) if sh_print else None
    logger.info(f'df_toPCA columns: {df_toPCA.columns}')
    # del sdft
    # gc.collect()

#saving Pyspark Dataframe to parquet
logger.info(f'Inicio do save em parquet')
t1=time.time()
# temp_path= '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA' + '/data/tmp'
sdfPath = save_etapas_dir + "spark_df_toPCA_Quad_"+str(read_quad)
df_toPCA.write.parquet(sdfPath,mode = 'overwrite')
t2=time.time()
logger.info(f'Tempo para salvar em parquet {t2-t1:.2f}s {(t2-t1)/60:.2f}m')

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'Read sdft per day and Gen df_toPCA'
proc_dic[ri]['subetapa'] = f'save df_toPCA for quad = {q} in parquet'
proc_dic[ri]['tempo'] = t2-t1

spark.stop()

tf=time.time()
logger.info(f'Tempo total do programa {tf-ti:.2f}s {(tf-ti)/60:.2f}m')
print (f'Tempo total do programa {tf-ti:.2f}s {(tf-ti)/60:.2f}m') #if sh_print else None

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'Read sdft per day and Gen df_toPCA'
proc_dic[ri]['subetapa'] = f'{nome_prog} time execution total'
proc_dic[ri]['tempo'] = tf-ti

time_file = process_time_dir + "process_times.pkl"

update_procTime_file(proc_dic, time_file)
print (f'process_time file: {time_file}')
print ("FIM")