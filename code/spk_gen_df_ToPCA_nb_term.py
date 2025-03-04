import os
import gc
import time
import datetime
import pandas as pd
import logging
import numpy as np

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
    .config("spark.memory.offHeap.enabled","false") \
    .config("spark.shuffle.compress", "true") \
    .config("spark.shuffle.spill.compress", "true") \
    .config("spark.driver.maxResultSize", "300g")\
    .config("spark.memory.fraction", "0.1") \
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
    .getOrCreate()

# Set log level to ERROR to suppress INFO and WARN messages
spark.sparkContext.setLogLevel("ERROR") 

from functions.functions_pca import cria_logger, list_files_to_read, get_bandsDates, gen_dfToPCA_filter, gen_sdfToPCA_filter, gen_arr_from_img_band_wCoords

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
t = datetime.datetime.now().strftime('%Y%m%d_%H%M_')

nome_prog = "spk_gen_df_ToPCA_nb_term" #os.path.basename(__file__)
print (f'{a}: ######## INICIO {nome_prog} {os.path.basename(__file__)} ##########')

#cria logger
nome_log = t+nome_prog.split('.')[0]+'.log'
log_dir = base_dir + 'code/logs/'

logger = cria_logger(log_dir, nome_log)
logger.info(f'######## INICIO {nome_prog} ##########')

spark_local_dir = spark.conf.get("spark.local.dir")
logger.info(f"spark.local.dir: {spark_local_dir}")
print (spark_local_dir)

# List all configurations
conf_list = spark.sparkContext.getConf().getAll()

# Print all configurations
logger.info(f'SPARK configurarion')
logger.info(f'defaultParallelism: {spark.sparkContext.defaultParallelism}')
logger.info(f'spark.sql.files.openCostInBytes: {spark.conf.get("spark.sql.files.openCostInBytes")}')
for conf in conf_list:
    logger.info(f"{conf[0]}: {conf[1]}")
    print (f"{conf[0]}: {conf[1]}")

# Get all bands
all_bands = ['B04', 'B03', 'B02', 'B08', 'EVI', 'NDVI']

read_dir = base_dir + 'data/Cassio/S2-16D_V2_012014_20220728_/'
#read_dir = '/scratch/flavia/S2-16D_V2_012014_20220728_/'
name_img = 'S2-16D_V2_012014'
band_tile_img_files = list_files_to_read(read_dir, name_img)
bands, dates= get_bandsDates(band_tile_img_files, tile=1)
cur_dir = os.getcwd()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
info = f'1. bands: {bands},\ndates: {dates}'
print (f'{a}: {info}')
logger.info(info)

from functions.functions_pca import load_image_files3, check_perc_nan, gen_sdf_from_img_band,gen_arr_from_img_band_wCoords
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

i=0 

for t in dates[:6]:
    print (f'******************** {i}: {t} *******************')
    band_img_files = band_tile_img_files
    band_img_file_to_load=[x for x in band_img_files if t in x.split('/')[-1]]
    band_img_file_to_load
    
    image_band_dic = {}
    pos=-1
    image_band_dic = load_image_files3(band_img_file_to_load, pos=pos)
    
    bands = image_band_dic.keys()
    print ( bands)#, band_img_file_to_load) 
    logger.info(f'Generating columns of image for {t}')
    t1=time.time()
    perc_day_nan= check_perc_nan(image_band_dic, logger)
    t2=time.time()
    logger.info(f'Tempo para verificar percentual de nan nos tiffs: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
    
    t1=time.time()
    img_bands = []
    img_bands = np.dstack([image_band_dic[x] for x in bands])
    del image_band_dic
    gc.collect()
    
    cols_name = [x+'_'+t for x in bands]
    t2=time.time()
    logger.info(f'Columns names: {cols_name}')
    print (f'Tempo para gerar img_bands of {t}: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
    logger.info(f'Tempo para gerar img_bands of {t}: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
    
    t1 = time.time()
    values  = gen_arr_from_img_band_wCoords(img_bands, cols_name, ar_pos=0, sp=0, sh_print=0, logger=logger)
    t2 = time.time()
    logger.info(f'Tempo para gerar values with coords and bands: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
    print (f'Tempo para gerar values with coords and bands: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
    
    print (f'values: {values.shape}, {type(values)}, {type(values[0,0])}')

    del img_bands
    gc.collect()

    print (cols_name)
    
    schema = StructType([StructField(x, IntegerType(), True) for x in ['coords_0','coords_1']+cols_name])
    print (f'schema: {schema}')
    
    t1=time.time()
    sdft = spark.createDataFrame(values, schema=schema)
    sdft.persist(StorageLevel.DISK_ONLY)
    t2=time.time()
    logger.info(f'Tempo para criar spark sdft {t2-t1}s, {(t2-t1)/60:2f}m')
    t2=time.time()
    sdft=sdft.repartition(500)
    t4=time.time()
    
    logger.info(f'Tempo para repartition spark sdft {t4-t2}s, {(t4-t2)/60:2f}m')
    sdft.unpersist()
    print(f'Tempo para criar criar sdft {t2-t1}s, {(t2-t1)/60:2f}m')
    print(f'Tempo para criar repartition sdft {t4-t2}s, {(t4-t2)/60:2f}m')
    
    del values
    gc.collect()

    # merge sdft with df_toPCA
    ar_pos=0
    t1=time.time()
    if i==0:   
        window_spec = Window.orderBy('coords_0','coords_1')    
        # Add a sequential index column
        sdft = sdft.withColumn("orig_index", row_number().over(window_spec))
        df_toPCA = sdft #.copy()
        # df_toPCA.persist(StorageLevel.MEMORY_AND_DISK)
        df_toPCA.persist(StorageLevel.DISK_ONLY)
        i+=1
        t2=time.time()
    else:
        if ar_pos:
            sdft = sdft.drop('coords', axis=1)
        df_toPCA = df_toPCA.join(sdft, on=['coords_0', 'coords_1'], how='inner')
        i+=1
    t2=time.time()
    
    info = f'tempo para adicionar sdft da matriz de bandas da imagen {t} : {t2-t1:.2f}'
    print (info) 
    logger.info(info)
    logger.info(f'tiffs of {t} inseridos no df_toPCA')
    logger.info(f'Tempo parcial do total do inseridos até {t} {(t2-ti)/60:.2f}')
    print(f'****** Tempo parcial do total do inseridos até {t} {(t2-ti)/60:.2f}')

    print (df_toPCA.columns)
    del sdft
    gc.collect()

#saving Pyspark Dataframe to parquet
logger.info(f'Inicio do save em parquet')
t1=time.time()
temp_path= '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA' + '/data/tmp'
sdfPath = temp_path + "/spark_sdf_toPCA2"
df_toPCA.write.parquet(sdfPath,mode = 'overwrite')
t2=time.time()
logger.info(f'Tempo para salvar em parquet {t2-t1:.2f}s {(t2-t1)/60:.2f}m')

tf=time.time()
logger.info(f'Tempo total do programa {tf-ti:.2f}s {(tf-ti)/60:.2f}m')
spark.stop()

