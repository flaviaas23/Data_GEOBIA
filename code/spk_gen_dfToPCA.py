# Programa para gerar o df to pca com spark e salvar em parquet
# 20240907: fiz download do exacta e fiz algumas alteracoes

import os
import gc
import time
import datetime
import pandas as pd
import logging

#for terminal
#from code.functions.functions_pca import list_files_to_read, get_bandsDates, gen_dfToPCA_filter

# Set the environment variable for spark
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
base_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/'

from pyspark.sql import SparkSession
# Step 1: Start a Spark session
# senao inicializar uma sessao é inicializada qdo chama o sparky
# for exacta:    .config("spark.driver.memory", "90g") \
#testes:
    # .config("spark.memory.fraction", "0.7") \
    # .config("spark.memory.storageFraction", "0.3") \
# spark = SparkSession.builder \
#     .master('local[*]') \
#     .appName("PySpark to PCA") \
#     .config("spark.local.dir", base_dir+"data/tmp_spark") \
#     .config("spark.executorEnv.PYARROW_IGNORE_TIMEZONE", "1") \
#     .config("spark.driverEnv.PYARROW_IGNORE_TIMEZONE", "1") \
#     .config("spark.driver.memory", "90g") \
#     .config("spark.executor.memory", "6g") \
#     .config("spark.shuffle.spill.compress", "true") \
#     .config("spark.driver.maxResultSize", "60g")\
#     .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
#     .getOrCreate()

#testar estas 2 opcoes de conf devido ao erro de timeout heartbeatInterval (default 10000)
# .set("spark.executor.heartbeatInterval", "200000") \ 
# .set("spark.network.timeout", "300000")
spark = SparkSession.builder \
    .master('local[*]') \
    .appName("PySpark to PCA") \
    .config("spark.local.dir", base_dir+"data/tmp_spark") \
    .config("spark.executorEnv.PYARROW_IGNORE_TIMEZONE", "1") \
    .config("spark.driverEnv.PYARROW_IGNORE_TIMEZONE", "1") \
    .config("spark.driver.memory", "150g") \
    .config("spark.executor.memory", "140g") \
    .config("spark.shuffle.spill.compress", "true") \
    .config("spark.driver.maxResultSize", "140g")\
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
    .getOrCreate()


# Set log level to ERROR to suppress INFO and WARN messages
spark.sparkContext.setLogLevel("ERROR") 

# 128 MB per partition 134217728 *5
spark.conf.set("spark.sql.files.maxPartitionBytes", 671088640)  
from functions.functions_pca import cria_logger, list_files_to_read, get_bandsDates, gen_dfToPCA_filter, gen_sdfToPCA_filter
#imports for spark
import pyspark.pandas as ps


ti=time.time()

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
t = datetime.datetime.now().strftime('%Y%m%d_%H%M_')

# nome_prog = 'terminal_spk_gen_dfToPCA'
#b = datetime.datetime.now().strftime('%H:%M:%S')
nome_prog = os.path.basename(__file__)
print (f'{a}: ######## INICIO {nome_prog} ##########')


nome_log = t+nome_prog.split('.')[0]+'.log'
log_dir = base_dir + 'code/logs/'
# logging.basicConfig(level=logging.INFO, 
#                     format='%(asctime)s - %(levelname)s - %(message)s',
#                     handlers=[
#                         logging.FileHandler(log_dir+nome_log, mode='a'),  # Salva no arquivo, mode='a'
#                         logging.StreamHandler()             # Mostra no console
#                     ])
# logging.info(f'0. INICIO {nome_prog}')
logger = cria_logger(log_dir, nome_log)
logger.info(f'######## INICIO {nome_prog} ##########')

spark_local_dir = spark.conf.get("spark.local.dir")
logger.info(f"spark.local.dir: {spark_local_dir}")

####
# List all configurations
conf_list = spark.sparkContext.getConf().getAll()

# Print all configurations
logger.info(f'SPARK configurarion ')
for conf in conf_list:
    logger.info(f"{conf[0]}: {conf[1]}")

# spark.stop()
# quit()
####
#read band image files
all_bands = ['B04', 'B03', 'B02', 'B08', 'EVI', 'NDVI']

read_dir = base_dir + 'data/Cassio/S2-16D_V2_012014_20220728_/'
#read_dir = '/scratch/flavia/S2-16D_V2_012014_20220728_/'
name_img = 'S2-16D_V2_012014'
band_tile_img_files = list_files_to_read(read_dir, name_img)
bands, dates= get_bandsDates(band_tile_img_files, tile=1)
cur_dir = os.getcwd()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
info = f'1. bands: {bands}, dates: {dates}'
# print (f'{a}: {info}')
# logging.info(info)
logger.info(info)

#gera o dataframe depois converte em spark dataframe
gen_df=0
if (gen_df):
    dates= ['20220728']
    t1=time.time()
    df_toPCA, max_bt, min_bt = gen_dfToPCA_filter(dates, band_tile_img_files, dask=2, n_part=63, ar_pos=0, sh_print=1)
    t2=time.time()
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a}: Time to gen df_toPCA: {t2-t1:.2f}')
    
    t1=time.time()
    sdf_toPCA = spark.createDataFrame(df_toPCA)
    t2=time.time()
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a}: Time to gen sdf_toPCA: {t2-t1:.2f}')

    del df_toPCA
    gc.collect()
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (sdf_toPCA.show())

#gerar o df em dataframe spark 
# primeiro gera as imagens em pandas spark e depois converte para spark dataframe
#em implementacao
#dates= ['20220728', '20220829']
t1=time.time()
sdf_toPCA, max_bt, min_bt = gen_sdfToPCA_filter(dates, band_tile_img_files, dask=3, n_part=63, ar_pos=0, sh_print=1, logger=logger)
t2=time.time()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
info =  f'2. Time to gen sdf_toPCA: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m'
# print (f'{a}: {info}')
# logging.info(info)
logger.info(info)
#logger.info(f'sdf_toPCA shape {sdf_toPCA.shape}\n {sdf_toPCA.head()}')

# sdf_toPCA = sdf_toPCA.spark.repartition(1000)
# logger.info(f'{sdf_toPCA.to_spark().rdd.getNumPartitions()}')

#save sdf_toPCA in parquet files
# sdfPath = '/scratch/flavia/' + "pca/sdf_toPCA"
save_sdf_toPCA=1
if save_sdf_toPCA:
    sdfPath = base_dir + "data/tmp/pca/sdf_toPCA_simulado"
    info = f'Saving sdf_toPCA in {sdfPath}'
    # logging.info(info)
    logger.info(info)
    t1=time.time()
    sdf_toPCA.to_parquet(
        sdfPath,
        mode = 'overwrite',
        )
    t2=time.time()
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    info = f'2.1 Time to save sdf_toPCA: {t2-t1:.2f}s {(t2-t1)/60:.2f}m'
    print (f'{a}: {info}')
    # logging.info(info)
    logger.info(info)

#gen a spark sdf 
save_spark_sdf_toPCA = 0
if save_spark_sdf_toPCA:
    t1=time.time()
    # spark_sdf_toPCA = sdf_toPCA.to_spark()
    spark_sdf_toPCA = sdf_toPCA.to_spark(index_col="index")
    t2=time.time()
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    info = f'3. Time to convert sdf_toPCA to spark_sdf_toPCA: {t2-t1:.2f}s {(t2-t1)/60:.2f}m'
    # print (f'{a}: info')
    # logging.info(info)
    logger.info(info)
    del sdf_toPCA
    gc.collect()

    #saving spark sdf
    # sdfPath = '/scratch/flavia/' + "pca/spark_sdf_toPCA"
    sdfPath = base_dir + "data/tmp/pca/spark_sdf_toPCA_simulado"
    info = f'Saving spark_sdf_toPCA in {sdfPath}'
    # logging.info(info)
    logger.info(info)
    # print (f'{a}: {info}')
    t1=time.time()
    spark_sdf_toPCA.write.mode("overwrite").parquet(sdfPath)
    t2=time.time()

    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    info = f'3.1 Time to save spark sdf_toPCA: {t2-t1:.2f}s {(t2-t1)/60:.2f}m'
    # print (f'{a}: {info}')
    # logging.info(info)
    logger.info(info)
#close sparky session

del sdf_toPCA
gc.collect()
spark.stop()

tf=time.time()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
info = f'Tempo de execução do {nome_prog}: {tf-ti:.2f} s {(tf-ti)/60:.2f} m'
# print (f'{a}: {info}')
# logging.info(info)
logger.info(info)
logger.info(f'######## FIM {nome_prog} ##########')
