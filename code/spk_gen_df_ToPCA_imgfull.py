# 20250508: programa para juntar os df_toPCA da 
#           imagem completa

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
'''
https://medium.com/@SingaramPalaniappan/optimizing-spark-performance-a-look-at-spark-sql-shuffle-compress-7970e64e9003
1. spark.sql.shuffle.compress
2. spark.sql.shuffle.spill.compress

spark.sql.shuffle.compress determines whether Spark compresses data when shuffling 
between nodes. When this property is set to true, Spark will use compression to reduce 
the amount of data transferred over the network. This can help speed up shuffles.

spark.sql.shuffle.spill.compress determines whether Spark compresses spilled data. 
When a shuffle partition is too large to fit in memory, Spark will spill it to disk, 
which can be a slow process. By setting this property to true, Spark will compress 
the spilled data to reduce disk I/O and improve performance.

By default, Spark uses lz4 codec for compression.

Both of these properties can have a significant impact on Spark’s performance and 
resource utilization.
'''
# Set the environment variable for spark
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
base_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/'

ti=time.time()

from pyspark.sql import SparkSession
from pyspark import StorageLevel

spark = SparkSession.builder \
    .master('local[*]') \
    .appName("PySpark to PCA") \
    .config("spark.local.dir", base_dir+"data/tmp_spark") \
    .config("spark.executorEnv.PYARROW_IGNORE_TIMEZONE", "1") \
    .config("spark.driverEnv.PYARROW_IGNORE_TIMEZONE", "1") \
    .config("spark.driver.memory", "300g") \
    .config("spark.executor.memory", "300g") \
    .config("spark.driver.maxResultSize", "300g")\
    .config("spark.sql.shuffle.spill.compress", "true")\
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
    .getOrCreate()

# from pyspark import SparkContext
# from pyspark import SparkConf

# Set log level to ERROR to suppress INFO and WARN messages
spark.sparkContext.setLogLevel("ERROR") 

from functions.functions_pca import cria_logger, update_procTime_file

# Chama o garbage collector da JVM
spark._jvm.System.gc()

# Inicializa o parser
parser = argparse.ArgumentParser(description="Program Gen df_toPCA image full")

# Define os argumentos
parser.add_argument("-bd", '--base_dir', type=str, help="Diretorio base", default='')
parser.add_argument("-sd", '--save_dir', type=str, help="Dir base para salvar saidas de cada etapa", default='data/tmp2/')
# parser.add_argument("-td", '--tif_dir', type=str, help="Dir dos tiffs", default='data/Cassio/S2-16D_V2_012014_20220728_/')
# parser.add_argument("-q", '--quadrante', type=int, help="Numero do quadrante da imagem [0-all,1,2,3,4,9]", default=1)
parser.add_argument("-ld", '--log_dir', type=str, help="Dir do log", default='code/logs/')
parser.add_argument("-i", '--name_img', type=str, help="Nome da imagem", default='S2-16D_V2_012014')
parser.add_argument("-sp", '--sh_print', type=int, help="Show prints", default=1)
parser.add_argument("-pd", '--process_time_dir', type=str, help="Dir para df com os tempos de processamento", default='data/tmp2/')

args = parser.parse_args()

save_etapas_dir = base_dir + args.save_dir if base_dir else args.save_dir + args.name_img +'/'
# tif_dir = base_dir + args.tif_dir if base_dir else args.tif_dir
# read_quad = args.quadrante 
log_dir = base_dir + args.log_dir if base_dir else args.log_dir
name_img = args.name_img
process_time_dir = base_dir + args.process_time_dir
sh_print = args.sh_print

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
t = datetime.datetime.now().strftime('%Y%m%d_%H%M_')

nome_prog = os.path.basename(__file__) #"spk_gen_df_ToPCA_nb_term_save_sdft" #os.path.basename(__file__)
print (f'{a}: ######## INICIO {nome_prog} ##########') #if sh_print else None
print (f'\n{a}: base_dir = {base_dir}\n args: sd={save_etapas_dir} \ntd={tif_dir} read_quad={read_quad} ld={log_dir} i={name_img} \npd={process_time_dir} sp={type(args.sh_print)}') if sh_print else None

#cria logger
nome_log = t + nome_prog.split('.')[0]+'.log'
# log_dir = base_dir + 'code/logs/'

logger = cria_logger(log_dir, nome_log)
logger.info(f'######## INICIO {nome_prog} ##########')
logger.info(f'args: sd={save_etapas_dir} ld={log_dir} i={name_img} pd={process_time_dir} sp={args.sh_print}')

spark_local_dir = spark.conf.get("spark.local.dir")
logger.info(f"spark.local.dir: {spark_local_dir}")

# List all configurations
conf_list = spark.sparkContext.getConf().getAll()

# Print all configurations
logger.info(f'SPARK configurarion')
logger.info(f'defaultParallelism: {spark.sparkContext.defaultParallelism}')
logger.info(f'spark.sql.files.openCostInBytes: {spark.conf.get("spark.sql.files.openCostInBytes")}')
for conf in conf_list:
    logger.info(f"{conf[0]}: {conf[1]}")

ri=0        #indice do dicionario com os tempos de cada subetapa
proc_dic = {}
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'Generate df_toPCA image full'    

cur_dir = os.getcwd()

from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

sdfPath1 = save_etapas_dir + 'spark_df_toPCA_Full_1'
sdfPath2 = save_etapas_dir + 'spark_df_toPCA_Full_2'

# read 1st df_toPCA
tr1 = time.time()
df_toPCA = spark.read.parquet(sdfPath1) #sdft #.copy()
tr2 = time.time()

proc_dic[ri]['subetapa'] = f'read first df_toPCA of {sdfPath1}'
proc_dic[ri]['tempo'] = tr2-tr1

dft_columns1 = df_toPCA.columns

logger.info(f'Tempo para ler df_toPCA1 parquet de {sdfPath1}: {tr2-tr1:.2f}s, {(tr2-tr1)/60:.2f}m')
logger.info(f'df_toPCA1 columns: {dft_columns1}')
print (f'df_toPCA1 columns {dft_columns1}')

# Read 2nd df_toPCA
tr1 = time.time()
dft = spark.read.parquet(sdfPath2)
tr2 = time.time()

proc_dic[ri]['subetapa'] = f'read 2nd df_toPCA of {sdfPath2}'
proc_dic[ri]['tempo'] = tr2-tr1

dft_columns2 = dft.columns
dft = dft.select([c for c in dft.columns if c != 'orig_index'])

logger.info(f'Tempo para ler df_toPCA2 parquet de {sdfPath2}: {tr2-tr1:.2f}s, {(tr2-tr1)/60:.2f}m')
logger.info(f'df_toPCA2 columns: {dft_columns2}')
print (f'df_toPCA2 columns {dft_columns2}')

# Join dfs to pca
t1 = time.time()
df_toPCA = df_toPCA.join(dft, on=['coords_0', 'coords_1'], how='inner')
t2=time.time()

del dft
gc.collect()
spark._jvm.System.gc() #20241210

dft_columns = df_toPCA.columns

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'Gen df_toPCA image full'
proc_dic[ri]['subetapa'] = f'join df_toPCA1 and df_toPCA2'
proc_dic[ri]['tempo'] = tr2-tr1

logger.info(f'Tempo do join df_toPCA1 and df_toPCA2 {(t2-t1)/60:.2f}')
logger.info(f'df_toPCA columns: {dft_columns}')
print(f'****** Tempo do join df_toPCA1 and df_toPCA2 {(t2-t1)/60:.2f}') #if sh_print else None
print (f'df_toPCA2 columns {dft_columns}')

#saving Pyspark Dataframe to parquet
logger.info(f'Inicio do save em parquet df_toPCA Full')

sdfPath = save_etapas_dir + "spark_df_toPCA_Full"
t1=time.time()
print (f'salvando em {sdfPath}')
df_toPCA.write.parquet(sdfPath,mode = 'overwrite')
t2=time.time()

tf = t2

logger.info(f'Tempo para salvar em parquet {t2-t1:.2f}s {(t2-t1)/60:.2f}m')

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'Generate df_toPCA image full'  
proc_dic[ri]['subetapa'] = f'save df_toPCA Full'
proc_dic[ri]['tempo'] = t2-t1

spark.stop()
logger.info(f'Tempo total do programa {tf-ti:.2f}s {(tf-ti)/60:.2f}m')
print (f'Tempo total do programa {tf-ti:.2f}s {(t2-ti)/60:.2f}m') #if sh_print else None

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'Generate df_toPCA image full'
proc_dic[ri]['subetapa'] = f'{nome_prog} time execution total'
proc_dic[ri]['tempo'] = tf-ti

time_file = process_time_dir + "process_times.pkl"

update_procTime_file(proc_dic, time_file)
print (f'process_time file: {time_file}')
print ("FIM")




