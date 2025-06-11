# 20250509: Flavia Schneider
# 20250509: Programa para gerar o dataframe para PCA com os pares e impares
#           alternados para cada linha da matriz da imagem, que no df gerado
#           a partir da imagem, cada linha da matriz da imagem é bloco de 10560 
#           linhas no df.
# 20250520: inclusao de opcao gerar spark_df_toPCA da imagem com pixels pares e
#           impares: 
#           -pi 1 para linhas par com pixels impar-par e linha impar com pixels par-impar
#           -pi 2 para linha par com pixels par-impar e linha impar com pixels impar-par
#           -pi 0 para todas as linhas

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
from pyspark.sql import SparkSession, Window
# from pyspark import StorageLevel
from pyspark.sql.functions import row_number,col, floor # monotonically_increasing_id, 

# # Initialize Spark session
# spark = SparkSession.builder.appName("SplitRows").getOrCreate()

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

from functions.functions_pca import cria_logger, update_procTime_file

def process_blocks_spark(df, n_rows=10560, even=2):
    half = n_rows // 2

    # Add row number to track global index
    # w = Window.orderBy(monotonically_increasing_id())  # Or another column if you have a natural order
    w = Window.orderBy("coords_0", "coords_1")  # Or another column if you have a natural order
    df_with_index = df.withColumn("row_num", row_number().over(w) - 1)

    # Get block number and position in block
    df_indexed = df_with_index.withColumn("block_id", floor("row_num") / n_rows)
    df_indexed = df_indexed.withColumn("within_block_pos", col("row_num") % n_rows)

    # Apply conditions:
    # 1. within_block_pos < half and even index (mod 2 == 0)
    # 2. within_block_pos >= half and odd index (mod 2 == 1)
    if even==2: #par
        filtered = df_indexed.filter(
            ((col("within_block_pos") < half) & (col("within_block_pos") % 2 == 0)) |
            ((col("within_block_pos") >= half) & (col("within_block_pos") % 2 == 1))
        )
    elif even==1: #impar
        filtered = df_indexed.filter(
        ((col("within_block_pos") < half) & (col("within_block_pos") % 2 == 1)) |
        ((col("within_block_pos") >= half) & (col("within_block_pos") % 2 == 0))
        )

    return filtered.drop("row_num", "block_id", "within_block_pos")

# Inicializa o parser
parser = argparse.ArgumentParser(description="Program segment image")

# Define os argumentos
parser.add_argument("-bd", '--base_dir', type=str, help="Diretorio base", default='')
parser.add_argument("-sd", '--save_dir', type=str, help="Dir base para salvar saidas de cada etapa", default='data/tmp2/')
# parser.add_argument("-td", '--tif_dir', type=str, help="Dir dos tiffs", default='data/Cassio/S2-16D_V2_012014_20220728_/')
parser.add_argument("-q", '--quadrante', type=int, help="Numero do quadrante da imagem [0-all,1,2,3,4,9]", default=1)
parser.add_argument("-pi", '--par_impar', type=int, help="pixels par_impar por linha [0,1,2]", default=0)
parser.add_argument("-ld", '--log_dir', type=str, help="Dir do log", default='code/logs/')
parser.add_argument("-i", '--name_img', type=str, help="Nome da imagem", default='S2-16D_V2_012014')
parser.add_argument("-sp", '--sh_print', type=int, help="Show prints", default=0)
parser.add_argument("-pd", '--process_time_dir', type=str, help="Dir para df com os tempos de processamento", default='data/tmp2/')
# parser.add_argument("-rf", '--READ_df_features', type=int, help="Read or create df with features", default=0 )
# parser.add_argument("-nc", '--num_components', type=int, help="Read or create df with features", default=4 )
args = parser.parse_args()

ti=time.time()


# base_dir = args.base_dir
save_etapas_dir = base_dir + args.save_dir if base_dir else args.save_dir + args.name_img +'/'
# tif_dir = base_dir + args.tif_dir if base_dir else args.tif_dir
read_quad = args.quadrante 
par_impar = args.par_impar
log_dir = base_dir + args.log_dir if base_dir else args.log_dir
name_img = args.name_img
process_time_dir = base_dir + args.process_time_dir
sh_print = args.sh_print
# READ_df_features = args.READ_df_features
# n_components = args.num_components

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
t = datetime.datetime.now().strftime('%Y%m%d_%H%M_')

nome_prog = os.path.basename(__file__) #"spk_gen_df_ToPCA_nb_term_from_sdft" #os.path.basename(__file__)
print (f'{a}: ######## INICIO {nome_prog} {os.path.basename(__file__)} ##########') if sh_print else None

#cria logger
nome_log = t+nome_prog.split('.')[0]+'.log'
# log_dir = base_dir + 'code/logs/'

logger = cria_logger(log_dir, nome_log)
logger.info(f'######## INICIO {nome_prog} ##########')

# # Example DataFrame (replace this with your actual DataFrame)
# data = [(i,) for i in range(1, 50000)]  # Example data
# columns = ["value"]
# df = spark.createDataFrame(data, columns)

ri=0        #indice do dicionario com os tempos de cada subetapa
proc_dic = {}
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'gen df_toPCA_pares_impares'

if read_quad == 9:
    # Read the DataFrame from Parquet
    sdfPath = save_etapas_dir + "spark_df_toPCA_Full" 
else:
    # Read the DataFrame from Parquet for quadrant
    sdfPath = save_etapas_dir + "spark_df_toPCA_Quad_" + str(read_quad)

spark_sdf_toPCA = spark.read.parquet(sdfPath)

# Sort by coords_0 and coords_1
t1=time.time()
spark_sdf_toPCA = spark_sdf_toPCA.orderBy(spark_sdf_toPCA["coords_0"].asc(), spark_sdf_toPCA["coords_1"].asc())
t2=time.time()
print (f'1. Tempo para sort no spark df by coords_0 and  coords_1: {t2-t1:.2f}s {(t2-t1)/60:.2f}m') if sh_print else None
logger.info(f'1. Tempo para sort no spark df by coords_0 and  coords_1: {t2-t1:.2f}s {(t2-t1)/60:.2f}m')

ri+=1       #indice do dicionario com os tempos de cada subetapa
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'PCA df image days'
proc_dic[ri]['subetapa'] = 'sort no spark df by coords_0 and  coords_1'
proc_dic[ri]['tempo'] = t2-t1
# Add a unique ID column
# df = df.withColumn("row_id", monotonically_increasing_id())

# Define block size and half block size
n_rows = 10560
half_block = n_rows // 2

processed_df = process_blocks_spark(spark_sdf_toPCA, n_rows=n_rows, even=par_impar)

spark_sdf_toPCA.unpersist() 
del spark_sdf_toPCA
gc.collect()
spark._jvm.System.gc()

if read_quad == 9:
    # Save the processed DataFrame to Parquet
    if par_impar == 1:    #linha par com pixels impar-par, linha impar com pixels par-impar
        sdfPath = save_etapas_dir + "spark_df_toPCA_Full_impar_par"        
    elif par_impar == 2:   # linha par com pixels par-impar, linha impar com pixels impar-par
        # gen df with linhas pares(even) pixels par, linhas impares pixels impares
        sdfPath = save_etapas_dir + "spark_df_toPCA_Full_par_impar"
else:
    # Save the processed DataFrame to Parquet for quadrant    
    if par_impar == 1:    
         # gen df with linhas pares(even) pixels impar-par, linhas impares pixels par-impar
        sdfPath = save_etapas_dir + "spark_df_toPCA_Quad_" + str(read_quad) + "_impar_par" 
    if par_impar == 2:
        # gen df with linhas pares(even) pixels par-impar, linhas impares pixels impares-par
        sdfPath = save_etapas_dir + "spark_df_toPCA_Quad_" + str(read_quad) + "_par_impar"

print (f'processed_df: {processed_df.count()}') if sh_print else None
# processed_df.show(5)
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (f'{a}:saving processed_df to {sdfPath}') if sh_print else None
# processed_df = processed_df.persist(StorageLevel.MEMORY_AND_DISK)
# Save the processed DataFrame to Parquet
t2=time.time()
processed_df.write.parquet(sdfPath, mode='overwrite')
t3=time.time()
print (f'2. Tempo para gerar o spark df com os pares e impares: {t3-t2:.2f}s {(t3-t2)/60:.2f}m') if sh_print else None
logger.info(f'2. Tempo para gerar o spark df com os pares e impares: {t3-t2:.2f}s {(t3-t2)/60:.2f}m')
ri+=1       #indice do dicionario com os tempos de cada subetapa
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'gen df_toPCA_pares_impares'
proc_dic[ri]['subetapa'] = 'gerar spark df com os pares e impares'
proc_dic[ri]['tempo'] = t3-t2

#close sparky session
tf=time.time()
print (f'Tempo total do {nome_prog}: {tf-ti:.2f}s, {(tf-ti)/60:.2f}m') if sh_print else None
spark.stop()

time_file = process_time_dir + "process_times.pkl"

update_procTime_file(proc_dic, time_file)
print (f'process_time file: {time_file}') if sh_print else None
logger.info(f'process_time file: {time_file}')
print (f'{a}: ######## FIM {nome_prog} ##########') if sh_print else None
logger.info(f'######## FIM {nome_prog} ##########')