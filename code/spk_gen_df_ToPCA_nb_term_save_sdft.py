# 20240924: programa foi criado a partir do spk_gen_df_ToPCA_nb_term, que foi criado a partir
#           do spk_gen_df_ToPCA baixado do exacta
#           faz a criacao do sdft por data com todas as bandas para ser lido na proxima etapa e 
#           inseridos no df_toPCA
# 20241127: ajustado para incluir a criação do df com os tempos de processamento, 
#           passagem dos parametros de configuração por argumentos 
# 20241209: dividir a imagem por 4 e fazer o sdft por data
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
#.config("spark.shuffle.spill.compress", "false") \
# .config("spark.memory.offHeap.enabled","false") \
# .config("spark.memory.fraction", "0.1") \
# spark = SparkSession.builder \
#     .master('local[*]') \
#     .appName("PySpark to PCA") \
#     .config("spark.local.dir", base_dir+"data/tmp_spark") \
#     .config("spark.executorEnv.PYARROW_IGNORE_TIMEZONE", "1") \
#     .config("spark.driverEnv.PYARROW_IGNORE_TIMEZONE", "1") \
#     .config("spark.driver.memory", "300g") \
#     .config("spark.executor.memory", "300g") \
#     .config("spark.memory.offHeap.enabled","false") \
#     .config("spark.shuffle.compress", "true") \
#     .config("spark.shuffle.spill.compress", "true") \
#spark.sql.shuffle.spill.compress
#     .config("spark.driver.maxResultSize", "300g")\
#     .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
#     .config("spark.executor.instances", "2") \  # Número de executores 20241203 incluido e depois removido
#     .getOrCreate()
spark = SparkSession.builder \
    .master('local[*]') \
    .appName("PySpark to PCA") \
    .config("spark.local.dir", base_dir+"data/tmp_spark") \
    .config("spark.executorEnv.PYARROW_IGNORE_TIMEZONE", "1") \
    .config("spark.driverEnv.PYARROW_IGNORE_TIMEZONE", "1") \
    .config("spark.driver.memory", "200g") \
    .config("spark.executor.memory", "200g") \
    .config("spark.driver.maxResultSize", "200g")\
    .config("spark.sql.shuffle.spill.compress", "true")\
    .getOrCreate()

from pyspark import SparkContext
from pyspark import SparkConf

# Chama o garbage collector da JVM
spark._jvm.System.gc()

# Set log level to ERROR to suppress INFO and WARN messages
spark.sparkContext.setLogLevel("ERROR") 

from functions.functions_pca import cria_logger, list_files_to_read, get_bandsDates,\
                                    gen_dfToPCA_filter, gen_sdfToPCA_filter, \
                                    gen_arr_from_img_band_wCoords, update_procTime_file

# Inicializa o parser
parser = argparse.ArgumentParser(description="Program segment image")

# Define os argumentos
parser.add_argument("-bd", '--base_dir', type=str, help="Diretorio base", default='')
parser.add_argument("-sd", '--save_dir', type=str, help="Dir base para salvar saidas de cada etapa", default='data/tmp2/')
parser.add_argument("-td", '--tif_dir', type=str, help="Dir dos tiffs", default='data/Cassio/S2-16D_V2_012014_20220728_/')
parser.add_argument("-q", '--quadrante', type=int, help="Numero do quadrante da imagem [0-all,1,2,3,4]", default=1)
parser.add_argument("-ld", '--log_dir', type=str, help="Dir do log", default='code/logs/')
parser.add_argument("-i", '--name_img', type=str, help="Nome da imagem", default='S2-16D_V2_012014')
parser.add_argument("-sp", '--sh_print', type=int, help="Show prints", default=1)
parser.add_argument("-pd", '--process_time_dir', type=str, help="Dir para df com os tempos de processamento", default='data/tmp2/')

args = parser.parse_args()
 
# base_dir = args.base_dir
save_etapas_dir = base_dir + args.save_dir if base_dir else args.save_dir + args.name_img +'/'
tif_dir = base_dir + args.tif_dir if base_dir else args.tif_dir
read_quad = args.quadrante 
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
logger.info(f'args: sd={save_etapas_dir} td={tif_dir} ld={log_dir} i={name_img} pd={process_time_dir} sp={args.sh_print}')

spark_local_dir = spark.conf.get("spark.local.dir")
logger.info(f"spark.local.dir: {spark_local_dir}")
# print (spark_local_dir) if sh_print else None

# List all configurations
conf_list = spark.sparkContext.getConf().getAll()

# Print all configurations
logger.info(f'SPARK configurarion')
logger.info(f'defaultParallelism: {spark.sparkContext.defaultParallelism}')
logger.info(f'spark.sql.files.openCostInBytes: {spark.conf.get("spark.sql.files.openCostInBytes")}')
for conf in conf_list:
    logger.info(f"{conf[0]}: {conf[1]}")
    # print (f"{conf[0]}: {conf[1]}") if sh_print else None

ri=0        #indice do dicionario com os tempos de cada subetapa
proc_dic = {}
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'Generate sdft per day'

# Get all bands
# all_bands = ['B04', 'B03', 'B02', 'B08', 'EVI', 'NDVI']

# tif_dir = base_dir + 'data/Cassio/S2-16D_V2_012014_20220728_/'
#read_dir = '/scratch/flavia/S2-16D_V2_012014_20220728_/'
# name_img = 'S2-16D_V2_012014'
band_tile_img_files = list_files_to_read(tif_dir, name_img)
bands, dates = get_bandsDates(band_tile_img_files, tile=0)  #tile=0 pos_date=-1, pos band=-2
cur_dir = os.getcwd()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
info = f'1. bands: {bands},\ndates: {dates}'
print (f'{a}: {info}') if sh_print else None
logger.info(info)

# spark.stop()
# quit()

# from functions.functions_pca import load_image_files3, check_perc_nan, gen_sdf_from_img_band,gen_arr_from_img_band_wCoords
from functions.functions_pca import load_image_files3, check_perc_nan
from functions.functions_segmentation import  get_quadrant_coords
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

# dates = ['20220728', '20220829', '20220830', '20220901', '20220902', '20220903']#, '20220904']
band_img_files = band_tile_img_files
i=0
ini=0

#number of quadrants of image
q_number =  4 #number of quadrants of image
quadrants = [read_quad] if read_quad else range(1,q_number+1)
# quadrants = [2,3,4] #para test

for t in dates[ini:]:#['2023-09-28']:#dates[ini:]:
    # t='2023-09-28'
    print (f'******************** {i}: {t} *******************') if sh_print else None
    tt1 = time.time()
    # band_img_files = band_tile_img_files
    band_img_file_to_load = [x for x in band_img_files if t in x.split('/')[-1]]
    # band_img_file_to_load
    
    image_band_dic = {}
    pos = -2 # position of band in name file
    t1=time.time()
    image_band_dic = load_image_files3(band_img_file_to_load, pos=pos)
    t2=time.time()
    print (f'Tempo de load dos tiffs: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
    bands = list(image_band_dic.keys())
    print (f'bands = {bands}') if sh_print else None #, band_img_file_to_load) 

    imgSize = image_band_dic[bands[0]].shape[0]

    logger.info(f'Generating columns of image for {t}, imgSize = {imgSize}')
    t1=time.time()
    perc_day_nan= check_perc_nan(image_band_dic, logger)
    t2=time.time()
    #20241209: tem q descartar se percentual for alto , definir alto
    logger.info(f'Tempo para verificar percentual de nan nos tiffs: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
    logger.info(f'Percentual de nan nos tiffs: {perc_day_nan}%')
    print (f'Percentual de nan nos tiffs {t}, imgSize= {imgSize}: {perc_day_nan}%') #if sh_print else None
    
    proc_dic[ri]['subetapa'] = 'tiffs nan percentage verification'
    proc_dic[ri]['tempo'] = t2-t1

    for q in quadrants:

        rowi, rowf, coli, colf = get_quadrant_coords(q, imgSize)
        a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        print (f'{a}: Quadrante Q{q} {rowi}:{rowf} {coli}:{colf}') if sh_print else None

        #gerar a imagem tif do quadrante
        t1=time.time()
        image_band_dic_q={}
        for x in bands:
            image_band_dic_q[x] = image_band_dic[x][rowi:rowf, coli:colf]
        t2 = time.time()

        img_bands = []
        img_bands = np.dstack([image_band_dic_q[x]for x in bands]) 
        t3 = time.time()
                                       
        # img_bands = np.dstack([image_band_dic[x] for x in bands])
        # del image_band_dic
        # gc.collect()
        t1=time.time()
        perc_day_nan_q= check_perc_nan(image_band_dic_q, logger)
        t2=time.time()
        logger.info(f'Tempo para verificar percentual de nan no quadrante {q} dos tiffs: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
        logger.info(f'Percentual de nan no quadrante {q} dos tiffs: {perc_day_nan_q}%')
        print (f'Percentual de nan no quadrante {q} tiffs {t}, imgSize= {image_band_dic_q[bands[0]].shape[0]}: {perc_day_nan_q}%') #if sh_print else None
    

        del image_band_dic_q
        gc.collect()

        cols_name = [x+'_'+t for x in bands]
        logger.info(f'Columns names: {cols_name}')
        print (f'Tempo para gerar img_band_dic_q of {t} for quad {q}: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m') if sh_print else None
        print (f'Tempo para gerar img_bands of {t} for quad {q}: {t3-t2:.2f}s, {(t3-t2)/60:.2f}m') if sh_print else None
        logger.info(f'Tempo para gerar img_band_dic_q of {t} for quad {q}: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
        logger.info(f'Tempo para gerar img_bands of {t} for quad {q}: {t3-t2:.2f}s, {(t3-t2)/60:.2f}m')

        ri+=1
        proc_dic[ri]={} if ri not in proc_dic else None
        proc_dic[ri]['etapa'] = 'Generate sdft per day'
        proc_dic[ri]['subetapa'] = 'gen img_band_dic_q of {t} for quad {q}'
        proc_dic[ri]['tempo'] = t2-t1

        ri+=1
        proc_dic[ri]={} if ri not in proc_dic else None
        proc_dic[ri]['etapa'] = 'Generate sdft per day'
        proc_dic[ri]['subetapa'] = 'gen img_bands of {t} for quad {q}'
        proc_dic[ri]['tempo'] = t3-t2

        t1 = time.time()
        values  = gen_arr_from_img_band_wCoords(img_bands, cols_name, ar_pos=0, sp=0, sh_print=0, logger=logger)
        t2 = time.time()
        logger.info(f'Tempo para gerar values with coords and bands: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
        print (f'Tempo para gerar values with coords and bands: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')  if sh_print else None

        print (f'values: {values.shape}, {type(values)}, {type(values[0,0])}')  if sh_print else None

        ri+=1
        proc_dic[ri]={} if ri not in proc_dic else None
        proc_dic[ri]['etapa'] = 'Generate sdft per day'
        proc_dic[ri]['subetapa'] = 'gen values with coords and bands of {t} for quad {q}'
        proc_dic[ri]['tempo'] = t2-t1

        del img_bands
        gc.collect()

        print (cols_name)  if sh_print else None
        a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        print (a) if sh_print else None


        schema = StructType([StructField(x, IntegerType(), True) for x in ['coords_0','coords_1']+cols_name])
        # print (f'\nantes de criar spark df, schema: \n{schema}\n') if sh_print else None

        t1=time.time()
        sdft = spark.createDataFrame(values, schema=schema)
        t2 = time.time()

        sdft.persist(StorageLevel.DISK_ONLY)
        t3 = time.time()
        # sdft.persist(StorageLevel.MEMORY_AND_DISK)

        t2 = time.time()
        logger.info(f'Tempo para criar spark sdft {t} for quad {q} {t2-t1}s, {(t2-t1)/60:2f}m')

        del values
        gc.collect()

        ri+=1
        proc_dic[ri]={} if ri not in proc_dic else None
        proc_dic[ri]['etapa'] = 'Generate sdft per day'
        proc_dic[ri]['subetapa'] = 'Create spark sdft of {t} for quad {q}'
        proc_dic[ri]['tempo'] = t2-t1

        # t2=time.time()
        # sdft=sdft.repartition(40)
        # t4=time.time()

        # logger.info(f'Tempo para repartition spark sdft {t} {t4-t2}s, {(t4-t2)/60:2f}m')
        # sdft.unpersist()
        print(f'Tempo para criar criar sdft de {t} for quad {q} {t2-t1}s, {(t2-t1)/60:2f}m') if sh_print else None
        # print(f'Tempo para criar repartition sdft {t} {t4-t2}s, {(t4-t2)/60:2f}m')   
    
        # merge sdft with df_toPCA
        ar_pos=0
        t1=time.time()
        if i==0:   
            window_spec = Window.orderBy('coords_0','coords_1')    
            # Add a sequential index column
            sdft = sdft.withColumn("orig_index", row_number().over(window_spec))
            # df_toPCA = sdft #.copy()

            # i+=1      # acrescentar só na mudanca de t
            t2=time.time()
            info = f'Tempo sort coords e para adicionar orig_index  {t} : {t2-t1:.2f}'
            print (info)  if sh_print else None
            logger.info(info)

            ri+=1
            proc_dic[ri]={} if ri not in proc_dic else None
            proc_dic[ri]['etapa'] = 'Generate sdft per day'
            proc_dic[ri]['subetapa'] = 'sort coords and added orig_index of {t} for {q}'
            proc_dic[ri]['tempo'] = t2-t1

        # is_empty = sdft.rdd.isEmpty()
        # print ('is_empty=',is_empty,'\n',sdft.printSchema())  if sh_print else None
        # print(f"Linhas no DataFrame: {sdft.count()}")
        
        #saving Pyspark Dataframe to parquet
        logger.info(f'Inicio do save em parquet de {t}')
        t1=time.time()
        # temp_path= '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA' + '/data/tmp'
        # temp_path= base_dir + 'data/tmp'
        # sdfPath = temp_path + "spark_sdft_"+ t
        # sdfPath = save_etapas_dir + "spark_sdft_"+ t #
        sdfPath = save_etapas_dir + "spark_sdft_"+ t +'/Quad_'+str(q) + '/' #20241209 incluido o Quad
        print (f'\n***** dir:  {sdfPath} em parquet') if sh_print else None

        a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        print (a) if sh_print else None 
        # Verificar se o diretório existe e, se não existir, criá-lo, nao precisa fazer isso
        # Caminho do diretório
        # diretorio = Path(sdfPath)
        # diretorio.mkdir(parents=True, exist_ok=True)
        # print(f"Diretório '{sdfPath}' está disponível.")

        # df_toPCA.write.parquet(sdfPath, mode = 'overwrite')
        #20241203 comented below
        # sdft.write.parquet(sdfPath, mode = 'overwrite')
    
        #20241203 : added verification below, nao reconheceu o statusTracker().getExecutorInfos()
        # sc = spark.sparkContext
        # try:
        #     print("Spark Context test.")
        #     sc.statusTracker().getExecutorInfos()
        #     print("Spark Context está ativo.")
        # except Exception as e:
        #     print("Spark Context foi finalizado:", e)

        #20241203 added 2 lines below
        num_partitions = 10  # Tente aumentar o número de partições
        sdft.repartition(num_partitions).write.parquet(sdfPath, mode='overwrite')

        t2=time.time()
        sdft.unpersist() #20241203
        del sdft
        gc.collect()
        spark._jvm.System.gc() #20241204


        logger.info(f'Tempo para salvar {t} em parquet {t2-t1:.2f}s {(t2-t1)/60:.2f}m')
        print (f'Tempo para salvar {t} em parquet {t2-t1:.2f}s {(t2-t1)/60:.2f}m') if sh_print else None

        ri+=1
        proc_dic[ri]={} if ri not in proc_dic else None
        proc_dic[ri]['etapa'] = 'Generate sdft per day'
        proc_dic[ri]['subetapa'] = f'save sdft of {t} and Quad {q} in parquet'
        proc_dic[ri]['tempo'] = t2-t1

        # i+=1
        # spark.stop()
        print (f'fim {q} of {t}')
    
    del image_band_dic
    gc.collect()
        
    i+=1
    tt2 = time.time()
    ri+=1
    proc_dic[ri]={} if ri not in proc_dic else None
    proc_dic[ri]['etapa'] = 'Generate sdft per day'
    proc_dic[ri]['subetapa'] = f'Tempo para gen and save sdfts of {t}'
    proc_dic[ri]['tempo'] = tt2-tt1
    print (f'Tempo para gen and save sdfts of {t}') if sh_print else None

tf = time.time()
logger.info(f'Tempo total do programa {tf-ti:.2f}s {(tf-ti)/60:.2f}m')
print (f'Tempo total do programa {tf-ti:.2f}s {(tf-ti)/60:.2f}m') #if sh_print else None
spark.stop()

ri+=1
proc_dic[ri]={} if ri not in proc_dic else None
proc_dic[ri]['etapa'] = 'Generate sdft per day'
proc_dic[ri]['subetapa'] = f'{nome_prog} time execution total'
proc_dic[ri]['tempo'] = tf-ti

time_file = process_time_dir + "process_times.pkl"

update_procTime_file(proc_dic, time_file)
print (f'process_time file: {time_file}')
print ("FIM")