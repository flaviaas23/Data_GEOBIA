# Programa para Fazer o PCA no df to PCA com spark
#20240927: baixado do exacta e alterado para fazer funcionar no laptop
# 20241210: alterado para ser executado com 1/4 do tile
#           Falta terminar de incluir so tempos de processamento no df
#           colocar os nomes parciais dos diretorios como variaveis no 
#           inicio do programa
# 20250120: Caso o programa dê erro de execucao apos salvar o df with features
#           executá-lo novamente com a opcao -rf 1
#           Talvez seja melhor ajustá-lo para ser executado 2 vezes, a primeira 
#           para salvar o df with featres e a segunda pra ler o df with features 
#           e executar o pca
# 20250508: inclusao de opcao para executar o pca no df da  imagem completa: -q 9
# 20250510: inclusao de opcao para executar o pca no df da imagem com pixels pares e
#           impares: 
#           -pi 1 para linhas par com pixels impar-par e linha impar com pixels par-impar
#           -pi 2 para linha par com pixels par-impar e linha impar com pixels impar-par
#           -pi 0 para todas as linhas
#           fazer as alteracoes do chat pata ver se melhora a performance.
# 20250523: colocando para salvar o df com features scaled 
#           incluindo a opcao -rs para ler o df com features scaled e fazer o pca
#           -rf 0 -rs 1 para ler o df com features scaled e nao ler o df com features
# 20250524: alterado para usar o PCA com a matriz de componentes        
#           -cm 1 para usar a matriz de componentes
import os
import gc
import time
import datetime
import pandas as pd
import logging
import numpy as np
import argparse
import pickle

import findspark
findspark.init()
#for terminal
#from code.functions.functions_pca import list_files_to_read, get_bandsDates, gen_dfToPCA_filter

# Set the environment variable for spark
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
base_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/'
ti=time.time()

from functions_pca import list_files_to_read, get_bandsDates, \
                          gen_dfToPCA_filter, gen_sdfToPCA_filter,\
                          save_to_pickle \
                          
#imports for spark
# import pyspark.pandas as ps
from pyspark.sql import SparkSession
from pyspark import StorageLevel

from pyspark.sql import functions as F
from pyspark.sql.functions import when, col, array, udf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StandardScaler

from pyspark.ml.feature import PCA
from pyspark.ml.feature import PCAModel


# Step 1: Start a Spark session
# senao inicializar uma sessao é inicializada qdo chama o sparky
spark = SparkSession.builder \
    .master('local[*]') \
    .appName("PySpark to PCA") \
    .config("spark.local.dir", base_dir+"data/tmp_spark") \
    .config("spark.executorEnv.PYARROW_IGNORE_TIMEZONE", "1") \
    .config("spark.driverEnv.PYARROW_IGNORE_TIMEZONE", "1") \
    .config("spark.driver.memory", "200g") \
    .config("spark.executor.memory", "200g") \
    .config("spark.driver.maxResultSize", "200g")\
    .config("spark.sql.shuffle.spill.compress", "true") \
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
    .config("spark.driver.extraJavaOptions", "-Djava.io.tmpdir=" + base_dir + "data/tmp_java")\
    .config("spark.executor.extraJavaOptions", "-Djava.io.tmpdir=" + base_dir + "data/tmp_java")\
    .getOrCreate()

# Set log level to ERROR to suppress INFO and WARN messages
spark.sparkContext.setLogLevel("ERROR") #comented on 20250508
# spark.sparkContext.setLogLevel("DEBUG")
# spark.conf.set("spark.sql.shuffle.partitions", "200")
print (f"Spark version: {spark.version}, spark.io.tmpdir:")
print(spark.sparkContext._jvm.java.lang.System.getProperty("java.io.tmpdir"))

from functions.functions_pca import cria_logger, update_procTime_file
# to multiply feature vector by PCA components matrix
from pyspark.ml.linalg import DenseMatrix, DenseVector, Vectors
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType
# Optional: convert to Vector
from pyspark.ml.linalg import Vectors, VectorUDT
# Step 3: define UDF to multiply feature vector by PCA components matrix

def to_vector(lst):
    return Vectors.dense(lst)

to_vector_udf = udf(to_vector, VectorUDT())

def log_jvm_memory(spark, stage_desc=""):
    """Logs JVM memory usage from Spark's perspective."""
    runtime = spark._jvm.java.lang.Runtime.getRuntime()
    max_mem = runtime.maxMemory() / (1024**2)
    total_mem = runtime.totalMemory() / (1024**2)
    free_mem = runtime.freeMemory() / (1024**2)
    used_mem = total_mem - free_mem

    print(f"[MEMORY] JVM {stage_desc}:")
    print(f"         Max Memory:   {max_mem:.2f} MB")
    print(f"         Total Memory: {total_mem:.2f} MB")
    print(f"         Free Memory:  {free_mem:.2f} MB")
    print(f"         Used Memory:  {used_mem:.2f} MB\n")

# 20241210 bloco de parse incluido
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
parser.add_argument("-rf", '--READ_df_features', type=int, help="Read or create df with features", default=0 )
parser.add_argument("-rs", '--READ_df_features_scaled', type=int, help="Read or create df with features scaled", default=0 )
parser.add_argument("-nc", '--num_components', type=int, help="Read or create df with features", default=4 )
parser.add_argument("-cm", '--components_matrix', type=int, help="Use components matrix to calc pca", default=0 )
args = parser.parse_args()

if args.base_dir: 
    base_dir = args.base_dir
    print (f'base_dir: {base_dir}') if args.sh_print else None
save_etapas_dir = base_dir + args.save_dir if base_dir else args.save_dir + args.name_img +'/'
# tif_dir = base_dir + args.tif_dir if base_dir else args.tif_dir
read_quad = args.quadrante 
par_impar = args.par_impar
log_dir = base_dir + args.log_dir if base_dir else args.log_dir
name_img = args.name_img
process_time_dir = base_dir + args.process_time_dir
sh_print = args.sh_print
READ_df_features = args.READ_df_features
READ_SCALE = args.READ_df_features_scaled
n_components = args.num_components
use_compMat = args.components_matrix

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
t = datetime.datetime.now().strftime('%Y%m%d_%H%M_')

nome_prog = os.path.basename(__file__) #"spk_gen_df_ToPCA_nb_term_from_sdft" #os.path.basename(__file__)
print (f'{a}: ######## INICIO {nome_prog} {os.path.basename(__file__)} ##########') if sh_print else None

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
proc_dic[ri]['etapa'] = 'PCA df image days'

# #read spark df in exacta
# base_dir = '/scratch/flavia/'
# sdfPath =  base_dir + "pca/spark_sdf_toPCA"
# READ_df_features=0 #20241210 : commented
if not READ_df_features:  # gen df with features
    
    # temp_path= base_dir + 'data/tmp'  #20241210 commented
    # sdfPath = temp_path + "/spark_sdf_toPCA2_6d" #20241210 commented
    # sdfPath = save_etapas_dir + "spark_df_toPCA_Quad_" + str(read_quad) #20250121:comentei para usar o df sem os nsns do 2023-01-01
    if read_quad==9:
        sdfPath = save_etapas_dir + "spark_df_toPCA_Full"
    else:
        if par_impar==0:
            sdfPath = save_etapas_dir + "spark_df_toPCA_Quad_" + str(read_quad) 
        elif par_impar==1:
            sdfPath = save_etapas_dir + "spark_df_toPCA_Quad_" + str(read_quad) + "_impar_par"
        elif par_impar==2:
            sdfPath = save_etapas_dir + "spark_df_toPCA_Quad_" + str(read_quad) + "_par_impar"

    logger.info(f'Creating df_features from {sdfPath}' )
    t1=time.time()
    spark_sdf_toPCA = spark.read.parquet(sdfPath)
    t2=time.time()
    logger.info(f'1. Tempo para ler spark df: {t2-t1:.2f}s {(t2-t1)/60:.2f}m')
    print (f'1. Tempo para ler spark df: {t2-t1:.2f}s {(t2-t1)/60:.2f}m') if sh_print else None
    log_jvm_memory(spark, "after loading data")

    proc_dic[ri]['subetapa'] = f'Creating df_features from {sdfPath}'
    proc_dic[ri]['tempo'] = t2-t1

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

    #replace nans
    t1=time.time()
    # Replace -9999 with NaN in all columns
    spark_sdf_toPCA = spark_sdf_toPCA.select([when(col(c) == -9999, F.lit(None)).otherwise(col(c)).alias(c) for c in spark_sdf_toPCA.columns])
    spark_sdf_toPCA = spark_sdf_toPCA.select([when(col(c) == -32768, F.lit(None)).otherwise(col(c)).alias(c) for c in spark_sdf_toPCA.columns])
    t2=time.time()
    print (f'1. Tempo para replace -9999/-32768 no spark df: {t2-t1:.2f}s {(t2-t1)/60:.2f}m') if sh_print else None
    logger.info(f'1. Tempo para replace -9999/-32768 no spark df: {t2-t1:.2f}s {(t2-t1)/60:.2f}m')

    ri+=1       #indice do dicionario com os tempos de cada subetapa
    proc_dic[ri]={} if ri not in proc_dic else None
    proc_dic[ri]['etapa'] = 'PCA df image days'
    proc_dic[ri]['subetapa'] = 'replace -9999/-32768 no spark df'
    proc_dic[ri]['tempo'] = t2-t1

    # #drop the nans
    t1=time.time()
    spark_sdf_toPCA = spark_sdf_toPCA.na.drop()
    #spark_sdf_toPCA.show()
    t2=time.time()
    print (f'2. Tempo para drop na spark df: {t2-t1}') if sh_print else None
    logger.info(f'2. Tempo para drop na spark df: {t2-t1}')

    ri+=1       #indice do dicionario com os tempos de cada subetapa
    proc_dic[ri]={} if ri not in proc_dic else None
    proc_dic[ri]['etapa'] = 'PCA df image days'
    proc_dic[ri]['subetapa'] = 'drop no spark df'
    proc_dic[ri]['tempo'] = t2-t1

    spark_sdf_toPCA = spark_sdf_toPCA.persist(StorageLevel.DISK_ONLY)

    pcaColumns = spark_sdf_toPCA.columns[2:]
    print (f'pcaColumns: {pcaColumns}') if sh_print else None
    logger.info(f'pcaColumns: {pcaColumns}')

    # assembler nao suporta nan's
    
    t1=time.time()
    assembler = VectorAssembler(inputCols=pcaColumns, outputCol="features")
    # df_with_features = assembler.transform(spark_sdf_toPCA)
    df_with_features = assembler.transform(spark_sdf_toPCA).select("coords_0", "coords_1", "features")
    t2=time.time()
    print (f'3. Tempo para criar col de array to pca: {t2-t1:.2f}s {(t2-t1)/60:.2f}m') if sh_print else None
    # print (f'3.1 df_with features: {df_with_features.show(n=10)}')
    # print (f'3.2 dtypes: {df_with_features.dtypes}')

    logger.info(f'3. Tempo para criar col de array to pca: {t2-t1:.2f}s {(t2-t1)/60:.2f}m')
    log_jvm_memory(spark, "after VectorAssembler")
    # logger.info (f'3.1 df_with features: {df_with_features.show(n=10)}')
    # logger.info (f'3.2 dtypes: {df_with_features.dtypes}')
    

    ri+=1       #indice do dicionario com os tempos de cada subetapa
    proc_dic[ri]={} if ri not in proc_dic else None
    proc_dic[ri]['etapa'] = 'PCA df image days'
    proc_dic[ri]['subetapa'] = 'df_with_features criar col de array to pca'
    proc_dic[ri]['tempo'] = t2-t1

    # 20250508: droping column above with .select("coords_0", "coords_1", "features") 
    columnsToDrop = pcaColumns #['coords_0', 'coords_1']
    print (f'df_with_features columns antes drop columns: {df_with_features.columns}') if sh_print else None
    logger.info(f'df_with_features columns antes drop columns: {df_with_features.columns}')
    # mudei a forma de selecao das colunas, nao preciso fazer o drop
    # df_with_features = df_with_features.drop(*columnsToDrop)
    # print (f'df_with_features columns after drop columns: {df_with_features.columns}') if sh_print else None
    # logger.info(f'df_with_features columns after drop columns: {df_with_features.columns}')

    df_with_features.printSchema() #20250508
    #saving spark Dataframe to parquet
    # t1=time.time()        #20241210 commented
    # sdfPath = base_dir + "data/tmp/df_with_features"  #20241210 commented
    if read_quad==9:        # image full
        sdfPath = save_etapas_dir + "df_with_features_Full/" 
    else:
        if par_impar==0:
            sdfPath = save_etapas_dir + "df_with_features_Quad_" + str(read_quad) +"/"
        elif par_impar==1:
            sdfPath = save_etapas_dir + "df_with_features_Quad_" + str(read_quad) + "_impar_par/"
        elif par_impar==2:
            sdfPath = save_etapas_dir + "df_with_features_Quad_" + str(read_quad) + "_par_impar/"
    
    log_jvm_memory(spark, "before persist df_with features")

    t1=time.time()
    #20250520: inclui o persist no df_with features
    df_with_features = df_with_features.persist(StorageLevel.DISK_ONLY)
    # 202250520: mudei este del para cá, vamos ver se vai dar certo
    t2=time.time()

    log_jvm_memory(spark, "after persist df_with features")

    spark_sdf_toPCA.unpersist()
    del spark_sdf_toPCA
    gc.collect()
    spark._jvm.System.gc() #20241210
    log_jvm_memory(spark, "after unpersist and del spark_sdf_toPCA")
    t3=time.time()

    print (f'8. Tempo para fazer o persist do df_with_features {t2-t1:.2f}s {(t2-t1)/60:.2f}m') if sh_print else None
    print (f'8. Tempo para fazer o unpersist do spark_sdf_toPCA {t3-t2:.2f}s {(t3-t2)/60:.2f}m') if sh_print else None
    num_partitions = 10  
    # 20250508: do programa spk_gen_df_toPCA_nb_term_save_sdft.py copiei para 
    #           comparar com o q está aqui
    # sdft.repartition(num_partitions).write.parquet(sdfPath, mode='overwrite')
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a}: saving df_with_features in: {sdfPath}') if sh_print else None
    
    t1=time.time()
    # df_with_features.repartition(num_partitions).write.mode("overwrite").parquet(sdfPath)
    df_with_features.coalesce(10).write.mode("overwrite").parquet(sdfPath)
    t2=time.time()
    print (f'9. Tempo para salvar df_with_features for quad {read_quad}: {(t2-t1)/60:.2f}') if sh_print else None
    logger.info(f'9. Tempo para salvar df_with_features for quad {read_quad}: {t2-t1:.2f}')
    print (f'df_with_features path: {sdfPath}') if sh_print else None
    
    logger.info(f'df_with_features path: {sdfPath}')
    log_jvm_memory(spark, "after saving parquet")

    ri+=1       #indice do dicionario com os tempos de cada subetapa
    proc_dic[ri]={} if ri not in proc_dic else None
    proc_dic[ri]['etapa'] = 'PCA df image days'
    proc_dic[ri]['subetapa'] = 'save df_with_features'
    proc_dic[ri]['tempo'] = t2-t1    
    
    #20250520 movi o bloco do del para antes do save
    # spark_sdf_toPCA.unpersist()
    # del spark_sdf_toPCA
    # gc.collect()
    # spark._jvm.System.gc() #20241210
    # log_jvm_memory(spark, "after del spark_sdf_toPCA")

    df_with_features.unpersist(blocking=True)
    gc.collect()
    spark._jvm.System.gc() #20241204

    # sair d programa e mandar ler o df features para fazer o pca

if READ_df_features and not READ_SCALE: #read df with features, 1st to gen df with features scaled
    # sdfPath = base_dir + "data/tmp/df_with_features"
    if read_quad==9:        # image full
        sdfPath = save_etapas_dir + "df_with_features_Full/" 
    else:
        
        if par_impar==0:
            sdfPath = save_etapas_dir + "df_with_features_Quad_" + str(read_quad) + "/"
        elif par_impar==1:
            sdfPath = save_etapas_dir + "df_with_features_Quad_" + str(read_quad) + "_impar_par/"
        elif par_impar==2:
            sdfPath = save_etapas_dir + "df_with_features_Quad_" + str(read_quad) + "_par_impar/"
 
    print (f'lendo df_with_features em: {sdfPath}') if sh_print else None
    t1 = time.time()
    df_with_features = spark.read.parquet(sdfPath)
    t2 = time.time()
    logger.info(f'Tempo para ler df_with_features  {t2-t1:.2f}')
    print (f'Tempo para ler df_with_features  {t2-t1:.2f}') if sh_print else None
    logger.info(f'df_with_features columns: {df_with_features.columns}')
    print (f'df_with_features columns: {df_with_features.columns}') if sh_print else None

    # Sort by coords_0 and coords_1
    t1=time.time()
    df_with_features = df_with_features.orderBy(df_with_features["coords_0"].asc(), df_with_features["coords_1"].asc())
    t2=time.time()
    print (f'1. Tempo para sort no df_with_features by coords_0 and coords_1: {t2-t1:.2f}s {(t2-t1)/60:.2f}m') if sh_print else None
    logger.info(f'1. Tempo para sort no df_with_features by coords_0 and coords_1: {t2-t1:.2f}s {(t2-t1)/60:.2f}m')

# columnsToDrop = ['coords_0', 'coords_1']
# df_with_features = df_with_features.drop(*columnsToDrop)
# print (f'df_with_features columns after drop coords columns: {df_with_features.columns}')
# logger.info(f'df_with_features columns after drop coords columns: {df_with_features.columns}')
# gc.collect()

SCALER = 1
# Apply PCA without scaler
# 20241210 ainda nao alterei o if not scaler para fazer por quadrante
if not SCALER:
    t1=time.time()
    n_components = 6
    pca = PCA(k=n_components, inputCol="features", outputCol="pca_features")
    pca_model = pca.fit(df_with_features)
    df_with_pca = pca_model.transform(df_with_features)
    t2=time.time()
    #df_with_pca.show()
    print (f'4. Tempo para run pca: {t2-t1:.2f} s {(t2-t1)/60:.2f} m') if sh_print else None
    logger.info(f'4. Tempo para run pca: {t2-t1:.2f} s {(t2-t1)/60:.2f} m')
    # Show the PCA results
    # print (f'4.1 {df_with_pca.select("pca_features").show(n=10, truncate=False)}')

    print (f'4.2 explainedVariance: {pca_model.explainedVariance}') if sh_print else None
    logger.info(f'4.2 explainedVariance: {pca_model.explainedVariance}')

    #save pca, model pca and df_with_PCA
    t1=time.time()
    pcaPath = base_dir + 'data/tmp/spark_pca'
    pca.write().overwrite().save(pcaPath)

    modelPath = base_dir + "data/tmp/spark_pca-model"
    pca_model.write().overwrite().save(modelPath)
    t2=time.time()
    print (f'5.1 Tempo para salvar pca e model pca: {t2-t1:.2f} s {(t2-t1)/60:.2f} m') if sh_print else None

    #saving spark sdf with pca to parquet
    sdfPath = base_dir + "data/tmp/spark_sdf_with_PCA"
    t1=time.time()
    df_with_pca.write.mode("overwrite").parquet(sdfPath)
    t2=time.time()
    print (f'5.2 Time to save spark_sdf_with_PCA: {t2-t1:.2f} s {(t2-t1)/60:.2f} m') if sh_print else None
    del df_with_pca
    gc.collect()
 
# Scale df_toPCA
if SCALER and READ_df_features:
    #20250510: alterar aqui para ler o df com os features com par e impar tb
    # n_components = 4 #20241210 commented, passed via args
    if READ_SCALE:
        if read_quad==9:        # image full
            sdfPath = save_etapas_dir + "df_with_features_scaled_Full/" 
        else:
            if par_impar==0:
                sdfPath = save_etapas_dir + "df_with_features_scaled_Quad_" + str(read_quad) +"/"
            elif par_impar==1:
                sdfPath = save_etapas_dir + "df_with_features_scaled_Quad_" + str(read_quad) + "_impar_par/"
            elif par_impar==2:
                sdfPath = save_etapas_dir + "df_with_features_scaled_Quad_" + str(read_quad) + "_par_impar/"
        # df_with_features = spark.read.parquet(sdfPath)
        # df_with_features = df_with_features.select("coords_0", "coords_1", "features")
        # df_with_features = df_with_features.persist(StorageLevel.DISK_ONLY)
        # df_with_features.show(n=10)
        # print (f'df_with_features columns: {df_with_features.columns}') if sh_print else None

        df_with_features = spark.read.parquet(sdfPath)
        print (f'Read df_with_features_scaled columns: {df_with_features.columns}') if sh_print else None
        logger.info(f'df_with_features_scaled columns: {df_with_features.columns}')
    
    else: # 20250510: se not read_scaler fazer o df com features scaled

        t1 = time.time()
        standardScaler = StandardScaler()
        standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")
        t2 = time.time()
        print (f'6.1 standardScaler: {standardScaler}') if sh_print else None
        logger.info(f'6.1 standardScaler: {standardScaler}')
        scaler_model = standardScaler.fit(df_with_features)
        df_with_features = scaler_model.transform(df_with_features).select("coords_0", "coords_1", "features_scaled")
        t3 = time.time()
        print (f'6.2 Time to do standardScaler: {t2-t1:.2f} s {(t2-t1)/60:.2f} m') if sh_print else None
        print (f'6.3 Time to do Scaler: {t3-t2:.2f}s {(t3-t2)/60:.2f} m') if sh_print else None
        print (f'6.4 df_with_features_scaled columns: {df_with_features.columns}') if sh_print else None
        # print (f'6.4 df_with_features_scaled : {df_with_features.show(n=10)}')

        logger.info(f'6.2 Time to do standardScaler: {t2-t1:.2f} s {(t2-t1)/60:.2f} m')
        logger.info(f'6.3 Time to do Scaler: {t3-t2:.2f} s {(t3-t2)/60:.2f} m')
        # logger.info(f'6.4 df_with_features_scaled : {df_with_features.show(n=10)}')

        # drop the features column
        # usando o select para fazer o drop
        # df_with_features = df_with_features.drop("features")

        #save df_with_features scaled

        if read_quad==9:        # image full
            sdfPath = save_etapas_dir + "df_with_features_scaled_Full/" 
        else:
            if par_impar==0:
                sdfPath = save_etapas_dir + "df_with_features_scaled_Quad_" + str(read_quad) +"/"
            elif par_impar==1:
                sdfPath = save_etapas_dir + "df_with_features_scaled_Quad_" + str(read_quad) + "_impar_par/"
            elif par_impar==2:
                sdfPath = save_etapas_dir + "df_with_features_scaled_Quad_" + str(read_quad) + "_par_impar/"
    
        print (f'saving df_with_features_scaled in: {sdfPath}') if sh_print else None
        t1=time.time()
        df_with_features.write.mode("overwrite").parquet(sdfPath)
        # 20250523: testar o save com o coalesce
        # df_with_features.coalesce(10).write.mode("overwrite").parquet(sdfPath)

        # 20250523: tentar opcao abaixo para ver se tem melhor performance, olhar a explicacao do chat 
        # https://stackoverflow.com/questions/57155855/how-we-save-a-huge-pyspark-dataframe
        # df_with_features.withColumn("par_id",col('id_source')%50) \
        #                 .repartition(50, 'par_id').write.format('parquet') \
        #                 .save(sdfPath, mode='overwrite')
        t2=time.time()
        print (f'9.1 Tempo para salvar df_with_features_scaled: {t2-t1:.2f}s {(t2-t1)/60:.2f}m') #if sh_print else None
        print (f'df_with_features_scaled path: {sdfPath}') #if sh_print else None

        logger.info(f'9.1 df_with_features_scaled path: {sdfPath}')
        logger.info(f'9.1 Tempo para salvar df_with_features_scaled: {t2-t1:.2f}s {(t2-t1)/60:.2f}m')

    ### run PCA on scaled features
    print ("Apply pca on df_with features scaled") if sh_print else None
    t1 = time.time()
    
    # Apply PCA
    n_components = n_components
    pca_scaled = PCA(k=n_components, inputCol="features_scaled", outputCol="pca_features_scaled")
    pca_model_scaled = pca_scaled.fit(df_with_features)
    df_with_pca_scaled = pca_model_scaled.transform(df_with_features).select("coords_0", "coords_1", "pca_features_scaled")
    t2 = time.time()

    components_matrix = pca_model_scaled.pc  # This is a DenseMatrix object
    t3 = time.time()
    print (f'6.5 Tempo para run pca on df_with_pca_scaled: {t2-t1:.2f} s {(t2-t1)/60:.2f} m') if sh_print else None
    logger.info(f'6.5 Tempo para run pca on df_with_pca_scaled: {t2-t1:.2f} s {(t2-t1)/60:.2f} m')
    print (f'Tempo para gerar components matrix {t3-t2:.2f}s {(t3-t2)/60:.2f}')
    logger.info(f'Tempo para gerar components matrix {t3-t2:.2f}s {(t3-t2)/60:.2f}')

    # Convert to numpy array
    components_matrix_np = np.array(components_matrix.toArray())

    #save components matrix
    if read_quad==9:        # image full
        compMatrixPath = save_etapas_dir + "spark_pca_scaled_compMatrix_Full.pkl"
    else:
        if par_impar==0:
            compMatrixPath = save_etapas_dir + "spark_pca_scaled_compMatrix_Quad_" + str(read_quad)+".pkl"
        elif par_impar==1:
            compMatrixPath = save_etapas_dir + "spark_pca_scaled_compMatrix_Quad_" + str(read_quad)+"_impar_par.pkl"
        elif par_impar==2:
            compMatrixPath = save_etapas_dir + "spark_pca_scaled_compMatrix_Quad_" + str(read_quad)+"_par_impar.pkl"
        
    obj_dic = {}
    obj_dic= {
        "components_matrix": components_matrix,  
        "components_matrix_np": components_matrix_np
        
       }  
    # Save the object to a file using pickle
    save_to_pickle(obj_dic, compMatrixPath)  
    print (f'6.4 saving components_matrix: {compMatrixPath}') if sh_print else None 
    logger.info(f'6.4 saving components_matrix: {compMatrixPath}') if sh_print else None 

    # 1. Unpersist if it's cached (safe even if it wasn't)
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a} 7.1 Unpersist df_with_features with blocking') if sh_print else None
    logger.info(f'7.1 Unpersist df_with_features')
    t1 = time.time()
    df_with_features.unpersist(blocking=True)
    del df_with_features
    gc.collect()
    spark._jvm.System.gc() #20241204
    t2 = time.time()
    a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    print (f'{a} 7.1 fim Unpersist e del df_with_features{t2-t1:.2f}s {(t2-t1)/60:.2f}m') if sh_print else None
    logger.info(f'7.1 fim Unpersist e del df_with_features {t2-t1:.2f}s {(t2-t1)/60:.2f}m')

    # print (f'7.2 df_with_pca_scaled\n: {df_with_pca_scaled.show(n=10)}')
    # logger.info(f'7.2 df_with_pca_scaled\n: {df_with_pca_scaled.show(n=10)}')

    # print (f'7.3 pca_model.explainedVariance: {pca_model.explainedVariance}')
    # logger.info(f'7.3 pca_model.explainedVariance: {pca_model.explainedVariance}')

    print (f'7.4 pca_model_scaled.explainedVariance: {pca_model_scaled.explainedVariance}') #if sh_print else None
    logger.info(f'7.4 pca_model_scaled.explainedVariance: {pca_model_scaled.explainedVariance}')
    pca_expl = pca_model_scaled.explainedVariance
    df_expl = pd.DataFrame([pca_expl], columns=['c'+str(i) for i in range(len(pca_expl))])
    df_expl['Explain Total'] = df_expl.sum(axis=1)
    if read_quad==9:        # image full
        explPath = save_etapas_dir + "spark_pca_expl_Full.pkl"
    else:
        if par_impar==0:
            explPath = save_etapas_dir + "spark_pca_expl_Quad_" +str(read_quad)+".pkl"
        elif par_impar==1:
            explPath = save_etapas_dir + "spark_pca_expl_Quad_" +str(read_quad)+"_impar_par.pkl"
        elif par_impar==2:
            explPath = save_etapas_dir + "spark_pca_expl_Quad_" +str(read_quad)+"_par_impar.pkl"
    df_expl.to_pickle(explPath)
    print (f'7.4 df explainedVariance: {df_expl}') if sh_print else None

    ### Save pca_model_scaled, pca_scaled, and spark_sdf_with_pca_scaled
    #salve model
    print (f'8. Saving pca_scaled and pca_scaled-model') if sh_print else None
    logger.info(f'8. Saving pca_scaled and pca_scaled-model')
    # pcaPath = base_dir + "data/tmp/spark_pca_scaled"  #20241210 commented 
    if read_quad==9:        # image full
        pcaPath = save_etapas_dir + "spark_pca_scaled_Full"
        modelPath =  save_etapas_dir + "spark_pca_scaled-model_Full"
    else:
        if par_impar==0:
            pcaPath = save_etapas_dir + "spark_pca_scaled_Quad_" +str(read_quad) 
            # modelPath = base_dir + "data/tmp/spark_pca_scaled-model" #20241210 commented 
            modelPath =  save_etapas_dir + "spark_pca_scaled-model_Quad_" + str(read_quad) 
        elif par_impar==1:
            pcaPath = save_etapas_dir + "spark_pca_scaled_Quad_" +str(read_quad) + "_impar_par"
            modelPath =  save_etapas_dir + "spark_pca_scaled-model_Quad_" + str(read_quad) + "_impar_par"
        elif par_impar==2:
            pcaPath = save_etapas_dir + "spark_pca_scaled_Quad_" +str(read_quad) + "_par_impar"
            modelPath =  save_etapas_dir + "spark_pca_scaled-model_Quad_" + str(read_quad) + "_par_impar"

    pca_scaled.write().overwrite().save(pcaPath)
    pca_model_scaled.write().overwrite().save(modelPath)

    print (f'9.0 paths for pca_scaled {pcaPath} and \npca_model_scaled {modelPath} ') if sh_print else None

    #saving spark Dataframe to parquet
    # t1=time.time()
    # sdfPath = base_dir + "data/tmp/spark_sdf_with_pca_scaled"
    # # Set checkpoint directory
    # spark.sparkContext.setCheckpointDir(base_dir + "data/tmp/checkpoint_dir")

    # df_with_pca_scaled = df_with_pca_scaled.checkpoint()
    # t2=time.time()
    # print (f'9.0.1 Tempo para fazer o perist do df_with_pca_scaled {t2-t1:.2f}s {(t2-t1)/60:.2f}m')
    # logger.info(f'9.0.1 Tempo para fazer o perist do df_with_pca_scaled {t2-t1}s {(t2-t1)/60:.2f}m')
    
    # t1 = time.time()
    # df_with_pca_scaled = df_with_pca_scaled.repartition(20) 
    # t2 = time.time()
    # print (f'9.0.2 Tempo para fazer o repartition do df_with_pca_scaled {t2-t1:.2f}s {(t2-t1)/60:.2f}m')
    # logger.info(f'9.0.2 Tempo para fazer o repartition do df_with_pca_scaled {t2-t1}s {(t2-t1)/60:.2f}m')

    if read_quad==9:        # image full
        sdfPath = save_etapas_dir + "df_pca_scaled_Full"
    else:
        if par_impar==0:
            # sdfPath = base_dir + "data/tmp/df_with_pca_scaled"  #20241210 commented
            sdfPath = save_etapas_dir + "df_pca_scaled_Quad_" +  str(read_quad) 
        elif par_impar==1:
            sdfPath = save_etapas_dir + "df_pca_scaled_Quad_" +  str(read_quad) + "_impar_par"
        elif par_impar==2:
            sdfPath = save_etapas_dir + "df_pca_scaled_Quad_" +  str(read_quad) + "_par_impar"

    t1=time.time()
    # 20250523 comentei o save abaixo para ver se funciona com coalesce
    # df_with_pca_scaled.write.mode("overwrite").parquet(sdfPath) #20250523
    df_with_pca_scaled.coalesce(10).write.mode("overwrite").parquet(sdfPath)
    # df_with_pca_scaled.withColumn("par_id",col('id_source')%50) \
    #                   .repartition(50, 'par_id')
    #                   .write.format('parquet') \
    #                   .save(sdfPath, mode='overwrite')
    t2=time.time()
    print (f'9.1 Tempo para salvar df_with_pca_scaled: {t2-t1:.2f}s {(t2-t1)/60:.2f}m') #if sh_print else None
    print (f'df_with_pca_scaled path: {sdfPath}') #if sh_print else None

    logger.info(f'9.1 df_with_pca_scaled path: {sdfPath}')
    logger.info(f'9.1 Tempo para salvar df_with_pca_scaled: {t2-t1:.2f}s {(t2-t1)/60:.2f}m')

    # 1. Unpersist if it's cached (safe even if it wasn't)
    df_with_pca_scaled.unpersist(blocking=True)
    del df_with_pca_scaled
    gc.collect()
    spark._jvm.System.gc() #20241204

#close sparky session
tf=time.time()
print (f'Tempo total do {nome_prog}: {tf-ti:.2f}s, {(tf-ti)/60:.2f}m') if sh_print else None
spark.stop()

time_file = process_time_dir + "process_times.pkl"

update_procTime_file(proc_dic, time_file)
print (f'process_time file: {time_file}')