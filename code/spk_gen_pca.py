# Programa para Fazer o PCA no df to PCA com spark
#20240927: baixado do exacta e alterado para fazer funcionar no laptop
# 20241210: alterado para ser executado com 1/4 do tile
#           Falta terminar de incluir so tempos de processamento no df
#           colocar os nomes parciais dos diretorios como variaveis no 
#           inicio do programa
# 20250120: Caso o programa dê erro de execucao apos salvar o df with features
#           executá-lo novamente com a opcao -rf 1
#           Talvez seja melhor ajustá-lo para ser executado 2 vezes, a primeira 
#           para salvar o df with featres e a segunda pra ler o df with feautures 
#           e executar o pca
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
#for terminal
#from code.functions.functions_pca import list_files_to_read, get_bandsDates, gen_dfToPCA_filter

# Set the environment variable for spark
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
base_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/'
ti=time.time()

from functions_pca import list_files_to_read, get_bandsDates, \
                          gen_dfToPCA_filter, gen_sdfToPCA_filter
                          
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
    .config("spark.driver.memory", "400g") \
    .config("spark.executor.memory", "400g") \
    .config("spark.driver.maxResultSize", "400g")\
    .config("spark.sql.shuffle.spill.compress", "true") \
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
    .getOrCreate()

# Set log level to ERROR to suppress INFO and WARN messages
spark.sparkContext.setLogLevel("ERROR") 
# spark.conf.set("spark.sql.shuffle.partitions", "200")

from functions.functions_pca import cria_logger, update_procTime_file

# 20241210 bloco de parse incluido
# Inicializa o parser
parser = argparse.ArgumentParser(description="Program segment image")

# Define os argumentos
parser.add_argument("-bd", '--base_dir', type=str, help="Diretorio base", default='')
parser.add_argument("-sd", '--save_dir', type=str, help="Dir base para salvar saidas de cada etapa", default='data/tmp2/')
# parser.add_argument("-td", '--tif_dir', type=str, help="Dir dos tiffs", default='data/Cassio/S2-16D_V2_012014_20220728_/')
parser.add_argument("-q", '--quadrante', type=int, help="Numero do quadrante da imagem [0-all,1,2,3,4]", default=1)
parser.add_argument("-ld", '--log_dir', type=str, help="Dir do log", default='code/logs/')
parser.add_argument("-i", '--name_img', type=str, help="Nome da imagem", default='S2-16D_V2_012014')
parser.add_argument("-sp", '--sh_print', type=int, help="Show prints", default=0)
parser.add_argument("-pd", '--process_time_dir', type=str, help="Dir para df com os tempos de processamento", default='data/tmp2/')
parser.add_argument("-rf", '--READ_df_features', type=int, help="Read or create df with features", default=0 )
parser.add_argument("-nc", '--num_components', type=int, help="Read or create df with features", default=4 )
args = parser.parse_args()
 
# base_dir = args.base_dir
save_etapas_dir = base_dir + args.save_dir if base_dir else args.save_dir + args.name_img +'/'
# tif_dir = base_dir + args.tif_dir if base_dir else args.tif_dir
read_quad = args.quadrante 
log_dir = base_dir + args.log_dir if base_dir else args.log_dir
name_img = args.name_img
process_time_dir = base_dir + args.process_time_dir
sh_print = args.sh_print
READ_df_features = args.READ_df_features
n_components = args.num_components

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
if not READ_df_features:
    
    # temp_path= base_dir + 'data/tmp'  #20241210 commented
    # sdfPath = temp_path + "/spark_sdf_toPCA2_6d" #20241210 commented
    # sdfPath = save_etapas_dir + "spark_df_toPCA_Quad_" + str(read_quad) #20250121:comentei para usar o df sem os nsns do 2023-01-01
    sdfPath = save_etapas_dir + "spark_df_toPCA_Quad_" + str(read_quad) 
    logger.info(f'Creating df_features from {sdfPath}' )
    t1=time.time()
    spark_sdf_toPCA = spark.read.parquet(sdfPath)
    t2=time.time()
    logger.info(f'1. Tempo para ler spark df: {t2-t1:.2f}s {(t2-t1)/60:.2f}m')
    print (f'1. Tempo para ler spark df: {t2-t1:.2f}s {(t2-t1)/60:.2f}m') if sh_print else None

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

    pcaColumns = spark_sdf_toPCA.columns[2:]
    print (f'pcaColumns: {pcaColumns}') if sh_print else None
    logger.info(f'pcaColumns: {pcaColumns}')

    # assembler nao suporta nan's
    t1=time.time()
    assembler = VectorAssembler(inputCols=pcaColumns, outputCol="features")
    df_with_features = assembler.transform(spark_sdf_toPCA)
    t2=time.time()
    print (f'3. Tempo para criar col de array to pca: {t2-t1:.2f}s {(t2-t1)/60:.2f}m') if sh_print else None
    # print (f'3.1 df_with features: {df_with_features.show(n=10)}')
    # print (f'3.2 dtypes: {df_with_features.dtypes}')

    logger.info(f'3. Tempo para criar col de array to pca: {t2-t1:.2f}s {(t2-t1)/60:.2f}m')
    # logger.info (f'3.1 df_with features: {df_with_features.show(n=10)}')
    # logger.info (f'3.2 dtypes: {df_with_features.dtypes}')
    del spark_sdf_toPCA
    gc.collect()
    spark._jvm.System.gc() #20241210

    ri+=1       #indice do dicionario com os tempos de cada subetapa
    proc_dic[ri]={} if ri not in proc_dic else None
    proc_dic[ri]['etapa'] = 'PCA df image days'
    proc_dic[ri]['subetapa'] = 'df_with_features criar col de array to pca'
    proc_dic[ri]['tempo'] = t2-t1

    columnsToDrop = pcaColumns #['coords_0', 'coords_1']
    # print (f'df_with_features columns antes drop columns: {df_with_features.columns}') if sh_print else None
    logger.info(f'df_with_features columns antes drop columns: {df_with_features.columns}')
    df_with_features = df_with_features.drop(*columnsToDrop)
    print (f'df_with_features columns after drop columns: {df_with_features.columns}') if sh_print else None
    logger.info(f'df_with_features columns after drop columns: {df_with_features.columns}')

    #saving spark Dataframe to parquet
    # t1=time.time()        #20241210 commented
    # sdfPath = base_dir + "data/tmp/df_with_features"  #20241210 commented
    sdfPath = save_etapas_dir + "df_with_features_Quad_" + str(read_quad) +"/"
    
    df_with_features.write.mode("overwrite").parquet(sdfPath)
    t2=time.time()
    print (f'9. Tempo para salvar df_with_features for quad {read_quad}: {(t2-t1)/60:.2f}') if sh_print else None
    logger.info(f'9. Tempo para salvar df_with_features for quad {read_quad}: {t2-t1:.2f}')
    print (f'df_with_features path: {sdfPath}') if sh_print else None
    logger.info(f'df_with_features path: {sdfPath}')
    
    ri+=1       #indice do dicionario com os tempos de cada subetapa
    proc_dic[ri]={} if ri not in proc_dic else None
    proc_dic[ri]['etapa'] = 'PCA df image days'
    proc_dic[ri]['subetapa'] = 'save df_with_features'
    proc_dic[ri]['tempo'] = t2-t1    
    
    gc.collect()
    spark._jvm.System.gc() #20241204

    # sair d programa e mandar ler o df features para fazer o pca


if READ_df_features:
    # sdfPath = base_dir + "data/tmp/df_with_features"
    sdfPath = save_etapas_dir + "df_with_features_Quad_" + str(read_quad) + "/"
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
    # n_components = 4 #20241210 commented, passed via args
    t1 = time.time()
    standardScaler = StandardScaler()
    standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")
    t2 = time.time()
    print (f'6.1 standardScaler: {standardScaler}') if sh_print else None
    logger.info(f'6.1 standardScaler: {standardScaler}')
    scaler_model = standardScaler.fit(df_with_features)
    df_with_features = scaler_model.transform(df_with_features)
    t3 = time.time()
    print (f'6.2 Time to do standardScaler: {t2-t1:.2f} s {(t2-t1)/60:.2f} m') if sh_print else None
    print (f'6.3 Time to do Scaler: {t3-t2:.2f}s {(t3-t2)/60:.2f} m') if sh_print else None
    # print (f'6.4 df_with_features_scaled : {df_with_features.show(n=10)}')

    logger.info(f'6.2 Time to do standardScaler: {t2-t1:.2f} s {(t2-t1)/60:.2f} m')
    logger.info(f'6.3 Time to do Scaler: {t3-t2:.2f} s {(t3-t2)/60:.2f} m')
    # logger.info(f'6.4 df_with_features_scaled : {df_with_features.show(n=10)}')

    ### run PCA on scaled features
    print ("Apply pca on df_with features scaled") if sh_print else None
    t1 = time.time()
    # Apply PCA
    n_components = 4
    pca_scaled = PCA(k=n_components, inputCol="features_scaled", outputCol="pca_features_scaled")
    pca_model_scaled = pca_scaled.fit(df_with_features)
    df_with_pca_scaled = pca_model_scaled.transform(df_with_features)
    t2 = time.time()
    del df_with_features
    gc.collect()
    spark._jvm.System.gc() #20241204


    print (f'7.1 Tempo para run pca on df_with_pca_scaled: {t2-t1:.2f} s {(t2-t1)/60:.2f} m') if sh_print else None
    logger.info(f'7.1 Tempo para run pca on df_with_pca_scaled: {t2-t1:.2f} s {(t2-t1)/60:.2f} m')

    # print (f'7.2 df_with_pca_scaled\n: {df_with_pca_scaled.show(n=10)}')
    # logger.info(f'7.2 df_with_pca_scaled\n: {df_with_pca_scaled.show(n=10)}')

    # print (f'7.3 pca_model.explainedVariance: {pca_model.explainedVariance}')
    # logger.info(f'7.3 pca_model.explainedVariance: {pca_model.explainedVariance}')

    print (f'7.4 pca_model_scaled.explainedVariance: {pca_model_scaled.explainedVariance}') #if sh_print else None
    logger.info(f'7.4 pca_model_scaled.explainedVariance: {pca_model_scaled.explainedVariance}')
    pca_expl = pca_model_scaled.explainedVariance
    df_expl = pd.DataFrame([pca_expl], columns=['c'+str(i) for i in range(len(pca_expl))])
    df_expl['Explain Total'] = df_expl.sum(axis=1)
    explPath = save_etapas_dir + "spark_pca_expl_Quad_" +str(read_quad)+".pkl"
    df_expl.to_pickle(explPath)
    print (f'7.4 df explainedVariance: {df_expl}') if sh_print else None

    ### Save pca_model_scaled, pca_scaled, and spark_sdf_with_pca_scaled
    #salve model
    print (f'8. Saving pca_scaled and pca_scaled-model') if sh_print else None
    logger.info(f'8. Saving pca_scaled and pca_scaled-model')
    # pcaPath = base_dir + "data/tmp/spark_pca_scaled"  #20241210 commented 
    pcaPath = save_etapas_dir + "spark_pca_scaled_Quad_" +str(read_quad) 
    
    pca_scaled.write().overwrite().save(pcaPath)

    # modelPath = base_dir + "data/tmp/spark_pca_scaled-model" #20241210 commented 
    modelPath =  save_etapas_dir + "spark_pca_scaled-model_Quad_" + str(read_quad) 
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

    # sdfPath = base_dir + "data/tmp/df_with_pca_scaled"  #20241210 commented
    sdfPath = save_etapas_dir + "df_pca_scaled_Quad_" +  str(read_quad) 

    t1=time.time()
    df_with_pca_scaled.write.mode("overwrite").parquet(sdfPath)
    t2=time.time()
    print (f'9.1 Tempo para salvar df_with_pca_scaled: {t2-t1:.2f}s {(t2-t1)/60:.2f}m') #if sh_print else None
    print (f'df_with_pca_scaled path: {sdfPath}') #if sh_print else None

    logger.info(f'9.1 df_with_pca_scaled path: {sdfPath}')
    logger.info(f'9.1 Tempo para salvar df_with_pca_scaled: {t2-t1:.2f}s {(t2-t1)/60:.2f}m')

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