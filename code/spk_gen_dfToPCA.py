# Programa para testar o spark
#%%
import os
import gc
import time
import pandas as pd
from code.functions.functions_pca import list_files_to_read, get_bandsDates, gen_dfToPCA_filter
#for terminal
#from code.functions.functions_pca import list_files_to_read, get_bandsDates, gen_dfToPCA_filter
#%%
# Set the environment variable for spark
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
#%%
#imports for spark
import pyspark.pandas as ps
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import when, col, array, udf
from pyspark.ml.feature import PCA
from pyspark.ml.feature import PCAModel

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors

#%%

# Step 1: Start a Spark session
# senao inicializar uma sessao é inicializada qdo chama o sparky
spark = SparkSession.builder \
    .appName("PySpark to PCA") \
    .config("spark.executorEnv.PYARROW_IGNORE_TIMEZONE", "1") \
    .config("spark.driverEnv.PYARROW_IGNORE_TIMEZONE", "1") \
    .config("spark.driver.memory", "9g") \
    .getOrCreate()
#%%

#read band image files
all_bands = ['B04', 'B03', 'B02', 'B08', 'EVI', 'NDVI']
read_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/Cassio/S2-16D_V2_012014_20220728_/'
name_img = 'S2-16D_V2_012014'
band_tile_img_files = list_files_to_read(read_dir, name_img)
bands, dates= get_bandsDates(band_tile_img_files, tile=1)
cur_dir = os.getcwd()
#%%
dates= ['20220728']
t1=time.time()
df_toPCA, max_bt, min_bt = gen_dfToPCA_filter(dates, band_tile_img_files, dask=2, n_part=63, ar_pos=0, sh_print=1)
t2=time.time()
print (f'Time to gen df_toPCA: {t2-t1}')
#%%
t1=time.time()
sdf_toPCA = spark.createDataFrame(df_toPCA)
t2=time.time()
print (f'Time to gen sdf_toPCA: {t2-t1}')
#%%
del df_toPCA
gc.collect()
print (sdf_toPCA.head())
#%%
sdf_toPCA_coords = sdf_toPCA
#%% #replace nans
t1=time.time()
# Replace -9999 with NaN in all columns
sdf_toPCA = sdf_toPCA.select([when(col(c) == -9999, F.lit(None)).otherwise(col(c)).alias(c) for c in sdf_toPCA.columns])
sdf_toPCA = sdf_toPCA.select([when(col(c) == -32768, F.lit(None)).otherwise(col(c)).alias(c) for c in sdf_toPCA.columns])
t2=time.time()
t2-t1

#%% 
# #drop the nans
sdf_toPCA = sdf_toPCA.na.drop()
sdf_toPCA.show()

#%%
#PCA
bands_name =['b1', 'b2', 'b3', 'b4']
# assembler nao suporta nan's
assembler = VectorAssembler(inputCols=bands_name, outputCol="features")
df_with_features=assembler.transform(sdf_toPCA)
df_with_features.show()
df_with_features.dtypes

#%%
# Apply PCA
pca = PCA(k=2, inputCol="features", outputCol="pca_features")
pca_model = pca.fit(df_with_features)
df_with_pca = pca_model.transform(df_with_features)
df_with_pca.show()
#%%
# Show the PCA results
df_with_pca.select("pca_features").show(truncate=False)

#%%
pca_model.explainedVariance
pca_model.transform(df_with_features).collect()[1].pca_features



#%%
bands_name =['b1', 'b2', 'b3', 'b4']
pca = PCA(k=4, inputCol="features")

#pca = PCA(k=2)#, inputCol=bands_name)
pca.setOutputCol("pca_features")
#%%

#%%
model = pca.fit(df_with_features.select("features"))
model.getK()

#%%
model.setOutputCol("output")

#%%
model.transform(df_with_features).collect()[1].output
#DenseVector([1.648..., -4.013...])
#%%
model.explainedVariance

#%%
#close sparky session
spark.stop()

