# Programa para testar o spark
#%%
import pandas as pd
import numpy as np
#%%
# Set the environment variable for spark
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
import pyspark.pandas as ps
#%%
#imports for spark
from pyspark.sql import SparkSession
#%%
# Step 1: Start a Spark ssession
spark = SparkSession.builder \
    .appName("Pandas to PySpark DataFrame") \
    .getOrCreate()
#%% exemplo 1
# Step 2: Create a Pandas DataFrame
data = {'col1': [1, 2, 3, 4],
        'col2': ['A', 'B', 'C', 'D']}
pandas_df = pd.DataFrame(data)

# Step 3: Convert Pandas DataFrame to PySpark DataFrame
spark_df = spark.createDataFrame(pandas_df)
# Show the PySpark DataFrame
spark_df.show()
#close sparky session
spark.stop()
#%% exemplo2
import pyspark.pandas as ps
# senao inicializar uma sessao é inicializada qdo chama o sparky

d = {'col1': [1, 2], 'col2': [3, 4]}
df = ps.DataFrame(data=d, columns=['col1', 'col2'])
print (df)
df.dtypes

#%% To enforce a single dtype:
import numpy as np
df = ps.DataFrame(data=d, dtype=np.int8)
df.dtypes
#%% Constructing DataFrame from numpy ndarray with Pandas index:
ps.DataFrame(data=np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]]),
    index=pd.Index([1, 4]), columns=['a', 'b', 'c', 'd', 'e'])

#%% #Constructing DataFrame from numpy ndarray with pandas-on-Spark index:
ps.DataFrame(data=np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]]),
    index=ps.Index([1, 4]), columns=['a', 'b', 'c', 'd', 'e'])
# %% #Constructing DataFrame from Pandas DataFrame with Pandas index:
pdf = pd.DataFrame(data=np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]]),
    columns=['a', 'b', 'c', 'd', 'e'])
    
ps_df = ps.DataFrame(data=pdf, index=pd.Index([1, 4]))
type(ps_df)
#pyspark.pandas.frame.DataFrame

# %% #Constructing DataFrame from Spark DataFrame with Pandas index:
#Enable ‘compute.ops_on_diff_frames’ to combine Spark DataFrame and pandas-on-Spark index
sdf = spark.createDataFrame([("Data", 1), ("Bricks", 2)], ["x", "y"])
type(sdf)
#pyspark.sql.dataframe.DataFrame

with ps.option_context("compute.ops_on_diff_frames", True):
    ps_sdf = ps.DataFrame(data=sdf, index=pd.Index([0, 1, 2]))
#%%  ######### Testes PCA spark

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import PCA
from pyspark.ml.feature import PCAModel
#%%
data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
    (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
    (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
    
sdf = spark.createDataFrame(data,["features"])
sdf
#%%

pca = PCA(k=2, inputCol="features")
pca.setOutputCol("pca_features")
#%%
model = pca.fit(sdf)
model.getK()

#%%
model.setOutputCol("output")

#%%
model.transform(sdf).collect()[0].output
#DenseVector([1.648..., -4.013...])
#%%
model.explainedVariance
# DenseVector([0.794..., 0.205...])
#%%
import os
temp_path= os.getcwd() + '/data/tmp'
pcaPath = temp_path + "/pca_spk"
pcaPath
#%%
pca.save(pcaPath)
#%%
loadedPca = PCA.load(pcaPath)
loadedPca.getK() == pca.getK()
#True
modelPath = temp_path + "/pca-model"
model.save(modelPath)

#%%

loadedModel = PCAModel.load(modelPath) #NOK
#loadedModel = model.load(modelPath)

loadedModel.pc == model.pc
# True
#%%
loadedModel.explainedVariance == model.explainedVariance
model.explainedVariance

# True
#%%
loadedModel.transform(sdf).take(1) == model.transform(sdf).take(1)
# True
#%%
#teste para gerar spark df da matrix
t='20220728'
band_img_files=band_tile_img_files
band_img_file_to_load=[x for x in band_img_files if t in x]
image_band_dic = {}
image_band_dic = load_image_files3(band_img_file_to_load, pos=-1)
bands = image_band_dic.keys()
print ( bands, band_img_file_to_load)
#%%
img_bands = np.dstack([image_band_dic[x] for x in list(bands)[:2]])
cols_name = [x+'_'+t for x in list(bands)[:2]]
#%%
#gen df for images bands of the t day
dft = pd.DataFrame()
t1=time.time()
#dft = gen_sdf_from_img_band(img_bands, cols_name, ar_pos=0, sp=2, sh_print=0)
t2=time.time()
print (t2-t1)

#%%
#testar a funcao acima aqui
sh_print=1
num_rows, num_cols, cols_df = img_bands.shape
print (f'matrix rows={num_rows}, cols={num_cols}, bands={cols_df}') if sh_print else None

#%%
# Flatten the matrix values and reshape into a 2D array
values = img_bands.reshape(-1, cols_df)
#%%
# Create a DataFrame from the index positions and values
#dft = pd.DataFrame(values, columns=bands_name)#, index=positions)
sdft = ps.DataFrame(values, columns=cols_name)
#%%
# Create a list of index positions [i, j]
#positions = [[i, j] for i in range(num_rows) for j in range(num_cols)] 
pos_0=np.repeat(np.arange(img_bands.shape[0]), img_bands.shape[1])
pos_1=np.tile(np.arange(img_bands.shape[1]), img_bands.shape[0])


#%%
#pyspark pandas dataframe doesn't support ndarray
pos_0 = pos_0.tolist()
pos_1 = pos_1.tolist()
#%%
sdftM.insert(0,'coords_0', pos_0)
sdftM.insert(1,'coords_1', pos_1)
#%%
# Convert to PySpark DataFrame
spark_df = sdftM.to_spark()

#%%
#%%
#close sparky session
spark.stop()
# %%
