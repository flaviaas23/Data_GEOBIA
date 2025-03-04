
base_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/'

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster #, ward
from scipy.spatial.distance import squareform

t1 = time.time()
matrix_dist_sel = zarr_group['matrix_dist_sel']
t2 = time.time()
print (f'{a}: 2.1 Tempo para read empty matrix de distancia: {(t2-t1):.2f}, {(t2-t1)/60:.2f}')
print (f'{a}: 2.1.1 matrix_dist_sel.shape: {matrix_dist_sel.shape}, {matrix_dist_sel.dtype}')


tmp_dir=base_dir + 'data/tmp/pca/'
t1 = time.time()
matrix_dist_sel[:] = 1 - matrix_sim_sel[:]  #[:] [:10000, 0:10000] 
t2=time.time()
print (f'{a}: 2.2 Tempo para gerar matrix de distancia: {(t2-t1):.2f}, {(t2-t1)/60:.2f}') #130.83, 2.18
del matrix_sim_sel
gc.collect()

t1=time.time()
dist_vector_matrix_dist_sel = squareform(matrix_dist_sel)
t2=time.time()
print (f'Tempo squareform {t2-t1}s, {(t2-t1)/60:.2f}m') #1.86m

t1=time.time()
linkage_matrix = linkage(dist_vector_matrix_dist_sel, method='ward')
t2=time.time()
print (f'Tempo linkage hierarchical clustering {t2-t1}s, {(t2-t1)/60:.2f}m') #12.11m

n_clusters=30
dic_cluster_ski_ms2 = {}
sse_ski_ms2 = []
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (a)
t1=time.time()
for n in range (2, n_clusters+1):   
    # print (f'n = {n}')    
    # clusters_sel = fcluster(linkage_matrix, n, criterion='distance')        
    # clusters_sel = fcluster(linkage_matrix, n, criterion='maxclust')        
    # dic_cluster_ski_ms2[str(n)] = clusters_sel.tolist()
    dic_cluster_ski_ms2[str(n)] = fcluster(linkage_matrix, n, criterion='maxclust')        



a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (a)

t1=time.time()
for n in dic_cluster_ski_ms2.keys():
    print (f'n = {n}')        
    # Calcular o SSE
    sse = 0
    clusters_sel = dic_cluster_ski_ms2[str(n)]
    unique_labels = np.unique(clusters_sel)
    # print (f'n={n}, num labels = {len(unique_labels)}')
    ts1 = time.time()
    for label in unique_labels:
        # Obter os índices dos pontos no cluster atual
        cluster_indices = np.where(clusters_sel == label)[0]        
        # # Extrair as distâncias internas do cluster da matriz de distâncias
        intra_cluster_distances = matrix_dist_sel[np.ix_(cluster_indices, cluster_indices)]        
        # # Calcular o SSE somando o quadrado das distâncias e dividindo pelo número de pares
        sse += np.sum(intra_cluster_distances ** 2) / 2  # Dividir por 2 para evitar contagem duplicada        
    ts2 = time.time()
    print(f"n = {n}, Tempo sse {ts2-ts1}s, SSE: {sse}")
    sse_ski_ms2.append(sse)
t2=time.time()

#for debug
n=2
sse = 0
clusters_sel = dic_cluster_ski_ms2[str(n)]
unique_labels = np.unique(clusters_sel)

label=2
t1=time.time()
cluster_indices = np.where(dic_cluster_ski_ms2[str(n)] == label)[0]  
t2=time.time()

del clusters_sel
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (a)
t1=time.time()
intra_cluster_distances = matrix_dist_sel[np.ix_(cluster_indices, cluster_indices)] 
t2=time.time()
t2-t1

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (a)
t1=time.time()
sse += np.sum(intra_cluster_distances ** 2) / 2 
t2=time.time()
t2-t1

sse = 0
for label in unique_labels:
    cluster_indices = np.where(labels == label)[0] 
    print (f'label = {label}\n\tcluster_indices = {cluster_indices}')
    intra_cluster_distances = dist_matrix[np.ix_(cluster_indices, cluster_indices)]
    print( f'\tintra_cluster_distances = {intra_cluster_distances}')
    print (f'\tsse= {np.sum(intra_cluster_distances ** 2)}')
    sse += np.sum(intra_cluster_distances ** 2) / 2
    print (f'\tsse= {sse}')
t2= time.time()


#fazer com numpy memmap
tmp_dir=base_dir + 'data/tmp/pca/'
filename = tmp_dir + 'matrix_dist_sel.npy'
temp_filename = tmp_dir + 'temp_matrix.npy'
len_cluster = len(matrix_dist_sel)
matrix_shape = matrix_dist_sel.shape  #matrix_dist_sel  dist_matrix


# Step 1: Save the array to a file
np.save(filename, matrix_dist_sel)  #dist_matrix  matrix_dist_sel

# Step 2: Load it as a memory-mapped array
matrix_dist_memmap = np.load(filename, mmap_mode="r+")

del matrix_dist_sel
gc.collect()

sh_print =0
n=2
sse_ski_ms2=[]
t_i= time.time()
for n in dic_cluster_ski_ms2.keys():
    # print (f'n = {n}')        
    # Calcular o SSE
    clusters_sel = dic_cluster_ski_ms2[str(n)]
    unique_labels = np.unique(clusters_sel)
    # matrix_dist = np.memmap(filename, dtype='float16', mode='w+', shape=matrix_shape)
    sse = 0
    intra_cluster_distances={}
    ti=time.time()
    for label in unique_labels:
        t0=time.time()
        cluster_indices =  np.where(clusters_sel == label)[0] 
        # print (f'label = {label}\n\tcluster_indices = {len(cluster_indices)}') if sh_print else None
        print (f'label = {label}\n\tcluster_indices = {cluster_indices}') if sh_print else None
        t1=time.time()
        intra_cluster_distances[label] = matrix_dist_memmap[np.ix_(cluster_indices, cluster_indices)] 
        # intra_cluster_distances = matrix_dist_test[np.ix_(cluster_indices, cluster_indices)] 
        intra_cluster_distances = intra_cluster_distances.astype(np.float32)
        print (f'\tintra_cluster = \n\t{intra_cluster_distances}') if sh_print else None
        t2=time.time()
        sse_tmp =  np.sum(intra_cluster_distances ** 2) / 2
        t3=time.time()
        #melhor fazer considerando a matrix toda e dividir por 2
        # sse_tmp2 = np.sum(np.triu(intra_cluster_distances) ** 2)
        # t4=time.time()
        if (sh_print):
            print (f'\tTempo para obter os indices ={t1-t0}s')
            print (f'\tTempo para obter intra_cluster_distances e converter para float32 {t2-t1}s')
            print (f'\tTempo para obter sse_tmp {sse_tmp} via matrix total ={t3-t2}s')
            # print (f'\tTempo para obter sse {sse_tmp2} via np.triu ={t4-t3}s')
        sse+=sse_tmp
    sse_ski_ms2.append(sse)
    tf=time.time()
    print (f'n= {n}, Tempo total para obter sse via matrix ={tf-ti}s')



t_f=time.time()
print (f'Tempo total para obter sse para cada n via matrix ={t_f-t_i:.2f}s,{(t_f-t_i)/60:.2f}m')

tf-ti


obj_dic={}
obj_dic = {
    "dic_cluster_ski": dic_cluster_ski_ms2,
    "sse_ski": sse_ski_ms2,
    # "dic_cluster_rd": dic_cluster_rd_ms2,
    # "sse_rd": sse_rd_ms2,
    # "rd_state":rd_state 
    #"n_opt": n_opt
}
id=0
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
t1=time.time()
# file_to_save = base_dir + 'pca_snic_cluster/clara_ms'+str(id)+'.pkl'
if KNN:
    file_to_save = base_dir + 'data/tmp/pca_snic_cluster/clara_ms_knn_'+str(id)+'_20241117.pkl'
else:
    file_to_save = base_dir + 'data/tmp/pca_snic_cluster/clara_ms_'+str(id)+'.pkl'
save_to_pickle(obj_dic, file_to_save)
t2=time.time()
print (f'{a}: 3.2 Tempo para salvar cluster clara: {t2-t1:.2f}, {(t2-t1)/60:.2f}')
print (f'{a}: 3.3 clara com matriz de similaridade salvo {file_to_save}')
del obj_dic
gc.collect()



#funcao para calcular intra inter e sse de um cluster
def calc_inter_intra_sse(matrix_dist_memmap, dic_cluster_ski_ms2, n, sh_print=True):
    clusters_sel = dic_cluster_ski_ms2[str(n)]
    unique_labels = np.unique(clusters_sel)
    # matrix_dist = np.memmap(filename, dtype='float16', mode='w+', shape=matrix_shape)
    sse = 0
    intra_cluster =[]
    inter_cluster =[]
    ti=time.time()
    for label in unique_labels:
        t0=time.time()
        cluster_indices =  np.where(clusters_sel == label)[0] 
        different_cluster_indices = np.where(clusters_sel != label)[0]
        # print (f'label = {label}\n\tcluster_indices = {len(cluster_indices)}') if sh_print else None
        print (f'label = {label}\n\tcluster_indices = {cluster_indices}') if sh_print else None
        print (f'\tdifferent_cluster_indices = {different_cluster_indices}') if sh_print else None
        t1=time.time()
        intra_cluster_distances = matrix_dist_memmap[np.ix_(cluster_indices, cluster_indices)] 
        # intra_cluster_distances = matrix_dist_test[np.ix_(cluster_indices, cluster_indices)] 
        #obtem max distancia dentro do mesmo cluster
        intra_cluster.append(np.max(intra_cluster_distances))
        intra_cluster_distances = intra_cluster_distances.astype(np.float32)
        print (f'\tintra_cluster = \n\t{intra_cluster_distances}') if sh_print else None
        t2=time.time()
        sse_tmp =  np.sum(intra_cluster_distances ** 2) / 2
        t3=time.time()
        #melhor fazer considerando a matrix toda e dividir por 2
        # sse_tmp2 = np.sum(np.triu(intra_cluster_distances) ** 2)
        # t4=time.time()
        if (sh_print):
            print (f'\tTempo para obter os indices ={t1-t0}s')
            print (f'\tTempo para obter intra_cluster_distances e converter para float32 {t2-t1}s')
            print (f'\tTempo para obter sse_tmp {sse_tmp} via matrix total ={t3-t2}s')
            # print (f'\tTempo para obter sse {sse_tmp2} via np.triu ={t4-t3}s')
        sse+=sse_tmp
        del intra_cluster_distances
        gc.collect()
        #calcula o inter cluster
        # Compute intercluster distance (minimum distance to other clusters)
        t4=time.time()
        different_cluster_indices = np.where(clusters_sel != label)[0]
        inter_cluster_distances = matrix_dist_memmap[np.ix_(different_cluster_indices, different_cluster_indices)]
        #exluir a diagonal de zeros
        inter_cluster_distances =  np.ma.masked_array(inter_cluster_distances, mask=np.eye(inter_cluster_distances.shape[0], dtype=bool))
        inter_cluster.append(np.min(inter_cluster_distances))
        del inter_cluster_distances
        gc.collect()
        t5=time.time()
        print (f'\tTempo para obter inter cluster for {label} ={t5-t4}s') if sh_print else None
        print (f'\tlabel={label} inter={inter_cluster}') if sh_print else None
        #acho que a opcao abaixo nao funciona
        # t5=time.time()
        # inter_cluster_distances = matrix_dist_memmap[cluster_indices,cluster_indices] 
        # inter_cluster_distances[cluster_indices,cluster_indices] = np.inf
        # inter_cluster2.append(np.min(inter_cluster_distances))
        # t6=time.time()
        # print (f'\tTempo para obter inter cluster2 for {label} ={t6-t5}s')
        # print (f'\tlabel={label} inter={inter_cluster2}')
        # del inter_cluster_distances
        # gc.collect()
    #sse_ski_ms2.append(sse)
    tf=time.time()
    print (f'n= {n}, Tempo total para obter sse via matrix ={tf-ti}s')
    return sse, intra_cluster, inter_cluster

masked_matrix = np.ma.masked_array(matrix, mask=np.eye(matrix.shape[0], dtype=bool))

############### fazer com spark
# usando a funcao q criei


from functions_pca import gen_sdf_from_img_band
cols_name = ['distance']
dft = gen_sdf_from_img_band(dist_matrix, cols_name, ar_pos=0, sh_print=1)

# from chat
from pyspark.sql import SparkSession


# Iniciar sessão Spark
spark = SparkSession.builder.appName("Calculo_SSE").getOrCreate()

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (a)
# Transformar a matriz de distância em um DataFrame do Spark
t1=time.time()
indices = np.indices(matrix_dist_sel.shape)
rows, cols = indices[0].flatten(), indices[1].flatten()
data = matrix_dist_sel.flatten()
t2=time.time()
print (f'Time to transform matrix in df {t2-t1}')

# Criar um DataFrame com as coordenadas e valores da matriz
df_matrix = spark.createDataFrame(zip(rows, cols, data), ["row", "col", "distance"])
df_matrix.cache() 

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StructType, StructField

#Criar df com a partesuperior da matrix, nao inclui a diagonal de zeros
rows, cols = np.triu_indices_from(matrix_dist_sel, k=1)  #matrix_dist_sel #dist_matrix
distances = matrix_dist_sel[rows, cols] #dist_matrix
# upper_triangle_data = list(zip(rows, cols, distances))
upper_triangle_data = [(int(r), int(c), int(d)) for r, c, d in zip(rows, cols, distances)]
# schema = StructType([
    # StructField("coords_0", IntegerType(), True),
    # StructField("coords_1", IntegerType(), True),
    # StructField("distance", IntegerType(), True)
# ])
# Define the schema for the DataFrame
schema = StructType([
    StructField("coords_0", IntegerType(), True),
    StructField("coords_1", IntegerType(), True),
    StructField("distance", FloatType(), True)
])

df_matrix_upper = spark.createDataFrame(upper_triangle_data, schema=schema)


sse_ski_ms2 = []
from pyspark.sql import functions as F
#codigo abaixo ok
for n in dic_cluster_ski_ms2.keys():
    print(f'n = {n}')        
    sse = 0
    clusters_sel = dic_cluster_ski_ms2[str(n)]
    unique_labels = np.unique(clusters_sel)
    
    ts1 = time.time()
    
    # Criar um DataFrame do Spark com os índices dos clusters para cada ponto
    df_clusters = spark.createDataFrame(zip(range(len(clusters_sel)), clusters_sel), ["index", "cluster_label"])
    
    for label in unique_labels:
        # Obter índices do cluster atual, clusters_sel deve ser do tipo np.array
        cluster_indices = np.where(clusters_sel == label)[0]
        
        # Criar um DataFrame Spark com os índices do cluster
        df_cluster_indices = spark.createDataFrame([(int(i),) for i in cluster_indices], ["index"])
        
        # Filtrar a matriz para obter apenas as distâncias internas ao cluster df_matrix
        intra_cluster_distances = (df_matrix_upper
                                   .join(df_cluster_indices.alias("i"), F.col("coords_0") == F.col("i.index"))
                                   .join(df_cluster_indices.alias("j"), F.col("coords_1") == F.col("j.index"))
                                   .select("distance"))        
        # Calcular o SSE parcial para o cluster
        sse_partial = intra_cluster_distances.withColumn("squared_distance", F.col("distance") ** 2).agg(F.sum("squared_distance") / 2).collect()[0][0]        
        # Somar ao SSE total
        sse += sse_partial if sse_partial else 0
    # ts2 = time.time()
    # print(f"n = {n}, Tempo sse {ts2 - ts1}s, SSE: {sse}")


    sse_ski_ms2.append(sse)
#

#fazer conforme o exemplo
# Função para gerar chunks de dados
def generate_upper_triangle_df():
    coords = np.triu_indices_from(matrix_dist_sel, k=1)
    for i in range(len(dist_vector_matrix_dist_sel)):
        yield int(coords[0][i]), int(coords[1][i]), float(dist_vector_matrix_dist_sel[i])

# Criar o DataFrame PySpark em partes
t1=time.time()
df_matrix_upper = spark.createDataFrame(generate_upper_triangle_df(), schema=schema)
t2=time.time()
print (f'Tempo Criar df matrix_upper: {t2-t1}, {(t2-t1)/60:.2f}')

#####  exemplo

dist_vector = np.array([1, 2, 5, 3, 4, 8, 2, 6, 7, 3])
dist_matrix = squareform(dist_vector)
# >>> dist_matrix
# array([[0, 1, 2, 5, 3],
#        [1, 0, 4, 8, 2],
#        [2, 4, 0, 6, 7],
#        [5, 8, 6, 0, 3],
#        [3, 2, 7, 3, 0]])

#dist_vector = np.array([1, 2, 5, 3,4, 4, 8, 2,1, 6, 7, 2, 3, 5, 3])
#array([[0, 1, 2, 5, 3, 4],
#        [1, 0, 4, 8, 2, 1],
#        [2, 4, 0, 6, 7, 2],
#        [5, 8, 6, 0, 3, 5],
#        [3, 2, 7, 3, 0, 3],
#        [4, 1, 2, 5, 3, 0])

dist_vector = squareform(dist_matrix)

Z = linkage(dist_vector, method='ward')

n_clusters = 3
labels = fcluster(Z, t=n_clusters, criterion='maxclust')

n=3
unique_labels = np.unique(labels)
labels
array([1, 1, 2, 3, 1], dtype=int32)

# Função para gerar chunks de dados
def generate_upper_triangle_df():
    coords = np.triu_indices_from(dist_matrix, k=1)
    for i in range(len(dist_vector)):
        yield int(coords[0][i]), int(coords[1][i]), float(dist_vector[i])

# Criar o DataFrame PySpark em partes
t1=time.time()
df_matrix_upper = spark.createDataFrame(generate_upper_triangle_df(), schema=schema)
t2=time.time()
print (f'Tempo Criar df matrix_upper: {t2-t1}, {(t2-t1)/60:.2f}')

sse_ski_ms2 = []
from pyspark.sql import functions as F
#codigo abaixo ok
for n in [3]:
    print(f'n = {n}')        
    sse = 0
    clusters_sel = labels #dic_cluster_ski_ms2[str(n)]
    unique_labels = np.unique(clusters_sel)
    clusters_sel = [int(label) for label in clusters_sel]
    ts1 = time.time()    
    # Criar um DataFrame do Spark com os índices dos clusters para cada ponto
    df_clusters = spark.createDataFrame(zip(range(len(clusters_sel)), clusters_sel), ["index", "cluster_label"])    
    for label in unique_labels:
        # Obter índices do cluster atual, clusters_sel deve ser do tipo np.array
        cluster_indices = np.where(clusters_sel == label)[0]        
        # Criar um DataFrame Spark com os índices do cluster
        df_cluster_indices = spark.createDataFrame([(int(i),) for i in cluster_indices], ["index"])        
        # Filtrar a matriz para obter apenas as distâncias internas ao cluster df_matrix
        intra_cluster_distances = (df_matrix_upper
                                   .join(df_cluster_indices.alias("i"), F.col("coords_0") == F.col("i.index"))
                                   .join(df_cluster_indices.alias("j"), F.col("coords_1") == F.col("j.index"))
                                   .select("distance"))        
        print (f'intra_cluster_distances:\n{intra_cluster_distances}')
        # Calcular o SSE parcial para o cluster
        sse_partial = intra_cluster_distances.withColumn("squared_distance", F.col("distance") ** 2).agg(F.sum("squared_distance")).collect()[0][0]        
        print (f'label = {label}, sse_partial = {sse_partial}')
        # Somar ao SSE total
        sse += sse_partial if sse_partial else 0

###### fim exemplo

# outra tentativa:


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

# Tamanho do bloco
chunk_size = 100000  # Ajuste conforme a memória disponível

# Função para gerar chunks de dados
def generate_upper_triangle_df(matrix, dist_vector, start, end):
    coords = np.triu_indices_from(matrix, k=1)
    for i in range(start, end):
        yield Row(coords_0=int(coords[0][i]), coords_1=int(coords[1][i]), distance=float(dist_vector[i]))

# Calcular o número de elementos na matriz triangular superior
num_elements = len(dist_vector_matrix_dist_sel)

# Loop para criar DataFrames PySpark em blocos e concatená-los
df_matrix_upper = None
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print (a)
t1=time.time()
for start in range(0, num_elements, chunk_size):
    end = min(start + chunk_size, num_elements)
    chunk_rdd = spark.sparkContext.parallelize(generate_upper_triangle_df(matrix_dist_sel, dist_vector_matrix_dist_sel, start, end))   
    # Converter para DataFrame
    chunk_df = spark.createDataFrame(chunk_rdd)
    # Concatenar ao DataFrame principal
    if df_matrix_upper is None:
        df_matrix_upper = chunk_df
    else:
        df_matrix_upper = df_matrix_upper.union(chunk_df)


t2=time.time()


#### 20241119 #######

>>> from functions.functions_cluster import _silhouette_reduce, calc_inter_intra_cluster
>>> df_ski_ms_test2 = calc_inter_intra_cluster(matrix_dist_test, dic_cluster, str(n_opt_ski_ms), metric='precomputed')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: calc_inter_intra_cluster() got an unexpected keyword argument 'metric'
>>> 
>>> def calc_inter_intra_cluster(arraybands_sel, dic_cluster, n_opt, metric = "euclidean", sh_print=False): 
...     le = LabelEncoder()
...     labels = le.fit_transform(dic_cluster[n_opt])
...     n_samples = len(labels)
...     label_freqs = np.bincount(labels)
...     # check_number_of_labels(len(le.classes_), n_samples)    
...     # metric = "euclidean"  'precomputed' if arraybands_sel is a distance matrix
...     kwds = {}    
...     kwds["metric"] = metric
...     reduce_func = functools.partial(
...         _silhouette_reduce, labels = labels, label_freqs = label_freqs
...     )
...     print (f'arraybands_sel : {arraybands_sel}') if sh_print else None
...     results = zip(*pairwise_distances_chunked(arraybands_sel, reduce_func=reduce_func, **kwds))
...     intra_clust_dists, inter_clust_dists = results
...     intra_clust_dists = np.concatenate(intra_clust_dists)
...     inter_clust_dists = np.concatenate(inter_clust_dists)
...     print (f'intra {intra_clust_dists}') if sh_print else None
...     df = pd.DataFrame({'label': labels, 'inter': inter_clust_dists, 'intra': intra_clust_dists })
...     return df
... 
>>> 
>>> df_ski_ms_test3 = calc_inter_intra_cluster(matrix_dist_test, dic_cluster, str(n_opt_ski_ms), metric='precomputed')

>>> df_ski_ms_test2
   label     inter     intra
0      0  0.479980  0.899414
1      0  0.740039  0.299805
2      1  0.749878  0.800781
3      2  1.000000  0.000000
4      0  0.740039  0.299805
5      0  0.740039  0.299805
6      1  0.625122  0.400391
7      1  0.749878  0.800781
8      1  0.625122  0.400391
9      1  0.625122  0.400391
>>> dic_cluster
{'6': array([2, 2, 3, 6, 2, 2, 3, 3, 3, 3], dtype=int32)}
>>> matrix_dist_test
memmap([[0.    , 0.2998, 0.6   , 1.    , 0.2998, 0.2998, 0.4   , 0.6   ,
         0.4   , 0.4   ],
        [0.2998, 0.    , 0.8   , 1.    , 0.    , 0.    , 0.7   , 0.8   ,
         0.7   , 0.7   ],
        [0.6   , 0.8   , 0.    , 1.    , 0.8   , 0.8   , 0.2002, 0.2002,
         0.2002, 0.2002],
        [1.    , 1.    , 1.    , 0.    , 1.    , 1.    , 1.    , 1.    ,
         1.    , 1.    ],
        [0.2998, 0.    , 0.8   , 1.    , 0.    , 0.    , 0.7   , 0.8   ,
         0.7   , 0.7   ],
        [0.2998, 0.    , 0.8   , 1.    , 0.    , 0.    , 0.7   , 0.8   ,
         0.7   , 0.7   ],
        [0.4   , 0.7   , 0.2002, 1.    , 0.7   , 0.7   , 0.    , 0.2002,
         0.    , 0.    ],
        [0.6   , 0.8   , 0.2002, 1.    , 0.8   , 0.8   , 0.2002, 0.    ,
         0.2002, 0.2002],
        [0.4   , 0.7   , 0.2002, 1.    , 0.7   , 0.7   , 0.    , 0.2002,
         0.    , 0.    ],
        [0.4   , 0.7   , 0.2002, 1.    , 0.7   , 0.7   , 0.    , 0.2002,
         0.    , 0.    ]], dtype=float16)
>>> matrix_dist_test[:8,:8]
memmap([[0.    , 0.2998, 0.6   , 1.    , 0.2998, 0.2998, 0.4   , 0.6   ],
        [0.2998, 0.    , 0.8   , 1.    , 0.    , 0.    , 0.7   , 0.8   ],
        [0.6   , 0.8   , 0.    , 1.    , 0.8   , 0.8   , 0.2002, 0.2002],
        [1.    , 1.    , 1.    , 0.    , 1.    , 1.    , 1.    , 1.    ],
        [0.2998, 0.    , 0.8   , 1.    , 0.    , 0.    , 0.7   , 0.8   ],
        [0.2998, 0.    , 0.8   , 1.    , 0.    , 0.    , 0.7   , 0.8   ],
        [0.4   , 0.7   , 0.2002, 1.    , 0.7   , 0.7   , 0.    , 0.2002],
        [0.6   , 0.8   , 0.2002, 1.    , 0.8   , 0.8   , 0.2002, 0.    ]],
       dtype=float16)
>>> labels = le.fit_transform(dic_cluster[str(n_opt_ski_ms)])
>>> labels
array([0, 0, 1, 2, 0, 0, 1, 1, 1, 1])
>>> label_freqs = np.bincount(labels)
>>> labels_freq
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'labels_freq' is not defined
>>> labels_freqs
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'labels_freqs' is not defined
>>> label_freqs
array([4, 5, 1])
>>> kwds = {}    
>>> kwds["metric"] = metric
>>> reduce_func = functools.partial(
...     _silhouette_reduce, labels = labels, label_freqs = label_freqs
... )
>>> metric
'euclidean'
>>> metric='precomputed'
>>> kwds = {} 
>>> kwds["metric"] = metric
>>> reduce_func = functools.partial(
...     _silhouette_reduce, labels = labels, label_freqs = label_freqs
... )
>>> metric
'precomputed'
>>> arraybands_sel= matrix_dist_test

>>> results = zip(*pairwise_distances_chunked(arraybands_sel, reduce_func=reduce_func, **kwds))
>>> results
<zip object at 0x1731526c0>
>>> intra_clust_dists, inter_clust_dists = results
>>> intra_clust_dists
(array([0.89941406, 0.29980469, 0.80078125, 0.        , 0.29980469,
       0.29980469, 0.40039062, 0.80078125, 0.40039062, 0.40039062]),)
>>> inter_clust_dists
(array([0.47998047, 0.74003906, 0.74987793, 1.        , 0.74003906,
       0.74003906, 0.62512207, 0.74987793, 0.62512207, 0.62512207]),)
>>> n= n_opt_ski_ms
>>> n
6

t1=time.time()
f_ski_ms_test3["silhouette"] = (f_ski_ms_test3["inter"] - f_ski_ms_test3["intra"]) / np.maximum(f_ski_ms_test3["inter"], f_ski_ms_test3["intra"])
t2=time.time()
t2-t1


##################### 20241204 re-testando a criacao dos sdfts, spark morrendo
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
nome_prog = "spk_gen_df_ToPCA_nb_term_save_sdft"
print (f'{a}: ######## INICIO {nome_prog} ##########')

#para o terminal:
save_etapas_dir = base_dir + 'data/tmp2/'
tif_dir = base_dir + 'data/Cassio/S2-16D_V2_012014_20220728_/'
log_dir = base_dir + 'code/logs/'
process_time_dir = base_dir + 'data/tmp2/'
name_img = 'S2-16D_V2_012014'
sh_print = True

t = datetime.datetime.now().strftime('%Y%m%d_%H%M_')
nome_log = t + nome_prog.split('.')[0]+'.log'
nome_log
logger = cria_logger(log_dir, nome_log)
logger.info(f'######## INICIO {nome_prog} ##########')
logger.info(f'args: sd={save_etapas_dir} td={tif_dir} ld={log_dir} i={name_img} pd={process_time_dir} sp={sh_print}')

spark_local_dir = spark.conf.get("spark.local.dir")

# List all configurations
conf_list = spark.sparkContext.getConf().getAll()

# Print all configurations
logger.info(f'SPARK configurarion')
logger.info(f'defaultParallelism: {spark.sparkContext.defaultParallelism}')
logger.info(f'spark.sql.files.openCostInBytes: {spark.conf.get("spark.sql.files.openCostInBytes")}')
for conf in conf_list:
    logger.info(f"{conf[0]}: {conf[1]}")

band_tile_img_files = list_files_to_read(tif_dir, name_img)
bands, dates = get_bandsDates(band_tile_img_files, tile=1)
cur_dir = os.getcwd()
a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
info = f'1. bands: {bands},\ndates: {dates}'
print (f'{a}: {info}') if sh_print else None
logger.info(info)

from functions.functions_pca import load_image_files3, check_perc_nan
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

dates = ['20220728', '20220829', '20220830', '20220901', '20220902', '20220903', '20220904']
band_img_files = band_tile_img_files
i=0 

###### o que está no for
t = dates[3]
t

band_img_file_to_load = [x for x in band_img_files if t in x.split('/')[-1]]
image_band_dic = {}
pos = -1
t1=time.time()
image_band_dic = load_image_files3(band_img_file_to_load, pos=pos)
t2=time.time()
print (f'Tempo de load dos tiffs: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')
bands = image_band_dic.keys()
print (bands) if sh_print else None #, band_img_file_to_load) 

t1=time.time()
perc_day_nan= check_perc_nan(image_band_dic, logger)
t2=time.time()

t2-t1

t1=time.time()
img_bands = []
img_bands = np.dstack([image_band_dic[x] for x in bands])
t2=time.time()
img_bands.shape

del image_band_dic
gc.collect()

cols_name = [x+'_'+t for x in bands]
logger.info(f'Columns names: {cols_name}')
print (f'Tempo para gerar img_bands of {t}: {t2-t1:.2f}s, {(t2-t1)/60:.2f}m')

t1 = time.time()
values  = gen_arr_from_img_band_wCoords(img_bands, cols_name, ar_pos=0, sp=0, sh_print=0, logger=logger)
t2=time.time()

values.shape

(t2-t1)/60
del img_bands
gc.collect()

schema = StructType([StructField(x, IntegerType(), True) for x in ['coords_0','coords_1']+cols_name])

t1=time.time()
sdft = spark.createDataFrame(values, schema=schema)
t2=time.time()

t2-t1
(t2-t1)/60

sdft.persist(StorageLevel.DISK_ONLY)
t3 = time.time()

t3-t2
del values
gc.collect()

ar_pos=0
t1=time.time()
if i==0:   
    window_spec = Window.orderBy('coords_0','coords_1')    
    # Add a sequential index column
    sdft = sdft.withColumn("orig_index", row_number().over(window_spec))
    # df_toPCA = sdft #.copy()    
    i+=1
    t2=time.time()
    info = f'Tempo sort coords e para adicionar orig_index  {t} : {t2-t1:.2f}'
    print (info)  if sh_print else None

a = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
a

t1=time.time()
sdfPath = save_etapas_dir + "spark_sdft_"+ t
print (f'\n***** dir:  {sdfPath} em parquet') if sh_print else None


t1=time.time()
num_partitions = 10  # Tente aumentar o número de partições
sdft.repartition(num_partitions).write.parquet(sdfPath, mode='overwrite')
t2=time.time()

t2-t1
(t2-t1)/60

sdft.unpersist() #20241203
spark.catalog.clearCache()
del sdft
gc.collect()
spark._jvm.System.gc() #20241204

