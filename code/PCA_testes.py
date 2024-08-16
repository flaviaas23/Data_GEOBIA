'''
Programa para testes
'''

''''
teste de merge com pca_components e df_toPCA
09/07/2024
20240808: codigo testado e funcionando no log/20240717_saida_terminal_windows
'''

#### Imports
#from dask.distributed import Client
#from sklearn.preprocessing import scale
import gc
import datetime
from dask_ml.decomposition import PCA
from dask_ml.preprocessing import StandardScaler
from dask.diagnostics import ProgressBar
from dask.array import from_zarr

ProgressBar().register()
from code.functions.functions_pca import *
from functions_segmentation import *
seed = random.seed(999) # para gerar sempre com a mesma seed

ti=time.time()
sh_print_n1=1
sh_print_n2=1
# Read df_toPCA
cur_dir = os.getcwd()
print (f'{cur_dir}')
read_parquet = cur_dir + '/data/parquet_npart111514/'
name_img = 'S2-16D_V2_012014'
f_name = name_img+'_df_toPCA_full_npart111514'

# 1. READ df_toPCA
print ("1. ******** READ df_toPCA\n")
t1=time.time()
#read_cols=['orig_index',  'coords_0',  'coords_1']#preciso ler todas as cols
                                                    # para fazer o drop dos nans
df_toPCA = dd.read_parquet(read_parquet+f_name, engine='pyarrow', calculate_divisions=True)#, chunksize="100 MiB") #gather_statistics=Truechunksize="5 MiB",
t2=time.time()
print (f'1. read time parquet files: {(t2-t1)} s {(t2-t1)/60} m') if sh_print_n1 else None
#print ("df_toPCA\n", df_toPCA.tail(2))  if sh_print_n2 else None
print (f'##### df_toPCA READ npartition = {df_toPCA.npartitions}')#\n df_toPCA tail = {df_toPCA.tail(2)}')

#2. Repartition
print ("\n2. ******** REPARTITION df_toPCA")
t1=time.time()
df_toPCA = df_toPCA.repartition(partition_size="100MB")
t2=time.time()
print (f'\n   df_toPCA tempo repartition  = {t2-t1}s, {(t2-t1)/60}m')
#print (f'df_toPCA n_part: {df_toPCA.npartitions}') #63

#3. Substituir os nans (-9999 e -32768) do df por np.nan
print ("3. ******** Substituir -9999/-32768 por NANs")
t1 = time.time()
df_toPCA = df_toPCA.mask(df_toPCA == -9999, np.nan)
df_toPCA = df_toPCA.mask(df_toPCA == -32768, np.nan)
t2 = time.time()
print (f'   .1 tempo para replace -9999 e -32768 = {t2-t1}s')

#3.1 Gerar df_Coords
print ("\n3.1 ******** Gerar df_toPCA_coords ******\n")
t1=time.time()
df_toPCA['coords_1'] = df_toPCA['coords_1'].astype('category')
t2=time.time()
print (f'   .1: df_toPCA category: {t2-t1}, {df_toPCA.coords_1.cat.known}')
df_toPCA_coords = gen_dfCoords(df_toPCA)
#print (f'.2 df_toPCA_coords category after astype int64, category and set_category: {t2-t1}')
print (f'   .3: df_toPCA_coords category {df_toPCA_coords.coords_1.cat.known}') #as_known()
#print ("df_toPCA_coords\n", df_toPCA_coords.tail(2))  if sh_print_n2 else None
#print ("   .4: df_toPCA_coords types\n", df_toPCA_coords.dtypes)  if sh_print_n2 else None
#df
print (f'cols df_toPCA_coords: {df_toPCA_coords.columns}')

#4. Faz o drop das linhas com Nans
print ("\n4. ******** Drop NA")
t1=time.time()
df_toPCA = df_toPCA.dropna()
t2=time.time()
print (f'\n   .1 tempo para fazer dropna no df_toPCA: {t2-t1}s') if sh_print_n1 else None

#5. Read pca_components_df
print ("\n5. ********** READ PCA_COMPONENTS\n")
f_name = name_img+'_pca_df_npart111514'
t1=time.time()
pca_components_df2 = dd.read_parquet(read_parquet+f_name, 
                                    engine='pyarrow', calculate_divisions=True)
t2=time.time()
print (f'\n   .1 read time pca parquet files {(t2-t1)} s {(t2-t1)/60} m') if sh_print_n1 else None
#print ("pca_components_df read\n", pca_components_df2.tail(2))  if sh_print_n2 else None
#print (f'   .2 type pca_components_df2 antes repartition {type(pca_components_df2)}')
#6. Repartition pca_components_df2
print ("\n6. ******** Repartition pca_components_df") 
t1=time.time()
npart=df_toPCA.npartitions
pca_components_df2 = pca_components_df2.repartition(npartitions=npart)#partition_size="100MB")
t2=time.time()
print (f'\n   .1 tempo repartition pca_components_df2 loc = {t2-t1}s, {(t2-t1)/60}m')

#7. Compara dfs
print ("\n7. ******** compara DFs ********\n")
print (f'\n   .2 type pca_components_df2 apos repartition {type(pca_components_df2)}')
print (f'   .3 type df_toPCA {type(df_toPCA)}')
print (f'   .4 index df_toPCA:\n{df_toPCA.index}')

# print (f'rows df_toPCA: {df_toPCA.index.pipe(len)}') #rows df_toPCA: 111456171
# print (f'rows pca_components_df2: {pca_components_df2.index.pipe(len)}') #rows pca_components_df2: 111456171

# print (f'   .4 dtypes df_toPCA:\n{df_toPCA.dtypes}')
# print (f'\n   .5 dtypes pca_components:\n{pca_components_df2.dtypes} ')

#8. cria pca_components com o index do df_toPCA
print ("\n8. ********* Cria pca df com index do df_toPCA *******")
#opcao 1 perde o indice
t1=time.time()
n_compon = 3
comps_sel = ['c'+str(x) for x in range(1,n_compon+1)]
#print (comps_sel)
#opcao com copy
pca_comps_df_index = df_toPCA[['orig_index']].copy()
n_part = df_toPCA.npartitions
chunks_pca = pca_comps_df_index.to_dask_array(lengths=True)
for c in comps_sel:
    #pca_comps_df_index[c] = pca_components_df2[c].values
    
    colc = pca_components_df2[c].to_dask_array(lengths=True)
    #linha abaixo nao está funcionando
    pca_comps_df_index[c] = da.from_array(colc, chunks=chunks_pca)

#orig_index_da=da.from_array(orig_index, chunks=pca_comps_df_index.to_dask_array(lengths=True).chunksize[0])
#'''
#opcoes abaixo tb deram o mesmo resultado do tail, nao funcionou
#pca_comps_df_index = pca_components_df2.map_partitions(lambda x: x).values.to_dask_dataframe(index=df_toPCA.index)
#pca_comps_df_index = pca_components_df2.values.to_dask_dataframe(index=df_toPCA.index)
t2=time.time()
print (f'\n   .1 Tempo para gerar df_comp_index: {t2-t1}')
print (f'   .2 index pca_comps_df_index:\n{pca_comps_df_index.index}')
print (f'   .3 n partitions pca_comps_df_index: {pca_comps_df_index.npartitions}')
#print (f'   .4 tail da primeira particao:\n{pca_comps_df_index.partitions[0].compute()}')#.tail(2)}')
print (f'   .5 columns do pca comps index:\n {pca_comps_df_index.columns}')
#print (f'   .6 tail do pca comps index:\n {pca_comps_df_index.tail(2)}') #nao funciona
'''
t1=time.time()
pca_comps_df_index.repartition(npartitions=npart)#partition_size="100MB")
t2=time.time()
print (f'   .2 Tempo para repartition df_comp_index: {t2-t1}')
#'''

#'''
''' 
#opcao 2 tb perde o indice
cols_drop = df_toPCA.columns[1:]
df_toPCA = df_toPCA.drop(columns=cols_drop)

print (df_toPCA.columns)
#print (f'rows df_toPCA: {df_toPCA.index.pipe(len)}')
for c in comps_sel:
    comp = pca_components_df2[c].to_dask_array()
    df_toPCA[c] = comp #pca_components_df2[c].values
print (df_toPCA.columns)
#print (f'rows df_toPCA: {df_toPCA.index.pipe(len)}')
#'''

#9. Fazer merge pca com index com df_coords
print ("\n9. ********* Merge pca comp index com df_toPCACoords *********\n")
t1=time.time()
pca_components_df_merge= df_toPCA_coords.merge(pca_comps_df_index, how='left',on='orig_index', shuffle="tasks") #mudei de join para merge 9/7/24 
#pca_components_df_merge=df_toPCA_coords.reset_index().set_index("index").merge(pca_comps_df_index.reset_index().set_index("orig_index"), left_index=True, right_index=True,how="outer").compute()
#pca_components_df_merge = dd.concat([df_toPCA_coords, pca_components_df], axis=1)
t2=time.time()
ddf_merge = ddf_merge.sort_values(['coords_0', 'coords_1'])
print (f'\n   .1 Tempo para gerar df_components_merge: {t2-t1}')
print (f'   .2 df_components_merge category {pca_components_df_merge.coords_1.cat.known}')

t1=time.time()
pca_components_df_merge.repartition(npartitions=npart)#partition_size="100MB")
t2=time.time()
print (f'   .3 Tempo para repartition df_components_merge: {t2-t1}')

#10.  Gerar imagens com pivot table 
print ("\n10. ********* pivot do df merge *********\n")
t1=time.time()
img_comp_dic = {}
img_comp_df = {}
for c in comps_sel:
    tp1=time.time()
    #img_comp_dic[c] = pca_components_df_merge.pivot_table(index='coords_0', columns='coords_1', values=c).values
    img_comp_df[c]= pca_components_df_merge.pivot_table(index='coords_0', columns='coords_1', values=c)
    #img_comp_dic[c] = img_comp_df.to_dask_array(lengths=True)
    img_comp_dic[c] = img_comp_df.values
    #img_comp_v= pca_components_df_merge.pivot_table(index='coords_0', columns='coords_1', values=c)
    tp2=time.time()
    print (f'\n   .1 tempo para pivot de {c}: {tp2-tp1}')
    print (f'\n   .2 type img_comp_dic {c}: {type(img_comp_dic[c])}')
#'''
tf=time.time()
print (f'\n   .1 tempo programa para gerar imagens: {(tf-ti)/60}m')

#11. Salvar imagens
print ("\n11. ********** Salvar imagens para cada component *********\n")
save_path = cur_dir + '/data/parquet_npart111514/'
t1=time.time()
comps= list(img_comp_dic.keys())
t2=time.time()
print (f'\n   .1 tempo para keys: {t2-t1} comps: {comps}')


#opcao para salvar em parquet
print ("\n   .2 ********** salvando imagens em PARQUET **********")
t1=time.time()   
c=comps[0]
cols=img_comp_df[c].columns
cols_dic={n:str(n) for n in cols}
img_comp_df[c] = img_comp_df[c].rename(columns=cols_dic)
t2=time.time()
t_rename_cols = t2-t1

t1=time.time()
img_comp_df[c].to_parquet(save_path+f_name+'df_parquet', engine='pyarrow')
t2=time.time()
t_to_parquet = t2-t1
t_save_parquet = t_rename_cols + t_to_parquet

print (f'   .3 tempo para rename cols: {t_rename_cols}') 
print (f'   .4 tempo para to_parquet: {t_to_parquet}')    
print (f'   .5 tempo total para salvar como df em parquet: {t_save_parquet}s, {t_save_parquet/60}m')
#'''

#opcao para salvar em zarr
c=comps[0]
f_name = name_img+'_img_pca_npart111514_'+str(c)+'_zarr'
t1=time.time()
#img_comp_dic[c].compute_chunk_sizes() 
t2=time.time()
print (f'   2. tempo to compute chunk sizes of {c}: {t2-t1}')
print (f'   3. saving img_pca {c} to {save_path+f_name}') if sh_print_n2 else None

t1=time.time()
#img_comp_dic[c].to_zarr(save_path+f_name)
t2=time.time()

#
#print (df_toPCA.tail(2))
#print (f'rows pca_components_df2: {df_toPCA.index.pipe(len)}') #rows pca_components_df2: 111456171
#print (f'rows pca_components_df2: {pca_components_df.index.pipe(len)}') #rows pca_components_df2: 111456171
del df_toPCA, df_toPCA_coords, 
del pca_components_df2, pca_comps_df_index, pca_components_df_merge
#'''


del img_comp_df, img_comp_dic