''''
Created:17/06/2024
from notebooks/notebooks/Test_PCA_img_full_working-Copy1.ipynb
'''
#%%
#### Imports
#from dask.distributed import Client
#from sklearn.preprocessing import scale
from dask_ml.decomposition import PCA
from dask_ml.preprocessing import StandardScaler
from functions_pca import *
#### Function

#### Functions get_bandsDates and get_dates_lower

#%%

#### PROGRAM
# from dask.distributed import Client
# client = Client(processes=False)
# #%%
# client.close()
#%%
# for tile image
sh_print_n1 = 1
sh_print_n2 = 1
SAVE_TO_PARQUET = 0
read_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/Cassio/S2-16D_V2_012014_20220728_/'
name_img = 'S2-16D_V2_012014' # 'S2-16D_V2_012014_20220728_'
all_bands = ['B04', 'B03', 'B02', 'B08', 'EVI', 'NDVI']

#%%
band_tile_img_files = list_files_to_read(read_dir, name_img)
print (f'len(band_tile_img_files): {len(band_tile_img_files)}')

bands, dates= get_bandsDates(band_tile_img_files, tile=1)
print (bands, dates[-3:], len(dates))
cur_dir = os.getcwd()
print (f'{cur_dir}')
#%%
df_toPCA = pd.DataFrame()
max_bt = {}
min_bt = {}
read_cols=[]
#%%
if SAVE_TO_PARQUET:
    #gen df from matrix bands files
    t0=time.time()
    #df_toPCA2, max_bt, min_bt = gen_dfToPCA_filter(dates, band_img_files, sh_print) #ex image
    df_toPCA, max_bt, min_bt = gen_dfToPCA_filter(dates, band_tile_img_files, dask=1, ar_pos=0, sh_print=1)
    #band_tile_img_files
    t1=time.time()
    print (f'gen df_toPCA: {t1-t0}s {(t1-t0)/60} min') if sh_print_n1 else None #995.8754930496216 #2786.018984079361
    print (df_toPCA.shape)  if sh_print_n2 else None
    print (max_bt, min_bt)  if sh_print_n2 else None
    print ("df_toPCA\n", df_toPCA.head(2))  if sh_print_n2 else None
    # save df to parquet file
    save_path = cur_dir + '/data/parquet_npart111514/'
    obj_dic = {}
    obj_dic = {
        "max_bt": max_bt, 
        "min_bt": min_bt,
        #"segments_slic_sel": segments_slic_sel,
        }
    #save_path+name_img+'_'+dates[-1]
    file_to_save = save_path+name_img+'_max_min_bt_npart111514.pkl'
    print (f'saving max min bt to {file_to_save}') if sh_print_n2 else None
    save_to_pickle(obj_dic, file_to_save)
    
    #f_name = name_img+'_df_toPCA_full.parquet'
    f_name = name_img+'_df_toPCA_full_npart111514'
    print (f'saving df to {save_path+f_name}') if sh_print_n2 else None
    t1-time.time()
    df_toPCA.to_parquet(save_path+f_name, engine='pyarrow')
    t2=time.time()
    print (f'time to save df to parquet file: {t2-t1}s, {(t2-t1)/60} m') if sh_print_n1 else None

else: #read parquet files
    #### read parquet files
    read_parquet = cur_dir + '/data/parquet_npart111514/'
    #f_name = name_img+'_df_toPCA_full.parquet'
    f_name = name_img+'_df_toPCA_full_npart111514'
    
    t1=time.time()
    #read_cols=['orig_index',  'coords_0',  'coords_1']#preciso ler todas as cols
                                                    # para fazer o drop dos nans
    if read_cols:
        df_toPCA = dd.read_parquet(read_parquet+f_name, engine='pyarrow', columns=read_cols)
    else:
        df_toPCA = dd.read_parquet(read_parquet+f_name, engine='pyarrow')
    t2=time.time()
    print (f'read time parquet files {(t2-t1)} s {(t2-t1)/60} m') if sh_print_n1 else None
    print ("df_toPCA\n", df_toPCA.head(2))  if sh_print_n2 else None
#%%
#### Remover os nans (-9999 e -32768) do df 
#df_toPCA2.replace({-9999: np.nan, -32768: np.nan}, inplace=True)
df_toPCA = df_toPCA.mask(df_toPCA == -9999, np.nan)
df_toPCA = df_toPCA.mask(df_toPCA == -32768, np.nan)

print (f'{df_toPCA.tail(2)}') if sh_print_n2 else None
#%%
filter_nans=0
if filter_nans:
    # Filter rows with NaN values to save 
    t1=time.time()
    rows_with_nan = df_toPCA[df_toPCA.isnull().any(axis=1)]
    #rows_with_nan.shape, 121898*100/1440000 = 8.46513888888889%
    #rows_with_nan2
    t2=time.time()
    print (f'tempo para filtrar nan rows : {t2-t1}') if sh_print_n1 else None
    ## #%%
    t1=time.time()
    columns_with_nan = df_toPCA.columns[df_toPCA.isnull().any()]
    t2=time.time()
    print (f'tempo para filtrar nan columns : {t2-t1}') if sh_print_n1 else None
    # #%%
    t1=time.time()
    df_with_nan = rows_with_nan[columns_with_nan]
    #df_with_nan2#.loc[:6]
    index_nan=df_with_nan.index
    t2=time.time()
    print (f'tempo para fazer df nan rows : {t2-t1}') if sh_print_n1 else None
# pegar as colunas com as coordenadas dos pixels 
# antes de remover as linhas com nans e nomear o index
cols_coords= df_toPCA.columns[1:3]
df_toPCA_coords = df_toPCA[cols_coords].copy()
df_toPCA_coords.index = df_toPCA_coords.index.rename('index_orig')
print ("df_toPCA_coords\n", df_toPCA_coords.tail(2))  if sh_print_n2 else None
#df_toPCA_coords['index_orig'] = df_toPCA.index

#%%
#Faz o drop das linhas que sobraram com Nans
t1=time.time()
df_toPCA = df_toPCA.dropna()
t2=time.time()
print (f'tempo para fazer dropna: {t2-t1}') if sh_print_n1 else None
#print (df_toPCA.tail(2)) if sh_print_n2 else None
#%%
do_PCA=0 #0 se for ler o pca dos aqrquivos, 1 se for fazer
if do_PCA:

    t1=time.time()
    colB_ini = 3
    #df_toPCA2 = norm_columns(df_toPCA, colB_ini, min_bt, max_bt)
    cols_to_pca=df_toPCA.columns[colB_ini:]
    dX = df_toPCA[cols_to_pca].to_dask_array()
#%%
test1=False
if test1:
    t1=time.time()
    #scale nao está funcionando...
    dX_scaled = scale(dX.astype('float64'), axis=0, with_std=False)

    t2=time.time()
    print ("time to scale nans: ",t2-t1)
#del dX
#%%
test2=False
if test2:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(with_std=False)
    
    t1=time.time()
    dX_scaled2 = scaler.fit_transform(dX.astype('float64'))
    t2=time.time()

#%%
if do_PCA:
    scaler = StandardScaler(with_std=False)
    print (scaler)
    t1 = time.time()
    dX_scaled = scaler.fit_transform(dX)
    t2 = time.time()
    print (f'tempo para fazer o scale no dX: {t2-t1}')
#%%
#run pca with dx
# func_random = 1 usar a funcao randomized svd do georges 
# func_random = 2 usar o random do pca dask
func_random = 3 
if func_random == 1:  #run randomized svd dp georges
    #usando a funcao do georges
    t1=time.time()
    #u_p, s_p, vh_p
    U, s, Vh = randomized_svd_g(dX_scaled, K=3,oversampling=5, power_iterations=2)
    t2=time.time()
    print (f'tempo para randomized_svd: {t2-t1}') if sh_print_n1 else None

    ### #%% # Gets the principal components of pca
    t1=time.time()
    U_fit =U
    U_fit *= s
    t2=time.time()
    print (f'tempo para U_fit: {t2-t1}') if sh_print_n1 else None
    cols=['c1', 'c2', 'c3']
    pca_components_df = dd.io.from_dask_array(U_fit, columns=cols)
    print (pca_omponents_df.head(2)) if sh_print_n2 else None
#%%
n_compon = 3
if func_random == 2:
    #oversampling=5, power_iterations=2
    t1=time.time()
    random_pca = PCA(n_components=n_compon, svd_solver='randomized', \
                        random_state=999, iterated_power=2,  whiten=False)
    t2=time.time()
    print (f'tempo para configurar o random pca do dask: {t2-t1}') if sh_print_n1 else None
    #%%
    # Gets the principal components of pca
    #principalComponents = pca.fit_transform(dX)
    #principalComponents_df = pd.DataFrame(principalComponents).loc[:,:5]
    t1=time.time()
    principalComponents = random_pca.fit_transform(dX)
    t2=time.time()
    print (f'tempo para obter principalComponents do random pca do dask: {t2-t1}s {(t2-t1)/60} m') if sh_print_n1 else None
    #principalComponents_df = pd.DataFrame(principalComponents)
    cols=['c1', 'c2', 'c3']
    pca_components_df = dd.io.from_dask_array(principalComponents, columns=cols)
    t3=time.time()
    print (f'tempo para gerar df das principalComponents do random pca do dask: {t3-t2}') if sh_print_n1 else None
    print ("pca_components_df\n", pca_components_df.tail(2))  if sh_print_n2 else None

SAVE_PCA_DF=0
if SAVE_PCA_DF:
    save_path = cur_dir + '/data/parquet_npart111514/'
    f_name = name_img+'_pca_df_npart111514'
    print (f'saving pca_components df to {save_path+f_name}') if sh_print_n2 else None
    t1-time.time()
    pca_components_df.to_parquet(save_path+f_name, engine='pyarrow')
    t2=time.time()
    print (f'time to save pca_components df to parquet file: {t2-t1} s, {(t2-t1)/60} m') if sh_print_n1 else None

else: #read pca parquet files
    #### read parquet files
    read_parquet = cur_dir + '/data/parquet_npart111514/'
    #f_name = name_img+'_df_toPCA_full.parquet'
    f_name = name_img+'_pca_df_npart111514'

    t1=time.time()
    pca_components_df = dd.read_parquet(read_parquet+f_name, 
                                        engine='pyarrow')
    t2=time.time()
    print (f'read time pca parquet files {(t2-t1)} s {(t2-t1)/60} m') if sh_print_n1 else None
    print ("pca_components_df read\n", pca_components_df.tail(2))  if sh_print_n2 else None


#%%
# pegar as colunas com as coordenadas dos pixels e o index do df_PCA
# que teve as linhas com nans removidas
# cols_coords= df_toPCA.columns[1:3]
# df_toPCA_coords = df_toPCA[cols_coords].copy()
# #df_toPCA_coords['index_orig'] = df_toPCA.index
# df_toPCA_coords.index = df_toPCA_coords.index.rename('index_orig')

# adicionar o index original do df_toPCA ao pca_components_df
#pca_components_df=principalComponents_df.copy()
t1=time.time()
# divisions=list(pca_components_df.divisions)
# divisions=divisions.sort()
divisions=[]
divisions=pca_components_df.divisions
#divisions=np.asarray(divisions, dtype=np.int64)
divisions=list(divisions)
print (f'len divisions: {len(divisions)}')
pca_components_df['index_orig'] = df_toPCA['orig_index'].values #df_toPCA.index desta forma nao funcionou
pca_components_df = pca_components_df.set_index('index_orig', sorted=False)#, divisions=divisions)
t2=time.time()
print (f'time to set index in pca_components_df: {t2-t1} s {(t2-t1)/60}') if sh_print_n1 else None
print ('pca_components df after insert index original',pca_components_df.tail(2))
del df_toPCA
print ("deleted df_toPCA") if sh_print_n2 else None
#%%
#cria df com as coordendas e as 3 componentes para gerar uma matriz que será passada no slic
#usando o index como link
t1=time.time()
#pca_components_df=pd.merge(df_toPCA_coords, pca_components_df,left_index=True, right_index=True)
pca_components_df = df_toPCA_coords.merge(pca_components_df, how='left')

t2=time.time()
#pca_components_df = pca_components_df.rename(columns={0:'c0', 1:'c1', 2:'c2', 3:'c3',4:'c4', 5:'c5'})
print (f'time to gen df_pca: {t2-t1}') if sh_print_n1 else None#, t4-t3)
print (pca_components_df.tail(2))#,props_df_sel_ts_pca.shape
del df_toPCA_coords
print ("deleted df_toPCA_coords") if sh_print_n2 else None

#salvar este df
SAVE_PCA_DF=1
if SAVE_PCA_DF:
    save_path = cur_dir + '/data/parquet_npart111514/'
    f_name = name_img+'_pca_df_index_npart111514'
    print (f'saving pca df to {save_path+f_name}') if sh_print_n2 else None
    t1-time.time()
    pca_components_df.to_parquet(save_path+f_name, engine='pyarrow')
    t2=time.time()
    print (f'time to save pca_components_df with index to parquet file: {t2-t1} s, {(t2-t1)/60} m') if sh_print_n1 else None


img_comp_dic={}
comps_sel = list(pca_components_df.columns[2:])
print (comps_sel)
pca_components_df = pca_components_df.categorize(columns=['coords_1'])
# %%
#criacao de matrix com as n componentes PCA por posicao da matriz
#cria uma matriz para cada componente e depois junta com dstack apenas as 3 primeiras uma img_pca
# nao funcionou juntar as colunas das componentes num valor [c1,c2,c3] e gerar a matriz
# o valor gerado fica com lista , nao como array, olhar os testes.
t1=time.time()
for c in comps_sel:
    img_comp_dic[c] = pca_components_df.pivot_table(index='coords_0', columns='coords_1', values=c).values

img_pca = np.dstack((img_comp_dic[comps_sel[0]], img_comp_dic[comps_sel[1]], img_comp_dic[comps_sel[2]]))

# matrix_c1 = pca_components_df.pivot(index='coords_0', columns='coords_1', values='c0').values    
# matrix_c2 = pca_components_df.pivot(index='coords_0', columns='coords_1', values='c1').values  
# matrix_c3 = pca_components_df.pivot(index='coords_0', columns='coords_1', values='c2').values   
# img_pca = np.dstack((matrix_c1, matrix_c2, matrix_c3))
#0.42520880699157715
t2=time.time()
print (f'time to gen img_pca: {t2-t1}') if sh_print_n1 else None

#%%
#salvar img_pca
for c in comps_sel:
    save_path = cur_dir + '/data/parquet_npart111514/'
    f_name = name_img+'_img_pca_npart111514_'+c
    print (f'saving img_pca to {save_path+f_name}') if sh_print_n2 else None
    t1-time.time()
    img_comp_dic[c].to_parquet(save_path+f_name, engine='pyarrow')
    t2=time.time()
    print (f'time to save img pca {c} to parquet file: {t2-t1}s, {(t2-t1)/60} m') if sh_print_n1 else None

#fazer snic
