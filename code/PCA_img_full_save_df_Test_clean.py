''''
Created:17/06/2024
from notebooks/notebooks/Test_PCA_img_full_working-Copy1.ipynb
save in parquet files the PCA df to be used to generate the pca images
'''
#%%

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

#### Function

#### Functions get_bandsDates and get_dates_lower

#%%

#### PROGRAM
#melhor nao usar o client , consome recurso e termina matando o programa
# from dask.distributed import Client
# print ('start the client/n')
# client = Client(processes=False)  #Client(processes=False)

# print ("###### Client: ######\n", client) 
# #%%
# client.close()
# quit()
#%%
# for tile image
print (" ********************************* inicio PROGRAM *********************************")
t_ini=time.time()
sh_print_n1 = 1
sh_print_n2 = 1
SAVE_TO_PARQUET = 0
read_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/Cassio/S2-16D_V2_012014_20220728_/'
name_img = 'S2-16D_V2_012014' # 'S2-16D_V2_012014_20220728_'
all_bands = ['B04', 'B03', 'B02', 'B08', 'EVI', 'NDVI']

#%%
band_tile_img_files = list_files_to_read(read_dir, name_img)
print (f'len(band_tile_img_files): {len(band_tile_img_files)}') if sh_print_n2 else None

bands, dates= get_bandsDates(band_tile_img_files, tile=1)
print (bands, dates[-3:], len(dates), '\n') if sh_print_n2 else None
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
        df_toPCA = dd.read_parquet(read_parquet+f_name, engine='pyarrow', calculate_divisions=True)#, chunksize="100 MiB") #gather_statistics=Truechunksize="5 MiB",
    t2=time.time()
    print (f'##### read time parquet files: {(t2-t1)} s {(t2-t1)/60} m') if sh_print_n1 else None
    #print ("df_toPCA\n", df_toPCA.head(2))  if sh_print_n2 else None
    print (f'##### df_toPCA READ npartition = {df_toPCA.npartitions}')#\n df_toPCA tail = {df_toPCA.tail(2)}')

nrows=0
if nrows:
    t1=time.time()    
    nrows=80000000 # 100000000#0000 #111514  111.513.599
    df_toPCA = df_toPCA.loc[:nrows]
    t2=time.time()
    print (f'##### tempo loc = {t2-t1}\t type from loc {type(df_toPCA)}') #vem sem category

t1=time.time()
df_toPCA = df_toPCA.repartition(partition_size="100MB")
t2=time.time()
print (f'\ndf_toPCA tempo repartition  = {t2-t1}')
print (f'df_toPCA n_part: {df_toPCA.npartitions}')
   
if 0: ######testes com tamanho de particoes 
    ######testes com tamanho de particoes 
    nrows=80000000#80000000 #111514  111.513.599
    t1=time.time()
    df_toPCA_head =df_toPCA.head(100)
    t2=time.time()
    print (f'tempo head = {t2-t1}')

    df_toPCA_loc = df_toPCA#.loc[:nrows]
    t2=time.time()
    print (f'tempo loc = {t2-t1}\ntype df fromhead {type(df_toPCA_head)}, type from loc {type(df_toPCA_loc)}')

    #print ("\n*************************** Tempos SEM Repartition ***************************\n")
    test=2
    # 1
    if test==1:
        t1=time.time()
        df_toPCA_loc = df_toPCA_loc.astype('category').persist().categorize(columns=['coords_1'])
        t2=time.time()
        print (f'test1 tempo para category: {t2-t1}')
        t1=time.time()
        df_toPCA_loc = df_toPCA_loc.astype('category').categorize(columns=['coords_1'])
        t2=time.time()
        print (f'test1 tempo para categorize sem persist: {t2-t1}')  #melhor sem persist
    # 2:
    test=4
    if test==2: #melhor desempenho
        t1=time.time()
        #print (f'\ncat.known: {df_toPCA_loc.coords_1.cat.known}')  #nao tem esta inofrmacao antes do astype
        df_toPCA_loc['coords_1'] = df_toPCA_loc['coords_1'].astype('category')
        t2=time.time()
        print (f'test2 tempo para astype category: {t2-t1} \ntest2 cat.known: {df_toPCA_loc.coords_1.cat.known}')
        df_toPCA_loc['coords_1'] = df_toPCA_loc['coords_1'].cat.set_categories(df_toPCA_loc['coords_1'].head(1).cat.categories)
        t3=time.time()

        print (f'test2 tempo para set category: {t3-t1}')
        print (f'test2 cat.known: {df_toPCA_loc.coords_1.cat.known}\n ')
        print (f'test2 categories: {df_toPCA_loc.coords_1.head(1).cat.categories}')
    # 3:
    test=4
    if test==3:  #pior desempenho
        t1=time.time()
        df_toPCA_loc = df_toPCA_loc.astype({'coords_1': "category"})
        df_toPCA_loc = df_toPCA_loc.persist()
        df_toPCA_loc = df_toPCA_loc.categorize(columns=['coords_1'])
        t2=time.time()

        print (f'test3 tempo para category: {t2-t1}')

    if test==2:
        t1=time.time()
        img_comp_dic = df_toPCA_loc.pivot_table(index='coords_0', columns='coords_1', values='B08_20220728').values
        t2=time.time()
        print (f'test2 tempo para pivot_table SEM repartition: {t2-t1}')

    #### Fazendo o repartition para ver o tempo to categorize
    print ("\n*************************** Tempos COM Repartition ***************************\n")
    t1=time.time()
    df_toPCA_loc = df_toPCA_loc.repartition(partition_size="100MB")
    t2=time.time()
    print (f'tempo repartition df_toPCA_loc = {t2-t1}')

    divisions=[]
    divisions = df_toPCA_loc.divisions
    #divisions=np.asarray(divisions, dtype=np.int64)
    df_toPCA_loc_div = list(divisions)
    df_toPCA_loc_npart = len(df_toPCA_loc_div) - 1
    print (f'len divisions df_to_PCA_coords: {len(df_toPCA_loc_div)}, n_part: {df_toPCA_loc_npart} {df_toPCA_loc.npartitions}')
    test=2
    # 1
    if test==1:
        t1=time.time()
        df_toPCA_loc = df_toPCA_loc.astype('category').persist().categorize(columns=['coords_1'])
        t2=time.time()
        print (f'\ntest1 tempo para categorize com persist: {t2-t1}')
        t1=time.time()
        df_toPCA_loc = df_toPCA_loc.astype('category').categorize(columns=['coords_1'])
        t2=time.time()
        print (f'test1 tempo para categorize sem persist: {t2-t1}')
    # 2:
    test=2
    if test==2:
        t1=time.time()
        df_toPCA_loc['coords_1'] = df_toPCA_loc['coords_1'].astype('category')
        t2=time.time()
        print (f'test2 tempo para astype category: {t2-t1},\n')
        print (f'\ncat.known: {df_toPCA_loc.coords_1.cat.known}')
        t1=time.time()
        df_toPCA_loc['coords_1'] = df_toPCA_loc['coords_1'].cat.set_categories(df_toPCA_loc['coords_1'].head(1).cat.categories)
        t2=time.time()
        print (f'test2 tempo para set_category: {t2-t1}, {df_toPCA_loc.coords_1.cat.known}')
        print (f'cat.known: {df_toPCA_loc.coords_1.cat.known}\n ')
        #print (f'categories: {df_toPCA_loc.coords_1.head(1).cat.categories}')
    # 3:
    test=2
    if test==3:
        t1=time.time()
        df_toPCA_loc = df_toPCA_loc.astype({'coords_1': "category"})
        df_toPCA_loc = df_toPCA_loc.persist()
        df_toPCA_loc = df_toPCA_loc.categorize(columns=['coords_1'])
        t2=time.time()

        print (f'test3 tempo para category: {t2-t1}')

    t1=time.time()
    print ("\n*********** COORDS ********")
    df_toPCA_loc = df_toPCA_loc.mask(df_toPCA_loc == -9999, np.nan)
    df_toPCA_loc = df_toPCA_loc.mask(df_toPCA_loc == -32768, np.nan)

    cols_coords= df_toPCA_loc.columns[1:3]
    df_toPCA_coords = df_toPCA_loc[cols_coords].copy()
    t2=time.time()
    print (f'df_toPCA_coords cat.known: {t2-t1}, {df_toPCA_coords.coords_1.cat.known}')

    df_toPCA_coords['coords_0']= df_toPCA_coords['coords_0'].astype('int64')
    df_toPCA_coords['coords_1']= df_toPCA_coords['coords_1'].astype('int64')
    df_toPCA_coords['coords_1'] = df_toPCA_coords['coords_1'].astype('category')
    df_toPCA_coords['coords_1'] = df_toPCA_coords['coords_1'].cat.set_categories(df_toPCA_coords['coords_1'].head(1).cat.categories)
    #print (f'df_toPCA_coords category after astype int64: {t2-t1}')
    print (f'df_toPCA_coords cat.known: {df_toPCA_coords.coords_1.cat.known}')
    print ("df_toPCA_coords types\n", df_toPCA_coords.dtypes)  if sh_print_n2 else None
    print (f'df_toPCA_coords partitions: {df_toPCA_coords.npartitions}, nrows: {df_toPCA_coords.index.pipe(len)}')

    t1=time.time()
    df_toPCA_loc = df_toPCA_loc.dropna()
    t2=time.time()
    print (f'\ntempo para fazer dropna no df_toPCA: {t2-t1}') if sh_print_n1 else None
    
    print ("\n######## reading PCA COMPONENTS \n")

    read_parquet = cur_dir + '/data/parquet_npart111514/'
    #f_name = name_img+'_df_toPCA_full.parquet'
    f_name = name_img+'_pca_df_npart111514'

    t1=time.time()
    pca_components_df2 = dd.read_parquet(read_parquet+f_name, 
                                        engine='pyarrow', calculate_divisions=True)
    t2=time.time()
    print (f'\nread time pca parquet files {(t2-t1)} s {(t2-t1)/60} m') if sh_print_n1 else None
    #print ("pca_components_df read\n", pca_components_df2.tail(2))  if sh_print_n2 else None
    print (f'pca_components_df2 npartition = {pca_components_df2.npartitions}')#\n pca_components_df2 tail = {df_toPCA.tail(2)}')


    print ("###### generating pca_components_df with df_toPCA & pca_components_df calculated\n")
    
    t1=time.time()
    npart=df_toPCA.npartitions
    pca_components_df2 = pca_components_df2.repartition(npartitions=npart)#partition_size="100MB")
    t2=time.time()
    print (f'\ntempo repartition pca_components_df2  = {t2-t1}')
    print (f'n_part df_toPCA pca_components_df2 repatitioned: {npart} {pca_components_df2.npartitions}')
    print (f'nrows df_toPCA after drop: {df_toPCA_loc.index.pipe(len)}, nrows pca_components_df2:{pca_components_df2.index.pipe(len)}')
    #print (f'pca_components_df2columns {pca_components_df2.columns}')
    
    pca_components_df = df_toPCA_loc[['orig_index']].copy()
    #c='B08_20220728'
    comps_sel = ['c1','c2','c3']#['c'+str(x) for x in range(1,n_compon+1)]
    for c in comps_sel: 
        #pca_components_df['B08_20220728'] = df_toPCA['B08_20220728'].values
        pca_components_df[c] = pca_components_df2[c].values

    del pca_components_df2
    gc.collect()
    print (f"type pca_components_df: {type(pca_components_df)}")

    #print (f'npartitions pca_components_df: {pca_components_df.npartitions}')#, nrows: {pca_components_df.index.pipe(len)}')

    pca_components_df_merge= df_toPCA_coords.merge(pca_components_df, how='left')#.reset_index() #on= ['index_orig'],
    
    print (f'npartitions pca_components_df_merge: {pca_components_df_merge.npartitions}')#, nrows: {pca_components_df_merge.index.pipe(len)}')
    print (f'pca_components_df_merge cat.known: {pca_components_df_merge.coords_1.cat.known}')
    print (f'pca_components_df_merge columns: {pca_components_df_merge.columns}')

    t3=time.time()
    #img_comp_dic_df = pca_components_df_merge.pivot_table(index='coords_0', columns='coords_1', values='B08_20220728')#.values
    img_comp_dic={}
    for c in comps_sel:
        img_comp_dic[c] = pca_components_df_merge.pivot_table(index='coords_0', columns='coords_1', values=c).values
    t4=time.time()
   
    # if 0:
    #     t2=time.time()
    #     img_comp_dic_arr= img_comp_dic_df.to_dask_array(lengths=True)
    #     t3=time.time()
    #     img_comp_dic_val = img_comp_dic_df.values
    #     t4=time.time()
    
    print (f'test tempo para PIVOT_TABLE COM repartition: {t2-t1}')

    print ("####################### ZARR #####################\n")
    # print (f'tempo para TO_DASK_ARRAY COM repartition: {t3-t2}')
    # print (f'tempo para VALUES COM repartition: {t4-t3}')
    print (f'tempo para pivot table of VALUES all comps {comps_sel} COM repartition: {t4-t3}')
    # print (f'type img_comp_dic_arr: {type(img_comp_dic_arr)}')
    # print (f'type img_comp_dic_val: {type(img_comp_dic_val)}')
    # print (f'chunk size {img_comp_dic_arr.chunksize}')   
    #print (f'chunks: {img_comp_dic_arr.chunks}')
    print (f'type img_comp_dic {c}: {type(img_comp_dic[c])}')
    print (f'{c} chunks: {img_comp_dic[c].chunks}')
    #ZARR
    #t_to_dask_array = t3-t2
    t_values = t4-t3
    save_path = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/parquet_npart111514/teste/'
    # f_name = 'img_pca_'
    # print (f'dir save zarr: {save_path+f_name}cmd_zarr')
    # t1=time.time()
    # img_comp_dic_arr.compute_chunk_sizes() 
    # t2=time.time()
    # print (f'\ntempo para fazer o compute_chunk_size do dask array para salvar como zarr: {t2-t1}')
    # t_compute_chunk_sizes = t2-t1
    # t3=time.time()
    # img_comp_dic_arr.to_zarr(save_path+f_name+'arr_cmd_zarr')
    # t4=time.time()
    # t_to_zarr=t4-t3
    # t_save_zarr_arr = t_to_dask_array + t_compute_chunk_sizes + t_to_zarr
    #print ("#### ZARR ##### ")
 
    f_name = 'img_pca_'
    for c in comps_sel:
        t1=time.time()
        #img_comp_dic_val.compute_chunk_sizes() 
        img_comp_dic[c].compute_chunk_sizes() 
        t2=time.time()
        print (f'\ntempo para fazer o compute_chunk_size do {c} values para salvar como zarr: {t2-t1}')
        t_compute_chunk_sizes_v = t2-t1
        t3=time.time()
        #img_comp_dic_val.to_zarr(save_path+f_name+'val_cmd_zarr')
        img_comp_dic[c].to_zarr(save_path+f_name+str(c)+'_cmd_zarr')
        t4=time.time()
        t_to_zarr_v = t4-t3

    t_save_zarr_val = t_values + t_compute_chunk_sizes_v + t_to_zarr_v
    
    # del img_comp_dic_arr, img_comp_dic_val
    del img_comp_dic


    #PARQUET
    # print ("###### PARQUET ###### ")
    # print ("\npegando o num de colunas e salvando como df em parquet")
    # t1=time.time()
    # ncols = img_comp_dic_df.index.pipe(len)
    # t2=time.time()
    # t_index_pipe = t2-t1
    # cols = {n+1:str(n+1) for n in range(ncols)}
    
    # t1=time.time()   
    # cols_=img_comp_dic_df.columns
    # cols_dic={n:str(n) for n in cols_}
    # img_comp_dic_df = img_comp_dic_df.rename(columns=cols_dic)
    # t2=time.time()
    # t_rename_cols = t2-t1

    # print (f'size do cols do range {len(cols)}, size das cols do df: {len(cols_dic)}')
    # print ("###### Sumary TO SAVE PARQUET #######")
    # print (f'tempo do index.pipe: {t2-t1} {ncols} {len(cols)}')
    # print (f'tempo para renomear as colunas: {t_rename_cols}')
    
    #print (save_path+f_name+'cmd_parquet')
    # t1=time.time()
    # img_comp_dic_df.to_parquet(save_path+f_name+'df_cmd_parquet', engine='pyarrow')
    # t2=time.time()
    # t_to_parquet = t2-t1
    # t_save_parquet = t_index_pipe + t_rename_cols + t_to_parquet

    # print (f'tempo para to_parquet: {t_to_parquet}')    
    # print (f'\ntempo total para salvar como df em parquet: {t_save_parquet}s, {t_save_parquet/60}m')

    # print ("\n###### Sumary TO SAVE ZARR #######")
    # print (f'tempo para to_dask_array = {t_to_dask_array}')
    # print (f'tempo para compute chunk sizes = {t_compute_chunk_sizes}')
    # print (f'tempo para to_zarr = {t_to_zarr}')
    # print (f'tempo TOTAL ZARR com to_dask_array = {t_save_zarr_arr}s, {t_save_zarr_arr/60}m\n')

    print (f'tempo para values array = {t_values}')
    print (f'tempo para compute chunk sizes = {t_compute_chunk_sizes_v}')
    print (f'tempo para to_zarr = {t_to_zarr_v}')
    print (f'tempo TOTAL ZARR com values = {t_save_zarr_val}s, {t_save_zarr_val/60}m\n')

    n_part_merge=pca_components_df_merge.npartitions
    f_name = name_img+'_pca_df_merge_npart111514_'+str(n_part_merge)
    print (f'saving pca df merge to {save_path+f_name}') if sh_print_n2 else None
    # tentando resolver problema do index
    pca_components_df_merge._meta.index.name = None
    t1=time.time()
    pca_components_df_merge.to_parquet(save_path+f_name, engine='pyarrow')
    t2=time.time()
    print (f'time to save pca_components_df with index and coords to parquet files: {t2-t1} s, {(t2-t1)/60} m') if sh_print_n1 else None


    del df_toPCA, df_toPCA_loc, df_toPCA_head, df_toPCA_coords#, img_comp_dic_df

    #tempos para ler e ter imagem pca em dask array
    # ZARR
    if 1:
        #read parquet e get the array
        t1=time.time()  
        save_path+f_name+str(c)+'_cmd_zarr'
        #img_read_z = from_zarr(save_path+f_name+'val_cmd_zarr')
        img_read_z = from_zarr(save_path+f_name+str(c)+'_cmd_zarr')
        t2=time.time()
        t_read_z_val = t2-t1
        print (f'tempo to read zarr from c values {t2-t1}, {(t2-t1)/60}')
        print (type(img_read_z))#.compute()
        del img_read_z

    if 0:
        #read parquet e get the array
        t1=time.time()  
        img_read_z = from_zarr(save_path+f_name+'arr_cmd_zarr')
        t2=time.time()
        t_read_z_arr = t2-t1
        print (f'tempo to read zarr from dask array: {t_read_z_arr}, {(t_read_z_arr)/60}')
        print (type(img_read_z))#.compute()
        del img_read_z

        #PARQUET
        
        t1=time.time()
        img_read_p = dd.read_parquet(save_path+'img_pca_df_cmd_parquet', engine='pyarrow', calculate_divisions=True)
        t2=time.time()
        t_read_df_parquet = t2-t1
        print (f'tempo to read parquet from df: {t2-t1}, {(t2-t1)/60}')
        t3=time.time()
        img_read_ar_p = img_read_p.to_dask_array(lengths=True)
        t4=time.time()
        t_read_todask = t4-t3
        print (f'tempo to read to_dask_array: {t_read_todask}, {(t_read_todask)/60}')
        t_img_read_parquet =  t_read_df_parquet + t_read_todask
        print (f'tempo to read e gen img dask array from parquet: {t_img_read_parquet}')
        print (type(img_read_p))#.compute()
        del img_read_p

    print ("#### tempos para ter img pca em dask array ##### \n")
    #print (f'tempo to READ ZARR from dask array: {t_read_z_arr}s, tempo to save: {(t_save_zarr_arr)/60}m')
    print (f'tempo to READ ZARR from values: {t_read_z_val}s, tempo to save: {(t_save_zarr_val)/60}m')
    #print (f'tempo to READ e gen img dask array from PARQUET: {t_img_read_parquet}s, tempo to save:{t_save_parquet}')

    t_fim=time.time()
    print (f'tempo para executar o teste: {(t_fim-t_ini)/60}m')
    e = datetime.datetime.now()
    print ("Current date and time = %s" % e)

    quit()
    #### fim dos testes
#%%
#### Remover os nans (-9999 e -32768) do df 
#df_toPCA2.replace({-9999: np.nan, -32768: np.nan}, inplace=True)
t1 = time.time()
df_toPCA = df_toPCA.mask(df_toPCA == -9999, np.nan)
df_toPCA = df_toPCA.mask(df_toPCA == -32768, np.nan)
t2 = time.time()
print (f'tempo para replace -9999 e -32768 = {t2-t1}')
#print (f'{df_toPCA.tail(2)}') if sh_print_n2 else None
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
    #index_nan=df_with_nan.index
    t2=time.time()
    print (f'tempo para fazer df nan rows : {t2-t1}') if sh_print_n1 else None

print ("********************************** COORDS **********************************")
# pegar as colunas com as coordenadas dos pixels 
# antes de remover as linhas com nans e nomear o index
df_toPCA['coords_1'] = df_toPCA['coords_1'].astype('category')
print (f'df_toPCA category: {t2-t1}, {df_toPCA.coords_1.cat.known}')
cols_coords= df_toPCA.columns[1:3]
df_toPCA_coords = df_toPCA[cols_coords].copy()
print (f'df_toPCA_coords category: {t2-t1}, {df_toPCA_coords.coords_1.cat.known}')
#df_toPCA_coords.index = df_toPCA_coords.index.rename('index_orig')
df_toPCA_coords['coords_0']= df_toPCA_coords['coords_0'].astype('int64')
df_toPCA_coords['coords_1']= df_toPCA_coords['coords_1'].astype('int64')
df_toPCA_coords['coords_1'] = df_toPCA_coords['coords_1'].astype('category')
df_toPCA_coords['coords_1'] = df_toPCA_coords['coords_1'].cat.set_categories(df_toPCA_coords['coords_1'].head(1).cat.categories)
print (f'df_toPCA_coords category after astype int64, category and set_category: {t2-t1}')
print (f'{df_toPCA_coords.coords_1.cat.as_known()}')
#print ("df_toPCA_coords\n", df_toPCA_coords.tail(2))  if sh_print_n2 else None
#print ("df_toPCA_coords types\n", df_toPCA_coords.dtypes)  if sh_print_n2 else None
#df_toPCA_coords['index_orig'] = df_toPCA.index

# divisions=[]
# divisions = df_toPCA_coords.divisions
# #divisions=np.asarray(divisions, dtype=np.int64)
# df_toPCA_coords_div = list(divisions)
# df_toPCA_coords_npart = len(df_toPCA_coords_div) - 1
# print (f'len divisions df_to_PCA_coords: {len(df_toPCA_coords_div)}, n_part: {df_toPCA_coords_npart} {df_toPCA_coords.npartitions}')
print (f'df_toPCA_coords n_part: {df_toPCA_coords.npartitions}')

#%%
#Faz o drop das linhas que sobraram com Nans
t1=time.time()
df_toPCA = df_toPCA.dropna()
t2=time.time()
print (f'\ntempo para fazer dropna no df_toPCA: {t2-t1}') if sh_print_n1 else None
#print (df_toPCA.tail(2)) if sh_print_n2 else None
#%%
do_PCA=0        #0 se for ler o pca dos aqrquivos, 1 se for fazer
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
# func_random = 3 nao fazer PCA
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

print ("********************************** PCA COMPONENTS **********************************")
SAVE_PCA_DF=0           #1: save files
                        #0: read df to pca parquet files 
if SAVE_PCA_DF == 1:
    save_path = cur_dir + '/data/parquet_npart111514/'
    f_name = name_img+'_pca_df_npart111514'
    print (f'saving pca_components df to {save_path+f_name}') if sh_print_n2 else None
    t1-time.time()
    pca_components_df.to_parquet(save_path+f_name, engine='pyarrow')
    t2=time.time()
    print (f'time to save pca_components df to parquet file: {t2-t1} s, {(t2-t1)/60} m') if sh_print_n1 else None

elif SAVE_PCA_DF == 0: #read pca parquet files
    #### read parquet files
    read_parquet = cur_dir + '/data/parquet_npart111514/'
    #f_name = name_img+'_df_toPCA_full.parquet'
    f_name = name_img+'_pca_df_npart111514'

    t1=time.time()
    pca_components_df2 = dd.read_parquet(read_parquet+f_name, 
                                        engine='pyarrow', calculate_divisions=True)
    t2=time.time()
    print (f'\nread time pca parquet files {(t2-t1)} s {(t2-t1)/60} m') if sh_print_n1 else None
    #print ("pca_components_df read\n", pca_components_df2.tail(2))  if sh_print_n2 else None
    print (f'pca_components_df2 npartition = {pca_components_df2.npartitions}')#\n pca_components_df2 tail = {df_toPCA.tail(2)}')

if nrows:
    #nrows=10000000 #111514  111.513.599
    #print (f'nrows ={nrows}')

    pca_components_df2 = pca_components_df2.loc[:nrows]
    t2=time.time()
    print (f'tempo loc {nrows} pca_components_df2= {t2-t1}\t ')

print (f'type from pca_components_df2 {type(pca_components_df2)}')

t1=time.time()
npart=df_toPCA.npartitions
pca_components_df2 = pca_components_df2.repartition(npartitions=npart)#partition_size="100MB")
t2=time.time()
print (f'tempo repartition pca_components_df2 loc = {t2-t1}')
print (f'n_part df_toPCA pca_components_df2: {npart} {pca_components_df2.npartitions}')


#%%
'''
# pegar as colunas com as coordenadas dos pixels e o index do df_PCA
# que teve as linhas com nans removidas
# cols_coords= df_toPCA.columns[1:3]
# df_toPCA_coords = df_toPCA[cols_coords].copy()
# #df_toPCA_coords['index_orig'] = df_toPCA.index
# df_toPCA_coords.index = df_toPCA_coords.index.rename('index_orig')

# adicionar o index original do df_toPCA ao pca_components_df
#pca_components_df=principalComponents_df.copy()

# divisions=list(pca_components_df.divisions)
# divisions=divisions.sort()
# divisions=[]
# divisions = pca_components_df2.divisions
# #divisions=np.asarray(divisions, dtype=np.int64)
# pca_comp_div = list(divisions)
# pca_comp_npart = len(divisions) - 1
#print (f'len divisions pca_components_df2 initial: {len(pca_comp_div)}, n_part: {pca_comp_npart} {pca_components_df2.npartitions}')
#'''
print (f'pca_components_df2 n_part: {pca_components_df2.npartitions}')

#tentar da forma baixo para manter o indice do df_toPCA
#pca_components_df['index_orig'] = df_toPCA['orig_index'].astype('int64').values #df_toPCA.index desta forma nao funcionou
t1=time.time()
pca_components_df = df_toPCA[['orig_index']].copy()
del df_toPCA
gc.collect()
#pca_components_df.index=pca_components_df.index.rename('index_orig')
n_compon = 3
comps_sel = ['c'+str(x) for x in range(1,n_compon+1)]
for c in comps_sel:
    pca_components_df[c] = pca_components_df2[c].values

'''
#tentar sem setar o index travou depois dele
#pca_components_df = pca_components_df.set_index('index_orig', sorted=False)#, divisions=divisions)
# pca_components_df['coords_0']= pca_components_df['coords_0'].astype('int64')
# pca_components_df['coords_1']= pca_components_df['coords_1'].astype('int64')
#'''
t2=time.time()
print (f'\ntime to copy orig_index in pca_components_df: {t2-t1} s {(t2-t1)/60}') if sh_print_n1 else None
'''
#print ('pca_components df after insert col index original from df_toPCA:\n',pca_components_df.tail(2))
#print ('pca_components_df dtypes before merge:\n',pca_components_df.dtypes) if sh_print_n2 else None
#print (f"type pca_components_df: {type(pca_components_df)}")
#print ("Fazer repartition in pca_components_df\n") if sh_print_n1 else None
#t1=time.time()
#pca_components_df = pca_components_df.repartition(npartitions=df_toPCA_coords_npart)
#t2=time.time()
#print (f'time to repartition in pca_components_df: {t2-t1} s {(t2-t1)/60}') if sh_print_n1 else None
#'''
del pca_components_df2
gc.collect()

print ("deleted df_toPCA, pca_components_df2") if sh_print_n2 else None

SAVE_PCA_DF=0
if SAVE_PCA_DF==1:
    save_path = cur_dir + '/data/parquet_npart111514/'
    f_name = name_img+'_pca_df_index_npart111514'
    print (f'saving pca df index to {save_path+f_name}') if sh_print_n2 else None
    t1-time.time()
    pca_components_df.to_parquet(save_path+f_name, engine='pyarrow')
    t2=time.time()
    print (f'time to save pca_components_df with index and coords to parquet files: {t2-t1} s, {(t2-t1)/60} m') if sh_print_n1 else None

#%%
'''
# t1=time.time()
# divisions=[]
# divisions=pca_components_df.divisions
# #divisions=np.asarray(divisions, dtype=np.int64)
# divisions=list(divisions)
# t2=time.time()
#'''
t2=time.time()
n_part_pca = pca_components_df.npartitions
t3=time.time()

print (f'tempo npartitions={t3-t2}')
print (f'len divisions pca_components df: {len(divisions)}, {n_part_pca}')

#cria df com as coordendas e as 3 componentes para gerar uma matriz que será passada no slic
#usando o index como link
print ("********************************** MERGE **********************************")
t1=time.time()
#pca_components_df=pd.merge(df_toPCA_coords, pca_components_df,left_index=True, right_index=True)
#pca_components_df = df_toPCA_coords.merge(pca_components_df, how='left')
#pca_components_df_merge= dd.merge(df_toPCA_coords, pca_components_df, how='left') #  on= 'index_orig',

#### teste com um pequeno df 55756799
if 0:
    nrows = 100
    df_coords = df_toPCA_coords.loc[:nrows] #dd.from_pandas(df_toPCA_coords.loc[:nrows], npartitions=df_toPCA_coords_npart)
    print (f'\n***** COORDS ******\n type of df_toPCA_coords: {type(df_toPCA_coords)}\n',
           f'type df_coords: {type(df_coords)}\n')
    
    df_pca = pca_components_df.loc[:(nrows-1)] # dd.from_pandas(pca_components_df.loc[:(nrows-1)], npartitions=df_toPCA_coords_npart)
    print (f'\n***** PCA COMPONENTS DF ******\ntype pca_components_df: {type(pca_components_df)}\n',
        f'type pca_components_df: {type(df_pca)}\n')
    pca_components_df_merge= df_coords.merge( df_pca,  how='left')  #on= 'index_orig', .reset_index()

    del df_coords, df_pca
    gc.collect()
    print (f'rows: {nrows}')
    #print ("df_toPCA_coords.tail:\n",df_toPCA_coords.tail())
#### fim do teste
#pca_components_df_merge= df_toPCA_coords.merge(pca_components_df, how='left') # .reset_index() on= ['index_orig'],
#05/07/2024 tentar com o join para ver se nado dar erro qdo for salvar to_parquet
pca_components_df_merge= df_toPCA_coords.merge(pca_components_df, how='left') #mudei de join para merge 9/7/24 
#pca_components_df_merge = dd.concat([df_toPCA_coords, pca_components_df], axis=1)
t2=time.time()
#pca_components_df = pca_components_df.rename(columns={0:'c0', 1:'c1', 2:'c2', 3:'c3',4:'c4', 5:'c5'})
print ("******** PCA COMPONENTS DF *********\n")
print (f'time to gen pca_components_df_merge: {t2-t1}') if sh_print_n1 else None#, t4-t3)
#print ('\npca_components_df_merge.tail():\n',pca_components_df_merge.tail(),'\n') if sh_print_n2 else None
print (f' pca merge npartition: {pca_components_df_merge.npartitions} ')

# print (pca_components_df.tail(2))
# print (df_toPCA_coords.tail(2))
t1=time.time()
print (f'len pca components={pca_components_df.index.pipe(len)}\nlen pca_coords={df_toPCA_coords.index.pipe(len)}')

print (f'\nlen index pca merge={pca_components_df_merge.index.pipe(len)},\n ')
t2=time.time()

print (f'tempo para len de index {t2-t1}')
del df_toPCA_coords, pca_components_df
gc.collect()
print ("\ndeleted df_toPCA_coords, pca_components_df\n") if sh_print_n2 else None

#print (pca_components_df.tail(2)) if sh_print_n2 else None#,props_df_sel_ts_pca.shape
# pca_components_df_merge['coords_0']= pca_components_df_merge['coords_0'].astype('int64')
# pca_components_df_merge['coords_1']= pca_components_df_merge['coords_1'].astype('int64')
# pca_components_df_merge['orig_index']= pca_components_df_merge['orig_index'].astype('int64')
# divisions=[]
# divisions=pca_components_df_merge.divisions
# #divisions=np.asarray(divisions, dtype=np.int64)
# divisions=list(divisions)
# print (f'len divisions pca_components_merge df: {len(divisions)}')
print (f'pca_components_df_merge.dtypes:\n{pca_components_df_merge.dtypes}\n{type(pca_components_df_merge)}') if sh_print_n2 else None
print (f'pca_components_df_merge category: {pca_components_df_merge.coords_1.cat.known}')
#pca_components_df_merge = pca_components_df_merge.repartition(npartitions=df_toPCA_coords_npart)
#pca_components_df_merge =  pca_components_df_merge.fillna(-9999)

#print ('\npca_components_df_merge.tail():\n',pca_components_df_merge.tail(),'\n') if sh_print_n2 else None
#salvar este df
SAVE_PCA_DF=0       #1: save df pca  with index to parquet files
                    #2: read the parquet files
if SAVE_PCA_DF==1:
    #olhar se components_df nao está deletado
    save_path = cur_dir + '/data/parquet_npart111514/'
    f_name = name_img+'_pca_df_index_npart111514'
    print (f'saving pca df index to {save_path+f_name}') if sh_print_n2 else None
    t1-time.time()
    pca_components_df.to_parquet(save_path+f_name, engine='pyarrow')
    t2=time.time()
    print (f'time to save pca_components_df with index and coords to parquet files: {t2-t1} s, {(t2-t1)/60} m') if sh_print_n1 else None

elif SAVE_PCA_DF==2: # read parquet files
    #### read parquet files
    read_parquet = cur_dir + '/data/parquet_npart111514/'
    #f_name = name_img+'_df_toPCA_full.parquet'
    f_name = name_img+'_pca_df_merge_npart111514'

    t1=time.time()
    pca_components_df_merge = dd.read_parquet(read_parquet+f_name, 
                                        engine='pyarrow')
    t2=time.time()
    print (f'read time pca df merge parquet files {(t2-t1)} s {(t2-t1)/60} m') if sh_print_n1 else None
    print ("pca_components_df_merge read\n", pca_components_df_merge.tail(2))  if sh_print_n2 else None


img_comp_dic={}
n_compon = 3
comps_sel = ['c'+str(x) for x in range(1,n_compon+1)]
#comps_sel = list(pca_components_df.columns[2:])
print (f'components: {comps_sel}, next categorize')

#pca_components_df_merge = pca_components_df_merge.categorize(columns=['coords_1'])

# t1=time.time()  
# pca_components_df_merge['coords_1'] = pca_components_df_merge['coords_1'].astype('category') #nao funciona sozinho
# pca_components_df_merge['coords_1'] = pca_components_df_merge['coords_1'].cat.set_categories(pca_components_df_merge['coords_1'].head(1).cat.categories)

t2=time.time()

# print (f'tempo para pca_components_df_merge category: {t2-t1}')
# print (f'pca_components_df_merge category: {pca_components_df_merge.coords_1.cat.known}')
n_part_merge = pca_components_df_merge.npartitions
# print (f'pca_components_df_merge npartitions: {n_part_merge}, {n_part_pca}')

#print (f'{pca_components_df_merge.coords_1.cat.as_known()}')
#testar:
'''
test=1
# 1
if test==1:
    t1=time.time()
    pca_components_df_merge = pca_components_df_merge.astype('category').persist().categorize(columns=['coords_1'],"local")
    t2=time.time()
# 2:
if test==2:
    t1=time.time()
    pca_components_df_merge['coords_1'] = pca_components_df_merge['coords_1'].astype('category')
    t2=time.time()
# 3:
if test==3:
    t1=time.time()
    pca_components_df_merge = pca_components_df_merge.astype({'coords_1': "category"})
    pca_components_df_merge = pca_components_df_merge.persist()
    pca_components_df_merge = pca_components_df_merge.categorize(columns=['coords_1'])
    t2=time.time()

print (f'tempo para category: {t2-t1}')
'''
# %%
#criacao de matrix com as n componentes PCA por posicao da matriz
#cria uma matriz para cada componente e depois junta com dstack apenas as 3 primeiras uma img_pca
# nao funcionou juntar as colunas das componentes num valor [c1,c2,c3] e gerar a matriz
# o valor gerado fica com lista , nao como array, olhar os testes.
print ("\ngerar img_comp_dic para cada component")
t1=time.time()
for c in comps_sel:
    tp1=time.time()
    img_comp_dic[c] = pca_components_df_merge.pivot_table(index='coords_0', columns='coords_1', values=c).values
    tp2=time.time()
    print (f'tempo para pivot de {c}: {tp2-t1}')

print (f'type img_comp_dic:{type(img_comp_dic[c])}')
#por enqaunto nao vou usar o img_pca
#img_pca = np.dstack((img_comp_dic[comps_sel[0]], img_comp_dic[comps_sel[1]], img_comp_dic[comps_sel[2]]))

# matrix_c1 = pca_components_df.pivot(index='coords_0', columns='coords_1', values='c0').values    
# matrix_c2 = pca_components_df.pivot(index='coords_0', columns='coords_1', values='c1').values  
# matrix_c3 = pca_components_df.pivot(index='coords_0', columns='coords_1', values='c2').values   
# img_pca = np.dstack((matrix_c1, matrix_c2, matrix_c3))
#0.42520880699157715
t2=time.time()
print (f'time to gen img_pca: {t2-t1}') if sh_print_n1 else None

#%%
#salvar img_pca
SAVE_IMG_PCA=1
if SAVE_IMG_PCA:
    for c in comps_sel:
        save_path = cur_dir + '/data/parquet_npart111514/'
        f_name = name_img+'_img_pca_npart111514_'+str(c)+'_zarr'
        t1=time.time()
        img_comp_dic[c].compute_chunk_sizes() 
        t2=time.time()
        t_compute_chunk_sizes = t2-t1
        print (f'tempo to compute chunk sizes of {c}: {t2-t1}')
        print (f'saving img_pca {c} to {save_path+f_name}') if sh_print_n2 else None
        t1-time.time()
        #img_comp_dic[c].to_parquet(save_path+f_name, engine='pyarrow')
        #nao é bom salvar com pickle, leva muito tempo para ler
        #save_to_pickle(img_comp_dic[c], save_path+f_name+'.pkl')
        img_comp_dic[c].to_zarr(save_path+f_name)
        t2=time.time()
        print (f'time to save img pca {c} to parquet file: {t2-t1}s, {(t2-t1)/60} m') if sh_print_n1 else None

    del img_comp_dic
    gc.collect()
    print (f'del img_comp_dic e gc.collect\n')
#print ('pca_components_df_merge.tail(2):\n',pca_components_df_merge.tail(2)) if sh_print_n2 else None
print 
SAVE_PCA_DF=1
if SAVE_PCA_DF==1:
    save_path = cur_dir + '/data/parquet_npart111514/'
    f_name = name_img+'_pca_df_merge_npart111514_'+str(n_part_merge)
    print (f'saving pca df merge to {save_path+f_name}') if sh_print_n2 else None
    t1=time.time()
    pca_components_df_merge.to_parquet(save_path+f_name, engine='pyarrow')
    t2=time.time()
    print (f'time to save pca_components_df with index and coords to parquet files: {t2-t1} s, {(t2-t1)/60} m') if sh_print_n1 else None

#del nos dask q sobraram
del pca_components_df_merge #, img_comp_dic
gc.collect()
t_fim=time.time()
print (f'del pca_components_df_merge e gc.collect\ntempo_programa={t_fim-t_ini}s {(t_fim-t_ini)/60}\nFIM')
#client.close()
#fazer snic
