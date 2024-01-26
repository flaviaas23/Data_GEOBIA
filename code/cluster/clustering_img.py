#%%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pickle
from tqdm import tqdm 

class Clustering:
    #def __init__(self, raw_dir='../../Data_MHTS/', raw_file='' ):
    def __init__(self, raw_dir, raw_file ):
        self.raw_dir = raw_dir #'../../../../Data_MHTS/'
        self.raw_file = raw_file #'raw.githubusercontent.com_Nixtla_transfer-learning-time-series_main_datasets_tourism.csv'
        

    def load_cluster(self):
        '''
        load cluster data
        '''           
        with open(self.raw_dir+self.raw_file, 'rb') as handle: 
            b = pickle.load(handle)
        dic_cluster= {**b}
        
        return dic_cluster
    
    def load_similarity(self, dir_sim, pickle_file_sim):
        '''
        load cluster similarity data
        '''           
        with open(dir_sim+pickle_file_sim, 'rb') as handle: 
            b = pickle.load(handle)
        dic_cluster_sim= {**b}
        
        return dic_cluster_sim
    
    def gen_df_tourism_cluster(self, dic_cluster, Y_df, dataset='tourism'):
        ''''
        add cluster info to dataframe and returns it and the hierarchies
        '''
        n_clusters = list(dic_cluster.keys())
        #print ("n_clusters: ", n_clusters )
        data_file = dic_cluster[n_clusters[0]]["data_file_name"] #'Tourism_bottom_pivot'
        print ("data_file: ", data_file)
        df_dataTocluster = pd.read_pickle(data_file)
        result=Y_df.copy()
        
        for n in n_clusters:
            #df_dataTocluster['cluster'+n]=['cluster_'+str(x) for x in dic_cluster[n]['cluster']]
            cluster_df_tmp= pd.DataFrame()
            cluster_df_tmp['unique_id']=df_dataTocluster['unique_id']
            cluster_df_tmp['Cluster'+str(n)]=['cluster'+str(n)+'_'+str(x) for x in dic_cluster[n]['cluster']]
            result=pd.merge(result, cluster_df_tmp, on='unique_id')
            del cluster_df_tmp

        result = result.drop('unique_id', axis=1)
        #spec = [['Country']]
        spec=[]
        if dataset == 'tourism':
            for n in n_clusters:
                spec.append(['Country', 'Cluster'+str(n)])
        elif dataset == 'gef':
            for n in n_clusters:
                spec.append(['Level0', 'Cluster'+str(n)])
        return result, spec
    
    def cria_SimilarityMatrix(self, dic_cluster):
        '''
        Generate a frequency/similarity/Co-association matrix based on 
        silhouette of clustering of time series
        '''
        n_clusters = list(dic_cluster.keys())  
        nrow = len(dic_cluster[n_clusters[0]]['cluster'])
        print ("nrow= {}, len nclusters = {}".format(nrow, len(n_clusters)))
        s = (nrow, nrow)
        freq_matrix= np.zeros(s)
        for n in n_clusters:
            #print ("n = ",n)
            sil = dic_cluster[n]['sample_silhouette_values']
            cluster = dic_cluster[n]['cluster']
            #print ("sil= ",sil,"\ncluster = ",cluster)
            for i in range(0, (nrow)):            
                #print ("i = ",i)
                for j in range(0, nrow):
                    #print ("j = ",j , cluster[i], cluster[j], sil[i], sil[j])
                    if cluster[i] == cluster[j]:

                        freq = (sil[i]+sil[j]+2)/4
                        freq_matrix[i,j] += freq
                        #print ("j = ",j , cluster[i], cluster[j], sil[i], sil[j], freq)

            #print ("freq_matrix = \n", freq_matrix)
        freq_matrix= freq_matrix/len(n_clusters)
        #print ("freq_matrix = \n", freq_matrix)
        return freq_matrix            
        
    def cria_SimilarityMatrix_freq(self, dic_cluster):
        '''
        Generate a frequency/similarity/Co-association matrix based on 
        frequency of point are together in the clustering of time series
        '''
        n_clusters = list(dic_cluster.keys())  
        nrow = len(dic_cluster[n_clusters[0]]['cluster'])
        print ("nrow= {}, len nclusters = {}".format(nrow, len(n_clusters)))
        s = (nrow, nrow)
        freq_matrix= np.zeros(s)
        for n in n_clusters:
            #print ("n = ",n)
            #sil = dic_cluster[n]['sample_silhouette_values']
            cluster = dic_cluster[n]['cluster']
            #print ("sil= ",sil,"\ncluster = ",cluster)
            for i in range(0, (nrow)):            
                #print ("i = ",i)
                for j in range(0, nrow):
                    #print ("j = ",j , cluster[i], cluster[j], sil[i], sil[j])
                    if cluster[i] == cluster[j]:

                        #freq = (sil[i]+sil[j]+2)/4
                        freq_matrix[i,j] += 1 # freq
                        #print ("j = ",j , cluster[i], cluster[j], sil[i], sil[j], freq)

            #print ("freq_matrix = \n", freq_matrix)
        freq_matrix= freq_matrix/len(n_clusters)
        #print ("freq_matrix = \n", freq_matrix)
        return freq_matrix        
    
    def save_similarity_matrix(self, dic_cluster, similarity_matrix, type_similarity):
        '''        
        Save to similarity matrix to pickle file
        '''
        obj_cluster = {
            "dic_cluster": dic_cluster,
            "similarity_matrix": similarity_matrix
        }
       
        pickle_file = self.raw_file.split(".")[0]
        file_to_save = self.raw_dir+pickle_file+type_similarity+'_similarity_matrix.pkl'
        with open(file_to_save, 'wb') as handle:
            pickle.dump(obj_cluster, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print ("file save: ", file_to_save)

        return
    
    def save_cluster_ensemble(self, obj_cluster, pickle_file_ensemble):
        '''
        save to a pickle file cluster ensemble info
        '''    
        pickle_file_ensemble = pickle_file_ensemble.split(".")[0]+'_ensemble.pkl'

        with open(self.raw_dir+pickle_file_ensemble, 'wb') as handle:
            pickle.dump(obj_cluster, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print ("file saved: ", self.raw_dir+pickle_file_ensemble)
        
        return

    def cria_grupos_kmeans(self, n, dist_sim_matrix):
        ''''
        Cria cluster based on distance matrix
        '''
        kmeans = KMeans(n_clusters=n)
        kmeans_fit = kmeans.fit(dist_sim_matrix)
        clusters_centers = kmeans_fit.cluster_centers_
        cluster_labels = kmeans_fit.labels_

        return cluster_labels, clusters_centers

    def cria_obj_grupos_matrix(self, n_clusters, dist_sim_matrix,\
                               pickle_file_sim, pickle_data_file_name):
        ''''
        Group based on dist matrix received
        '''    
        obj_cluster = {}
        #n_clusters = list(dic_cluster_sim['dic_cluster'].keys())
        #print (dist_sim_matrix)
        for n in tqdm(n_clusters):
            #print ("n= ", n)
            clusters, clusters_center, = self.cria_grupos_kmeans(n, dist_sim_matrix)
            #print ("clusters: ", clusters)
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(dist_sim_matrix, clusters)
            
            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(dist_sim_matrix, clusters)
            
            obj_cluster[n] = {
                # "data": df_cluster_sample,# amostra
                "data_file_name": pickle_data_file_name, #PROCESSED_DIR+data_file+'.pkl',
                "data_file_similarity": pickle_file_sim,
                #"seed": seed,
                "distance_metric": "1 - similarity_matrix",
                "cluster": clusters,     # [] resultado do cluster 
                "clusters_centers": clusters_center,
                #"dias_sample": num_dias,     #dias usados do sample
                "silhouette_avg":silhouette_avg, # silhouette_avg
                "sample_silhouette_values":sample_silhouette_values,        #[] resultado do silhoute para cada ponto do cluster
            }
        return obj_cluster
        
    def add_column_df(self,df, level_to_filter, label_to_filter, column_to_set, value):
        '''
        Add a column and value to a multiindex df  
        '''
        
        # Specify the level and label to filter rows based on the multi-index
        # level_to_filter = 'level'#'Index1'
        # label_to_filter = 'H_Cluster17' #'B'

        # # Column where you want to set the values
        # column_to_set = 'Sil_avg'

        # # New value to set in the specified column
        # value = 42
        if column_to_set not in df.columns:
            df[column_to_set]=pd.NA
        
        # Use .xs() to retrieve the cross-section of the DataFrame
        cross_section = df.xs(key=label_to_filter, level=level_to_filter)
        
        # Update the specified column in the cross-section
        cross_section[column_to_set] = value
        
        # Update the original DataFrame with the modified cross-section
        df.loc[df.index.get_level_values(level_to_filter) == label_to_filter, :] = cross_section.values

        return df    

    def add_cluster_info_to_eval_df(self, eval_df, strategy, d_metric, cluster_method):
        ''''
        Add columns cluster info to evaluation table
        '''
        
        eval_df = eval_df.reset_index()
        eval_df.insert(loc=1, column='Strategy', value= strategy)
        eval_df.insert(loc=3, column='DistMetric', value=d_metric)
        eval_df.insert(loc=4, column='ClusterMethod', value=cluster_method)
        eval_df.set_index(['level', 'Strategy', 'metric','DistMetric'], inplace=True)
        eval_df = eval_df.sort_index(level=['level', 'Strategy', 'metric'])
    
        return eval_df
            
    def select_clusters(self, dic_cluster, typeofmeas='mean', sil_meas_thr=0.45):
        ''''
        Select clusters based on value of sillouette
        mean, median , return which groups of clusters
        '''
        print ("select_cluster sil_meas_thr: ", sil_meas_thr)
        media=[]
        mediana=[]
        desvio=[]
        for k in dic_cluster.keys():
            sil_values = dic_cluster[k]['sample_silhouette_values']
            media.append(np.mean(sil_values))
            mediana.append(np.median(sil_values))
            desvio.append(np.std(sil_values))

        if typeofmeas == 'mean':
            clusters = [i+2 for i, valor in enumerate(media) if valor >= sil_meas_thr]
        elif typeofmeas == 'median':
            clusters = [i+2 for i, valor in enumerate(media) if valor >= sil_meas_thr]
        
        stats_df = pd.DataFrame([media, mediana,desvio]).T
        stats_df=stats_df.rename(columns={0:'media',1:'mediana',2:'desvio'})
        return clusters, stats_df
        
    ####
    #Novas funcoes para serem usadas com arquivos de imagem
    def get_labels(clusters, len_arraybands_list):
        ''''
        retorna os labels de cada elemento baseado no seu indice
        '''

        labels = []
        for elemento_procurado in range(len_arraybands_list): 
            for i, subarray in enumerate(clusters):
                if elemento_procurado in subarray:
                    #indice_subarray = i
                    labels.append(i)
                    break
        
        return labels
            
            
            
# %%
