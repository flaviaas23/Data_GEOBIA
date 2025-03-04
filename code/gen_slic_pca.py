
# Generate slic/snic segments using pca images
# funcao gerada e executada no exacta
import numpy as np
from math import ceil

import pandas as pd
import psutil
import time
import sys
import os
import glob
import datetime

from itertools import product

import pickle
import random
from tqdm import tqdm

import gc
import dask.array as da
import dask.dataframe as dd 
from dask_ml.decomposition import PCA
from dask.diagnostics import ProgressBar
from dask.array import from_zarr

from timeit import default_timer as timer

#for slic
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

import imageio.v2 as imageio
from skimage.measure import regionprops, regionprops_table

from pysnic.algorithms.snic import snic
#For snic SP
from pysnic.algorithms.snic import compute_grid
from pysnic.ndim.operations_collections import nd_computations
from itertools import chain
from pysnic.metric.snic import create_augmented_snic_distance
from timeit import default_timer as timer

from functions_pca import save_to_pickle

## read image and mask pca
a = datetime.datetime.now().strftime('%d %m %Y %H:%M:%S')
b = datetime.datetime.now().strftime('%H:%M:%S')
print (f'1. {a}: INICIO')
print (f'1. {b}: reading files')
save_path = '/scratch/flavia/pca/'
# with open(save_path+'img_pca_sel.pkl', 'rb') as handle: 
#     img_pca = pickle.load(handle)

#mask for slic
with open(save_path+'img_pca_sel_mask.pkl', 'rb') as handle: 
    img_pca_mask = pickle.load(handle)

a = datetime.datetime.now().strftime('%H:%M:%S')
print (f'2. {a}: gen img pca')
n_components = 3
img_pca={}
pca_sel = ['c'+str(c+1)for c in range(n_components)]
for c in pca_sel:
    f_name = f'img_pca_{c}_npart_63_zarr'
    img_pca[c] = from_zarr(save_path+f_name)

img_sel_da = np.dstack((img_pca[pca_sel[0]], img_pca[pca_sel[1]], img_pca[pca_sel[2]]))

img_pca_np = img_sel_da.compute()

del img_sel_da, img_pca
gc.collect()

img_pca_np = img_pca_np.astype(np.float32)

print (f'   2.1 img_pca_np type: {type(img_pca_np)} {img_pca_np.dtype}')

img_pca_np = img_pca_np.tolist() 

print (f'   2.2 img_pca_np list type: {type(img_pca_np)} {img_pca_np.dtype}')

# print ("3. saving img in pkl file")
# t1=time.time()
# save_to_pickle(img_pca_np, save_path+'img_pca_sel_f32.pkl')
# t2=time.time()
# print (f'slic time run {t2-t1}')

# gen slic segments
n_segms = 110000
compact = 2
sigma = 1 #0.1
connectivity= 1 #0

a = datetime.datetime.now().strftime('%H:%M:%S')
# print (f'4. {a}:Segmentacao img_pca')


#for slic

# print (f'4. {a}:Segmentacao slic img_pca')
# print (f'   4.1 n_segms = {n_segms}, compactness = {compact}, sigma = {sigma}, connectivity= {connectivity}')
# t1=time.time()
# segments_slic_sel = slic(img_pca_np, n_segments=n_segms, compactness=compact, sigma=sigma, \
#                              start_label=1,mask=img_pca_mask, enforce_connectivity=connectivity)

# t2=time.time()
# print (f'   4.2 slic time run {t2-t1}s {(t2-t1)/60}')

# For SNIC
#a = datetime.datetime.now().strftime('%H:%M:%S')
print (f'4. {a}: Segmentacao snic img_pca')

t1=time.time()
# compute grid
img_pca_shape = img_pca_np.shape
grid = compute_grid(img_pca_shape, n_segms)
seeds = list(chain.from_iterable(grid))
seed_len = len(seeds)    
print (f'   4.1 n_segms = {n_segms}, seed_len = {seed_len}, compactness = {compact}, img_pca_shape = {img_pca_shape}, connectivity= {connectivity}')
# choose a distance metric #se nao fornecido faz exatamente isso
distance_metric = create_augmented_snic_distance(img_pca_shape, seed_len, compact)    
#start = timer()
t2=time.time()
segments_snic_pca_sp, dist_snic_sp, centroids_snic_sp = snic(
                            img_pca_np,
                            #img_sel_norm.tolist(),
                            seeds,
                            compact, nd_computations["3"], distance_metric)#,
                            #update_func=lambda num_pixels: print("processed %05.2f%%" % (num_pixels * 100 / number_of_pixels)))    
t3=time.time()
print(f'   4.2 snic took: {t3-t2}s, {(t3 - t2)/60}m')


a = datetime.datetime.now().strftime('%H:%M:%S')
print (f'6. {a}: saving segments to pickle')
# save SLIC to pickle
#save_to_pickle(segments_slic_sel, save_path+'segments_slic_sel_pca.pkl')

#save SNIC
compact = 2
sigma = 1 #0.1
connectivity= 1 
test=f'n110k_c{compact}_s{sigma}_con{connectivity}_'
t1=time.time()
save_to_pickle(segments_snic_pca_sp, save_path+test+'segments_snic_sel_pca_sp.pkl')
t2=time.time()
save_to_pickle(dist_snic_sp, save_path+test+'dist_snic_sel_pca_sp.pkl')
t3=time.time()
save_to_pickle(centroids_snic_sp, save_path+test+'centroids_snic_sel_pca_sp.pkl')
t4=time.time()

print (f'   {a}:tempos para salvar SNIC\n\t segmentos: {t2-t1}\n\t distancia: {t3-t2}\n\t centroids: {t4-t3}')

a = datetime.datetime.now().strftime('%d %m %Y %H:%M:%S')
print (f'4. {a}: FIM')