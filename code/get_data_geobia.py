#%%
from wtss import WTSS


# %%
service = WTSS('https://brazildatacube.dpi.inpe.br/')#, access_token='change-me')
#%%
service
# %%
cbers4_coverage = service['CB4-16D-2']
cbers4_coverage
# %%
red_band = 'BAND15'
nir_band = 'BAND16'
# %%
time_series = cbers4_coverage.ts(attributes=(red_band, nir_band),
                                 latitude=-16.817,
                                 longitude=-52.079,
                                 start_date="2017-01-01",
                                 end_date="2019-12-31")
#%%
time_series.plot()
# %%
import pandas as pd
import plotly.express as px
#%%
cbers_df = pd.DataFrame({ 'BAND15': time_series.BAND15, 'BAND16': time_series.BAND16 }, 
                        index = pd.to_datetime(time_series.timeline))

cbers_df
# %%
fig = px.line(cbers_df, x=cbers_df.index, y=['BAND15', 'BAND16'], title='CBERS-4/AWFI (BAND 15 and 16)', labels={
    'index': 'Date',
    'value': 'Spectral Reflectance (scaled)'
})
#%%
fig.update_xaxes(rangeslider_visible=True)
#%%
fig.show()
# %%
#### PSTAC
import pystac_client
# %%
pystac_client.__version__
# %%
parameters = dict(access_token='change-me')
service_pystac = pystac_client.Client.open('https://brazildatacube.dpi.inpe.br/stac/', parameters=parameters)
# %%
bbox = (-45.9, -12.9, -45.4, -12.6)
item_search = service_pystac.search(collections=['CB4-16D-2'],
                             bbox=bbox,
                             datetime='2018-08-01/2019-07-31')

#%%
items = list(item_search.get_items())
item = items[0]
item
# %%
sorted(item.properties['eo:bands'], key=lambda band: band['name'])
#%%
collection = service_pystac.get_collection('CB4-16D-2')
collection
# %%
collection.get_items()
#%%
item_search = service_pystac.search(bbox=(-46.62597656250001,-13.19716452328198,
                                          -45.03570556640626,-12.297068292853805),
                             datetime='2018-08-01/2019-07-31',
                             collections=['CB4-16D-2'])
item_search

# %%
item_search.matched()
#%%
for item in item_search.get_items():
    print(item)
# %%
assets = item.assets
#Then, from the assets it is possible to traverse or access individual elements:

for k in assets.keys():
    print(k)

# %%
blue_asset = assets['BAND13']
blue_asset
#To iterate in the item's assets, use the following pattern:

for asset in assets.values():
    print(asset)

#%%

# Using RasterIO and NumPy
# The rasterio library can be used to read image files from the Brazil Data Cube' service on-the-fly and then to create NumPy arrays. The read method of an Item can be used to perform the reading and array creation:

import rasterio
#%%
with rasterio.open(assets['BAND16'].href) as nir_ds:
    nir = nir_ds.read(1)
# %%
os.getcwd()
#%%
from PIL import Image

dir_images = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/notebooks/images/'
img_tif = 'CB4-16D_V2_007004_20180728_EVI.tif'

# image=Image.open(dir_images+img_tif)

# image.show()
#%%
import numpy as np
image_array = np.array(image)
#%%
import pyreadr
#%%
file_sits = '/Users/flaviaschneider/Documents/flavia/Doutorado/Dados_GEOBIA/sitsdata/samples_matogrosso_mod13q1.rda'
#file_sits = '/Users/flaviaschneider/Documents/flavia/Doutorado/Dados_GEOBIA/sitsdata/point_mt_mod13q1.rda'
#%%
# Use a função read_rda para ler o arquivo
result = pyreadr.read_r(file_sits)


# %%
# import os
# os.environ["R_HOME"] = ""
import rpy2.robjects as robjects
#%%
from rpy2.robjects.packages import importr
#%%
from rpy2.robjects import pandas2ri
#%%
# Carregue o pacote necessário
base = importr('base')
#%%
# Carregue o arquivo .rda
robjects.r['load'](file_sits)

# %%
obj_name = 'samples_matogrosso_mod13q1'
obj = robjects.r[obj_name]
# %%
import pandas as pd
data_frame = pd.DataFrame(obj)

#%%
with (robjects.default_converter + pandas2ri.converter).context():
  r_from_pd_df = robjects.conversion.py2rpy(obj)

r_from_pd_df

# %%
from datetime import datetime, timedelta

# Defina a data de referência
data_referencia = datetime(1900, 1, 1)

# Valor a ser convertido
valor = 13405.000

# Converta para um objeto de data e hora
data_convertida = data_referencia + timedelta(days=valor)
