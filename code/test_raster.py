# from geopyspark.geopycontext import GeoPyContext 
# from geopyspark.geotrellis.constants import SPATIAL 
# from geopyspark.geotrellis.geotiff_rdd import get

# read_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/Cassio/S2-16D_V2_012014_20220728_/'
# geopysc = GeoPyContext(appName="rasterrdd-example", master="local")
# raster_rdd = get(geopysc=geopysc, rdd_type=SPATIAL, read_dir)

# import numpy as np
# import rasterio
# from geopyspark.geopycontext import GeoPyContext
# from geopyspark.geotrellis.constants import SPATIAL, ZOOM 
# from geopyspark.geotrellis.catalog import write
# from geopyspark.geotrellis.rdd import RasterRDD

# geopysc = GeoPyContext(appName="sentinel-ingest", master="local[*]")
#read_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/Cassio/S2-16D_V2_012014_20220728_/'

# #jp2s = ["/tmp/B01.jp2", "/tmp/B09.jp2", "/tmp/B10.jp2"]
# jp2s = [read_dir+"S2-16D_V2_012014_20220728_B04.tif", read_dir+"S2-16D_V2_012014_20220728_B03.tif", read_dir+"S2-16D_V2_012014_20220728_B02.tif"]
# arrs = []
# # Reading the jp2s with rasterio
# for jp2 in jp2s:
#     with rasterio.open(jp2) as f:
#         arrs.append(f.read(1))
# data = np.array(arrs, dtype=arrs[0].dtype)
# # saving the max and min values of the tile 
# with open(read_dir+'sentinel_stats.txt', 'w') as f: 
#     f.writelines([str(data.max()) + "\n", str(data.min())])

# if f.nodata:
#     no_data = f.nodata
# else:
#     no_data = 0

# bounds = f.bounds
# epsg_code = int(f.crs.to_dict()['init'][5:])
# # Creating the RasterRDD
# tile = {'data': data, 'no_data_value': no_data}
# extent = {'xmin': bounds.left, 'ymin': bounds.bottom, 'xmax': bounds.right, 'ymax': ˓→bounds.top}
# projected_extent = {'extent': extent, 'epsg': epsg_code}
# rdd = geopysc.pysc.parallelize([(projected_extent, tile)])
# raster_rdd = RasterRDD.from_numpy_rdd(geopysc, SPATIAL, rdd)
# metadata = raster_rdd.collect_metadata()
# laid_out = raster_rdd.tile_to_layout(metadata)
# reprojected = laid_out.reproject("EPSG:3857", scheme=ZOOM)
# pyramided = reprojected.pyramid(start_zoom=12, end_zoom=1)
# for tiled in pyramided:
# write("file:///tmp/sentinel-catalog", "sentinel-benchmark", tiled)

#%%


from pyspark import SparkContext


from shapely.geometry import box
import os


os.environ["SPARK_HOME"] = "/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/.venv_data_geobia/lib/python3.8/site-packages/pyspark"
print(os.environ["SPARK_HOME"])
os.environ["PYSPARK_PYTHON"]="python"
os.environ["PYTHONPATH"] = "/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/.venv_data_geobia/lib/python3.8/site-packages"
import geopyspark as gps
#os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]= "python"
#print(os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"])

#read_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/Cassio/S2-16D_V2_012014_20220728_/'
read_dir = '/Users/flaviaschneider/Documents/flavia/Data_GEOBIA/data/tmp/'
# Create the SparkContext
conf = gps.geopyspark_conf(appName="geopyspark-example", master="local[*]")
sc = SparkContext(conf=conf)

# Read in the NLCD tif that has been saved locally.
# This tif represents the state of Pennsylvania.
raster_layer = gps.geotiff.get(layer_type=gps.LayerType.SPATIAL,
                               #uri=read_dir+"S2-16D_V2_012014_20220728_B04.tif",
                               uri=read_dir+"NLCD2011_LC_Pennsylvania.zip",
                               num_partitions=100)
#%%
# Tile the rasters within the layer and reproject them to Web Mercator.
tiled_layer = raster_layer.tile_to_layout(layout=gps.GlobalLayout(), target_crs=3857)

# Creates a Polygon that covers roughly the north-west section of Philadelphia.
# This is the region that will be masked.
area_of_interest = box(-75.229225, 40.003686, -75.107345, 40.084375)

# Mask the tiles within the layer with the area of interest
masked = tiled_layer.mask(geometries=area_of_interest)

# We will now pyramid the masked TiledRasterLayer so that we can use it in a TMS server later.
pyramided_mask = masked.pyramid()

# Save each layer of the pyramid locally so that it can be accessed at a later time.
for pyramid in pyramided_mask.levels.values():
    gps.write(uri='file:///tmp/pa-nlcd-2011',
              layer_name='north-west-philly',
              tiled_raster_layer=pyramid)
