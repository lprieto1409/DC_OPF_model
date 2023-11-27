# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 18:06:40 2022

@author: afarabi
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.colors as colors
import matplotlib as mpl
import seaborn as sns
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy
import skgstat as skg #for variograms and kriging
from shapely.geometry import Point
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pykrige.kriging_tools as kt
import gstools as gs
import matplotlib.font_manager as font_manager #to standarize font
import warnings
from netCDF4 import Dataset
from skgstat import OrdinaryKriging
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Point
from osgeo import gdal, osr

#%MPI import
from mpi4py import MPI

warnings.filterwarnings('ignore')
skg.plotting.backend('matplotlib')

#Load data
folder = '/home/lprieto/CE791/Mini_project/folder/'
folder_1 = '/home/lprieto/CE791/Mini_project/folder_1/'
folder_ts = '/home/lprieto/CE791/Mini_project/folder_ts/'
folder_images = '/home/lprieto/CE791/Mini_project/Images/'
folder_network = '/home/lprieto/CE791/Mini_project/folder_network/'
KLumber = '/home/lprieto/CE791/Mini_project/KLumber/'
Kcf = '/home/lprieto/CE791/Mini_project/KCape/'
Kypd = '/home/lprieto/CE791/Mini_project/KYadkin/'
Kwo = '/home/lprieto/CE791/Mini_project/KWhite/'
Ktp = '/home/lprieto/CE791/Mini_project/KTarP/'
Kneuse = '/home/lprieto/CE791/Mini_project/KNeuse/'

#%% Watersheds
#North Carolina Boundaries
NC_bound = gpd.read_file(folder + 'NC_boundary.shp')
CapeFear_WS = gpd.read_file(folder_1 + 'Cape_fear_WS.shp')
Lumber_WS = gpd.read_file(folder_1 + 'Lumbert_WS.shp')
YadkinPeeDee_WS = gpd.read_file(folder_1 + 'Yadkin_Pee_Dee_WS.shp')
WhiteOak_WS = gpd.read_file(folder_1 + 'White_Oak_WS.shp')
TarPamlico_WS = gpd.read_file(folder_1 + 'Tar_Pamlico_WS.shp')
Neuse_WS = gpd.read_file(folder_1 + 'Neuse.shp')

#%% Substations in North Carolina
ST_M2S = pd.read_csv(folder_network + 'buses.csv') #Latest information about substations
ST_M2S_x = ST_M2S['x'] #Longitude coordinate
ST_M2S_y = ST_M2S['y'] #Latitude coordinate

#%% Timestep data
ST_time = pd.read_csv(folder+'M2S_timesteps.csv')

#%% High Water Mark Data
CapeFear_HWM = pd.read_csv(folder_1+'Cape_Fear_HWM.csv')
Lumber_HWM = pd.read_csv(folder_1+'Lumber_HWM.csv')
YadkinPeeDee_HWM = pd.read_csv(folder_1+'Yadkin_Pee_Dee_HWM.csv')
WhiteOak_HWM = pd.read_csv(folder_1+'White_Oak_HWM.csv')
TarPamlico_HWM = pd.read_csv(folder_1+'Tar_Pamlico_HWM.csv')
Neuse_HWM = pd.read_csv(folder_1+'Neuse_HWM.csv')

#%% Converting High Water Mark Data to Geopandas
proj = 'EPSG:4269'
Basins = ['CapeFear','Lumber','YadkinPeeDee', 'WhiteOak', 'TarPamlico', 'Neuse']

CapeFear_HWM['geometry'] = CapeFear_HWM.apply(lambda x: Point((float(x.Long), float(x.Lat))), axis=1)
CF_HWM = gpd.GeoDataFrame(CapeFear_HWM, geometry = 'geometry', crs = proj)

Lumber_HWM['geometry'] = Lumber_HWM.apply(lambda x: Point((float(x.Long), float(x.Lat))), axis=1)
Lumb_HWM = gpd.GeoDataFrame(Lumber_HWM, geometry = 'geometry', crs = proj)

YadkinPeeDee_HWM['geometry'] = YadkinPeeDee_HWM.apply(lambda x: Point((float(x.Long), float(x.Lat))), axis=1)
Yadkin_HWM = gpd.GeoDataFrame(YadkinPeeDee_HWM, geometry = 'geometry', crs = proj)

WhiteOak_HWM['geometry'] = WhiteOak_HWM.apply(lambda x: Point((float(x.Long), float(x.Lat))), axis=1)
WOak_HWM = gpd.GeoDataFrame(WhiteOak_HWM, geometry = 'geometry', crs = proj)

TarPamlico_HWM['geometry'] = TarPamlico_HWM.apply(lambda x: Point((float(x.Long), float(x.Lat))), axis=1)
TarPamli_HWM = gpd.GeoDataFrame(TarPamlico_HWM, geometry = 'geometry', crs = proj)

Neuse_HWM['geometry'] = Neuse_HWM.apply(lambda x: Point((float(x.Long), float(x.Lat))), axis=1)
Neu_HWM = gpd.GeoDataFrame(Neuse_HWM, geometry = 'geometry', crs = proj)

#%% Reprojecting the data (Based on EPSG from M2S substations)
CapeFear_WS = CapeFear_WS.to_crs(proj)
Lumber_WS = Lumber_WS.to_crs(proj)
YadkinPeeDee_WS = YadkinPeeDee_WS.to_crs(proj)
WhiteOak_WS = WhiteOak_WS.to_crs(proj)
TarPamlico_WS = TarPamlico_WS.to_crs(proj)
Neuse_WS = Neuse_WS.to_crs(proj)

#%% Time series flooding depth per watershed
TS_Capefear = pd.read_csv(folder_ts+'M2S_timesteps_CapeFear.csv')
TS_Lumber = pd.read_csv(folder_ts+'M2S_timesteps_Lumber.csv')
TS_YadkinPeeDee = pd.read_csv(folder_ts+'M2S_timesteps_YadkinPeeDee.csv')
TS_WhiteOak = pd.read_csv(folder_ts+'M2S_timesteps_WhiteOak.csv')
TS_TarPamlico = pd.read_csv(folder_ts+'M2S_timesteps_TarPamlico.csv')
TS_Neuse = pd.read_csv(folder_ts+'M2S_timesteps_Neuse.csv')



#%% MPI Calls
world_comm = MPI.COMM_WORLD
world_size = world_comm.Get_size()
my_rank = world_comm.Get_rank()

N = 10 #  % length of h_snap 

workloads = [ N // world_size for i in range(world_size) ]
for i in range( N % world_size ):
   workloads[i] += 1

my_start = 0

for i in range( my_rank ):
   my_start += workloads[i]

my_end = my_start + workloads[my_rank]


#%% Timestep list
#N = ST_time['N']
h_snap = ST_time['h'] #Formato h_0 ..... h_720

#%% HWM list
Lumber_hwm_ls = Lumber_HWM['Datetime']
CapeFear_hwm_ls = CapeFear_HWM['Datetime']
YadkingPeeDee_hwm_ls = YadkinPeeDee_HWM['Datetime']
WhiteOak_hwm_ls = WhiteOak_HWM['Datetime']
TarPamlico_hwm_ls = TarPamlico_HWM['Datetime']
Neuse_hwm_ls = Neuse_HWM['Datetime']

#%% Editing
#Color
cmap_kr = mpl.cm.viridis

#Font
font_number = 22
font_title = 30
font_cbar = 30
marker_size = 180
font_legend = 22


#%%-----------------NEUSE------------------------------------------------##
#%%Krig norm NEUSE  and extent
krig_NE_min = 0
krig_NE_max = 17.5

#gridx = np.arange(-79.50, -76.00, 0.01)
#gridy = np.arange(34.80, 36.50, 0.01)

top_left_lon = -79.50
top_left_lat = 36.50

extent_mat = (top_left_lon, top_left_lon + Neuse_WS.shape[1] * 0.4375, top_left_lat - Neuse_WS.shape[0] * 1.7, top_left_lat)

#%%Defining the grid
xv = np.arange(-79.50, -76.00, 0.01)
yv = np.arange(34.80, 36.50, 0.01)


xmin, xmax = min(xv), max(xv)
ymin, ymax = min(yv), max(yv)

# We determine the resolution in x direction:
res_x = 100
# We determine the resolution in y direction
# based on res_x to make a regular grid:
res_y = int((ymax - ymin)*res_x/(xmax - xmin))

# We build the grid:
xx,yy = np.mgrid[xmin:xmax:complex(res_x), 
                 ymin:ymax:complex(res_y)]

#Starting the timer for MPI's
start_time = MPI.Wtime()


#%% Starting with the kriging without cutting the raster file

#Segmenting the positive values
h_snap_NE = h_snap#h_snap.head(256)

for k in range(my_start,my_end):
    k = h_snap_NE[my_start]
    
    # Variogram
    V_NE = skg.Variogram(Neu_HWM[['Long', 'Lat']].values, Neu_HWM[str(k)].values, maxlag='median', normalize=False)
    V_NE_params = V_NE.describe()
    
    #Kriging
    ok = OrdinaryKriging(V_NE, min_points=5, max_points=20, mode='exact')
    
    field_NE = ok.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)
    s2_NE = ok.sigma.reshape(xx.shape)
    
    # plot
    
    fig, axs = plt.subplots(1, 2 ,figsize=(15, 10), sharey=True)
    
    ax = axs[0]
    
    # Contour fringes of the kriging process:
    ctr_hh = ax.contourf(xx, yy, field_NE,
                         vmin = krig_NE_min,
                         vmax = krig_NE_max,
                     #   range(0,13,1),
                         cmap = "viridis_r", 
                         alpha = 0.5,
                         extent = extent_mat)
    
    ax.set_title("Ordinary kriging estimation")
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    #plt.colorbar(ctr_hh, cax=cax, label = 'Flooding depth [ft]')
    
    # We add the location of points of our dataset
    Neu_HWM.plot(ax = ax, c = "black", marker= '.', markersize = 2)
    Neuse_WS.plot(ax = ax, color = 'bisque', edgecolor = 'dimgray',alpha = 0.4)
    
    ax = axs[1]
    
    # Contour fringes of the kriging error:
    ctr_err = ax.contourf(xx, yy, s2_NE,
                          #vmin = 0,
                          #vmax = 20,
                         #range(0,20,2),
                          cmap = "plasma",
                          alpha = 0.5,
                          extent = extent_mat)
    ax.set_title("Kriging error estimation")
    
    # We add the location of points of our dataset
    Neu_HWM.plot(ax = ax, c = "black", marker= '.', markersize = 2)
    Neuse_WS.plot(ax = ax, color = 'bisque', edgecolor = 'dimgray',alpha = 0.4)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    #plt.colorbar(ctr_err, cax=cax, label = 'Error [m]')
    #plt.show()
    
    #%%Ahora completando kriging para que aparezca en el resto del watershed
    
    # We initiale a mask with 0 everywhere with the similar shape as earlier
    maskin = xx - xx
    
    # In this mask, we assign True if the point is inside the watershed
    # and we assign False if the point is outside the watershed
    for i in range(maskin.shape[0]):
        for j in range(maskin.shape[1]):
            xi = xx[i][j]
            yj = yy[i][j]
            
            if Point(xi, yj).within(Neuse_WS["geometry"][0]):
                maskin[i, j] = 0
            else:
                maskin[i, j] = 1
    
    # We apply our mask to previous calculated arrays:
    depth_m_NE = np.ma.masked_array(field_NE, maskin)
    s2_m_NE = np.ma.masked_array(s2_NE, maskin)

    #%% We plot the results using the mask through the watershed:
    fig, axs = plt.subplots(1, 2 ,figsize=(15, 10), sharey=True)
    
    ax = axs[0]
    
    # Contour fringes of the kriging process:
    ctr_hh = ax.contourf(xx, yy, depth_m_NE, 
                         vmin = krig_NE_min,
                         vmax = krig_NE_max,
                     #   range(0, 13, 1),
                         cmap = "viridis_r", 
                     #   alpha = 0.9,
                         extent = extent_mat)
    
    ax.set_title("Ordinary kriging estimation_"+str(k))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    #plt.colorbar(ctr_hh, cax=cax, label = 'Flooding depth [ft]')
    
    # We add the location of points of our dataset
    Neu_HWM.plot(ax = ax, c = "black", marker= '.', markersize = 2)
    Neuse_WS.plot(ax = ax, color = 'bisque', edgecolor = 'dimgray',alpha = 0.4)
    
    ax = axs[1]
    
    #%% Contour fringes of the kriging error:
    ctr_err = ax.contourf(xx, yy, s2_m_NE,
                          #vmin = 0,
                          #vmax = 20,
                      #   range(0,20,2),
                          cmap = "plasma",
                      #   alpha = 0.9,
                          extent = extent_mat)
    
    ax.set_title("Kriging error_"+str(k))
    
    # We add the location of points of our dataset
    Neu_HWM.plot(ax = ax, c = "black", marker= '.', markersize = 2)
    Neuse_WS.plot(ax = ax, color = 'bisque', edgecolor = 'dimgray',alpha = 0.4)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    #plt.colorbar(ctr_err, cax=cax, label = 'Error [m]')
    
    #plt.show()
    
    #%%EXPORT TO RASTER
    
    # we determine the pixel size in x and y directions.
    dx = abs(xmax - xmin)/res_x
    dy = abs(ymax - ymin)/res_y
    
    # we determine paramaters associated to our image
    # top left x 
    # w-e pixel resolution
    # rotation, 0 if image is "north up"
    # top left y
    # n-s pixel resolution
    
    params =(xmin - dx/2, dx, 0, ymax + dy/2, 0, -dy)
    
    # the name of our geoTIFF
    tif_name = Kneuse+"Kriging_NE_"+str(k)+".tif"
    
    # Create/Open the raster
    output_raster = gdal.GetDriverByName('GTiff').Create(tif_name, res_x+1, res_y+1, 1 ,gdal.GDT_Float32)
    
    # Specify its coordinates
    output_raster.SetGeoTransform(params)
    
    # Establish its coordinate encoding:
    srs = osr.SpatialReference() 
    
    # Our projection system is specified:
    srs.ImportFromEPSG(4269)                     
    
    # Exports the coordinate system to the file
    output_raster.SetProjection(srs.ExportToWkt()) 
    
    # Writes my array to the raster after some transformation due to the resulting shape of the kriging:
    output_raster.GetRasterBand(1).WriteArray(np.transpose(np.flip(depth_m_NE[::-1]))) 
    output_raster.GetRasterBand(1).SetNoDataValue(0) 
    output_raster.FlushCache()
    my_start += 1
#finish
#Time Calculation

end_time = MPI.Wtime()
if my_rank == 0:
    print(" Time: " + str(end_time-start_time))
    f = open("Time.txt", "a")
    f.write(" Time: " + str(end_time-start_time) + "world size" + str(world_size))
    f.close()  
