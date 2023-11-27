# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 18:37:19 2023

@author: lprieto
"""
import pandas as pd
import numpy as np
import matplotlib
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
from pylab import axes
import pykrige.kriging_tools as kt
#from pykrige.ok import OrdinaryKriging
import gstools as gs
import matplotlib.font_manager as font_manager #to standarize font
import warnings
from netCDF4 import Dataset
from skgstat import OrdinaryKriging
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Point
from osgeo import gdal, osr, ogr

warnings.filterwarnings('ignore')
skg.plotting.backend('matplotlib')

#%% Folders
#Load data
folder = 'D:/NC State/2. Research/Shapefiles/'
folder_1 = 'D:/3. Lectures/Fall 2021/GIS/' 
#folder_2 = 'D:/2. Research/5. Latest data/1. Balanciong Authorities/'
#folder_3 = 'D:/NC State/2. Research/Shapefiles/Flood_depth_outputs/Cape_Fear/'
folder_ts = 'D:/5. Jordan_1st_paper/2. Datasets_HWM/'
folder_images = 'D:/5. Jordan_1st_paper/Images/latest/' #final images are in this folder
folder_network = 'D:/2. Research/5. Latest data/17. Figures/M2S_network/during/'
folder_ascii_lumber = 'D:/5. Jordan_1st_paper/3. Ascii files/1. Lumber/'
folder_ascii_cf = 'D:/5. Jordan_1st_paper/3. Ascii files/2. Cape Fear/'
folder_ascii_ypd = 'D:/5. Jordan_1st_paper/3. Ascii files/3. Yadkin Pee Dee/'
folder_ascii_wo = 'D:/5. Jordan_1st_paper/3. Ascii files/4. White Oak/'
folder_ascii_tp = 'D:/5. Jordan_1st_paper/3. Ascii files/5. Tar Pamlico/'
folder_ascii_neuse = 'D:/5. Jordan_1st_paper/3. Ascii files/6. Neuse/'

#%% Watersheds
#North Carolina Boundaries
NC_bound = gpd.read_file(folder + 'NC_boundary.shp')
CapeFear_WS = gpd.read_file(folder_1 + 'Cape_fear_WS.shp') #Cape Fear Watershed
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

#Raster background for images
#ds_LU = gdal.OpenEx(folder_images+'rast_lu_background.tif')
#gt = ds_LU.GetGeoTransform()
#nodata_LU = ds_LU.GetRasterBand(1).GetNoDataValue()
#data_LU = ds_LU.ReadAsArray()
#ds = None

#data_LU = np.ma.masked_values(data_LU, nodata_LU)

#ys, xs = data_LU.shape
#ulx, xres, _, uly, _, yres = gt
#extent = [ulx, ulx+xres*xs, uly, uly+yres*ys]

#%% Time series flooding depth per watershed
TS_Capefear = pd.read_csv(folder_ts+'M2S_timesteps_CapeFear.csv')
TS_Lumber = pd.read_csv(folder_ts+'M2S_timesteps_Lumber.csv')
TS_YadkinPeeDee = pd.read_csv(folder_ts+'M2S_timesteps_YadkinPeeDee.csv')
TS_WhiteOak = pd.read_csv(folder_ts+'M2S_timesteps_WhiteOak.csv')
TS_TarPamlico = pd.read_csv(folder_ts+'M2S_timesteps_TarPamlico.csv')
TS_Neuse = pd.read_csv(folder_ts+'M2S_timesteps_Neuse.csv')

#%% Timestep list
N = ST_time['N']
h_snap = ST_time['h'] #Formato h_0 ..... h_720

#%% HWM list
Lumber_hwm_ls = Lumber_HWM['Datetime']
CapeFear_hwm_ls = CapeFear_HWM['Datetime']
YadkingPeeDee_hwm_ls = YadkinPeeDee_HWM['Datetime']
WhiteOak_hwm_ls = WhiteOak_HWM['Datetime']
TarPamlico_hwm_ls = TarPamlico_HWM['Datetime']
Neuse_hwm_ls = Neuse_HWM['Datetime']

###############################################################################################################
##########################################-----LUMBER----------------##########################################
###############################################################################################################
#Date Time format
TS_Lumber['Date'] = pd.to_datetime(TS_Lumber['Date'])
TS_Capefear['Date'] = pd.to_datetime(TS_Capefear['Date'])
TS_YadkinPeeDee['Date'] = pd.to_datetime(TS_YadkinPeeDee['Date'])
TS_WhiteOak['Date'] = pd.to_datetime(TS_WhiteOak['Date'])
TS_TarPamlico['Date'] = pd.to_datetime(TS_TarPamlico['Date'])
TS_Neuse['Date'] = pd.to_datetime(TS_Neuse['Date'])

#For the scatter data

norm_lumber = colors.TwoSlopeNorm(vmin = 0, vcenter = 3, vmax = 6)


#Color
cmap_kr = mpl.cm.coolwarm#mpl.cm.viridis_r

#Font
font_number = 47#17
font_axis = 62#85
font_cbar = 57
marker_size = 180
font_title = 57#17
font_legend = 37#22
font_box = 52
folder_1 + 'Lumbert_WS.shp'

fig, axs = plt.subplots(2,3,figsize = (42,24))   

#Space between the plots
plt.subplots_adjust(hspace= 0.25, wspace = 0.38) 

#-------Pannel 1------#
for i in Lumber_hwm_ls:
    fig = sns.lineplot(x = "Date", y = str(i), palette='viridis',data = TS_Lumber[TS_Lumber[str(i)]>=0], ax=axs[0,0])

#x_dates = TS_Lumber['Date'].dt.strftime('%m-%d-%r').sort_values().unique()
#axs[0,0].set_xticklabels(labels=x_dates, rotation=45, ha='right',fontsize = font_number)
axs[0,0].tick_params(axis='x', direction = 'out', length=22, labelsize=font_number) #nuevo
axs[0,0].tick_params(axis='y', direction = 'out', length=22, labelsize=font_number)
axs[0,0].xaxis.set_major_locator(mdates.HourLocator(interval=95))
axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d\n%I:%M %p'))
axs[0,0].set_xlabel(' ', fontsize=12)
axs[0,0].set_ylabel('Flooding depth (ft)', fontsize=font_title, fontweight = 'bold', labelpad = 20)
axs[0,0].axvline(pd.Timestamp('2018-09-16 06:00:00'),ls='-.',color='k', lw = 6)
axs[0,0].axvline(pd.Timestamp('2018-09-16 23:00:00'),ls='-.',color='k', lw = 6)
axs[0,0].axvline(pd.Timestamp('2018-09-18 23:00:00'),ls='-.',color='k', lw = 6)
axs[0,0].axvline(pd.Timestamp('2018-09-20 23:00:00'),ls='-.',color='k', lw = 6)
axs[0,0].axvline(pd.Timestamp('2018-09-22 23:00:00'),ls='-.',color='k', lw = 6)
axs[0,0].annotate("a", xy = (0,0), xytext=(715,618),#xytext=(620,550), 
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))
axs[0,0].set_title('Lumber: Flood depth (ft)', family='Arial', style='normal',
                   size=font_title, loc= 'center', fontweight = 'bold')

#-------Pannel 2------#
Lumber_WS.plot(ax = axs[0,1], color = 'bisque', edgecolor = 'dimgray',alpha = 0.4)
HWM_b = axs[0,1].scatter(Lumb_HWM['Long'],Lumb_HWM['Lat'],norm = norm_lumber, cmap = cmap_kr, c =Lumb_HWM['h_150'], s = marker_size)
axs[0,1].set_xticklabels(labels=Lumb_HWM['Long'],fontsize = font_number)
axs[0,1].set_yticklabels(labels=Lumb_HWM['Lat'],fontsize = font_number)
axs[0,1].tick_params(axis='x', direction = 'out', length=22, labelsize=font_number)
axs[0,1].tick_params(axis='y', direction = 'out', length=22, labelsize=font_number)
axs[0,1].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
axs[0,1].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
axs[0,1].set_title('09/16 06:00 AM', family='Arial',
                   style='normal', size=font_title, loc= 'center', fontweight = 'bold')

axs[0,1].annotate("b", xy = (0,0), xytext=(588,618), #xytext=(20,550) cuando se quiere al otro extremo
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))

cb = plt.colorbar(HWM_b,location="right",fraction=0.04, pad=0.04,ax=axs[0,1])
cb.set_label("Flooding depth (ft)", family = 'Arial', size = font_cbar, style = 'normal', fontweight = 'bold', labelpad = 20)
cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out', length=22)
#-------Pannel 3------#
Lumber_WS.plot(ax = axs[0,2], color = 'bisque', edgecolor = 'dimgray',alpha = 0.4)
HWM_c = axs[0,2].scatter(Lumb_HWM['Long'],Lumb_HWM['Lat'],norm = norm_lumber, cmap = cmap_kr, c =Lumb_HWM['h_167'], s = marker_size)
axs[0,2].set_xticklabels(labels=Lumb_HWM['Long'],fontsize = font_number)
axs[0,2].set_yticklabels(labels=Lumb_HWM['Lat'],fontsize = font_number)
axs[0,2].tick_params(axis='x', direction = 'out', length=22, labelsize=font_number)
axs[0,2].tick_params(axis='y', direction = 'out', length=22, labelsize=font_number)
axs[0,2].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
axs[0,2].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
axs[0,2].set_title('09/16 11:00 PM', family='Arial',
                   style='normal', size=font_title, loc= 'center', fontweight = 'bold')

axs[0,2].annotate("c", xy = (0,0), xytext=(588,618), #xytext=(20,550) cuando se quiere al otro extremo
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))

cb = plt.colorbar(HWM_c,location="right",fraction=0.04, pad=0.04,ax=axs[0,2])
cb.set_label("Flooding depth (ft)", family = 'Arial', size = font_cbar, style = 'normal', fontweight = 'bold', labelpad = 20)
cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out', length=22)
#-------Pannel 4------#
Lumber_WS.plot(ax = axs[1,0], color = 'bisque', edgecolor = 'dimgray',alpha = 0.4)
HWM_d = axs[1,0].scatter(Lumb_HWM['Long'],Lumb_HWM['Lat'],norm = norm_lumber, cmap = cmap_kr, c =Lumb_HWM['h_215'], s = marker_size)
axs[1,0].set_xticklabels(labels=Lumb_HWM['Long'],fontsize = font_number)
axs[1,0].set_yticklabels(labels=Lumb_HWM['Lat'],fontsize = font_number)
axs[1,0].tick_params(axis='x', direction = 'out', length=22, labelsize=font_number)
axs[1,0].tick_params(axis='y', direction = 'out', length=22, labelsize=font_number)
axs[1,0].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
axs[1,0].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
axs[1,0].set_title('09/18 11:00 PM', family='Arial',
                   style='normal', size=font_title, loc= 'center', fontweight = 'bold')

axs[1,0].annotate("d", xy = (0,0), xytext=(588,618), #xytext=(20,550) cuando se quiere al otro extremo
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))

cb = plt.colorbar(HWM_d,location="right",fraction=0.04, pad=0.04,ax=axs[1,0])
cb.set_label("Flooding depth (ft)", family = 'Arial', size = font_cbar, style = 'normal', fontweight = 'bold', labelpad = 20)
cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out', length=22)
#-------Pannel 5------#
Lumber_WS.plot(ax = axs[1,1], color = 'bisque', edgecolor = 'dimgray',alpha = 0.4)
HWM_e = axs[1,1].scatter(Lumb_HWM['Long'],Lumb_HWM['Lat'],norm = norm_lumber, cmap = cmap_kr, c =Lumb_HWM['h_263'], s = marker_size)
axs[1,1].set_xticklabels(labels=Lumb_HWM['Long'],fontsize = font_number)
axs[1,1].set_yticklabels(labels=Lumb_HWM['Lat'],fontsize = font_number)
axs[1,1].tick_params(axis='x', direction = 'out', length=22, labelsize=font_number)
axs[1,1].tick_params(axis='y', direction = 'out', length=22, labelsize=font_number)
axs[1,1].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
axs[1,1].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
axs[1,1].set_title('09/20 11:00 PM', family='Arial',
                   style='normal', size=font_title, loc= 'center', fontweight = 'bold')

axs[1,1].annotate("e", xy = (0,0), xytext=(588,618), #xytext=(20,550) cuando se quiere al otro extremo
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))

cb = plt.colorbar(HWM_e,location="right",fraction=0.04, pad=0.04,ax=axs[1,1])
cb.set_label("Flooding depth (ft)", family = 'Arial', size = font_cbar, style = 'normal', fontweight = 'bold', labelpad = 20)
cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out', length=22)
#-------Pannel 6------#
Lumber_WS.plot(ax = axs[1,2], color = 'bisque', edgecolor = 'dimgray',alpha = 0.4)
HWM_f = axs[1,2].scatter(Lumb_HWM['Long'],Lumb_HWM['Lat'],norm = norm_lumber, cmap = cmap_kr, c =Lumb_HWM['h_311'], s = marker_size)
axs[1,2].set_xticklabels(labels=Lumb_HWM['Long'],fontsize = font_number)
axs[1,2].set_yticklabels(labels=Lumb_HWM['Lat'],fontsize = font_number)
axs[1,2].tick_params(axis='x', direction = 'out', length=22, labelsize=font_number)
axs[1,2].tick_params(axis='y', direction = 'out', length=22, labelsize=font_number)
axs[1,2].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
axs[1,2].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
axs[1,2].set_title('09/22 11:00 PM', family='Arial',
                   style='normal', size=font_title, loc= 'center', fontweight = 'bold')

axs[1,2].annotate("f", xy = (0,0), xytext=(600,618), #xytext=(20,550) cuando se quiere al otro extremo
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))

cb = plt.colorbar(HWM_f,location="right",fraction=0.04, pad=0.04,ax=axs[1,2])
cb.set_label("Flooding depth (ft)", family = 'Arial', size = font_cbar, style = 'normal', fontweight = 'bold', labelpad = 20)
cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out', length=22)

plt.tight_layout()
plt.savefig(folder_images+'Lumber_HWM_depth.png', dpi=150, bbox_inches='tight')
plt.show()
plt.clf()

###############################################################################################################
##########################################------------VARIOGRAM----------------################################
###############################################################################################################

fig, axs = plt.subplots(2,3,figsize = (40,24))    

#-------Pannel 1------#
for i in Lumber_hwm_ls:
    fig = sns.lineplot(x = "Date", y = str(i), palette='viridis',data = TS_Lumber[TS_Lumber[str(i)]>=0], ax=axs[0,0])

#x_dates = TS_Lumber['Date'].dt.strftime('%m-%d-%r').sort_values().unique()
#axs[0,0].set_xticklabels(labels=x_dates, rotation=45, ha='right',fontsize = font_number)
axs[0,0].tick_params(axis='x', labelsize=font_number, labelrotation = 45) #nuevo
axs[0,0].tick_params(axis='y', labelsize=font_number)
axs[0,0].xaxis.set_major_locator(mdates.HourLocator(interval=24))
#axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%r')) #nuevo con hora
axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d')) #nuevo
axs[0,0].set_xlabel(' ', fontsize=12)
axs[0,0].set_ylabel('Flooding depth (ft)', fontsize=font_title)
axs[0,0].set_title('a) Lumber: Flood depth (ft)', family='Arial',
                                   style='normal', size=font_title, loc= 'left')

#-------Pannel 2------#
# Variogram
V1 = skg.Variogram(Lumb_HWM[['Long', 'Lat']].values, Lumb_HWM['h_150'].values, maxlag='median', normalize=False)
V1.plot(show=False, hist = True, axes = axs[0,1])
axs[0,1].set_title('b) Variogram: 09/16 6:00 AM', family='Arial',
                                   style='normal', size=font_title, loc= 'left')
axs[0,1].tick_params(axis='x', labelsize=font_number)
axs[0,1].tick_params(axis='y', labelsize=font_number)
axs[0,1].set_xlabel('Lag (-)', fontsize=font_title)
axs[0,1].set_ylabel('Semivariance (matheron)', fontsize=font_title)

#-------Pannel 3------#
V2 = skg.Variogram(Lumb_HWM[['Long', 'Lat']].values, Lumb_HWM['h_167'].values, maxlag='median', normalize=False)
V2.plot(show=False, axes = axs[0,2])
axs[0,2].set_title('c) Variogram: 09/16 11:00 PM', family='Arial',
                                   style='normal', size=font_title, loc= 'left')
axs[0,2].tick_params(axis='x', labelsize=font_number)
axs[0,2].tick_params(axis='y', labelsize=font_number)
axs[0,2].set_xlabel('Lag (-)', fontsize=font_title)
axs[0,2].set_ylabel('Semivariance (matheron)', fontsize=font_title)

#-------Pannel 4------#
V3 = skg.Variogram(Lumb_HWM[['Long', 'Lat']].values, Lumb_HWM['h_215'].values, maxlag='median', normalize=False)
V3.plot(show=False, axes = axs[1,0])
axs[1,0].set_title('d) Variogram: 09/18 11:00 PM', family='Arial',
                                   style='normal', size=font_title, loc= 'left')
axs[1,0].tick_params(axis='x', labelsize=font_number)
axs[1,0].tick_params(axis='y', labelsize=font_number)
axs[1,0].set_xlabel('Lag (-)', fontsize=font_title)
axs[1,0].set_ylabel('Semivariance (matheron)', fontsize=font_title)

#-------Pannel 5------#
V4 = skg.Variogram(Lumb_HWM[['Long', 'Lat']].values, Lumb_HWM['h_263'].values, maxlag='median', normalize=False)
V4.plot(show=False, axes = axs[1,1])
axs[1,1].set_title('e) Variogram: 09/20 11:00 PM', family='Arial',
                                   style='normal', size=font_title, loc= 'left')
axs[1,1].tick_params(axis='x', labelsize=font_number)
axs[1,1].tick_params(axis='y', labelsize=font_number)
axs[1,1].set_xlabel('Lag (-)', fontsize=font_title)
axs[1,1].set_ylabel('Semivariance (matheron)', fontsize=font_title)

#-------Pannel 6------#
V5 = skg.Variogram(Lumb_HWM[['Long', 'Lat']].values, Lumb_HWM['h_311'].values, maxlag='median', normalize=False)
V5.plot(show=False, axes = axs[1,2])
axs[1,2].set_title('f) Variogram: 09/22 11:00 PM', family='Arial',
                                   style='normal', size=font_title, loc= 'left')
axs[1,2].tick_params(axis='x', labelsize=font_number)
axs[1,2].tick_params(axis='y', labelsize=font_number)
axs[1,2].set_xlabel('Lag (-)', fontsize=font_title)
axs[1,2].set_ylabel('Semivariance (matheron)', fontsize=font_title)

plt.tight_layout()
plt.savefig(folder_images+'Lumber_HWM_variogram.png', dpi=150, bbox_inches='tight')
plt.show()
plt.clf()

###############################################################################################################
###################################------------VARIOGRAM-WITH-MARKER-SIZE---------#############################
###############################################################################################################

fig, axs = plt.subplots(2,3,figsize = (42,24))  

#Space between the plots
plt.subplots_adjust(hspace= 0.35, wspace = 0.25) 

# Arial family for all the figures
matplotlib.rcParams['font.family'] = "Arial" 

#-------Pannel 1------#
for i in Lumber_hwm_ls:
    fig = sns.lineplot(x = "Date", y = str(i), palette='viridis',data = TS_Lumber[TS_Lumber[str(i)]>=0], ax=axs[0,0])

axs[0,0].tick_params(axis='x', direction = 'out', length=22, labelsize=font_number) #nuevo
axs[0,0].tick_params(axis='y', direction = 'out', length=22, labelsize=font_number)
axs[0,0].xaxis.set_major_locator(mdates.HourLocator(interval=95))
axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d\n%I:%M %p'))
axs[0,0].set_xlabel(' ', fontsize=12)
axs[0,0].set_ylabel('Flooding depth (ft)', fontsize=font_title, fontweight = 'bold', labelpad = 20)
axs[0,0].axvline(pd.Timestamp('2018-09-16 06:00:00'),ls='-.',color='k', lw = 6)
axs[0,0].axvline(pd.Timestamp('2018-09-16 23:00:00'),ls='-.',color='k', lw = 6)
axs[0,0].axvline(pd.Timestamp('2018-09-18 23:00:00'),ls='-.',color='k', lw = 6)
axs[0,0].axvline(pd.Timestamp('2018-09-20 23:00:00'),ls='-.',color='k', lw = 6)
axs[0,0].axvline(pd.Timestamp('2018-09-22 23:00:00'),ls='-.',color='k', lw = 6)
axs[0,0].annotate("a", xy = (0,0), xytext=(730,588),#xytext=(620,550), 
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))
axs[0,0].set_title('Lumber: Flood depth (ft)', family='Arial', style='normal',
                   size=font_title, loc= 'center', fontweight = 'bold')

#-------Pannel 2------#
V_model_1 = V1.to_gstools()

# get the empirical for the plot as well
bins, gamma = V1.get_empirical(bin_center=True)  

#plotting
V_model_1.plot(ax = axs[0,1], lw = 6)
axs[0,1].scatter(bins, gamma, s = marker_size, c = 'red')
axs[0,1].set_title('Variogram: 09/16 6:00 AM', family='Arial',
                   style='normal', size=font_title, loc= 'center', fontweight = 'bold')
axs[0,1].tick_params(axis='x', direction = 'out', length=22, labelsize=font_number)
axs[0,1].tick_params(axis='y', direction = 'out', length=22, labelsize=font_number)
axs[0,1].annotate("b", xy = (0,0), xytext=(730,588), #xytext=(20,550) cuando se quiere al otro extremo
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))
axs[0,1].set_xlabel('Lag (-)', fontsize=font_title)
axs[0,1].set_ylabel('Semivariance (matheron)', fontsize=font_title, fontweight = 'bold')
axs[0,1].legend(fontsize = font_legend, loc = 'lower right')
#-------Pannel 3------#
V_model_2 = V2.to_gstools()

# get the empirical for the plot as well
bins, gamma = V2.get_empirical(bin_center=True)  

#plotting
V_model_2.plot(ax = axs[0,2], lw = 6)
axs[0,2].scatter(bins, gamma,s = marker_size, c = 'red')
axs[0,2].set_title('Variogram: 09/16 11:00 PM', family='Arial',
                   style='normal', size=font_title, loc= 'center', fontweight = 'bold')
axs[0,2].tick_params(axis='x', direction = 'out', length=22, labelsize=font_number)
axs[0,2].tick_params(axis='y', direction = 'out', length=22, labelsize=font_number)
axs[0,2].annotate("c", xy = (0,0), xytext=(730,588), #xytext=(20,550) cuando se quiere al otro extremo
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))
axs[0,2].set_xlabel('Lag (-)', fontsize=font_title)
axs[0,2].set_ylabel('Semivariance (matheron)', fontsize=font_title, fontweight = 'bold')
axs[0,2].legend(fontsize = font_legend, loc = 'lower right')


#-------Pannel 4------#
V_model_3 = V3.to_gstools()

# get the empirical for the plot as well
bins, gamma = V3.get_empirical(bin_center=True)  

#plotting
V_model_3.plot(ax = axs[1,0], lw = 6)
axs[1,0].scatter(bins, gamma,s = marker_size, c = 'red')
axs[1,0].set_title('Variogram: 09/18 11:00 PM', family='Arial',
                   style='normal', size=font_title, loc= 'center', fontweight = 'bold')
axs[1,0].tick_params(axis='x', direction = 'out', length=22, labelsize=font_number)
axs[1,0].tick_params(axis='y', direction = 'out', length=22, labelsize=font_number)
axs[1,0].annotate("d", xy = (0,0), xytext=(730,588), #xytext=(20,550) cuando se quiere al otro extremo
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))
axs[1,0].set_xlabel('Lag (-)', fontsize=font_title)
axs[1,0].set_ylabel('Semivariance (matheron)', fontsize=font_title, fontweight = 'bold')
axs[1,0].legend(fontsize = font_legend, loc = 'lower right')


#-------Pannel 5------#
V_model_4 = V4.to_gstools()

# get the empirical for the plot as well
bins, gamma = V4.get_empirical(bin_center=True)  

#plotting
V_model_4.plot(ax = axs[1,1], lw = 6)
axs[1,1].scatter(bins, gamma,s = marker_size, c = 'red')
axs[1,1].set_title('Variogram: 09/20 11:00 PM', family='Arial',
                   style='normal', size=font_title, loc= 'center', fontweight = 'bold')
axs[1,1].tick_params(axis='x', direction = 'out', length=22, labelsize=font_number)
axs[1,1].tick_params(axis='y', direction = 'out', length=22, labelsize=font_number)
axs[1,1].annotate("e", xy = (0,0), xytext=(730,588), #xytext=(20,550) cuando se quiere al otro extremo
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))
axs[1,1].set_xlabel('Lag (-)', fontsize=font_title)
axs[1,1].set_ylabel('Semivariance (matheron)', fontsize=font_title, fontweight = 'bold')
axs[1,1].legend(fontsize = font_legend, loc = 'lower right')


#-------Pannel 6------#
V_model_5 = V5.to_gstools()

# get the empirical for the plot as well
bins, gamma = V5.get_empirical(bin_center=True)  

#plotting
V_model_5.plot(ax = axs[1,2], lw = 6)
axs[1,2].scatter(bins, gamma,s = marker_size, c = 'red')
axs[1,2].set_title('Variogram: 09/22 11:00 PM', family='Arial',
                   style='normal', size=font_title, loc= 'center', fontweight = 'bold')
axs[1,2].tick_params(axis='x', direction = 'out', length=22, labelsize=font_number)
axs[1,2].tick_params(axis='y', direction = 'out', length=22, labelsize=font_number)
axs[1,2].annotate("f", xy = (0,0), xytext=(730,588), #xytext=(20,550) cuando se quiere al otro extremo
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))
axs[1,2].set_xlabel('Lag (-)', fontsize=font_title)
axs[1,2].set_ylabel('Semivariance (matheron)', fontsize=font_title, fontweight = 'bold')
axs[1,2].legend(fontsize = font_legend, loc = 'lower right')


plt.tight_layout()
plt.savefig(folder_images+'Lumber_HWM_variogram_v2.png', dpi=150, bbox_inches='tight')
plt.show()
plt.clf()

###############################################################################################################
##########################################------------KRIGING----------------##################################
###############################################################################################################

#Krig norm
krig_min = 0#-2
krig_max = 6

gridx = np.arange(-79.75, -78.00, 0.01)
gridy = np.arange(33.80, 35.40, 0.01)
top_left_lon = -79.75
top_left_lat = 35.40
extent_mat = (top_left_lon, top_left_lon + Lumber_WS.shape[1] * 0.21875, top_left_lat - Lumber_WS.shape[0] * 1.6, top_left_lat)

xmin, xmax = min(gridx), max(gridx)
ymin, ymax = min(gridy), max(gridy)

# We determine the resolution in x direction:
res_x = 100
# We determine the resolution in y direction
# based on res_x to make a regular grid:
res_y = int((ymax - ymin)*res_x/(xmax - xmin))

# We build the grid:
xx,yy = np.mgrid[xmin:xmax:complex(res_x), 
                 ymin:ymax:complex(res_y)]

fig, axs = plt.subplots(2,3,figsize = (40,24))    
#-------Pannel 1------#
for i in Lumber_hwm_ls:
    fig = sns.lineplot(x = "Date", y = str(i), palette='viridis',data = TS_Lumber[TS_Lumber[str(i)]>=0], ax=axs[0,0])

axs[0,0].tick_params(axis='x', direction = 'out', length=22, labelsize=font_number) #nuevo
axs[0,0].tick_params(axis='y', direction = 'out', length=22, labelsize=font_number)
axs[0,0].xaxis.set_major_locator(mdates.HourLocator(interval=95))
axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d\n%r'))
axs[0,0].set_xlabel(' ', fontsize=12)
axs[0,0].set_ylabel('Flooding depth (ft)', fontsize=font_title, fontweight = 'bold')
axs[0,0].axvline(pd.Timestamp('2018-09-16 06:00:00'),ls='-.',color='k', lw = 6)
axs[0,0].axvline(pd.Timestamp('2018-09-16 23:00:00'),ls='-.',color='k', lw = 6)
axs[0,0].axvline(pd.Timestamp('2018-09-18 23:00:00'),ls='-.',color='k', lw = 6)
axs[0,0].axvline(pd.Timestamp('2018-09-20 23:00:00'),ls='-.',color='k', lw = 6)
axs[0,0].axvline(pd.Timestamp('2018-09-22 23:00:00'),ls='-.',color='k', lw = 6)
axs[0,0].annotate("a", xy = (0,0), xytext=(685,628),#xytext=(620,550), 
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))
axs[0,0].set_title('Lumber: Flood depth (ft)', family='Arial', style='normal',
                   size=font_title, loc= 'center', fontweight = 'bold')

#-------Pannel 2------#

#Kriging
ok1 = OrdinaryKriging(V1, min_points=5, max_points=20, mode='exact')
    
field1 = ok1.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)
s2_1 = ok1.sigma.reshape(xx.shape)

ax = axs[0,1]

# We add the location of points of our dataset
Lumb_HWM.plot(ax = ax, c = "black", marker= '.', markersize = 2)
#Background 
#ax.imshow(data_LU, vmin = 0, vmax = 6, extent = extent_mat, cmap = 'viridis_r')
#Watershed
Lumber_WS.plot(ax = ax, facecolor = 'none', edgecolor = 'black')

# Contour fringes of the kriging process:
ctr_hh = ax.contourf(xx, yy, field1,
                     vmin = krig_min,
                     vmax = krig_max,
                 #   range(0,13,1),
                     cmap = cmap_kr, 
                     alpha = 0.5,
                     extent = extent_mat)

ax.set_title("Ordinary kriging estimation: 09/16/18 6:00 AM")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ctr_hh, cax=cax, label = 'Flooding depth [ft]')

#-------Pannel 3------#

#Kriging
ok2 = OrdinaryKriging(V2, min_points=5, max_points=20, mode='exact')
    
field2 = ok2.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)
s2_2 = ok2.sigma.reshape(xx.shape)

ax = axs[0,2]

# We add the location of points of our dataset
Lumb_HWM.plot(ax = ax, c = "black", marker= '.', markersize = 2)
Lumber_WS.plot(ax = ax, color = 'white', edgecolor = 'black')

# Contour fringes of the kriging process:
ctr_hh = ax.contourf(xx, yy, field2,
                     vmin = krig_min,
                     vmax = krig_max,
                 #   range(0,13,1),
                     cmap = cmap_kr, 
                     alpha = 0.5,
                     extent = extent_mat)

ax.set_title("Ordinary kriging estimation: 09/16/18 11:00 PM")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ctr_hh, cax=cax, label = 'Flooding depth [ft]')

#-------Pannel 4------#

#Kriging
ok3 = OrdinaryKriging(V3, min_points=5, max_points=20, mode='exact')
    
field3 = ok3.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)
s2_3 = ok3.sigma.reshape(xx.shape)

ax = axs[1,0]

# We add the location of points of our dataset
Lumb_HWM.plot(ax = ax, c = "black", marker= '.', markersize = 2)
Lumber_WS.plot(ax = ax, color = 'white', edgecolor = 'black')

# Contour fringes of the kriging process:
ctr_hh = ax.contourf(xx, yy, field3,
                     vmin = krig_min,
                     vmax = krig_max,
                 #   range(0,13,1),
                     cmap = cmap_kr, 
                     alpha = 0.5,
                     extent = extent_mat)

ax.set_title("Ordinary kriging estimation: 09/18/18 11:00 PM")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ctr_hh, cax=cax, label = 'Flooding depth [ft]')

#-------Pannel 5------#

#Kriging
ok4 = OrdinaryKriging(V4, min_points=5, max_points=20, mode='exact')
    
field4 = ok4.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)
s2_4 = ok4.sigma.reshape(xx.shape)

ax = axs[1,1]

# We add the location of points of our dataset
Lumb_HWM.plot(ax = ax, c = "black", marker= '.', markersize = 2)
Lumber_WS.plot(ax = ax, color = 'white', edgecolor = 'black')

# Contour fringes of the kriging process:
ctr_hh = ax.contourf(xx, yy, field4,
                     vmin = krig_min,
                     vmax = krig_max,
                 #   range(0,13,1),
                     cmap = cmap_kr, 
                     alpha = 0.5,
                     extent = extent_mat)

ax.set_title("Ordinary kriging estimation: 09/20/18 11:00 PM")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ctr_hh, cax=cax, label = 'Flooding depth [ft]')

#-------Pannel 6------#

#Kriging
ok5 = OrdinaryKriging(V5, min_points=5, max_points=20, mode='exact')
    
field5 = ok5.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)
s2_5 = ok5.sigma.reshape(xx.shape)

ax = axs[1,2]

# We add the location of points of our dataset
Lumb_HWM.plot(ax = ax, c = "black", marker= '.', markersize = 2)
Lumber_WS.plot(ax = ax, color = 'white', edgecolor = 'black')

# Contour fringes of the kriging process:
ctr_hh = ax.contourf(xx, yy, field5,
                     vmin = krig_min,
                     vmax = krig_max,
                 #   range(0,13,1),
                     cmap = cmap_kr, 
                     alpha = 0.5,
                     extent = extent_mat)

ax.set_title("Ordinary kriging estimation: 09/22/18 11:00 PM")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ctr_hh, cax=cax, label = 'Flooding depth [ft]')

###############################################################################################################
##########################################------------KRIGING-MASK-----------##################################
###############################################################################################################

#%%Cutting the watersheds to the border
    
# We initiale a mask with 0 everywhere with the similar shape as earlier
maskin = xx - xx
    
# In this mask, we assign True if the point is inside the watershed
# and we assign False if the point is outside the watershed
for i in range(maskin.shape[0]):
    for j in range(maskin.shape[1]):
        xi = xx[i][j]
        yj = yy[i][j]
            
        if Point(xi, yj).within(Lumber_WS["geometry"][0]):
            maskin[i, j] = 0
        else:
            maskin[i, j] = 1
    

fig, axs = plt.subplots(2,3,figsize = (42,24))   

#Space between the plots
plt.subplots_adjust(hspace= 0.25, wspace = 0.38)  
#-------Pannel 1------#
for i in Lumber_hwm_ls:
    fig = sns.lineplot(x = "Date", y = str(i), palette='viridis',data = TS_Lumber[TS_Lumber[str(i)]>=0], ax=axs[0,0])

axs[0,0].tick_params(axis='x', direction = 'out', length=22, labelsize=font_number) #nuevo
axs[0,0].tick_params(axis='y', direction = 'out', length=22, labelsize=font_number)
axs[0,0].xaxis.set_major_locator(mdates.HourLocator(interval=95))
axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d\n%I:%M %p'))
axs[0,0].set_xlabel(' ', fontsize=12)
axs[0,0].set_ylabel('Flooding depth (ft)', fontsize=font_title, fontweight = 'bold', labelpad = 20)
axs[0,0].axvline(pd.Timestamp('2018-09-16 06:00:00'),ls='-.',color='k', lw = 6)
axs[0,0].axvline(pd.Timestamp('2018-09-16 23:00:00'),ls='-.',color='k', lw = 6)
axs[0,0].axvline(pd.Timestamp('2018-09-18 23:00:00'),ls='-.',color='k', lw = 6)
axs[0,0].axvline(pd.Timestamp('2018-09-20 23:00:00'),ls='-.',color='k', lw = 6)
axs[0,0].axvline(pd.Timestamp('2018-09-22 23:00:00'),ls='-.',color='k', lw = 6)
axs[0,0].annotate("a", xy = (0,0), xytext=(730,608),#xytext=(620,550), 
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))
axs[0,0].set_title('Lumber: Flood depth (ft)', family='Arial', style='normal',
                   size=font_title, loc= 'center', fontweight = 'bold')

#-------Pannel 2------#

#Kriging
ax = axs[0,1]

# We apply our mask to previous calculated arrays:
depth_ma_1 = np.ma.masked_array(field1, maskin)
s2_ma_1 = np.ma.masked_array(s2_1, maskin)

# We add the location of points of our dataset
Lumb_HWM.plot(ax = ax, c = "black", marker= '.', markersize = 14)
#Background 
#ax.imshow(data_LU, vmin = 0, vmax = 6, extent = extent_mat, cmap = 'viridis_r')
#Watershed
Lumber_WS.plot(ax = ax, facecolor = 'none', edgecolor = 'black')

# Contour fringes of the kriging process:
ctr_hh = ax.contourf(xx, yy, depth_ma_1, 
                     vmin = krig_min,
                     vmax = krig_max,
                     norm = norm_lumber,
                     #   range(0, 13, 1),
                     cmap = cmap_kr, 
                     #   alpha = 0.9,
                     extent = extent_mat)

#Tag
ax.annotate("b", xy = (0,0), xytext=(557,608), #xytext=(20,550) cuando se quiere al otro extremo
            xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
            bbox = dict(boxstyle = "square", fc = "black"))

#Colorbar
#Colorbar
sm_k = plt.cm.ScalarMappable(cmap=cmap_kr, norm=norm_lumber)
sm_k.set_array([])

cb = plt.colorbar(sm_k, location="right",fraction=0.04, pad=0.04,ax=ax)
cb.set_label("Flooding depth (ft)", family = 'Arial', size = font_cbar, style = 'normal', fontweight = 'bold', labelpad = 20)
cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out', length=22)

ax.tick_params(axis='x', direction = 'out', length=22, labelsize=font_number)
ax.tick_params(axis='y', direction = 'out', length=22, labelsize=font_number)
ax.set_title('Flood: 09/16 6:00 AM', family='Arial',
             style='normal', size=font_title, loc= 'center', fontweight = 'bold')


#-------Pannel 3------#

#Kriging
ax = axs[0,2]

# We apply our mask to previous calculated arrays:
depth_ma_2 = np.ma.masked_array(field2, maskin)
s2_ma_2 = np.ma.masked_array(s2_2, maskin)

# We add the location of points of our dataset
Lumb_HWM.plot(ax = ax, c = "black", marker= '.', markersize = 2)
#Background 
#ax.imshow(data_LU, vmin = 0, vmax = 6, extent = extent_mat, cmap = 'viridis_r')
#Watershed
Lumber_WS.plot(ax = ax, facecolor = 'none', edgecolor = 'black')

# Contour fringes of the kriging process:
ctr_hh = ax.contourf(xx, yy, depth_ma_2, 
                     vmin = krig_min,
                     vmax = krig_max,
                     norm = norm_lumber,
                     #   range(0, 13, 1),
                     cmap = cmap_kr, 
                     #   alpha = 0.9,
                     extent = extent_mat)

#Tag
ax.annotate("c", xy = (0,0), xytext=(557,608), #xytext=(20,550) cuando se quiere al otro extremo
            xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
            bbox = dict(boxstyle = "square", fc = "black"))

#Colorbar
sm_k2 = plt.cm.ScalarMappable(cmap=cmap_kr, norm=norm_lumber)
sm_k2.set_array([])

cb = plt.colorbar(sm_k2, location="right",fraction=0.04, pad=0.04,ax=ax)
cb.set_label("Flooding depth (ft)", family = 'Arial', size = font_cbar, style = 'normal', fontweight = 'bold', labelpad = 20)
cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out', length=22)

ax.tick_params(axis='x', direction = 'out', length=22, labelsize=font_number)
ax.tick_params(axis='y', direction = 'out', length=22, labelsize=font_number)
ax.set_title("Flood: 09/16/18 11:00 PM",family='Arial',
             style='normal', size=font_title, loc= 'center', fontweight = 'bold')
    

#-------Pannel 4------#

#Kriging
ax = axs[1,0]

# We apply our mask to previous calculated arrays:
depth_ma_3 = np.ma.masked_array(field3, maskin)
s2_ma_3 = np.ma.masked_array(s2_3, maskin)

# We add the location of points of our dataset
Lumb_HWM.plot(ax = ax, c = "black", marker= '.', markersize = 2)
#Background 
#ax.imshow(data_LU, vmin = 0, vmax = 6, extent = extent_mat, cmap = 'viridis_r')
#Watershed
Lumber_WS.plot(ax = ax, facecolor = 'none', edgecolor = 'black')

# Contour fringes of the kriging process:
ctr_hh = ax.contourf(xx, yy, depth_ma_3, 
                     vmin = krig_min,
                     vmax = krig_max,
                     norm = norm_lumber,
                     #   range(0, 13, 1),
                     cmap = cmap_kr, 
                     #   alpha = 0.9,
                     extent = extent_mat)

#Tag
ax.annotate("d", xy = (0,0), xytext=(557,608), #xytext=(20,550) cuando se quiere al otro extremo
            xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
            bbox = dict(boxstyle = "square", fc = "black"))

#Colorbar
sm_k3 = plt.cm.ScalarMappable(cmap=cmap_kr, norm=norm_lumber)
sm_k3.set_array([])

cb = plt.colorbar(sm_k3, location="right",fraction=0.04, pad=0.04,ax=ax)
cb.set_label("Flooding depth (ft)", family = 'Arial', size = font_cbar, style = 'normal', fontweight = 'bold', labelpad = 20)
cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out', length=22)

ax.tick_params(axis='x', direction = 'out', length=22, labelsize=font_number)
ax.tick_params(axis='y', direction = 'out', length=22, labelsize=font_number)
ax.set_title("Flood: 09/18/18 11:00 PM",family='Arial',
             style='normal', size=font_title, loc= 'center', fontweight = 'bold')    

#-------Pannel 5------#

#Kriging
ax = axs[1,1]

# We apply our mask to previous calculated arrays:
depth_ma_4 = np.ma.masked_array(field4, maskin)
s2_ma_4 = np.ma.masked_array(s2_4, maskin)

# We add the location of points of our dataset
Lumb_HWM.plot(ax = ax, c = "black", marker= '.', markersize = 2)
#Background 
#ax.imshow(data_LU, vmin = 0, vmax = 6, extent = extent_mat, cmap = 'viridis_r')
#Watershed
Lumber_WS.plot(ax = ax, facecolor = 'none', edgecolor = 'black')

# Contour fringes of the kriging process:
ctr_hh = ax.contourf(xx, yy, depth_ma_4, 
                     vmin = krig_min,
                     vmax = krig_max,
                     norm = norm_lumber,
                     #   range(0, 13, 1),
                     cmap = cmap_kr, 
                     #   alpha = 0.9,
                     extent = extent_mat)

#Tag
ax.annotate("e", xy = (0,0), xytext=(557,608), #xytext=(20,550) cuando se quiere al otro extremo
            xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
            bbox = dict(boxstyle = "square", fc = "black"))

#Colorbar
sm_k4 = plt.cm.ScalarMappable(cmap=cmap_kr, norm=norm_lumber)
sm_k4.set_array([])

cb = plt.colorbar(sm_k4, location="right",fraction=0.04, pad=0.04,ax=ax)
cb.set_label("Flooding depth (ft)", family = 'Arial', size = font_cbar, style = 'normal', fontweight = 'bold', labelpad = 20)
cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out', length=22)

ax.tick_params(axis='x', direction = 'out', length=22, labelsize=font_number)
ax.tick_params(axis='y', direction = 'out', length=22, labelsize=font_number)
ax.set_title("Flood: 09/20/18 11:00 PM",family='Arial',
             style='normal', size=font_title, loc= 'center', fontweight = 'bold')
    
#-------Pannel 6------#

#Kriging
ax = axs[1,2]

# We apply our mask to previous calculated arrays:
depth_ma_5 = np.ma.masked_array(field5, maskin)
s2_ma_5 = np.ma.masked_array(s2_5, maskin)

# We add the location of points of our dataset
Lumb_HWM.plot(ax = ax, c = "black", marker= '.', markersize = 2)
#Background 
#ax.imshow(data_LU, vmin = 0, vmax = 6, extent = extent_mat, cmap = 'viridis_r')
#Watershed
Lumber_WS.plot(ax = ax, facecolor = 'none', edgecolor = 'black')

# Contour fringes of the kriging process:
ctr_hh = ax.contourf(xx, yy, depth_ma_5, 
                     vmin = krig_min,
                     vmax = krig_max,
                     #   range(0, 13, 1),
                     norm = norm_lumber,
                     cmap = cmap_kr, 
                     #   alpha = 0.9,
                     extent = extent_mat)

#Tag
ax.annotate("f", xy = (0,0), xytext=(557,608), #xytext=(20,550) cuando se quiere al otro extremo
            xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
            bbox = dict(boxstyle = "square", fc = "black"))

#Colorbar
sm_k5 = plt.cm.ScalarMappable(cmap=cmap_kr, norm=norm_lumber)
sm_k5.set_array([])

cb = plt.colorbar(sm_k5, location="right",fraction=0.04, pad=0.04,ax=ax)
cb.set_label("Flooding depth (ft)", family = 'Arial', size = font_cbar, style = 'normal', fontweight = 'bold', labelpad = 20)
cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out', length=22)

ax.tick_params(axis='x', direction = 'out', length=22, labelsize=font_number)
ax.tick_params(axis='y', direction = 'out', length=22, labelsize=font_number)
ax.set_title("Flood: 09/22/18 11:00 PM",family='Arial',
             style='normal', size=font_title, loc= 'center', fontweight = 'bold')
        
#plt.clim(0, 6)
plt.tight_layout()
plt.savefig(folder_images+'Lumber_kriging_v2.png', dpi=150, bbox_inches='tight')
plt.show()
plt.clf()