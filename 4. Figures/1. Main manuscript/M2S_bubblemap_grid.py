# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 22:30:44 2023

@author: lprieto
"""
# libraries
import pypsa
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy
from shapely.geometry import Point
from pylab import axes
from matplotlib.lines import Line2D
import matplotlib as mpl
import matplotlib.font_manager as font_manager #to standarize font
import matplotlib.patches as mpatches

#folder location of the data
folder_github = 'D:/2. Research/5. Latest data/13. Collaboration flood/Images_1pp/Inputs/'
folder_images = 'D:/5. Jordan_1st_paper/Images/Latest/' #final images are in this folder
during = 'D:/2. Research/5. Latest data/17. Figures/M2S_network/during/'
gis = 'D:/18. 2nd paper/1. Drew/'

#Font for the figures
font = font_manager.FontProperties(family='Arial',
                                   style='normal', size=17.5)
size_tl = 28
size_tick = 24

#Font
font_number = 50#17
font_title = 85#25
font_cbar = 30
marker_size = 180
font_legend = 60#17
font_box = 88
multiplo = 15

#Boundaries
NC_bound = gpd.read_file(folder_github + 'NC_boundary.shp')

#Power Plants
Egrid_18 = pd.read_csv(folder_github+'Egrid_powerplants_v4.csv')

#Projection
proj = 'EPSG:4269'

Egrid_18['geometry'] = Egrid_18.apply(lambda x: Point((float(x.Long), float(x.Lat))), axis=1)
M2S_PP = gpd.GeoDataFrame(Egrid_18[['netcap_MW','fuel_type','Long','Lat','geometry','Colors']], geometry = 'geometry', crs = proj)
# M2S_PP.netcap_MW.astype(int)

# Save substations and power plants as shapefiles
M2S_PP.to_file(gis+'power_plants.shp')

#network
network = pypsa.Network()
network.import_from_csv_folder(csv_folder_name=during)

##############################################START THE PLOT###############################################################
fig, axs = plt.subplots(2,1,subplot_kw={"projection": ccrs.EqualEarth()}, figsize=(100, 50))    

# Arial family for all the figures
mpl.rcParams['font.family'] = "Arial"

##########################################-------PANNEL A--------##########################################################

#Adding states borders and the ocean
axs[0].add_feature(cartopy.feature.OCEAN)
axs[0].add_feature(cartopy.feature.STATES, edgecolor='gray', facecolors = 'white', alpha=0.99)

axs[0].scatter(M2S_PP['Long'],M2S_PP['Lat'], color = M2S_PP['Colors'], s = M2S_PP['netcap_MW']*multiplo, 
               transform = ccrs.PlateCarree(), edgecolors= 'black')

axs[0].annotate('a', xy = (0,0), xytext=(40,1420), #estas coordenadas de xytext solo en graficos PyPSA
                xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                bbox = dict(boxstyle = "square", fc = "black"))

axs[0].set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

#Manually create legend of the plot
#define handles and labels that will get added to legend
handles_1, labels_1 = axs[0].get_legend_handles_labels()#Manually create legend of the plot
#define patches and lines to add to legend
patch1 = mpatches.Patch(color='cyan', label='Nuclear')
patch2 = mpatches.Patch(color='grey', label='Coal')
patch3 = mpatches.Patch(color='yellow', label='Solar')
patch4 = mpatches.Patch(color='royalblue', label='Hydro')
patch5 = mpatches.Patch(color='darkcyan', label='Wind')
patch6 = mpatches.Patch(color='red', label='Nat. Gas')
patch7 = mpatches.Patch(color='darkviolet', label='Oil')
patch8 = mpatches.Patch(color='forestgreen', label='Biomass')

#add handles
handles_1.extend([patch1, patch2, patch3, patch4, patch5, patch6, patch7, patch8])

#legend
l1 = axs[0].legend(handles = handles_1, loc = 'lower left', bbox_to_anchor = (0.005,0.002), ncol=3, fontsize = font_legend)

#Adding total amount of unserved energy
#ax2 = axs[0].twinx()

#define handles and labels that will get added to legend
handles_2, labels_2 = axs[0].get_legend_handles_labels()#Manually create legend of the plot

#define patches and lines to add to legend
line_1 = Line2D([0], [0], marker = 'o', label='2575 MW', color='None', markerfacecolor='black', markersize=70)
line_2 = Line2D([0], [0], marker = 'o', label=' ', color='None', markerfacecolor='black', markersize=52)
line_3 = Line2D([0], [0], marker = 'o', label=' ', color='None', markerfacecolor='black', markersize=43)
line_4 = Line2D([0], [0], marker = 'o', label=' ', color='None', markerfacecolor='black', markersize=31)
line_5 = Line2D([0], [0], marker = 'o', label='1 MW', color='None', markerfacecolor='black', markersize=24)

#add handles
handles_2.extend([line_1, line_2, line_3, line_4, line_5])

#legend
l2 = axs[0].legend(handles = handles_2, loc = 'lower right', bbox_to_anchor = (1,0.002), ncol=1, fontsize = font_legend)

axs[0].add_artist(l1) #this is to add l1 again

##########################################-------PANNEL B--------##########################################################

network.plot(
    ax=axs[1],
    line_widths= 10,
    line_cmap = 'Reds_r',
    #line_cmap=plt.cm.jet,
    #title="M2S: North Carolina Grid Representation",
    bus_sizes=0.50e-3,
    bus_alpha=0.7,
    bus_colors='black',)

#NC_bound.plot(ax = ax, facecolor='white',edgecolor='black', alpha=0.15)

axs[1].annotate('b', xy = (0,0), xytext=(40,1420), #estas coordenadas de xytext solo en graficos PyPSA
             xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
             bbox = dict(boxstyle = "square", fc = "black"))

axs[1].add_feature(cartopy.feature.OCEAN)
axs[1].add_feature(cartopy.feature.STATES, edgecolor='gray' )
axs[1].set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

#Manually create legend of the plot
#define handles and labels that will get added to legend
handles_3, labels_3 = axs[1].get_legend_handles_labels()#Manually create legend of the plot

#define patches and lines to add to legend
line_6 = Line2D([0], [0], label='High voltage line', color='firebrick', lw=25)
line_7 = Line2D([0], [0], marker = 'o', label='Substation/node', color='None', markerfacecolor='black', markersize=44)

#add handles
handles_3.extend([line_6, line_7])

axs[1].legend(handles = handles_3, loc = 'lower right', bbox_to_anchor = (1,0.002), ncol=1, fontsize = font_legend)

fig.tight_layout()
plt.savefig(folder_images+'M2S_grid_final.png', dpi=150, bbox_inches='tight')
plt.show()
plt.clf()
