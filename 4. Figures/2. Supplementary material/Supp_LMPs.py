# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 21:04:21 2023

@author: lprieto
"""
import fiona
from shapely.geometry import shape,mapping, Point, Polygon, MultiPolygon
import geopandas as gpd
from geopandas.tools import sjoin
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
from shapely.geometry import Point
from pylab import axes
from matplotlib.lines import Line2D
import matplotlib.font_manager as font_manager #to standarize font
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors


#folder location of the data
folder_github = 'D:/2. Research/5. Latest data/13. Collaboration flood/Images_1pp/Inputs/'
folder_images = 'D:/5. Jordan_1st_paper/Images/Latest/' #final images are in this folder
folder_census = 'D:/5. Jordan_1st_paper/7. Shapefiles/'
during = 'D:/2. Research/5. Latest data/17. Figures/M2S_network/during/'
folder_slack = 'D:/5. Jordan_1st_paper/11. LMPS_slack/'
folder_shp = 'D:/5. Jordan_1st_paper/7. Shapefiles/'
folder_3 = 'D:/8. GitHub/M2S_line_outages/Outputs_2/'
folder_inputs = 'D:/15. Globus/RESULTS/INPUT_model/'
folder_slack_jk = 'D:/5. Jordan_1st_paper/11. LMPS_slack/Slack/'

#fonts
font_number = 160#17
font_title = 162#25
font_cbar = 30
marker_size = 900#180
marker_legend = 35
font_legend = 160#17
font_box = 160
font_text = 80
font_length = 80
font_ranges = 80

Pannels = ['a','b','c','d']
Captures = [6198,6216,6263,6312] #all of these captures are in the during range
Depth = ['0ft','1ft','2ft','3ft','4ft','5ft','6ft','7ft','8ft','9ft','10ft']
Time = ['09/16/18 6:00 AM','09/17/18 12:00 AM','09/18/18 11:00 PM','09/21/18 12:00 AM']

i = 0 #Depth
j = 0 #Pannels/captures
k = 0 #Base case capture
t = 0

## OJO: ACA DEBERIA EMPEZAR EL FOR LOOP
# Dataframes
M2S_st_flood_all = pd.read_csv(folder_inputs + 'ST_all_basins_flood.csv')
substations = pd.read_csv(during+'buses.csv')
ST = substations.rename(columns={"name":"Node"})
ST['Lat'] = substations['y']
ST['Long'] = substations['x']


# Initialize an empty list to store the Slack buses
M2S_slack_var  = []

# Assuming 'i' is defined and represents the starting index
for d in range(i, i+11):
    # Construct the file path
    file_path = folder_3 + 'slack_' + Depth[d] + '.csv'
    
    # Read the CSV file and append it to the list
    df_slack = pd.read_csv(file_path, header=0)
    M2S_slack_var.append(df_slack)


# Initialize an empty list to store the dual prices
M2S_duals  = []

# Assuming 'i' is defined and represents the starting index
for d in range(i, i+11):
    # Construct the file path
    file_path = folder_3 + 'duals_' + Depth[d] + '.csv'
    
    # Read the CSV file and append it to the list
    df_duals = pd.read_csv(file_path, header=0)
    M2S_duals.append(df_duals)

# Slack variables
df_slack_during = []

# Define the range of iterations (assuming i starts from 0)
for d in range(i, i+11):
    df_slack = M2S_slack_var[d].loc[(M2S_slack_var[d]['Time']<=6312)&(M2S_slack_var[d]['Time']>6118)&(M2S_slack_var[d]['Value']>2)]
    df_slack_during.append(df_slack)

# Merge the two dataframes on the 'Node' column
merged_df_sl = []

# Iterate over the DataFrames in df_slack_during
for df_slack in df_slack_during:
    # Merge the DataFrame with ST
    merged_df = df_slack.merge(ST[['Node', 'Lat', 'Long']], on='Node', how='left')
    # Append the merged DataFrame to the list
    merged_df_sl.append(merged_df)
    
# Final Slack dataframe
slack_all_m2s = merged_df_sl

# Duals from M2S
df_duals_during = []

for d in range(i, i+11):
    df_duals = M2S_duals[d].loc[(M2S_duals[d]['Time']<=6312)&(M2S_duals[d]['Time']>6118)]
    df_duals_during.append(df_duals)
    
# Merge the two dataframes on the 'Bus' column
merged_df = []

for df_dual in df_duals_during:
    merged_d = df_dual.merge(ST[['Node', 'Lat', 'Long']], left_on='Bus', right_on='Node', how='left')
    merged_df.append(merged_d)

# Final dataframe duals
duals_all_m2s = merged_df

# Outage data during Hurricane Florence
df_outages_nc = pd.read_csv(folder_shp + 'county_outages_hurricane_florence.csv')

#Boundaries
NC_bound = gpd.read_file(folder_github + 'NC_boundary.shp')
census_nc_data = gpd.read_file(folder_shp + 'Census_counties.shp')

# Assuming "NAMELSAD" is the common column between the two dataframes
census_nc = census_nc_data.merge(df_outages_nc, on="NAMELSAD")

# Extra columns
census_nc['Total_popu'][70] = 60203

#Projection
proj = 'EPSG:4269'

M2S_st_flood_all['geometry'] = M2S_st_flood_all.apply(lambda x: Point((float(x.x), float(x.y))), axis=1)
M2S_flood_all = gpd.GeoDataFrame(M2S_st_flood_all, geometry = 'geometry', crs = proj)

substations['geometry'] = substations.apply(lambda x: Point((float(x.x), float(x.y))), axis=1)
M2S_st = gpd.GeoDataFrame(substations, geometry = 'geometry', crs = proj)

# Initialize an empty list to store GeoDataFrames
M2S_duals_all = []

# Iterate over the DataFrames in duals_all_m2s
for df_duals in duals_all_m2s:
    # Convert 'Long' and 'Lat' columns to float and create a 'geometry' column
    df_duals['geometry'] = df_duals.apply(lambda x: Point((float(x.Long), float(x.Lat))), axis=1)
    # Create a GeoDataFrame
    gdf_duals = gpd.GeoDataFrame(df_duals, geometry='geometry', crs=proj)
    # Append the GeoDataFrame to the list
    M2S_duals_all.append(gdf_duals)

# Initialize an empty list to store GeoDataFrames
M2S_slack_all = []

# Iterate over the DataFrames in slack_all_m2s
for df_slack in slack_all_m2s:
    # Convert 'Long' and 'Lat' columns to float and create a 'geometry' column
    df_slack['geometry'] = df_slack.apply(lambda x: Point((float(x.Long), float(x.Lat))), axis=1)
    # Create a GeoDataFrame
    gdf_slack = gpd.GeoDataFrame(df_slack, geometry='geometry', crs=proj)
    # Append the GeoDataFrame to the list
    M2S_slack_all.append(gdf_slack)

# We start to plot here
wspacing = 0.04
hspacing = 0.1

# Norm and cmap for LMPs
cmap_lmp = mpl.cm.coolwarm

lmp_min = 0#-2000#-1200#-340 
lmp_max = 5000#5000#2500#740#630
lmp_center = 1500#3000#100#40 #200

norm_lmp = colors.TwoSlopeNorm(vmin = lmp_min, vcenter = lmp_center, vmax = lmp_max)

#################PLOTTING############################
for t in range(4):
    
    fig, axs = plt.subplots(4,3,subplot_kw={"projection": ccrs.EqualEarth()}, figsize=(150, 94), gridspec_kw={'wspace': wspacing, 'hspace': hspacing})

    # Arial family for all the figures
    mpl.rcParams['font.family'] = "Arial"
    
    # Set the figure title
    
    fig.suptitle("Date: "+Time[t], x=0.5, y=0.92, fontsize=font_title, fontweight='bold', ha='center', family='Arial')

    # FIRST COLUMN

    # 0ft

    ax = axs[0,0]

    # Captures in time
    LMPs_sp = M2S_duals_all[i].loc[(M2S_duals_all[i]['Time']==Captures[t])]

    plot_lmp = ax.scatter(LMPs_sp['Long'],LMPs_sp['Lat'],norm = norm_lmp, cmap = cmap_lmp, 
                                  c = LMPs_sp['Value'], s = marker_size, transform = ccrs.PlateCarree(), edgecolors= 'black')


    # Black box with flooding depth
    ax.annotate(Depth[i], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    # Ocean and state limits
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())


    # 1ft

    ax = axs[0,1]

    # Captures in time
    LMPs_sp = M2S_duals_all[i+1].loc[(M2S_duals_all[i+1]['Time']==Captures[t])]

    plot_lmp = ax.scatter(LMPs_sp['Long'],LMPs_sp['Lat'],norm = norm_lmp, cmap = cmap_lmp, 
                                  c = LMPs_sp['Value'], s = marker_size, transform = ccrs.PlateCarree(), edgecolors= 'black')


    # Black box with flooding depth
    ax.annotate(Depth[i+1], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    # Ocean and state limits
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())

    # 2ft

    ax = axs[0,2]

    # Captures in time
    LMPs_sp = M2S_duals_all[i+2].loc[(M2S_duals_all[i+2]['Time']==Captures[t])]

    plot_lmp = ax.scatter(LMPs_sp['Long'],LMPs_sp['Lat'],norm = norm_lmp, cmap = cmap_lmp, 
                                  c = LMPs_sp['Value'], s = marker_size, transform = ccrs.PlateCarree(), edgecolors= 'black')

    # Black box with flooding depth
    ax.annotate(Depth[i+2], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    # Ocean and state limits
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())

    # 3ft

    ax = axs[1,0]

    # Captures in time
    LMPs_sp = M2S_duals_all[i+3].loc[(M2S_duals_all[i+3]['Time']==Captures[t])]

    plot_lmp = ax.scatter(LMPs_sp['Long'],LMPs_sp['Lat'],norm = norm_lmp, cmap = cmap_lmp, 
                                  c = LMPs_sp['Value'], s = marker_size, transform = ccrs.PlateCarree(), edgecolors= 'black')

    # Black box with flooding depth
    ax.annotate(Depth[i+3], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    # Ocean and state limits
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())

    # 4ft

    ax = axs[1,1]

    # Captures in time
    LMPs_sp = M2S_duals_all[i+4].loc[(M2S_duals_all[i+4]['Time']==Captures[t])]

    plot_lmp = ax.scatter(LMPs_sp['Long'],LMPs_sp['Lat'],norm = norm_lmp, cmap = cmap_lmp, 
                                  c = LMPs_sp['Value'], s = marker_size, transform = ccrs.PlateCarree(), edgecolors= 'black')

    # Black box with flooding depth
    ax.annotate(Depth[i+4], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    # Ocean and state limits
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())

    # 5ft

    ax = axs[1,2]

    # Captures in time
    LMPs_sp = M2S_duals_all[i+5].loc[(M2S_duals_all[i+5]['Time']==Captures[t])]

    plot_lmp = ax.scatter(LMPs_sp['Long'],LMPs_sp['Lat'],norm = norm_lmp, cmap = cmap_lmp, 
                                  c = LMPs_sp['Value'], s = marker_size, transform = ccrs.PlateCarree(), edgecolors= 'black')

    # Black box with flooding depth
    ax.annotate(Depth[i+5], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    # Ocean and state limits
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())

    # 6ft
    ax = axs[2,0]

    # Captures in time
    LMPs_sp = M2S_duals_all[i+6].loc[(M2S_duals_all[i+6]['Time']==Captures[t])]

    plot_lmp = ax.scatter(LMPs_sp['Long'],LMPs_sp['Lat'],norm = norm_lmp, cmap = cmap_lmp, 
                                  c = LMPs_sp['Value'], s = marker_size, transform = ccrs.PlateCarree(), edgecolors= 'black')

    # Black box with flooding depth
    ax.annotate(Depth[i+6], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    # Ocean and state limits
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())

    # 7ft

    ax = axs[2,1]

    # Captures in time
    LMPs_sp = M2S_duals_all[i+7].loc[(M2S_duals_all[i+7]['Time']==Captures[t])]

    plot_lmp = ax.scatter(LMPs_sp['Long'],LMPs_sp['Lat'],norm = norm_lmp, cmap = cmap_lmp, 
                                  c = LMPs_sp['Value'], s = marker_size, transform = ccrs.PlateCarree(), edgecolors= 'black')

    # Black box with flooding depth
    ax.annotate(Depth[i+7], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    # Ocean and state limits
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())

    # 8ft

    ax = axs[2,2]

    # Captures in time
    LMPs_sp = M2S_duals_all[i+8].loc[(M2S_duals_all[i+8]['Time']==Captures[t])]

    plot_lmp = ax.scatter(LMPs_sp['Long'],LMPs_sp['Lat'],norm = norm_lmp, cmap = cmap_lmp, 
                                  c = LMPs_sp['Value'], s = marker_size, transform = ccrs.PlateCarree(), edgecolors= 'black')

    # Black box with flooding depth
    ax.annotate(Depth[i+8], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    # Ocean and state limits
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())

    # 9ft

    ax = axs[3,0]

    # Captures in time
    LMPs_sp = M2S_duals_all[i+9].loc[(M2S_duals_all[i+9]['Time']==Captures[t])]

    plot_lmp = ax.scatter(LMPs_sp['Long'],LMPs_sp['Lat'],norm = norm_lmp, cmap = cmap_lmp, 
                                  c = LMPs_sp['Value'], s = marker_size, transform = ccrs.PlateCarree(), edgecolors= 'black')

    # Black box with flooding depth
    ax.annotate(Depth[i+9], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    # Ocean and state limits
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())

    # 10ft

    ax = axs[3,1]

    # Captures in time
    LMPs_sp = M2S_duals_all[i+10].loc[(M2S_duals_all[i+10]['Time']==Captures[t])]

    plot_lmp = ax.scatter(LMPs_sp['Long'],LMPs_sp['Lat'],norm = norm_lmp, cmap = cmap_lmp, 
                                  c = LMPs_sp['Value'], s = marker_size, transform = ccrs.PlateCarree(), edgecolors= 'black')


    # Black box with flooding depth
    ax.annotate(Depth[i+10], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    # Ocean and state limits
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())
    
    ##############----LEGEND----###
    ax = axs[3,2]
    
    # Turn off the axes and set background color to white
    ax.set_axis_off()
    ax.set_facecolor('white')
    
    # Set the aspect ratio of axs[1,0] to match axs[0,0]
    axs[3,2].set_aspect(axs[0,1].get_aspect())
    
    cb = plt.colorbar(plot_lmp,location="right",fraction=0.0223, pad=0.04,ax=axs)
    cb.set_label("LMP $/MWh", family = 'Arial', size = font_title, style = 'normal', fontweight = 'bold', labelpad=30)
    cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out',length = font_length)
    

    #fig.tight_layout()
    plt.savefig(folder_images+'M2S_SP_LMPs_'+str(t)+'.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.clf()



        