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

#folder location of the data
folder_github = 'D:/2. Research/5. Latest data/13. Collaboration flood/Images_1pp/Inputs/'
folder_images = 'D:/5. Jordan_1st_paper/Images/Latest/' #final images are in this folder
folder_census = 'D:/5. Jordan_1st_paper/7. Shapefiles/'
during = 'D:/2. Research/5. Latest data/17. Figures/M2S_network/during/'
folder_slack = 'D:/5. Jordan_1st_paper/11. LMPS_slack/'
folder_3 = 'D:/8. GitHub/M2S_line_outages/Outputs_2/'
folder_inputs = 'D:/15. Globus/RESULTS/INPUT_model/'
folder_slack_jk = 'D:/5. Jordan_1st_paper/11. LMPS_slack/Slack/'
folder_lines = 'D:/5. Jordan_1st_paper/8. Lines/'
folder_lines_v2 = 'D:/8. GitHub/M2S_line_outages/Inputs/'

#Figure snapshots
Pannels = ['a','b','c','d']
Captures = [6198,6216,6263,6312] #all of these captures are in the during range
Depth = ['0ft','1ft','2ft','3ft','4ft','5ft','6ft','7ft','8ft','9ft','10ft']
Time = ['09/16/18 6:00 AM','09/17/18 12:00 AM','09/18/18 11:00 PM','09/21/18 12:00 AM']

#Color
rosybrown_co = (205/255, 155/255, 155/255) #RGB code

#Iterations between flood depths
i = 0 # For 2ft:0 and for 8ft:6
j = 0
k = 0

#Width of the lines
multiplo = 15
factor = 25
base = 2 #se puede cambiar

#Font
font_number = 50#17
font_title = 152#85#25
font_cbar = 30
marker_size = 180
font_legend = 150#48#17
marker_legend = 95#25
font_box = 140#88
font_text = 70

#Slack variables
substations = pd.read_csv(during+'buses.csv')

# Initialize an empty list to store the dataframes
M2S_trans = []

# Iterate over the depths and read the corresponding CSV file
for d in Depth:
    file_path = folder_3 + f'flow_{d}.csv'
    df = pd.read_csv(file_path, header=0)
    M2S_trans.append(df)

# Changes in power flows
M2S_trans_base = pd.read_csv(folder_3 + 'flow_' + Depth[k+10] + '.csv', header=0) #folder 4 using the changes in the OF

for idx, d in enumerate(Depth[i:i+11]):
    column_name = f'Diff between: {d} - No flood'
    M2S_trans[idx][column_name] = abs(M2S_trans[idx]['Value']) - abs(M2S_trans_base['Value'])

# Transmission line parameters
TL_params = pd.read_excel(folder_lines + 'NC_lines_parameters.xls', sheet_name = 'lines')

#Boundaries
NC_bound = gpd.read_file(folder_github + 'NC_boundary.shp')

#Projection
proj = 'EPSG:4269'

substations['geometry'] = substations.apply(lambda x: Point((float(x.x), float(x.y))), axis=1)
M2S_st = gpd.GeoDataFrame(substations, geometry = 'geometry', crs = proj)

########################################################################################################################################
#########################################TRANSMISSION LINES####################################################################################
# Transmission in MWh in each line
df_trans_during = []

for idx, df in enumerate(M2S_trans):
    df_during = df.loc[(df['Time'] <= 6312) & (df['Time'] > 6118)]  # 09/13 to 09/21
    df_trans_during.append(df_during)

df_trans_base_sinflood = M2S_trans_base.loc[(M2S_trans_base['Time']<=6312)&(M2S_trans_base['Time']>6118)]  #09/13 to 09/21

# Sort the DataFrame by the "Time" column in ascending order
df_trans_sorted = []

for df in df_trans_during:
    df_sorted = df.sort_values(by='Time', ascending=True)
    df_sorted = df_sorted.drop(columns=['Unnamed: 0'])
    df_trans_sorted.append(df_sorted)

df_base_sorted = df_trans_base_sinflood.sort_values(by='Time', ascending=True)
df_base_sorted = df_base_sorted.drop(columns=['Unnamed: 0'])

# Flooding impacts on transmission lines
df_trans_impacted = []

for d in Depth[i:i+11]:
    file_path = folder_lines_v2 + f'lines_{d}.csv'
    df = pd.read_csv(file_path)
    df_trans_impacted.append(df)


df_trans_impacted_during = []

for df in df_trans_impacted:
    df_I_during = df.loc[(df['Unnamed: 0'] <= 6313) & (df['Unnamed: 0'] > 6119)]
    df_I_during['Time'] = df_I_during['Unnamed: 0'] - 1
    df_trans_impacted_during.append(df_I_during)


# Re-organizing the impacted lines dataframe
sp_lines = df_trans_sorted[0].loc[df_trans_sorted[0]['Time'] == Captures[j]]
columns_to_melt = sp_lines['Line']

melted_df = []

for df in df_trans_impacted_during:
    melted_d = pd.melt(df, id_vars=['Time'], value_vars=columns_to_melt, var_name='Line', value_name='Value')
    melted_df.append(melted_d)

# Outage dataframe has to be replace: 1(out) and 0(no damage)
outage_df = melted_df

for df in outage_df:
    df['Value'] = df['Value'].replace({0: 1, 1: 0})

#################FOR LOOP############################
for j in range(4):
    # Snapshots of outages in time 
    df_out = []

    for df in outage_df:
        df_out_part = df.loc[df['Time'] == Captures[j]]
        df_out_sorted = df_out_part.sort_values('Line')
        df_out_sorted.reset_index(drop=True, inplace=True)
        df_out.append(df_out_sorted)

    # Line widths to see the impacted lines
    for df in df_out:
        df[f'width_{Captures[j]}'] = df['Value'] * factor

    # Network calling from PyPSA
    network = pypsa.Network()
    network.import_from_csv_folder(csv_folder_name=during)

    # Impacted lines
    network.lines_t.p0.loc['during_'+str(Captures[j])+'_'+Depth[i]] = df_out[i]['width_'+str(Captures[j])].to_numpy()
    network.lines_t.p0.loc['during_'+str(Captures[j])+'_'+Depth[i+1]] = df_out[i+1]['width_'+str(Captures[j])].to_numpy()
    network.lines_t.p0.loc['during_'+str(Captures[j])+'_'+Depth[i+2]] = df_out[i+2]['width_'+str(Captures[j])].to_numpy()
    network.lines_t.p0.loc['during_'+str(Captures[j])+'_'+Depth[i+3]] = df_out[i+3]['width_'+str(Captures[j])].to_numpy()
    network.lines_t.p0.loc['during_'+str(Captures[j])+'_'+Depth[i+4]] = df_out[i+4]['width_'+str(Captures[j])].to_numpy()
    network.lines_t.p0.loc['during_'+str(Captures[j])+'_'+Depth[i+5]] = df_out[i+5]['width_'+str(Captures[j])].to_numpy()
    network.lines_t.p0.loc['during_'+str(Captures[j])+'_'+Depth[i+6]] = df_out[i+6]['width_'+str(Captures[j])].to_numpy()
    network.lines_t.p0.loc['during_'+str(Captures[j])+'_'+Depth[i+7]] = df_out[i+7]['width_'+str(Captures[j])].to_numpy()
    network.lines_t.p0.loc['during_'+str(Captures[j])+'_'+Depth[i+8]] = df_out[i+8]['width_'+str(Captures[j])].to_numpy()
    network.lines_t.p0.loc['during_'+str(Captures[j])+'_'+Depth[i+9]] = df_out[i+9]['width_'+str(Captures[j])].to_numpy()
    network.lines_t.p0.loc['during_'+str(Captures[j])+'_'+Depth[i+10]] = df_out[i+10]['width_'+str(Captures[j])].to_numpy()

    line_capacity = network.lines.s_nom

    # IMPACTED LINES
    outage_m2s = []

    for idx, d in enumerate(Depth[i:i+11]):
        label = f'during_{Captures[j]}_{d}'
        outage_m2s.append(network.lines_t.p0.loc[label])

    # We start to plot here
    wspacing = 0.04
    hspacing = 0.1 

    fig, axs = plt.subplots(4,3,subplot_kw={"projection": ccrs.EqualEarth()}, figsize=(140, 94), gridspec_kw={'wspace': wspacing, 'hspace': hspacing})

    # Arial family for all the figures
    mpl.rcParams['font.family'] = "Arial"
    
    # Set the figure title
    
    fig.suptitle("Date: "+Time[j], x=0.5, y=0.92, fontsize=font_title, fontweight='bold', ha='center', family='Arial')

    # FIRST COLUMN

    # 0ft

    ax = axs[0,0]

    #Impacted lines
    network.plot(
        ax=ax,
        line_widths= outage_m2s[i],
        line_colors = 'red',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

    #Lines in the background 
    network.plot(
        ax=ax,
        line_widths= 1.7,
        line_colors = 'dimgray',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )
    #NC_bound.plot(ax = ax, facecolor='white',edgecolor='black', alpha=0.15)

    ax.annotate(Depth[0], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

    # Set the aspect ratio of axs[1,0] to match axs[0,0]
    axs[0,0].set_aspect(axs[0,1].get_aspect())

    # 1ft

    ax = axs[0,1]

    #Impacted lines
    network.plot(
        ax=ax,
        line_widths= outage_m2s[i+1],
        line_colors = 'red',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

    #Lines in the background 
    network.plot(
        ax=ax,
        line_widths= 1.7,
        line_colors = 'dimgray',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )
    #NC_bound.plot(ax = ax, facecolor='white',edgecolor='black', alpha=0.15)

    ax.annotate(Depth[1], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

    # Set the aspect ratio of axs[1,0] to match axs[0,0]
    axs[0,0].set_aspect(axs[0,1].get_aspect())

    # 2ft

    ax = axs[0,2]

    #Impacted lines
    network.plot(
        ax=ax,
        line_widths= outage_m2s[i+2],
        line_colors = 'red',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

    #Lines in the background 
    network.plot(
        ax=ax,
        line_widths= 1.7,
        line_colors = 'dimgray',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )
    #NC_bound.plot(ax = ax, facecolor='white',edgecolor='black', alpha=0.15)

    ax.annotate(Depth[2], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

    # Set the aspect ratio of axs[1,0] to match axs[0,0]
    axs[0,0].set_aspect(axs[0,1].get_aspect())

    # 3ft

    ax = axs[1,0]

    #Impacted lines
    network.plot(
        ax=ax,
        line_widths= outage_m2s[i+3],
        line_colors = 'red',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

    #Lines in the background 
    network.plot(
        ax=ax,
        line_widths= 1.7,
        line_colors = 'dimgray',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )
    #NC_bound.plot(ax = ax, facecolor='white',edgecolor='black', alpha=0.15)

    ax.annotate(Depth[3], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

    # Set the aspect ratio of axs[1,0] to match axs[0,0]
    axs[0,0].set_aspect(axs[0,1].get_aspect())

    # 4ft

    ax = axs[1,1]

    #Impacted lines
    network.plot(
        ax=ax,
        line_widths= outage_m2s[i+4],
        line_colors = 'red',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

    #Lines in the background 
    network.plot(
        ax=ax,
        line_widths= 1.7,
        line_colors = 'dimgray',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )
    #NC_bound.plot(ax = ax, facecolor='white',edgecolor='black', alpha=0.15)

    ax.annotate(Depth[4], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

    # Set the aspect ratio of axs[1,0] to match axs[0,0]
    axs[0,0].set_aspect(axs[0,1].get_aspect())

    # 5ft

    ax = axs[1,2]

    #Impacted lines
    network.plot(
        ax=ax,
        line_widths= outage_m2s[i+5],
        line_colors = 'red',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

    #Lines in the background 
    network.plot(
        ax=ax,
        line_widths= 1.7,
        line_colors = 'dimgray',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )
    #NC_bound.plot(ax = ax, facecolor='white',edgecolor='black', alpha=0.15)

    ax.annotate(Depth[5], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

    # Set the aspect ratio of axs[1,0] to match axs[0,0]
    axs[0,0].set_aspect(axs[0,1].get_aspect())

    # 6ft

    ax = axs[2,0]

    #Impacted lines
    network.plot(
        ax=ax,
        line_widths= outage_m2s[i+6],
        line_colors = 'red',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

    #Lines in the background 
    network.plot(
        ax=ax,
        line_widths= 1.7,
        line_colors = 'dimgray',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )
    #NC_bound.plot(ax = ax, facecolor='white',edgecolor='black', alpha=0.15)

    ax.annotate(Depth[6], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

    # Set the aspect ratio of axs[1,0] to match axs[0,0]
    axs[0,0].set_aspect(axs[0,1].get_aspect())

    # 7ft

    ax = axs[2,1]

    #Impacted lines
    network.plot(
        ax=ax,
        line_widths= outage_m2s[i+7],
        line_colors = 'red',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

    #Lines in the background 
    network.plot(
        ax=ax,
        line_widths= 1.7,
        line_colors = 'dimgray',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )
    #NC_bound.plot(ax = ax, facecolor='white',edgecolor='black', alpha=0.15)

    ax.annotate(Depth[7], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

    # Set the aspect ratio of axs[1,0] to match axs[0,0]
    axs[0,0].set_aspect(axs[0,1].get_aspect())

    # 8ft

    ax = axs[2,2]

    #Impacted lines
    network.plot(
        ax=ax,
        line_widths= outage_m2s[i+8],
        line_colors = 'red',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

    #Lines in the background 
    network.plot(
        ax=ax,
        line_widths= 1.7,
        line_colors = 'dimgray',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )
    #NC_bound.plot(ax = ax, facecolor='white',edgecolor='black', alpha=0.15)

    ax.annotate(Depth[8], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

    # Set the aspect ratio of axs[1,0] to match axs[0,0]
    axs[0,0].set_aspect(axs[0,1].get_aspect())

    # 9ft

    ax = axs[3,0]

    #Impacted lines
    network.plot(
        ax=ax,
        line_widths= outage_m2s[i+9],
        line_colors = 'red',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

    #Lines in the background 
    network.plot(
        ax=ax,
        line_widths= 1.7,
        line_colors = 'dimgray',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )
    #NC_bound.plot(ax = ax, facecolor='white',edgecolor='black', alpha=0.15)

    ax.annotate(Depth[9], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

    # Set the aspect ratio of axs[1,0] to match axs[0,0]
    axs[0,0].set_aspect(axs[0,1].get_aspect())

    # 10ft

    ax = axs[3,1]

    #Impacted lines
    network.plot(
        ax=ax,
        line_widths= outage_m2s[i+10],
        line_colors = 'red',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

    #Lines in the background 
    network.plot(
        ax=ax,
        line_widths= 1.7,
        line_colors = 'dimgray',
        #line_cmap=plt.cm.jet,
        #title="M2S: North Carolina Grid Representation",
        bus_sizes=0.15e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )
    #NC_bound.plot(ax = ax, facecolor='white',edgecolor='black', alpha=0.15)

    ax.annotate(Depth[10], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))

    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

    # Set the aspect ratio of axs[1,0] to match axs[0,0]
    axs[0,0].set_aspect(axs[0,1].get_aspect())
    
    # Legend
    ax = axs[3,2]
    
    # Turn off the axes and set background color to white
    ax.set_axis_off()
    ax.set_facecolor('white')
    
    # Set the aspect ratio of axs[1,0] to match axs[0,0]
    axs[3,2].set_aspect(axs[0,1].get_aspect())



    legend_elements = [Line2D([0], [0], color= 'dimgray', lw=7, label='Line'),
                       Line2D([0], [0], color= 'red', lw=42, label='Line'),
                       Line2D([0], [0], marker='o', color= None, label='Scatter',
                              markerfacecolor='black', markersize=marker_legend)]



    ax.legend(legend_elements,['Transmission lines','Impacted lines', 'Substations'],loc='lower right',prop={'size': font_legend})
    


    #fig.tight_layout()
    plt.savefig(folder_images+'M2S_supp_line_impacts_'+str(j)+'.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.clf()


