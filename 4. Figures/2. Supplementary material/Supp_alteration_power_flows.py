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
base = 2 #se puede cambiar

#Font
font_number = 160#50#17
font_title = 162#95#85#25
font_cbar = 30
marker_size = 180
font_legend = 160#48#17
font_box = 160#88
font_text = 80

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

# Alterations in power flows
APF_df = df_trans_sorted

#################FOR LOOP############################
for j in range(4):
    # Snapshots of outages in time 
    df_APF = []

    for df in APF_df:
        df_APF_part = df.loc[df['Time'] == Captures[j]]
        df_APF_sorted = df_APF_part.sort_values('Line')
        df_APF_sorted.reset_index(drop=True, inplace=True)
        df_APF.append(df_APF_sorted)


    # Network calling from PyPSA
    network = pypsa.Network()
    network.import_from_csv_folder(csv_folder_name=during)

    # Alterations in power flows (APF)
    network.lines_t.p0.loc['APF_'+str(Captures[j])+'_'+Depth[i]] = df_APF[i]['Diff between: '+ Depth[i] + ' - No flood'].to_numpy()
    network.lines_t.p0.loc['APF_'+str(Captures[j])+'_'+Depth[i+1]] = df_APF[i+1]['Diff between: '+ Depth[i+1] + ' - No flood'].to_numpy()
    network.lines_t.p0.loc['APF_'+str(Captures[j])+'_'+Depth[i+2]] = df_APF[i+2]['Diff between: '+ Depth[i+2] + ' - No flood'].to_numpy()
    network.lines_t.p0.loc['APF_'+str(Captures[j])+'_'+Depth[i+3]] = df_APF[i+3]['Diff between: '+ Depth[i+3] + ' - No flood'].to_numpy()
    network.lines_t.p0.loc['APF_'+str(Captures[j])+'_'+Depth[i+4]] = df_APF[i+4]['Diff between: '+ Depth[i+4] + ' - No flood'].to_numpy()
    network.lines_t.p0.loc['APF_'+str(Captures[j])+'_'+Depth[i+5]] = df_APF[i+5]['Diff between: '+ Depth[i+5] + ' - No flood'].to_numpy()
    network.lines_t.p0.loc['APF_'+str(Captures[j])+'_'+Depth[i+6]] = df_APF[i+6]['Diff between: '+ Depth[i+6] + ' - No flood'].to_numpy()
    network.lines_t.p0.loc['APF_'+str(Captures[j])+'_'+Depth[i+7]] = df_APF[i+7]['Diff between: '+ Depth[i+7] + ' - No flood'].to_numpy()
    network.lines_t.p0.loc['APF_'+str(Captures[j])+'_'+Depth[i+8]] = df_APF[i+8]['Diff between: '+ Depth[i+8] + ' - No flood'].to_numpy()
    network.lines_t.p0.loc['APF_'+str(Captures[j])+'_'+Depth[i+9]] = df_APF[i+9]['Diff between: '+ Depth[i+9] + ' - No flood'].to_numpy()
    network.lines_t.p0.loc['APF_'+str(Captures[j])+'_'+Depth[i+10]] = df_APF[i+10]['Diff between: '+ Depth[i+10] + ' - No flood'].to_numpy()
    
    network.lines_t.p0.loc['voltage'] =  TL_params['voltage'].to_numpy()
    
    line_capacity = network.lines.s_nom

    # IMPACTED LINES
    APF_m2s = []

    for idx, d in enumerate(Depth[i:i+11]):
        label = f'APF_{Captures[j]}_{d}'
        APF_m2s.append(network.lines_t.p0.loc[label])
    
    #voltage per line
    Voltage_class = network.lines_t.p0.loc['voltage']
    Voltage_class.describe()

    #limits
    Unserved_min = -3000 
    Unserved_max = 1500 #3000
    Unserved_center = 0
    cmap_unserved = mpl.cm.coolwarm#mpl.cm.bwr
    norm_unserved = colors.TwoSlopeNorm(vmin = Unserved_min, vcenter = Unserved_center, vmax = Unserved_max)

    # We start to plot here
    wspacing = 0.04
    hspacing = 0.1 

    fig, axs = plt.subplots(4,3,subplot_kw={"projection": ccrs.EqualEarth()}, figsize=(150, 94), gridspec_kw={'wspace': wspacing, 'hspace': hspacing})

    # Arial family for all the figures
    mpl.rcParams['font.family'] = "Arial"
    
    # Set the figure title
    
    fig.suptitle("Date: "+Time[j], x=0.5, y=0.92, fontsize=font_title, fontweight='bold', ha='center', family='Arial')

    # FIRST COLUMN

    # 0ft

    ax = axs[0,0]

    #Impacted lines
    plt_APF = network.plot(
        ax=ax,
        line_colors = APF_m2s[i],
        line_widths= (Voltage_class/(multiplo*(APF_m2s[i].mean()+0.001)))*(APF_m2s[i].mean()+0.001),
        line_cmap = cmap_unserved,
        line_norm = norm_unserved,
        bus_sizes=0.06e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

 
    ax.annotate(Depth[0], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))
    
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

    # 1ft

    ax = axs[0,1]

    #Impacted lines
    plt_APF = network.plot(
        ax=ax,
        line_colors = APF_m2s[i+1],
        line_widths= (Voltage_class/(multiplo*(APF_m2s[i+1].mean()+0.001)))*(APF_m2s[i+1].mean()+0.001),
        line_cmap = cmap_unserved,
        line_norm = norm_unserved,
        bus_sizes=0.06e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

 
    ax.annotate(Depth[1], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))
    
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

    # 2ft

    ax = axs[0,2]

    #Impacted lines
    plt_APF = network.plot(
        ax=ax,
        line_colors = APF_m2s[i+2],
        line_widths= (Voltage_class/(multiplo*(APF_m2s[i+2].mean()+0.001)))*(APF_m2s[i+2].mean()+0.001),
        line_cmap = cmap_unserved,
        line_norm = norm_unserved,
        bus_sizes=0.06e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

 
    ax.annotate(Depth[2], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))
    
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

    # 3ft

    ax = axs[1,0]

    #Impacted lines
    plt_APF = network.plot(
        ax=ax,
        line_colors = APF_m2s[i+3],
        line_widths= (Voltage_class/(multiplo*(APF_m2s[i+3].mean()+0.001)))*(APF_m2s[i+3].mean()+0.001),
        line_cmap = cmap_unserved,
        line_norm = norm_unserved,
        bus_sizes=0.06e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

 
    ax.annotate(Depth[3], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))
    
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original
    
    # 4ft
    
    ax = axs[1,1]

    #Impacted lines
    plt_APF = network.plot(
        ax=ax,
        line_colors = APF_m2s[i+4],
        line_widths= (Voltage_class/(multiplo*(APF_m2s[i+4].mean()+0.001)))*(APF_m2s[i+4].mean()+0.001),
        line_cmap = cmap_unserved,
        line_norm = norm_unserved,
        bus_sizes=0.06e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

 
    ax.annotate(Depth[4], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))
    
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original
        
    # 5ft

    ax = axs[1,2]

    #Impacted lines
    plt_APF = network.plot(
        ax=ax,
        line_colors = APF_m2s[i+5],
        line_widths= (Voltage_class/(multiplo*(APF_m2s[i+5].mean()+0.001)))*(APF_m2s[i+5].mean()+0.001),
        line_cmap = cmap_unserved,
        line_norm = norm_unserved,
        bus_sizes=0.06e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

 
    ax.annotate(Depth[5], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))
    
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

    # 6ft
    ax = axs[2,0]

    #Impacted lines
    plt_APF = network.plot(
        ax=ax,
        line_colors = APF_m2s[i+6],
        line_widths= (Voltage_class/(multiplo*(APF_m2s[i+6].mean()+0.001)))*(APF_m2s[i+6].mean()+0.001),
        line_cmap = cmap_unserved,
        line_norm = norm_unserved,
        bus_sizes=0.06e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

 
    ax.annotate(Depth[6], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))
    
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

    # 7ft

    ax = axs[2,1]

    #Impacted lines
    plt_APF = network.plot(
        ax=ax,
        line_colors = APF_m2s[i+7],
        line_widths= (Voltage_class/(multiplo*(APF_m2s[i+7].mean()+0.001)))*(APF_m2s[i+7].mean()+0.001),
        line_cmap = cmap_unserved,
        line_norm = norm_unserved,
        bus_sizes=0.06e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

 
    ax.annotate(Depth[7], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))
    
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original


    # 8ft

    ax = axs[2,2]

    #Impacted lines
    plt_APF = network.plot(
        ax=ax,
        line_colors = APF_m2s[i+8],
        line_widths= (Voltage_class/(multiplo*(APF_m2s[i+8].mean()+0.001)))*(APF_m2s[i+8].mean()+0.001),
        line_cmap = cmap_unserved,
        line_norm = norm_unserved,
        bus_sizes=0.06e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

 
    ax.annotate(Depth[8], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))
    
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

    # 9ft

    ax = axs[3,0]

    #Impacted lines
    plt_APF = network.plot(
        ax=ax,
        line_colors = APF_m2s[i+9],
        line_widths= (Voltage_class/(multiplo*(APF_m2s[i+9].mean()+0.001)))*(APF_m2s[i+9].mean()+0.001),
        line_cmap = cmap_unserved,
        line_norm = norm_unserved,
        bus_sizes=0.06e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

 
    ax.annotate(Depth[9], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))
    
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

    # 10ft

    ax = axs[3,1]

    #Impacted lines
    plt_APF = network.plot(
        ax=ax,
        line_colors = APF_m2s[i+10],
        line_widths= (Voltage_class/(multiplo*(APF_m2s[i+10].mean()+0.001)))*(APF_m2s[i+10].mean()+0.001),
        line_cmap = cmap_unserved,
        line_norm = norm_unserved,
        bus_sizes=0.06e-3,
        bus_alpha=0.7,
        bus_colors='black',
    )

 
    ax.annotate(Depth[10], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
                 xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                 bbox = dict(boxstyle = "square", fc = "black"))
    
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
    ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

    
    ###########------ LEGEND -----------#####################
    ax = axs[3,2]
    
    # Turn off the axes and set background color to white
    ax.set_axis_off()
    ax.set_facecolor('white')
    
    # Set the aspect ratio of axs[1,0] to match axs[0,0]
    axs[3,2].set_aspect(axs[0,1].get_aspect())
    
    # Set label for the colorbar
    cbar = plt.colorbar(plt_APF[1], location="right", fraction=0.0223, pad=0.04, ax=axs)
    cbar.set_label(label="Changes in power flows (MWh)", size=font_title, fontweight='bold', labelpad=30, y=0.50)
    cbar.ax.tick_params(axis='both', labelsize=font_number, length=20)
    
    #Manually create legend of the plot
    #define handles and labels that will get added to legend
    handles, labels = axs[0,0].get_legend_handles_labels()#Manually create legend of the plot

    #define patches and lines to add to legend
    line1 = Line2D([0], [0], label='115-230 kV', color='black', linewidth= 15, linestyle='-')
    line2 = Line2D([0], [0], label='230-500 kV', color='black', linewidth= 30, linestyle='-')
    line3 = Line2D([0], [0], label='500 >= kV', color='black', linewidth= 40, linestyle='-')

    #add handles
    handles.extend([line1,line2,line3])

    #legen
    ax.legend(handles = handles, loc = 'lower left', bbox_to_anchor = (0.01,0.002), ncol=1, fontsize = font_legend)


    #fig.tight_layout()
    plt.savefig(folder_images+'M2S_APF_'+str(j)+'.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.clf()


