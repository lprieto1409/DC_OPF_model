# -*- coding: utf-8 -*-
"""
Created on Sun May 28 17:13:54 2023

Corrected code showing 2ft: Height A

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
#folder_3 = 'D:/5. Jordan_1st_paper/Images/Latest_outputs/Outputs/' #Cambiar por ultima corrida
folder_3 = 'D:/8. GitHub/M2S_line_outages/Outputs_2/'
folder_inputs = 'D:/15. Globus/RESULTS/INPUT_model/'
folder_slack_jk = 'D:/5. Jordan_1st_paper/11. LMPS_slack/Slack/'
folder_lines = 'D:/5. Jordan_1st_paper/8. Lines/'
folder_lines_v2 = 'D:/8. GitHub/M2S_line_outages/'

#Figure snapshots
Pannels = ['a','b','c','d']
Captures = [6198,6216,6263,6312] #all of these captures are in the during range
Depth = ['0ft','1ft','2ft','3ft','4ft','5ft','6ft','7ft','8ft','9ft','10ft']
Time = ['09/16/18 6:00 AM','09/17/18 12:00 AM','09/18/18 11:00 PM','09/21/18 12:00 AM']

#Iterations between flood depths
i = 6 # For 2ft:0 and for 8ft:6
j = 0
k = 0

#Width of the lines
multiplo = 15
factor = 25
base = 2 #se puede cambiar

#Font
font_number = 70#50#17
font_title = 90#85#25
font_cbar = 40#30
marker_size = 180
font_legend = 60#48#17
font_box = 90#88
font_text = 80#70

#Slack variables
#M2S_st_flood_all = pd.read_csv(folder_inputs + 'ST_all_basins_flood.csv')
substations = pd.read_csv(during+'buses.csv')

M2S_trans = pd.read_csv(folder_3 + 'flow_'+Depth[i+2]+'.csv', header=0)
M2S_trans_base = pd.read_csv(folder_3 + 'flow_'+Depth[k+10]+'.csv', header=0) #folder 4 using the changes in the OF
M2S_trans['Diff between:'+ Depth[i+2] + '- No flood'] = abs(M2S_trans['Value'])-abs(M2S_trans_base['Value']) #Diferencia con el escenario base
TL_params = pd.read_excel(folder_lines + 'NC_lines_parameters.xls', sheet_name = 'lines')


#Boundaries
NC_bound = gpd.read_file(folder_github + 'NC_boundary.shp')

#Projection
proj = 'EPSG:4269'

#M2S_st_flood_all['geometry'] = M2S_st_flood_all.apply(lambda x: Point((float(x.x), float(x.y))), axis=1)
#M2S_flood_all = gpd.GeoDataFrame(M2S_st_flood_all, geometry = 'geometry', crs = proj)

substations['geometry'] = substations.apply(lambda x: Point((float(x.x), float(x.y))), axis=1)
M2S_st = gpd.GeoDataFrame(substations, geometry = 'geometry', crs = proj)

#duals_all_m2s['geometry'] = duals_all_m2s.apply(lambda x: Point((float(x.Long), float(x.Lat))), axis=1)
#M2S_duals_all = gpd.GeoDataFrame(duals_all_m2s, geometry = 'geometry', crs = proj)


########################################################################################################################################
#########################################TRANSMISSION LINES####################################################################################
# Transmission in MWh in each line
df_trans_during = M2S_trans.loc[(M2S_trans['Time']<=6312)&(M2S_trans['Time']>6118)]  #09/13 to 09/21
df_trans_base_sinflood = M2S_trans_base.loc[(M2S_trans_base['Time']<=6312)&(M2S_trans_base['Time']>6118)]  #09/13 to 09/21

# Sort the DataFrame by the "Time" column in ascending order
df_trans_sorted = df_trans_during.sort_values(by='Time', ascending=True)
df_trans_sorted = df_trans_sorted.drop(columns=['Unnamed: 0'])
df_base_sorted = df_trans_base_sinflood.sort_values(by='Time', ascending=True)
df_base_sorted = df_base_sorted.drop(columns=['Unnamed: 0'])

# Snapshots of base flow in time 
df_base_a = df_base_sorted.loc[df_trans_sorted['Time'] == Captures[j]]
df_base_a_sorted = df_base_a.sort_values('Line')
df_base_a_sorted.reset_index(drop=True, inplace=True)
df_base_b = df_base_sorted.loc[df_trans_sorted['Time'] == Captures[j+1]]
df_base_b_sorted = df_base_b.sort_values('Line')
df_base_b_sorted.reset_index(drop=True, inplace=True)
df_base_c = df_base_sorted.loc[df_trans_sorted['Time'] == Captures[j+2]]
df_base_c_sorted = df_base_c.sort_values('Line')
df_base_c_sorted.reset_index(drop=True, inplace=True)
df_base_d = df_base_sorted.loc[df_trans_sorted['Time'] == Captures[j+3]]
df_base_d_sorted = df_base_d.sort_values('Line')
df_base_d_sorted.reset_index(drop=True, inplace=True)

# Snapshots of flow in time 
df_flow_a = df_trans_sorted.loc[df_trans_sorted['Time'] == Captures[j]]
df_flow_a_sorted = df_flow_a.sort_values('Line')
df_flow_a_sorted.reset_index(drop=True, inplace=True)
df_flow_b = df_trans_sorted.loc[df_trans_sorted['Time'] == Captures[j+1]]
df_flow_b_sorted = df_flow_b.sort_values('Line')
df_flow_b_sorted.reset_index(drop=True, inplace=True)
df_flow_c = df_trans_sorted.loc[df_trans_sorted['Time'] == Captures[j+2]]
df_flow_c_sorted = df_flow_c.sort_values('Line')
df_flow_c_sorted.reset_index(drop=True, inplace=True)
df_flow_d = df_trans_sorted.loc[df_trans_sorted['Time'] == Captures[j+3]]
df_flow_d_sorted = df_flow_d.sort_values('Line')
df_flow_d_sorted.reset_index(drop=True, inplace=True)

# Flooding impacts on transmission lines
df_trans_impacted = pd.read_csv(folder_lines_v2 +'lines_'+Depth[i+2]+'.csv')
df_trans_impacted_during = df_trans_impacted.loc[(df_trans_impacted['Unnamed: 0']<=6313)&(df_trans_impacted['Unnamed: 0']>6119)]
df_trans_impacted_during['Time'] = df_trans_impacted_during['Unnamed: 0'] - 1

# Re-organizing the impacted lines dataframe
columns_to_melt = df_flow_a['Line']
melted_df = pd.melt(df_trans_impacted_during, id_vars=['Time'], value_vars=columns_to_melt, var_name='Line', value_name='Value') 

# Outage dataframe has to be replace: 1(out) and 0(no damage)
outage_df = melted_df
outage_df['Value'] = outage_df['Value'].replace({0: 1, 1: 0})

# Snapshots of outages in time 
df_out_a =  outage_df.loc[outage_df['Time'] == Captures[j]]
df_out_a_sorted = df_out_a.sort_values('Line')
df_out_a_sorted.reset_index(drop=True, inplace=True)
df_out_b =  outage_df.loc[outage_df['Time'] == Captures[j+1]]
df_out_b_sorted = df_out_b.sort_values('Line')
df_out_b_sorted.reset_index(drop=True, inplace=True)
df_out_c =  outage_df.loc[outage_df['Time'] == Captures[j+2]]
df_out_c_sorted = df_out_c.sort_values('Line')
df_out_c_sorted.reset_index(drop=True, inplace=True)
df_out_d =  outage_df.loc[outage_df['Time'] == Captures[j+3]]
df_out_d_sorted = df_out_d.sort_values('Line')
df_out_d_sorted.reset_index(drop=True, inplace=True)

# Line widths to see the impacted lines
df_out_a_sorted['width_'+str(Captures[j])] = df_out_a_sorted['Value']*factor
df_out_b_sorted['width_'+str(Captures[j+1])] = df_out_b_sorted['Value']*factor
df_out_c_sorted['width_'+str(Captures[j+2])] = df_out_c_sorted['Value']*factor
df_out_d_sorted['width_'+str(Captures[j+3])] = df_out_d_sorted['Value']*factor

# Network calling from PyPSA
network = pypsa.Network()
network.import_from_csv_folder(csv_folder_name=during)

# Impacted lines
network.lines_t.p0.loc['during a'] = df_out_a_sorted['width_'+str(Captures[j])].to_numpy()
network.lines_t.p0.loc['during b'] = df_out_b_sorted['width_'+str(Captures[j+1])].to_numpy()
network.lines_t.p0.loc['during c'] = df_out_c_sorted['width_'+str(Captures[j+2])].to_numpy()
network.lines_t.p0.loc['during d'] = df_out_d_sorted['width_'+str(Captures[j+3])].to_numpy()


line_capacity = network.lines.s_nom
outage_m2s_a = network.lines_t.p0.loc['during a']
outage_m2s_b = network.lines_t.p0.loc['during b']
outage_m2s_c = network.lines_t.p0.loc['during c']
outage_m2s_d = network.lines_t.p0.loc['during d']


########----CHANGES IN POWER FLOWS----################

#line loading
network.lines_t.p0.loc['Pannel_'+Pannels[j]+'_'+Depth[i+2]] = df_flow_a_sorted['Diff between:'+ Depth[i+2] + '- No flood'].to_numpy()
network.lines_t.p0.loc['Pannel_'+Pannels[j+1]+'_'+Depth[i+2]] = df_flow_b_sorted['Diff between:'+ Depth[i+2] + '- No flood'].to_numpy()
network.lines_t.p0.loc['Pannel_'+Pannels[j+2]+'_'+Depth[i+2]] = df_flow_c_sorted['Diff between:'+ Depth[i+2] + '- No flood'].to_numpy()
network.lines_t.p0.loc['Pannel_'+Pannels[j+3]+'_'+Depth[i+2]] = df_flow_d_sorted['Diff between:'+ Depth[i+2] + '- No flood'].to_numpy()
network.lines_t.p0.loc['voltage'] =  TL_params['voltage'].to_numpy()

#unserved energy
Unserved_energy_a = network.lines_t.p0.loc['Pannel_'+Pannels[j]+'_'+Depth[i+2]] #Total unserved energy in MWh
Unserved_energy_a.describe()
Unserved_energy_b = network.lines_t.p0.loc['Pannel_'+Pannels[j+1]+'_'+Depth[i+2]] #Total unserved energy in MWh
Unserved_energy_b.describe()
Unserved_energy_c = network.lines_t.p0.loc['Pannel_'+Pannels[j+2]+'_'+Depth[i+2]] #Total unserved energy in MWh
Unserved_energy_c.describe()
Unserved_energy_d = network.lines_t.p0.loc['Pannel_'+Pannels[j+3]+'_'+Depth[i+2]] #Total unserved energy in MWh
Unserved_energy_d.describe()

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

fig, axs = plt.subplots(4,2,subplot_kw={"projection": ccrs.EqualEarth()}, figsize=(90, 94), gridspec_kw={'wspace': wspacing, 'hspace': hspacing})

# Arial family for all the figures
mpl.rcParams['font.family'] = "Arial"

###FIRST PANNEL
ax = axs[0,0]

#Impacted lines
network.plot(
    ax=ax,
    line_widths= outage_m2s_a,
    line_colors= 'red',
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

ax.annotate(Pannels[j], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
             xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
             bbox = dict(boxstyle = "square", fc = "black"))

ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

# Set the aspect ratio of axs[1,0] to match axs[0,0]
axs[0,0].set_aspect(axs[0,1].get_aspect())


#ax.set_xlim(0, 4000000000000000)
#ax.set_title('b) North Carolina: Grid', loc = 'left',fontsize = size_tl, family ='Arial')#, fontweight = 'bold')

legend_elements = [Line2D([0], [0], color= 'dimgray', lw=7, label='Line'),
                   Line2D([0], [0], color= 'red', lw=22, label='Line'),
                   Line2D([0], [0], marker='o', color= None, label='Scatter',
                          markerfacecolor='black', markersize=25)]



ax.legend(legend_elements,['Transmission lines','Impacted lines', 'Substations'],loc='lower left',prop={'size': font_legend})
ax.set_title(Time[j], loc = 'center',fontsize = font_title, family ='Arial', fontweight = 'bold')

############ 2nd panel
ax = axs[1,0]

network.plot(
    ax=ax,
    line_widths= outage_m2s_b,
    line_colors = 'red',
    #line_cmap=plt.cm.jet,
    #title="M2S: North Carolina Grid Representation",
    bus_sizes=0.15e-3,
    bus_alpha=0.7,
    bus_colors='black',
)

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

ax.annotate(Pannels[j+1], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
             xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
             bbox = dict(boxstyle = "square", fc = "black"))

ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

# Set the aspect ratio of axs[1,0] to match axs[0,0]
axs[1,0].set_aspect(axs[0,1].get_aspect())

#ax.set_xlim(0, 4000000000000000)
#ax.set_title('b) North Carolina: Grid', loc = 'left',fontsize = size_tl, family ='Arial')#, fontweight = 'bold')

legend_elements = [Line2D([0], [0], color= 'dimgray', lw=7, label='Line'),
                   Line2D([0], [0], color= 'red', lw=22, label='Line'),
                   Line2D([0], [0], marker='o', color= None, label='Scatter',
                          markerfacecolor='black', markersize=25)]



ax.legend(legend_elements,['Transmission lines','Impacted lines', 'Substations'],loc='lower left',prop={'size': font_legend})
ax.set_title(Time[j+1], loc = 'center',fontsize = font_title, family ='Arial', fontweight = 'bold')


############ 3rd panel
ax = axs[2,0]

network.plot(
    ax=ax,
    line_widths= outage_m2s_c,
    line_colors = 'red',
    #line_cmap=plt.cm.jet,
    #title="M2S: North Carolina Grid Representation",
    bus_sizes=0.15e-3,
    bus_alpha=0.7,
    bus_colors='black',
)

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

ax.annotate(Pannels[j+2], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
             xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
             bbox = dict(boxstyle = "square", fc = "black"))

ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

# Set the aspect ratio of axs[1,0] to match axs[0,0]
axs[2,0].set_aspect(axs[0,1].get_aspect())

#ax.set_xlim(0, 4000000000000000)
#ax.set_title('b) North Carolina: Grid', loc = 'left',fontsize = size_tl, family ='Arial')#, fontweight = 'bold')

legend_elements = [Line2D([0], [0], color= 'dimgray', lw=7, label='Line'),
                   Line2D([0], [0], color= 'red', lw=22, label='Line'),
                   Line2D([0], [0], marker='o', color= None, label='Scatter',
                          markerfacecolor='black', markersize=25)]



ax.legend(legend_elements,['Transmission lines','Impacted lines', 'Substations'],loc='lower left',prop={'size': font_legend})
ax.set_title(Time[j+2], loc = 'center',fontsize = font_title, family ='Arial', fontweight = 'bold')

############ 4rd panel
ax = axs[3,0]

network.plot(
    ax=ax,
    line_widths= outage_m2s_d,
    line_colors = 'red',
    #line_cmap=plt.cm.jet,
    #title="M2S: North Carolina Grid Representation",
    bus_sizes=0.15e-3,
    bus_alpha=0.7,
    bus_colors='black',
)

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

ax.annotate(Pannels[j+3], xy = (0,0), xytext=(40,1010), #estas coordenadas de xytext solo en graficos PyPSA
             xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
             bbox = dict(boxstyle = "square", fc = "black"))

ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())#original

# Set the aspect ratio of axs[1,0] to match axs[0,0]
axs[3,0].set_aspect(axs[0,1].get_aspect())

#ax.set_xlim(0, 4000000000000000)
#ax.set_title('b) North Carolina: Grid', loc = 'left',fontsize = size_tl, family ='Arial')#, fontweight = 'bold')

legend_elements = [Line2D([0], [0], color= 'dimgray', lw=7, label='Line'),
                   Line2D([0], [0], color= 'red', lw=22, label='Line'),
                   Line2D([0], [0], marker='o', color= None, label='Scatter',
                          markerfacecolor='black', markersize=25)]



ax.legend(legend_elements,['Transmission lines','Impacted lines', 'Substations'],loc='lower left',prop={'size': font_legend})
ax.set_title(Time[j+3], loc = 'center',fontsize = font_title, family ='Arial', fontweight = 'bold')

####################FIRST SECOND COLUMN
ax = axs[0,1]

#Plotting the network
plot_a = network.plot(ax = ax,line_colors=(Unserved_energy_a),
line_widths = (Voltage_class/(multiplo*(Unserved_energy_a.mean()+0.001)))*(Unserved_energy_a.mean()+0.001),
line_cmap = cmap_unserved,
line_norm = norm_unserved,
bus_sizes = 0.06e-3,
bus_alpha = 0.7,
bus_colors ='black')
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())

ax.annotate(Pannels[j], xy = (0,0), xytext=(40,930), 
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))

#title
ax.set_title(Time[j], loc = 'center',fontsize = font_title, fontweight = 'bold', pad = 50)

#cbar = plt.colorbar(plot_a[1],location="right",fraction=0.0223, pad=0.04,ax = ax)
#cbar.set_label(label = "Changes in power flows (MWh)", size = font_title, fontweight = 'bold', labelpad = 5, y = 0.50)
#cbar.ax.tick_params(axis = 'both',labelsize=font_number, length = 20)

#Manually create legend of the plot
#define handles and labels that will get added to legend
handles, labels = axs[0,0].get_legend_handles_labels()#Manually create legend of the plot

#define patches and lines to add to legend
line1 = Line2D([0], [0], label='115-230 kV', color='black', linewidth= 10, linestyle='-')
line2 = Line2D([0], [0], label='230-500 kV', color='black', linewidth= 20, linestyle='-')
line3 = Line2D([0], [0], label='500 >= kV', color='black', linewidth= 30, linestyle='-')

#add handles
handles.extend([line1,line2,line3])

#legen
ax.legend(handles = handles, loc = 'lower left', bbox_to_anchor = (0.01,0.002), ncol=1, fontsize = font_text)

####################SECOND SECOND COLUMN
ax = axs[1,1]

#Plotting the network
plot_b = network.plot(ax = ax,line_colors=(Unserved_energy_b),
line_widths = (Voltage_class/(multiplo*(Unserved_energy_b.mean()+0.001)))*(Unserved_energy_b.mean()+0.001),
line_cmap = cmap_unserved,
line_norm = norm_unserved,
bus_sizes = 0.06e-3,
bus_alpha = 0.7,
bus_colors ='black')
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())

ax.annotate(Pannels[j+1], xy = (0,0), xytext=(40,930), 
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))

#title
ax.set_title(Time[j+1], loc = 'center',fontsize = font_title, fontweight = 'bold', pad = 50)

#cbar = plt.colorbar(plot_b[1],location="right",fraction=0.0223, pad=0.04,ax = ax)
#cbar.set_label(label = "Changes in power flows (MWh)", size = font_title, fontweight = 'bold', labelpad = 5, y = 0.50)
#cbar.ax.tick_params(axis = 'both',labelsize=font_number, length = 20)

#Manually create legend of the plot
#define handles and labels that will get added to legend
handles, labels = ax.get_legend_handles_labels()#Manually create legend of the plot

#define patches and lines to add to legend
line1 = Line2D([0], [0], label='115-230 kV', color='black', linewidth= 10, linestyle='-')
line2 = Line2D([0], [0], label='230-500 kV', color='black', linewidth= 20, linestyle='-')
line3 = Line2D([0], [0], label='500 >= kV', color='black', linewidth= 30, linestyle='-')

#add handles
handles.extend([line1,line2,line3])

#legen
ax.legend(handles = handles, loc = 'lower left', bbox_to_anchor = (0.01,0.002), ncol=1, fontsize = font_text)

####################THIRD SECOND COLUMN
ax = axs[2,1]

#Plotting the network
plot_c = network.plot(ax = ax,line_colors=(Unserved_energy_c),
line_widths = (Voltage_class/(multiplo*(Unserved_energy_c.mean()+0.001)))*(Unserved_energy_c.mean()+0.001),
line_cmap = cmap_unserved,
line_norm = norm_unserved,
bus_sizes = 0.06e-3,
bus_alpha = 0.7,
bus_colors ='black')
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())

ax.annotate(Pannels[j+2], xy = (0,0), xytext=(40,930), 
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))

#title
ax.set_title(Time[j+2], loc = 'center',fontsize = font_title, fontweight = 'bold', pad = 50)

#cbar = plt.colorbar(plot_c[1],location="right",fraction=0.0223, pad=0.04,ax = ax)
#cbar.set_label(label = "Changes in power flows (MWh)", size = font_title, fontweight = 'bold', labelpad = 5, y = 0.50)
#cbar.ax.tick_params(axis = 'both',labelsize=font_number, length = 20)

#Manually create legend of the plot
#define handles and labels that will get added to legend
handles, labels = ax.get_legend_handles_labels()#Manually create legend of the plot

#define patches and lines to add to legend
line1 = Line2D([0], [0], label='115-230 kV', color='black', linewidth= 10, linestyle='-')
line2 = Line2D([0], [0], label='230-500 kV', color='black', linewidth= 20, linestyle='-')
line3 = Line2D([0], [0], label='500 >= kV', color='black', linewidth= 30, linestyle='-')

#add handles
handles.extend([line1,line2,line3])

#legen
ax.legend(handles = handles, loc = 'lower left', bbox_to_anchor = (0.01,0.002), ncol=1, fontsize = font_text)

####################FOURTH SECOND COLUMN
ax = axs[3,1]

#Plotting the network
plot_d = network.plot(ax = ax,line_colors=(Unserved_energy_d),
line_widths = (Voltage_class/(multiplo*(Unserved_energy_d.mean()+0.001)))*(Unserved_energy_d.mean()+0.001),
line_cmap = cmap_unserved,
line_norm = norm_unserved,
bus_sizes = 0.06e-3,
bus_alpha = 0.7,
bus_colors ='black')
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES, edgecolor='gray' )
ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())

ax.annotate(Pannels[j+3], xy = (0,0), xytext=(40,930), 
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))

#title
ax.set_title(Time[j+3], loc = 'center',fontsize = font_title, fontweight = 'bold', pad = 50)

cbar = plt.colorbar(plot_d[1],location="right",fraction=0.0223, pad=0.04,ax = axs)
cbar.set_label(label = "Changes in power flows (MWh)", size = font_title, fontweight = 'bold', labelpad = 15, y = 0.50)
cbar.ax.tick_params(axis = 'both',labelsize=font_number, length = 20)

#Manually create legend of the plot
#define handles and labels that will get added to legend
handles, labels = ax.get_legend_handles_labels()#Manually create legend of the plot

#define patches and lines to add to legend
line1 = Line2D([0], [0], label='115-230 kV', color='black', linewidth= 10, linestyle='-')
line2 = Line2D([0], [0], label='230-500 kV', color='black', linewidth= 20, linestyle='-')
line3 = Line2D([0], [0], label='500 >= kV', color='black', linewidth= 30, linestyle='-')

#add handles
handles.extend([line1,line2,line3])

#legen
ax.legend(handles = handles, loc = 'lower left', bbox_to_anchor = (0.01,0.002), ncol=1, fontsize = font_text)


fig.tight_layout()
plt.savefig(folder_images+'M2S_impacts_'+Depth[i+2]+'_flood_abs.png', dpi=150, bbox_inches='tight')
plt.show()
plt.clf()
        