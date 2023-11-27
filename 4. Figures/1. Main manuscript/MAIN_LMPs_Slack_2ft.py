# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 18:48:31 2023

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
folder_shp = 'D:/5. Jordan_1st_paper/7. Shapefiles/'
during = 'D:/2. Research/5. Latest data/17. Figures/M2S_network/during/'
folder_3 = 'D:/8. GitHub/M2S_line_outages/Outputs_2/'
folder_images = 'D:/5. Jordan_1st_paper/Images/Latest/' #final images are in this folder    
folder_inputs = 'D:/8. GitHub/M2S_line_outages/Inputs/'
folder_svi = 'D:/5. Jordan_1st_paper/13. SVI index/'
folder_format = 'D:/5. Jordan_1st_paper/1. Text/2. Paper_tables/'

#fonts
font_number = 74#(2.2/1.8)*60#17
font_title = 86#(2.2/1.8)*70#25
font_cbar = 37#(2.2/1.8)*30
marker_size = 950#180
marker_legend = 70#35
font_legend = 60#(2.2/1.8)*48#17
font_box = 108#(2.2/1.8)*88
font_text = 74#(2.2/1.8)*60
font_length = 37#(2.2/1.8)*30
font_ranges = 70#40

Pannels = ['a','b','c','d']
Captures = [6198,6216,6263,6312] #all of these captures are in the during range
Depth = ['0ft','1ft','2ft','3ft','4ft','5ft','6ft','7ft','8ft','9ft','10ft']
Time = ['09/16/18 6:00 AM','09/17/18 12:00 AM','09/18/18 11:00 PM','09/21/18 12:00 AM']

i = 0 #Depth
j = 0 #Pannels/captures
k = 0 #Base case capture

#Projection
proj = 'EPSG:4269'

#Shapefiles
substations = pd.read_csv(during+'buses.csv')
substations['geometry'] = substations.apply(lambda x: Point((float(x.x), float(x.y))), axis=1)
M2S_st = gpd.GeoDataFrame(substations, geometry = 'geometry', crs = proj)
M2S_st['Bus'] = M2S_st['name']
df_st = M2S_st[['Bus','x','y']] #for the join table to add coordinates

# INPUTS
# Data load for every elevation
M2S_load_1 = []
  
for j in Depth:    # Iterate through depths 0 to 10
    df_load = pd.read_csv(folder_inputs + 'data_load_'+j+'_final.csv')
    df_load['Time'] = df_load.index
    M2S_load_1.append(df_load)

# Assuming M2S_load is a list of dataframes
M2S_load = []

for df in M2S_load_1:
    # Melt the dataframe
    melted_df = df.melt(id_vars=['Time'], var_name='Bus', value_name='Value')
    
    M2S_load.append(melted_df)

# Locational Marginal Prices for every elevation
M2S_duals = []
  
for j in Depth:    # Iterate through depths 0 to 10
    df_duals = pd.read_csv(folder_3 + 'duals_'+j+'.csv', header=0)
    M2S_duals.append(df_duals)
    
# Transmission flow for every elevation
M2S_trans = []
  
for j in Depth:    # Iterate through depths 0 to 10
    df_trans = pd.read_csv(folder_3 + 'flow_'+j+'.csv', header=0)
    M2S_trans.append(df_trans)

# Loss of load variables for every elevation
M2S_slack = []
  
for j in Depth:    # Iterate through depths 0 to 10
    df_slack = pd.read_csv(folder_3 + 'slack_'+j+'.csv', header=0)
    M2S_slack.append(df_slack)

# Extracting the values: During
##DUALS
M2S_duals_during = []

# Iterate through the list of dataframes in df_load_during
for df_duals in M2S_duals:
    filt_duals = df_duals.loc[(df_duals['Time'] <= 6312) & (df_duals['Time'] > 6118)]
    # Perform the join operation
    df_duals_coord = filt_duals.join(df_st.set_index('Bus'), on='Bus')
    
    # Create the 'geometry' column
    df_duals_coord['geometry'] = df_duals_coord.apply(lambda x: Point((float(x.x), float(x.y))), axis=1)
    
    # Create a GeoDataFrame and append it to the list
    M2S_duals_during.append(gpd.GeoDataFrame(df_duals_coord, geometry='geometry', crs=proj))


##LOAD
df_load_during = []

for df_load in M2S_load:
    filtered_df = df_load.loc[(df_load['Time'] <= 6312) & (df_load['Time'] > 6118)]
    df_load_during.append(filtered_df)

df_load_base = df_load_during[10]

##SLACK
df_slack_during = []

for df_slack in M2S_slack:
    filtered_df = df_slack.loc[(df_slack['Time'] <= 6312) & (df_slack['Time'] > 6118) & (df_slack['Value'] > 2)]
    df_slack_during.append(filtered_df)

# DIRECT UNSERVED ENERGY: IMPACTED SUBSTATIONS
for i in range(len(df_load_during)):
    df_load_during[i]['UE: No flood - ' + Depth[i]] = df_load_base['Value'] - df_load_during[i]['Value']

# Adding coordinates: duals and slack
#load - Direct Unserved Energy (DUE)
M2S_load_during_DUE = []

# Iterate through the list of dataframes in df_load_during
for df_load in df_load_during:
    # Perform the join operation
    df_load_ue_coord = df_load.join(df_st.set_index('Bus'), on='Bus')
    
    # Create the 'geometry' column
    df_load_ue_coord['geometry'] = df_load_ue_coord.apply(lambda x: Point((float(x.x), float(x.y))), axis=1)
    
    # Create a GeoDataFrame and append it to the list
    M2S_load_during_DUE.append(gpd.GeoDataFrame(df_load_ue_coord, geometry='geometry', crs=proj))

#slack - Indirect Unserved Energy (IUE)
M2S_slack_during_IUE = []

# Iterate through the list of dataframes in df_slack_during
for df_slack in df_slack_during:
    # Perform the join operation using the column "Node" in df_slack
    df_slack_coord = df_slack.join(df_st.set_index('Bus'), on='Node')
    
    # Create the 'geometry' column
    df_slack_coord['geometry'] = df_slack_coord.apply(lambda x: Point((float(x.x), float(x.y))), axis=1)
    
    # Column IUE
    df_slack_coord['IUE_MWh'] = df_slack_coord['Value']
    df_slack_coord.rename(columns = {"Node":"Bus"}, inplace = True)
    
    # Create a GeoDataFrame and append it to the list
    M2S_slack_during_IUE.append(gpd.GeoDataFrame(df_slack_coord, geometry='geometry', crs=proj))

# Assuming M2S_slack_during_IUE and M2S_load_during_DUE are lists of DataFrames
M2S_TUE_dfs = []

# Loop through each pair of DataFrames
for df1, df2 in zip(M2S_load_during_DUE, M2S_slack_during_IUE):
    # Perform the merge based on 'Bus' and 'Time'
    merged_df = pd.merge(df1, df2[['Bus', 'Time', 'IUE_MWh']], on=['Bus', 'Time'], how='left')
    # Fill NaN values in the "IUE_MWh" column with 0
    merged_df['IUE_MWh'] = merged_df['IUE_MWh'].fillna(0)
    # Append the merged DataFrame to the list
    M2S_TUE_dfs.append(merged_df)

# Total unserved energy due to flooding
M2S_TUE_data = []

for i in range(11):  
    # Assuming i takes values from 0 to 10
    df_sum = M2S_TUE_dfs[i][['Time','Bus','Value','IUE_MWh','x','y','geometry']]
    # Create a new dataframe with the desired column sum
    df_sum['TUE_MWh'] = M2S_TUE_dfs[i]['UE: No flood - ' + Depth[i]] + M2S_TUE_dfs[i]['IUE_MWh']
    # Add the new dataframe to the list
    M2S_TUE_data.append(df_sum)  
    # Save the new dataframe to a CSV file with the name based on Depth[i]
    df_sum.to_csv(folder_format + 'M2S_TUE_'+Depth[i]+'.csv', index=False)


# Outage data during Hurricane Florence
df_outages_nc = pd.read_csv(folder_shp + 'county_outages_hurricane_florence.csv')

#Boundaries
census_nc_data = gpd.read_file(folder_shp + 'Census_counties.shp')

# Assuming "NAMELSAD" is the common column between the two dataframes
census_nc = census_nc_data.merge(df_outages_nc, on="NAMELSAD")

# Which substations are in which counties
df_st_duals_2ft = gpd.sjoin(M2S_duals_during[2], census_nc, how="left", op="within")
df_st_duals_8ft = gpd.sjoin(M2S_duals_during[8], census_nc, how="left", op="within")

# For the table
df_st_duals_2ft.to_csv(folder_format + "M2S_duals_2ft.csv")
df_st_duals_8ft.to_csv(folder_format + "M2S_duals_8ft.csv")

# Extra columns
census_nc['Total_popu'][70] = 60203

# We start to plot here
wspacing = 0.20
hspacing = 0.01

fig, axs = plt.subplots(4,2,subplot_kw={"projection": ccrs.EqualEarth()}, figsize=(90, 92), gridspec_kw={'wspace': wspacing, 'hspace': hspacing})

# Arial family for all the figures
mpl.rcParams['font.family'] = "Arial"

###----------------------------FIRST COLUMN--------------#######################
##########################################-------PANNEL A--------##########################################################
cmap_lmp = mpl.cm.coolwarm
cmap_outages = mpl.cm.coolwarm #mismos colores para ambos

# Filter the data in the DataFrame
i=0
j=0
filtered_outage_data = census_nc['MWh_out_' + str(Captures[j])]
filtered_outage_data = filtered_outage_data[filtered_outage_data > 0]

# Limits
min_value = filtered_outage_data.min()
max_value = filtered_outage_data.max()

# Norms
norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)

# Location of the panel
ax = axs[0,0]

#Adding states borders and the ocean
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES, edgecolor='gray', facecolors = 'white', alpha=0.99)

# Counties with outage data
census_nc_filt = census_nc.loc[census_nc['MWh_out_' + str(Captures[j])] > 0]
census_nc_filt.plot(column='MWh_out_' + str(Captures[j]), ax=ax, legend=False, cmap=cmap_outages, norm=norm, transform=ccrs.PlateCarree())
# Set the aspect ratio of axs[1,0] to match axs[0,0]
axs[0,0].set_aspect(axs[0,1].get_aspect())

#Captures in time - Total Unserved Energy
M2S_TUE = M2S_TUE_data[i+2].loc[(M2S_TUE_data[i+2]['Time']==Captures[j])&(M2S_TUE_data[i+2]['TUE_MWh']>0)]

plot_lmp_a = ax.scatter(M2S_TUE['x'],M2S_TUE['y'],norm = norm, cmap = cmap_lmp, 
                              c = M2S_TUE['TUE_MWh'], s = marker_size, transform = ccrs.PlateCarree(), edgecolors= 'black', linewidths=5)

ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())

ax.annotate(Pannels[j], xy = (0,0), xytext=(40,930), 
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))

cb = plt.colorbar(plot_lmp_a,location="right",fraction=0.0223, pad=0.04,ax=ax)
cb.set_label("Outages in MWh", family = 'Arial', size = font_title, style = 'normal', fontweight = 'bold', labelpad = 35)
cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out',length = font_length)

ax.set_title(Time[j], loc = 'center',fontsize = font_title, family ='Arial', fontweight = 'bold')

### Legend for outages ###

# Create a single legend category with a range from 0 to 700

legend_labels = ['0-700 MWh','1100-1500 MWh']

# Create legend handles
handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_outages(norm(485)), markersize=marker_legend, label=legend_labels[0]),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_outages(norm(1329)), markersize=marker_legend, label=legend_labels[1])
]

# Add the legend to the plot
ax.legend(handles=handles, loc='lower left',title='Flooding-Related Outages',title_fontsize=font_ranges, frameon=False,labelspacing=0.5, prop={'size': font_ranges})

##########################################-------PANNEL B--------##########################################################
# Location of the panel
ax = axs[1,0]

#Adding states borders and the ocean
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES, edgecolor='gray', facecolors = 'white', alpha=0.99)

# Counties with outage data
census_nc_filt = census_nc.loc[census_nc['MWh_out_' + str(Captures[j+1])] > 0]
census_nc_filt.plot(column='MWh_out_' + str(Captures[j+1]), ax=ax, legend=False, cmap=cmap_outages, norm=norm, transform=ccrs.PlateCarree())
# Set the aspect ratio of axs[1,0] to match axs[0,0]
axs[1,0].set_aspect(axs[0,1].get_aspect())

#Captures in time
M2S_TUE = M2S_TUE_data[i+2].loc[(M2S_TUE_data[i+2]['Time']==Captures[j+1])&(M2S_TUE_data[i+2]['TUE_MWh']>0)]

plot_lmp_b = ax.scatter(M2S_TUE['x'],M2S_TUE['y'],norm = norm, cmap = cmap_lmp, 
                              c = M2S_TUE['TUE_MWh'], s = marker_size, transform = ccrs.PlateCarree(), edgecolors= 'black', linewidths=5)

ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())

ax.annotate(Pannels[j+1], xy = (0,0), xytext=(40,930), 
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))

cb = plt.colorbar(plot_lmp_b,location="right",fraction=0.0223, pad=0.04,ax=ax)
cb.set_label("Outages in MWh", family = 'Arial', size = font_title, style = 'normal', fontweight = 'bold', labelpad = 35)
cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out',length = font_length)

ax.set_title(Time[j+1], loc = 'center',fontsize = font_title, family ='Arial', fontweight = 'bold')

### Legend for outages ###

# Create a single legend category with a range from 0 to 700

legend_labels = ['0-700 MWh','1100-1500 MWh']

# Create legend handles
handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_outages(norm(485)), markersize=marker_legend, label=legend_labels[0]),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_outages(norm(1329)), markersize=marker_legend, label=legend_labels[1])
]

# Add the legend to the plot
ax.legend(handles=handles, loc='lower left',title='Flooding-Related Outages',title_fontsize=font_ranges, frameon=False,labelspacing=0.5, prop={'size': font_ranges})

##########################################-------PANNEL C--------##########################################################
# Location of the panel
ax = axs[2,0]

#Adding states borders and the ocean
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES, edgecolor='gray', facecolors = 'white', alpha=0.99)

# Counties with outage data
census_nc_filt = census_nc.loc[census_nc['MWh_out_' + str(Captures[j+2])] > 0]
census_nc_filt.plot(column='MWh_out_' + str(Captures[j+2]), ax=ax, legend=False, cmap=cmap_outages, norm=norm, transform=ccrs.PlateCarree())
# Set the aspect ratio of axs[1,0] to match axs[0,0]
axs[2,0].set_aspect(axs[0,1].get_aspect())

#Captures in time
#M2S_slack_gh = M2S_slack_all.loc[(M2S_slack_all['Time']==Captures[j+2])]
M2S_TUE = M2S_TUE_data[i+2].loc[(M2S_TUE_data[i+2]['Time']==Captures[j+2])&(M2S_TUE_data[i+2]['TUE_MWh']>0)]


plot_lmp_c = ax.scatter(M2S_TUE['x'],M2S_TUE['y'],norm = norm, cmap = cmap_lmp, 
                              c = M2S_TUE['TUE_MWh'], s = marker_size, transform = ccrs.PlateCarree(), edgecolors= 'black', linewidths=5)

ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())
ax.annotate(Pannels[j+2], xy = (0,0), xytext=(40,930), 
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))

cb = plt.colorbar(plot_lmp_c,location="right",fraction=0.0223, pad=0.04,ax=ax)
cb.set_label("Outages in MWh", family = 'Arial', size = font_title, style = 'normal', fontweight = 'bold', labelpad = 35)
cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out',length = font_length)

ax.set_title(Time[j+2], loc = 'center',fontsize = font_title, family ='Arial', fontweight = 'bold')

### Legend for outages ###
handles, labels = ax.get_legend_handles_labels()#Manually create legend of the plot

### Legend for outages ###

# Create a single legend category with a range from 0 to 700

legend_labels = ['0-700 MWh']

# Create legend handles
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_outages(norm(485)), markersize=marker_legend, label=legend_labels[0])]

# Add the legend to the plot
ax.legend(handles=handles, loc='lower left',title='Flooding-Related Outages',title_fontsize=font_ranges, frameon=False,labelspacing=0.5, prop={'size': font_ranges})

##########################################-------PANNEL D--------##########################################################
# Location of the panel
ax = axs[3,0]

#Adding states borders and the ocean
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES, edgecolor='gray', facecolors = 'white', alpha=0.99)

# Counties with outage data
census_nc_filt = census_nc.loc[census_nc['MWh_out_' + str(Captures[j+3])] > 0]
census_nc_filt.plot(column='MWh_out_' + str(Captures[j+3]), ax=ax, legend=False, cmap=cmap_outages, norm=norm, transform=ccrs.PlateCarree())
# Set the aspect ratio of axs[1,0] to match axs[0,0]
axs[3,0].set_aspect(axs[0,1].get_aspect())

#Captures in time
M2S_TUE = M2S_TUE_data[i+2].loc[(M2S_TUE_data[i+2]['Time']==Captures[j+3])&(M2S_TUE_data[i+2]['TUE_MWh']>0)]

plot_lmp_d = ax.scatter(M2S_TUE['x'],M2S_TUE['y'],norm = norm, cmap = cmap_lmp, 
                              c = M2S_TUE['TUE_MWh'], s = marker_size, transform = ccrs.PlateCarree(), edgecolors= 'black', linewidths=5)

ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())

ax.annotate(Pannels[j+3], xy = (0,0), xytext=(40,930), 
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))

cb = plt.colorbar(plot_lmp_d,location="right",fraction=0.0223, pad=0.04,ax=ax)
cb.set_label("Outages in MWh", family = 'Arial', size = font_title, style = 'normal', fontweight = 'bold', labelpad = 35)
cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out',length = font_length)

ax.set_title(Time[j+3], loc = 'center',fontsize = font_title, family ='Arial', fontweight = 'bold')

### Legend for outages ###

# Create a single legend category with a range from 0 to 700

legend_labels = ['0-700 MWh']

# Create legend handles
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_outages(norm(485)), markersize=marker_legend, label=legend_labels[0])]

# Add the legend to the plot
ax.legend(handles=handles, loc='lower left',title='Flooding-Related Outages',title_fontsize=font_ranges, frameon=False,labelspacing=0.5, prop={'size': font_ranges})

###----------------------------SECOND COLUMN--------------#######################
##########################################-------PANNEL A--------##########################################################
cmap_lmp = mpl.cm.coolwarm

lmp_min_a = 0#-2000#-1200#-340 
lmp_max_a = 5000#5000#2500#740#630
lmp_center_a = 1500#3000#100#40 #200

norm_lmp_a = colors.TwoSlopeNorm(vmin = lmp_min_a, vcenter = lmp_center_a, vmax = lmp_max_a)

# Location of the panel
ax = axs[0,1]

#Adding states borders and the ocean
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES, edgecolor='gray', facecolors = 'white', alpha=0.99)

# NC boundary
#NC_bound.plot(alpha=1, ax = ax, transform=ccrs.PlateCarree(), facecolor ='white')

# Captures in time
LMPs_sp = M2S_duals_during[i+2].loc[(M2S_duals_during[i+2]['Time']==Captures[j])]

plot_lmp_a = ax.scatter(LMPs_sp['x'],LMPs_sp['y'],norm = norm_lmp_a, cmap = cmap_lmp, 
                              c = LMPs_sp['Value'], s = marker_size, transform = ccrs.PlateCarree(), edgecolors= 'black')

ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())

ax.annotate(Pannels[j], xy = (0,0), xytext=(40,930), 
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))

cb = plt.colorbar(plot_lmp_a,location="right",fraction=0.0223, pad=0.04,ax=ax)
cb.set_label("LMP $/MWh", family = 'Arial', size = font_title, style = 'normal', fontweight = 'bold', labelpad = 35)
cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out',length = font_length)

ax.set_title(Time[j], loc = 'center',fontsize = font_title, family ='Arial', fontweight = 'bold')

##########################################-------PANNEL B--------##########################################################
cmap_lmp = mpl.cm.coolwarm

lmp_min_b = 0#-2000#-1200#-340 
lmp_max_b = 5000#5000#2500#740#630
lmp_center_b = 1500#3000#100#40 #200

norm_lmp_b = colors.TwoSlopeNorm(vmin = lmp_min_b, vcenter = lmp_center_b, vmax = lmp_max_b)

# Location of the panel
ax = axs[1,1]

#Adding states borders and the ocean
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES, edgecolor='gray', facecolors = 'white', alpha=0.99)

# NC boundary
#NC_bound.plot(alpha=1, ax = ax, transform=ccrs.PlateCarree(), facecolor ='white')

#Captures in time
LMPs_sp = M2S_duals_during[i+2].loc[(M2S_duals_during[i+2]['Time']==Captures[j+1])]

plot_lmp_b = ax.scatter(LMPs_sp['x'],LMPs_sp['y'],norm = norm_lmp_b, cmap = cmap_lmp, 
                              c = LMPs_sp['Value'], s = marker_size, transform = ccrs.PlateCarree(), edgecolors= 'black')

ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())
ax.annotate(Pannels[j+1], xy = (0,0), xytext=(40,930), 
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))

cb = plt.colorbar(plot_lmp_b,location="right",fraction=0.0223, pad=0.04,ax=ax)
cb.set_label("LMP $/MWh", family = 'Arial', size = font_title, style = 'normal', fontweight = 'bold', labelpad = 35)
cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out',length = font_length)

ax.set_title(Time[j+1], loc = 'center',fontsize = font_title, family ='Arial', fontweight = 'bold')

##########################################-------PANNEL C--------##########################################################
cmap_lmp = mpl.cm.coolwarm

lmp_min_c = 0#-2000#-1200#-340 
lmp_max_c = 5000#5000#2500#740#630
lmp_center_c = 1500#3000#100#40 #200

norm_lmp_c = colors.TwoSlopeNorm(vmin = lmp_min_c, vcenter = lmp_center_c, vmax = lmp_max_c)

# Location of the panel
ax = axs[2,1]

#Adding states borders and the ocean
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES, edgecolor='gray', facecolors = 'white', alpha=0.99)

# NC boundary
#NC_bound.plot(alpha=1, ax = ax, transform=ccrs.PlateCarree(), facecolor ='white')

#Captures in time
LMPs_sp = M2S_duals_during[i+2].loc[(M2S_duals_during[+2]['Time']==Captures[j+2])]

#LMPs_sp = M2S_duals_all.loc[(M2S_duals_all['Time']==Captures[j+2])]

plot_lmp_c = ax.scatter(LMPs_sp['x'],LMPs_sp['y'],norm = norm_lmp_c, cmap = cmap_lmp, 
                              c = LMPs_sp['Value'], s = marker_size, transform = ccrs.PlateCarree(), edgecolors= 'black')

ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())
ax.annotate(Pannels[j+2], xy = (0,0), xytext=(40,930), 
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))

cb = plt.colorbar(plot_lmp_c,location="right",fraction=0.0223, pad=0.04,ax=ax)
cb.set_label("LMP $/MWh", family = 'Arial', size = font_title, style = 'normal', fontweight = 'bold', labelpad = 35)
cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out',length = font_length)

ax.set_title(Time[j+2], loc = 'center',fontsize = font_title, family ='Arial', fontweight = 'bold')

##########################################-------PANNEL D--------##########################################################
cmap_lmp = mpl.cm.coolwarm

lmp_min_d = 0#-2000#-1200#-340 
lmp_max_d = 5000#5000#2500#740#630
lmp_center_d = 1500#3000#100#40 #200

norm_lmp_d = colors.TwoSlopeNorm(vmin = lmp_min_d, vcenter = lmp_center_d, vmax = lmp_max_d)

# Location of the panel
ax = axs[3,1]

#Adding states borders and the ocean
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES, edgecolor='gray', facecolors = 'white', alpha=0.99)

# NC boundary
#NC_bound.plot(alpha=1, ax = ax, transform=ccrs.PlateCarree(), facecolor ='white')

#Captures in time
LMPs_sp = M2S_duals_during[i+2].loc[(M2S_duals_during[i+2]['Time']==Captures[j+3])]

plot_lmp_d = ax.scatter(LMPs_sp['x'],LMPs_sp['y'],norm = norm_lmp_d, cmap = cmap_lmp, 
                              c = LMPs_sp['Value'], s = marker_size, transform = ccrs.PlateCarree(), edgecolors= 'black')

ax.set_extent([-83.76, -76, 33.7, 37], crs=ccrs.PlateCarree())
ax.annotate(Pannels[j+3], xy = (0,0), xytext=(40,930), 
                  xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
                  bbox = dict(boxstyle = "square", fc = "black"))

cb = plt.colorbar(plot_lmp_d,location="right",fraction=0.0223, pad=0.04,ax=ax)
cb.set_label("LMP $/MWh", family = 'Arial', size = font_title, style = 'normal', fontweight = 'bold', labelpad = 35)
cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out',length = font_length)

ax.set_title(Time[j+3], loc = 'center',fontsize = font_title, family ='Arial', fontweight = 'bold')


fig.tight_layout()
plt.savefig(folder_images+'M2S_LMPs_Slack'+Depth[i+2]+'_v3.png', dpi=150, bbox_inches='tight')
plt.show()
plt.clf()