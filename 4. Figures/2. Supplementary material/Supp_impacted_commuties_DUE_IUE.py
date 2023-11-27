# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 22:08:44 2023

@author: lprieto
"""
import fiona
from shapely.geometry import shape,mapping, Point, Polygon, MultiPolygon
import geopandas as gpd
from geopandas.tools import sjoin
import pandas as pd
import pypsa
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
from shapely.geometry import Point
from pylab import axes
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.font_manager as font_manager #to standarize font
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from math import sqrt
from matplotlib.legend_handler import HandlerTuple



#folder
folder_github = 'D:/2. Research/5. Latest data/13. Collaboration flood/Images_1pp/Inputs/'
folder_shp = 'D:/5. Jordan_1st_paper/7. Shapefiles/'
during = 'D:/2. Research/5. Latest data/17. Figures/M2S_network/during/'
folder_3 = 'D:/5. Jordan_1st_paper/Images/Latest_outputs/Outputs_3/' #Con los cambios
folder_images = 'D:/5. Jordan_1st_paper/Images/Latest/' #final images are in this folder    
folder_inputs = 'D:/5. Jordan_1st_paper/Descarga/M2S_line_outages/Inputs/'
folder_svi = 'D:/5. Jordan_1st_paper/13. SVI index/'

#fonts
font_box = 100#88 #texto con letra
font_text = 100#60
font_number = 100#60
font_title = 100#80
font_scatter = 500#500
marker_font = 100
font_tick = 50
font_legend = 100


#Snapshots
Pannels = ['a','b','c','d']
Race = ['American Indian', 'Asian', 'Black', 'Native Hawaiian', 'Some Other Race', 'Two or More Races', 'White']
Captures = [6198,6216,6263,6312] #all of these captures are in the during range
Depth = ['0ft','1ft','2ft','3ft','4ft','5ft','6ft','7ft','8ft','9ft','10ft']
Time = ['09/16/18 6:00 AM','09/17/18 12:00 AM','09/18/18 11:00 PM','09/21/18 12:00 AM']

#Flatten a list
def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

#Snaphots in for loop
i = 0 #Depth
j = 0 #Pannels/captures

#Projection
proj = 'EPSG:4269'

# SVI Index
svi_NC = pd.read_csv(folder_svi + 'svi_index.csv')

#Shapefiles
substations = pd.read_csv(during+'buses.csv')
substations['geometry'] = substations.apply(lambda x: Point((float(x.x), float(x.y))), axis=1)
M2S_st = gpd.GeoDataFrame(substations, geometry = 'geometry', crs = proj)
M2S_st['Bus'] = M2S_st['name']
df_st = M2S_st[['Bus','x','y']] #for the join table to add coordinates
census_nc = gpd.read_file(folder_shp + 'Census_counties.shp')
df_census_nc_lp = pd.read_excel(folder_shp + 'census_county_latest.xls', sheet_name = 'census_county')
census_nc['Impacted transmission'] =  df_census_nc_lp['Impacted_transmission_'+Depth[i]].values
census_nc['Color'] =  df_census_nc_lp['Color'].values
census_nc['White %'] = df_census_nc_lp['White %'].values*100
census_nc['Color %'] = df_census_nc_lp['Color %'].values*100

# Adding SVI index to shapefile
census_svi = census_nc.merge(svi_NC[['NAMELSAD','SVI_index']], on = "NAMELSAD")

#Boundaries
NC_bound = gpd.read_file(folder_github + 'NC_boundary.shp')

#ACA INICIA EL FOR LOOP

#Inputs

# 2ft
M2S_load = pd.read_csv(folder_inputs + 'data_load_'+Depth[i+2]+'_final.csv')
M2S_load['Time'] = M2S_load.index

# 8ft
M2S_load_8ft = pd.read_csv(folder_inputs + 'data_load_'+Depth[i+8]+'_final.csv')
M2S_load_8ft['Time'] = M2S_load_8ft.index

#Base case - No flood
M2S_load_10ft = pd.read_csv(folder_inputs + 'data_load_'+Depth[i+10]+'_final.csv')
M2S_load_10ft['Time'] = M2S_load.index

#Outputs
M2S_duals_2ft = pd.read_csv(folder_3 + 'duals_'+Depth[i+2]+'.csv', header=0)
M2S_trans_2ft = pd.read_csv(folder_3 + 'flow_'+Depth[i+2]+'.csv', header=0)
M2S_slack_2ft = pd.read_csv(folder_3 + 'slack_'+Depth[i+2]+'.csv', header=0)
M2S_slack_2 = M2S_slack_2ft.rename(columns={"Node": "Bus"})

#Outputs
M2S_duals_8ft = pd.read_csv(folder_3 + 'duals_'+Depth[i+8]+'.csv', header=0)
M2S_trans_8ft = pd.read_csv(folder_3 + 'flow_'+Depth[i+8]+'.csv', header=0)
M2S_slack_8ft = pd.read_csv(folder_3 + 'slack_'+Depth[i+8]+'.csv', header=0)
M2S_slack_8 = M2S_slack_8ft.rename(columns={"Node": "Bus"})

#Extracting the values: during - OUTAGES
#df_duals_during = M2S_duals.loc[(M2S_duals['Time']<=6311)&(M2S_duals['Time']>6118)&(M2S_duals['Value']>2000)]
df_slack_during = M2S_slack_2.loc[(M2S_slack_2['Time']<=6312)&(M2S_slack_2['Time']>6118)&(M2S_slack_2['Value']>2)]
df_load_during = M2S_load.loc[(M2S_load['Time']<=6312)&(M2S_load['Time']>6118)]
df_slack_during_8ft = M2S_slack_8.loc[(M2S_slack_8['Time']<=6312)&(M2S_slack_8['Time']>6118)&(M2S_slack_8['Value']>2)]
df_load_during_8ft = M2S_load_8ft.loc[(M2S_load_8ft['Time']<=6312)&(M2S_load_8ft['Time']>6118)]
df_load_base = M2S_load_10ft.loc[(M2S_load_10ft['Time']<=6312)&(M2S_load_10ft['Time']>6118)]

df_load_during['Timestep'] = 0
df_load_during_8ft['Timestep'] = 0
df_load_base['Timestep'] = 0
for m in range(6119,6312):
    df_load_during['Timestep'][m] = 'h_' + str(m)
    df_load_during_8ft['Timestep'][m] = 'h_' + str(m)
    df_load_base['Timestep'][m] = 'h_' + str(m)

#create an empty dataframe
m2s_load = pd.DataFrame()
m2s_load_8ft = pd.DataFrame()
m2s_base = pd.DataFrame()

#collecting the row information
ro_t = []
ro_tt = []
ro_bas_t = []
for j in range(193): #looking to put all the rows in a single value column
    row = df_load_during.iloc[j,:265].tolist()
    row_8 = df_load_during_8ft.iloc[j,:265].tolist()
    row_b = df_load_base.iloc[j,:265].tolist()
    ro_t.append(row)
    ro_tt.append(row_8)
    ro_bas_t.append(row_b)

Rows_m2s = flatten_list(ro_t)
Rows_8ft = flatten_list(ro_tt)
Rows_base = flatten_list(ro_bas_t)

#collecting time information
ti_t = []
ti_tt = []
ti_bas_t = []
for k in range(6119,6312):
    Time_sp = [df_load_during['Time'][k]]*265
    Time_8ft = [df_load_during_8ft['Time'][k]]*265
    Time_sp_b = [df_load_base['Time'][k]]*265
    ti_t.append(Time_sp)
    ti_tt.append(Time_8ft)
    ti_bas_t.append(Time_sp_b)

Time_m2s = flatten_list(ti_t)
Time_8ft = flatten_list(ti_tt)
Time_base = flatten_list(ti_bas_t)

# Bus data
b_t = [df_load_during.columns[:265].tolist()]*193
b_tt = [df_load_during_8ft.columns[:265].tolist()]*193
b_bas = [df_load_base.columns[:265].tolist()]*193
Bus_m2s = flatten_list(b_t) 
Bus_8ft = flatten_list(b_tt) 
Bus_base = flatten_list(b_bas) 

#filling the empty dataframes
m2s_load['Bus'] = Bus_m2s
m2s_base['Bus'] = Bus_base
m2s_load_8ft['Bus'] = Bus_8ft
m2s_load['Time'] = Time_m2s
m2s_base['Time'] = Time_base
m2s_load_8ft['Time'] = Time_8ft
m2s_load['Value'] = Rows_m2s
m2s_base['Value'] = Rows_base
m2s_load_8ft['Value'] = Rows_8ft

#Unserve energy due to substations (sudden changes in power flows)
m2s_load['UE: No flood -' + Depth[i+2]] = m2s_base['Value'] - m2s_load['Value']
m2s_load['UE: No flood -' + Depth[i+8]] = m2s_base['Value'] - m2s_load_8ft['Value']

#Adding coordinates: duals and slack
#load
df_load_ue_coord = m2s_load.join(df_st.set_index('Bus'), on='Bus')
df_load_ue_coord['geometry'] = df_load_ue_coord.apply(lambda x: Point((float(x.x), float(x.y))), axis=1)
M2S_load_during_UE = gpd.GeoDataFrame(df_load_ue_coord, geometry = 'geometry', crs = proj)

#slack
df_slack_coord = df_slack_during.join(df_st.set_index('Bus'), on='Bus')
df_slack_coord['geometry'] = df_slack_coord.apply(lambda x: Point((float(x.x), float(x.y))), axis=1)
M2S_slack_during = gpd.GeoDataFrame(df_slack_coord, geometry = 'geometry', crs = proj)

#slack for 8ft
df_slack_coord_8ft = df_slack_during_8ft.join(df_st.set_index('Bus'), on='Bus')
df_slack_coord_8ft['geometry'] = df_slack_coord_8ft.apply(lambda x: Point((float(x.x), float(x.y))), axis=1)
M2S_slack_during_8ft = gpd.GeoDataFrame(df_slack_coord_8ft, geometry = 'geometry', crs = proj)

#Adding county and census information
ST_census_slack =gpd.sjoin(M2S_slack_during, census_nc, how='left',op="within")
ST_census_slack_8ft =gpd.sjoin(M2S_slack_during_8ft, census_nc, how='left',op="within")
ST_census_load =gpd.sjoin(M2S_load_during_UE, census_nc, how='left',op="within")
ST_census_load_8ft  = ST_census_load

#Exporting table to assign unserve energy per racial group - SLACK (LOAD SHEDDING) --Indirect
ST_census_slack['White_unserve'] = (ST_census_slack['White']*ST_census_slack['Value'])/ST_census_slack['Total_popu']
ST_census_slack['Black_unserve'] = (ST_census_slack['Black']*ST_census_slack['Value'])/ST_census_slack['Total_popu']
ST_census_slack['AmeIndian_unserve'] = (ST_census_slack['Ame_indian']*ST_census_slack['Value'])/ST_census_slack['Total_popu']
ST_census_slack['Asian_unserve'] = (ST_census_slack['Asian']*ST_census_slack['Value'])/ST_census_slack['Total_popu']
ST_census_slack['NatHawaii_unserve'] = (ST_census_slack['Nat_Hawaii']*ST_census_slack['Value'])/ST_census_slack['Total_popu']
ST_census_slack['SO_unserve'] = (ST_census_slack['Some_other']*ST_census_slack['Value'])/ST_census_slack['Total_popu']
ST_census_slack['TOM_unserve'] = (ST_census_slack['two_more']*ST_census_slack['Value'])/ST_census_slack['Total_popu']

ST_census_slack_8ft['White_unserve'] = (ST_census_slack_8ft['White']*ST_census_slack_8ft['Value'])/ST_census_slack_8ft['Total_popu']
ST_census_slack_8ft['Black_unserve'] = (ST_census_slack_8ft['Black']*ST_census_slack_8ft['Value'])/ST_census_slack_8ft['Total_popu']
ST_census_slack_8ft['AmeIndian_unserve'] = (ST_census_slack_8ft['Ame_indian']*ST_census_slack_8ft['Value'])/ST_census_slack_8ft['Total_popu']
ST_census_slack_8ft['Asian_unserve'] = (ST_census_slack_8ft['Asian']*ST_census_slack_8ft['Value'])/ST_census_slack_8ft['Total_popu']
ST_census_slack_8ft['NatHawaii_unserve'] = (ST_census_slack_8ft['Nat_Hawaii']*ST_census_slack_8ft['Value'])/ST_census_slack_8ft['Total_popu']
ST_census_slack_8ft['SO_unserve'] = (ST_census_slack_8ft['Some_other']*ST_census_slack_8ft['Value'])/ST_census_slack_8ft['Total_popu']
ST_census_slack_8ft['TOM_unserve'] = (ST_census_slack_8ft['two_more']*ST_census_slack_8ft['Value'])/ST_census_slack_8ft['Total_popu']

#Exporting table to assign unserve energy per racial group - LOAD (FLOODED ASSETS) --Direct
ST_census_load_8ft['White_unserve'] = (ST_census_load_8ft['White']*ST_census_load_8ft['UE: No flood -' + Depth[i+8]])/ST_census_load_8ft['Total_popu']
ST_census_load_8ft['Black_unserve'] = (ST_census_load_8ft['Black']*ST_census_load_8ft['UE: No flood -' + Depth[i+8]])/ST_census_load_8ft['Total_popu']
ST_census_load_8ft['AmeIndian_unserve'] = (ST_census_load_8ft['Ame_indian']*ST_census_load_8ft['UE: No flood -' + Depth[i+8]])/ST_census_load_8ft['Total_popu']
ST_census_load_8ft['Asian_unserve'] = (ST_census_load_8ft['Asian']*ST_census_load_8ft['UE: No flood -' + Depth[i+8]])/ST_census_load_8ft['Total_popu']
ST_census_load_8ft['NatHawaii_unserve'] = (ST_census_load_8ft['Nat_Hawaii']*ST_census_load_8ft['UE: No flood -' + Depth[i+8]])/ST_census_load_8ft['Total_popu']
ST_census_load_8ft['SO_unserve'] = (ST_census_load_8ft['Some_other']*ST_census_load_8ft['UE: No flood -' + Depth[i+8]])/ST_census_load_8ft['Total_popu']
ST_census_load_8ft['TOM_unserve'] = (ST_census_load_8ft['two_more']*ST_census_load_8ft['UE: No flood -' + Depth[i+8]])/ST_census_load_8ft['Total_popu']

ST_census_load['White_unserve'] = (ST_census_load['White']*ST_census_load['UE: No flood -' + Depth[i+2]])/ST_census_load['Total_popu']
ST_census_load['Black_unserve'] = (ST_census_load['Black']*ST_census_load['UE: No flood -' + Depth[i+2]])/ST_census_load['Total_popu']
ST_census_load['AmeIndian_unserve'] = (ST_census_load['Ame_indian']*ST_census_load['UE: No flood -' + Depth[i+2]])/ST_census_load['Total_popu']
ST_census_load['Asian_unserve'] = (ST_census_load['Asian']*ST_census_load['UE: No flood -' + Depth[i+2]])/ST_census_load['Total_popu']
ST_census_load['NatHawaii_unserve'] = (ST_census_load['Nat_Hawaii']*ST_census_load['UE: No flood -' + Depth[i+2]])/ST_census_load['Total_popu']
ST_census_load['SO_unserve'] = (ST_census_load['Some_other']*ST_census_load['UE: No flood -' + Depth[i+2]])/ST_census_load['Total_popu']
ST_census_load['TOM_unserve'] = (ST_census_load['two_more']*ST_census_load['UE: No flood -' + Depth[i+2]])/ST_census_load['Total_popu']


#Pivot table race vs Value
#para outages por lmps usar data load, filtrar nodos y tiempo para que coincidan con los elegidos en ST_census_lmps
ST_pv_sl = pd.pivot_table(ST_census_slack, values = ['White_unserve','Black_unserve','AmeIndian_unserve','Asian_unserve','NatHawaii_unserve','SO_unserve','TOM_unserve'], index = 'NAMELSAD', aggfunc = np.mean)
ST_pv_sl_sum = ST_pv_sl.sum()

ST_pv_sl_8ft = pd.pivot_table(ST_census_slack_8ft, values = ['White_unserve','Black_unserve','AmeIndian_unserve','Asian_unserve','NatHawaii_unserve','SO_unserve','TOM_unserve'], index = 'NAMELSAD', aggfunc = np.mean)
ST_pv_sl_8ft_sum = ST_pv_sl_8ft.sum()

ST_pv_load = pd.pivot_table(ST_census_load, values = ['White_unserve','Black_unserve','AmeIndian_unserve','Asian_unserve','NatHawaii_unserve','SO_unserve','TOM_unserve'], index = 'NAMELSAD', aggfunc = np.mean)
ST_pv_load_sum = ST_pv_load.sum()

ST_pv_load_8ft = pd.pivot_table(ST_census_load_8ft, values = ['White_unserve','Black_unserve','AmeIndian_unserve','Asian_unserve','NatHawaii_unserve','SO_unserve','TOM_unserve'], index = 'NAMELSAD', aggfunc = np.mean)
ST_pv_load_8ft_sum = ST_pv_load_8ft.sum()

# Total unserved energy per county
UE_county_2ft = ST_pv_load.add(ST_pv_sl, fill_value=0)
UE_county_8ft = ST_pv_load_8ft.add(ST_pv_sl_8ft, fill_value=0)

# Get a geodataframe with total unserved energy
UE_county_shp_2ft = census_nc.merge(UE_county_2ft, on='NAMELSAD', how='outer')
UE_county_shp_8ft = census_nc.merge(UE_county_8ft, on='NAMELSAD', how='outer')

# Replace NaN values with 0
UE_county_shp_2ft.fillna(0, inplace=True)
UE_county_shp_8ft.fillna(0, inplace=True)

# List of columns to sum
columns_to_sum = ['White_unserve', 'Black_unserve', 'AmeIndian_unserve', 'Asian_unserve', 'NatHawaii_unserve', 'SO_unserve', 'TOM_unserve']

# Create new column as sum of specified columns
UE_county_shp_2ft = UE_county_shp_2ft.assign(Total_Unserved=UE_county_shp_2ft[columns_to_sum].sum(axis=1))
UE_county_shp_8ft = UE_county_shp_8ft.assign(Total_Unserved=UE_county_shp_8ft[columns_to_sum].sum(axis=1))

# Obtaining indirect (load shedding) and direct unserved energy (impacts on grid assets)
##Indirect
ST_pv_sl = ST_pv_sl.assign(Total_Unserved=ST_pv_sl[columns_to_sum].sum(axis=1)) 
ST_pv_sl.reset_index()
ST_pv_sl['Indir_UE'] = ST_pv_sl['Total_Unserved']
ST_pv_sl_8ft = ST_pv_sl_8ft.assign(Total_Unserved=ST_pv_sl_8ft[columns_to_sum].sum(axis=1))
ST_pv_sl_8ft.reset_index()
ST_pv_sl_8ft['Indir_UE'] = ST_pv_sl_8ft['Total_Unserved']

##Direct
ST_pv_load = ST_pv_load.assign(Total_Unserved=ST_pv_load[columns_to_sum].sum(axis=1)) 
ST_pv_load.reset_index()
ST_pv_load['Dir_UE'] = ST_pv_load['Total_Unserved']
ST_pv_load_8ft = ST_pv_load_8ft.assign(Total_Unserved=ST_pv_load_8ft[columns_to_sum].sum(axis=1))
ST_pv_load_8ft.reset_index()
ST_pv_load_8ft['Dir_UE'] = ST_pv_load_8ft['Total_Unserved']


# Reproject the geodataframes
census_nc = census_nc.to_crs('EPSG:4269')
census_svi = census_svi.to_crs('EPSG:4269')
UE_county_shp_2ft = UE_county_shp_2ft.to_crs('EPSG:4269')
UE_county_shp_8ft = UE_county_shp_8ft.to_crs('EPSG:4269')

# Calculate the centroid for each polygon
UE_county_shp_2ft['centroid'] = UE_county_shp_2ft.geometry.centroid
UE_county_shp_8ft['centroid'] = UE_county_shp_8ft.geometry.centroid

# Separate the x and y coordinates of the centroids
UE_county_shp_2ft['centroid_x'] = UE_county_shp_2ft['centroid'].x
UE_county_shp_2ft['centroid_y'] = UE_county_shp_2ft['centroid'].y

# Separate the x and y coordinates of the centroids
UE_county_shp_8ft['centroid_x'] = UE_county_shp_8ft['centroid'].x
UE_county_shp_8ft['centroid_y'] = UE_county_shp_8ft['centroid'].y

# Add direct and indirect UE in UE_county
UE_merge_2ft = pd.merge(UE_county_shp_2ft, ST_pv_load, on='NAMELSAD', how='inner')
UE_all_2ft = pd.merge(UE_merge_2ft, ST_pv_sl, on='NAMELSAD', how='inner')

UE_merge_8ft = pd.merge(UE_county_shp_8ft, ST_pv_load_8ft, on='NAMELSAD', how='inner')
UE_all_8ft = pd.merge(UE_merge_8ft, ST_pv_sl_8ft, on='NAMELSAD', how='inner')

# For the barplots
# create an empty dataframe to store percentages from unserved energy: load and slack 
m2s_percen_2ft = pd.DataFrame()
m2s_percen_2ft['Indirect: Load shedding'] = ST_pv_sl_sum
m2s_percen_2ft['Direct: Impacts on Load'] = ST_pv_load_sum
m2s_percen_2ft['Total Unserved'] = m2s_percen_2ft['Indirect: Load shedding'] + m2s_percen_2ft['Direct: Impacts on Load']
m2s_percen_2ft['% Indirect'] = (m2s_percen_2ft['Indirect: Load shedding']/m2s_percen_2ft['Total Unserved'])*100
m2s_percen_2ft['% Direct'] = (m2s_percen_2ft['Direct: Impacts on Load']/m2s_percen_2ft['Total Unserved'])*100

m2s_percen_8ft = pd.DataFrame()
m2s_percen_8ft['Indirect: Load shedding'] = ST_pv_sl_8ft_sum
m2s_percen_8ft['Direct: Impacts on Load'] = ST_pv_load_8ft_sum
m2s_percen_8ft['Total Unserved'] = m2s_percen_8ft['Indirect: Load shedding'] + m2s_percen_8ft['Direct: Impacts on Load']
m2s_percen_8ft['% Indirect'] = (m2s_percen_8ft['Indirect: Load shedding']/m2s_percen_8ft['Total Unserved'])*100
m2s_percen_8ft['% Direct'] = (m2s_percen_8ft['Direct: Impacts on Load']/m2s_percen_8ft['Total Unserved'])*100

#==============================================================GRIDSPEC (NOW WORKING)------------
# We start to plot here
wspacing = 0.1
hspacing = 0.15 

# Colormap
cmap_unserved = mpl.cm.coolwarm

# Plotting
fig, axs = plt.subplots(2,2,figsize=(88, 50),subplot_kw={"projection": ccrs.EqualEarth()}, gridspec_kw={'wspace': wspacing, 'hspace': hspacing})

# Arial family for all the figures
mpl.rcParams['font.family'] = "Arial"

#------FIRST COLUMN-----#
###FIRST PANNEL
ax = axs[0,0]

# Adding ocean, states and limits
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES)
ax.set_extent([-83.76, -75.8, 33.7, 37], crs=ccrs.PlateCarree())

# Define the normalization
norm_census = mcolors.Normalize(vmin=census_nc['Color %'].min(), vmax=census_nc['Color %'].max())
norm_census_svi = mcolors.Normalize(vmin=census_svi['SVI_index'].min(), vmax=census_svi['SVI_index'].max())

# Plot with normalization
census_svi.plot(column='SVI_index', ax=ax, legend=False, cmap=cmap_unserved, norm=norm_census_svi, transform=ccrs.PlateCarree()) 

ax.annotate('a', xy = (0,0), xytext=(40,1140), 
             xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
             bbox = dict(boxstyle = "square", fc = "black"))

# Bubble plot
sca = ax.scatter(UE_all_2ft['centroid_x'], UE_all_2ft ['centroid_y'], s=UE_all_2ft['Dir_UE']*70 ,transform = ccrs.PlateCarree(),color = 'black', marker = 'o',edgecolors= 'black')

# Create legend manually
marker_sizes = [UE_all_2ft['Dir_UE'].mean()*25, UE_all_2ft['Dir_UE'].mean()*15, UE_all_2ft['Dir_UE'].mean()*8]
legend_labels = ['458 MWh', ' ', '0 MWh']

# Create legend handles
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=size, label=label) for size, label in zip(marker_sizes, legend_labels)]

# Add the legend to the plot
ax.legend(handles=handles, loc='lower left', frameon=False,labelspacing=0.1, prop={'size': font_legend})

###SECOND PANNEL################################################
ax = axs[1,0]

# Adding ocean, states and limits
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES)
ax.set_extent([-83.76, -75.8, 33.7, 37], crs=ccrs.PlateCarree())

# Define the normalization
norm_census = mcolors.Normalize(vmin=census_nc['Color %'].min(), vmax=census_nc['Color %'].max())
norm_census_svi = mcolors.Normalize(vmin=census_svi['SVI_index'].min(), vmax=census_svi['SVI_index'].max())

# Plot with normalization
census_svi.plot(column='SVI_index', ax=ax, legend=False, cmap=cmap_unserved, norm=norm_census_svi, transform=ccrs.PlateCarree()) 

ax.annotate('c', xy = (0,0), xytext=(40,1140), 
             xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
             bbox = dict(boxstyle = "square", fc = "black"))

# Buble plot
sca = ax.scatter(UE_all_8ft['centroid_x'], UE_all_8ft ['centroid_y'], s=UE_all_8ft['Dir_UE']*70 ,transform = ccrs.PlateCarree(),color = 'black', marker = 'o',edgecolors= 'black')

# Create legend manually
marker_sizes = [UE_all_2ft['Dir_UE'].mean()*25, UE_all_2ft['Dir_UE'].mean()*15, UE_all_2ft['Dir_UE'].mean()*8]
legend_labels = ['458 MWh', ' ', '0 MWh']

# Create legend handles
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=size, label=label) for size, label in zip(marker_sizes, legend_labels)]

# Add the legend to the plot
ax.legend(handles=handles, loc='lower left', frameon=False,labelspacing=0.1, prop={'size': font_legend})

#------SECOND COLUMN-----#
###FIRST PANNEL
ax = axs[0,1]

# Adding ocean, states and limits
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES)
ax.set_extent([-83.76, -75.8, 33.7, 37], crs=ccrs.PlateCarree())

# Define the normalization
norm_census = mcolors.Normalize(vmin=census_nc['Color %'].min(), vmax=census_nc['Color %'].max())
norm_census_svi = mcolors.Normalize(vmin=census_svi['SVI_index'].min(), vmax=census_svi['SVI_index'].max())

# Plot with normalization
census_svi.plot(column='SVI_index', ax=ax, legend=False, cmap=cmap_unserved, norm=norm_census_svi, transform=ccrs.PlateCarree()) 

ax.annotate('b', xy = (0,0), xytext=(40,1140), 
             xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
             bbox = dict(boxstyle = "square", fc = "black"))

# Buble plot
scatt = ax.scatter(UE_all_2ft ['centroid_x'], UE_all_2ft ['centroid_y'], s=UE_all_2ft['Indir_UE']*70 ,transform = ccrs.PlateCarree(),color = 'green', marker = 'o',edgecolors= 'black')


# Create legend manually
marker_sizes = [UE_all_2ft['Dir_UE'].mean()*25, UE_all_2ft['Dir_UE'].mean()*15, UE_all_2ft['Dir_UE'].mean()*8]
legend_labels = ['458 MWh', ' ', '0 MWh']

# Create legend handles
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=size, label=label) for size, label in zip(marker_sizes, legend_labels)]

# Add the legend to the plot
ax.legend(handles=handles, loc='lower left', frameon=False,labelspacing=0.1, prop={'size': font_legend})

###SECOND PANNEL
ax = axs[1,1]

# Adding ocean, states and limits
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES)
ax.set_extent([-83.76, -75.8, 33.7, 37], crs=ccrs.PlateCarree())

# Define the normalization
norm_census = mcolors.Normalize(vmin=census_nc['Color %'].min(), vmax=census_nc['Color %'].max())
norm_census_svi = mcolors.Normalize(vmin=census_svi['SVI_index'].min(), vmax=census_svi['SVI_index'].max())

# Plot with normalization
census_svi.plot(column='SVI_index', ax=ax, legend=False, cmap=cmap_unserved, norm=norm_census_svi, transform=ccrs.PlateCarree()) 

ax.annotate('d', xy = (0,0), xytext=(40,1140), 
             xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
             bbox = dict(boxstyle = "square", fc = "black"))

#Colorbar
sm_k1 = plt.cm.ScalarMappable(cmap=cmap_unserved, norm=norm_census_svi)
sm_k1.set_array([])
cb = plt.colorbar(sm_k1, location="right",fraction=0.02, pad=0.04,ax=axs)
cb.set_label("Social Vulnerability Index (SVI)", family = 'Arial', size = font_title, style = 'normal',fontweight = 'bold', labelpad = 35)
cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out', length = font_tick)

# Buble plot
scatt = ax.scatter(UE_all_8ft ['centroid_x'], UE_all_8ft ['centroid_y'], s=UE_all_8ft['Indir_UE']*70 ,transform = ccrs.PlateCarree(),color = 'green', marker = 'o',edgecolors= 'black')


# Create legend manually
marker_sizes = [UE_all_2ft['Dir_UE'].mean()*25, UE_all_2ft['Dir_UE'].mean()*15, UE_all_2ft['Dir_UE'].mean()*8]
legend_labels = ['458 MWh', ' ', '0 MWh']

# Create legend handles
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=size, label=label) for size, label in zip(marker_sizes, legend_labels)]

# Add the legend to the plot
ax.legend(handles=handles, loc='lower left', frameon=False, labelspacing=0.1, prop={'size': font_legend})

#fig.tight_layout()
plt.savefig(folder_images+'UE_communities_dir_indir.png', dpi=150, bbox_inches='tight')
plt.show()
plt.clf()
