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
import matplotlib.patches as mpatches


#folder location of the data
folder_shp = 'D:/5. Jordan_1st_paper/7. Shapefiles/'
during = 'D:/2. Research/5. Latest data/17. Figures/M2S_network/during/'
folder_3 = 'D:/8. GitHub/M2S_line_outages/Outputs_2/'
folder_images = 'D:/5. Jordan_1st_paper/Images/Latest/' #final images are in this folder    
folder_inputs = 'D:/8. GitHub/M2S_line_outages/Inputs/'
folder_svi = 'D:/5. Jordan_1st_paper/13. SVI index/'
folder_format = 'D:/5. Jordan_1st_paper/1. Text/2. Paper_tables/'

#fonts
font_box = 88#88 #texto con letra
font_text = 80#60
font_number = 70#60
font_title = 85#80
font_scatter = 500#500
marker_font = 50
font_tick = 25
font_legend = 80


Pannels = ['a','b','c','d']
Race = ['American Indian', 'Asian', 'Black', 'Native Hawaiian', 'Some Other Race', 'Two or More Races', 'White']
Captures = [6198,6216,6263,6312] #all of these captures are in the during range
Depth = ['0ft','1ft','2ft','3ft','4ft','5ft','6ft','7ft','8ft','9ft','10ft']
Time = ['09/16/18 6:00 AM','09/17/18 12:00 AM','09/18/18 11:00 PM','09/21/18 12:00 AM']

i = 0 #Depth
j = 0 #Pannels/captures
k = 0 #Base case capture
t = 0

#Projection
proj = 'EPSG:4269'

#Shapefiles
substations = pd.read_csv(during+'buses.csv')
substations['geometry'] = substations.apply(lambda x: Point((float(x.x), float(x.y))), axis=1)
M2S_st = gpd.GeoDataFrame(substations, geometry = 'geometry', crs = proj)
M2S_st['Bus'] = M2S_st['name']
df_st = M2S_st[['Bus','x','y']] #for the join table to add coordinates

# SVI Index
svi_NC = pd.read_csv(folder_svi + 'svi_index.csv')

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
    df_sum['DUE_MWh'] = df_sum['TUE_MWh'] - df_sum['IUE_MWh']
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
census_svi = census_nc.merge(svi_NC[['NAMELSAD','SVI_index']], on = "NAMELSAD")

# Which substations are in which counties
df_st_duals_2ft = gpd.sjoin(M2S_duals_during[2], census_nc, how="left", op="within")
df_st_duals_8ft = gpd.sjoin(M2S_duals_during[8], census_nc, how="left", op="within")

# For the table
df_st_duals_2ft.to_csv(folder_format + "M2S_duals_2ft.csv")
df_st_duals_8ft.to_csv(folder_format + "M2S_duals_8ft.csv")

# Extra columns
census_nc['Total_popu'][70] = 60203
census_nc['Color'] = census_nc['Total_popu']-census_nc['White']
census_nc['POC %'] = (census_nc['Color']/census_nc['Total_popu'])*100
census_nc['POW %'] = (census_nc['White']/census_nc['Total_popu'])*100


#Adding county and census information only to the nodes that show TUE(MWh)
ST_census_TUE = []

for df in M2S_TUE_data:
    # Perform the spatial join
    st_census = gpd.sjoin(df, census_nc, how='left', op="within")
    st_census_f = st_census.loc[(st_census['TUE_MWh']>0)]
    ST_census_TUE.append(st_census_f)

#Agregated data by county
ST_census_agg_TUE = []

for df in ST_census_TUE:
    # Perform the spatial join
    st_census_ct = df.groupby(['Time','NAMELSAD','White','Black','Ame_indian','Asian','Nat_Hawaii',
                               'Some_other','two_more','Total_popu'])[['IUE_MWh', 'DUE_MWh', 'TUE_MWh']].sum().reset_index()
    
    st_census_ct['White_TUE'] = (st_census_ct['White'] * st_census_ct['TUE_MWh']) / st_census_ct['Total_popu']
    st_census_ct['Black_TUE'] = (st_census_ct['Black'] * st_census_ct['TUE_MWh']) / st_census_ct['Total_popu']
    st_census_ct['AmeIndian_TUE'] = (st_census_ct['Ame_indian'] * st_census_ct['TUE_MWh']) / st_census_ct['Total_popu']
    st_census_ct['Asian_TUE'] = (st_census_ct['Asian'] * st_census_ct['TUE_MWh']) / st_census_ct['Total_popu']
    st_census_ct['NatHawaii_TUE'] = (st_census_ct['Nat_Hawaii'] * st_census_ct['TUE_MWh']) / st_census_ct['Total_popu']
    st_census_ct['SO_TUE'] = (st_census_ct['Some_other'] * st_census_ct['TUE_MWh']) / st_census_ct['Total_popu']
    st_census_ct['TOM_TUE'] = (st_census_ct['two_more'] * st_census_ct['TUE_MWh']) / st_census_ct['Total_popu']
    st_census_ct['White_DUE'] = (st_census_ct['White'] * st_census_ct['DUE_MWh']) / st_census_ct['Total_popu']
    st_census_ct['Black_DUE'] = (st_census_ct['Black'] * st_census_ct['DUE_MWh']) / st_census_ct['Total_popu']
    st_census_ct['AmeIndian_DUE'] = (st_census_ct['Ame_indian'] * st_census_ct['DUE_MWh']) / st_census_ct['Total_popu']
    st_census_ct['Asian_DUE'] = (st_census_ct['Asian'] * st_census_ct['DUE_MWh']) / st_census_ct['Total_popu']
    st_census_ct['NatHawaii_DUE'] = (st_census_ct['Nat_Hawaii'] * st_census_ct['DUE_MWh']) / st_census_ct['Total_popu']
    st_census_ct['SO_DUE'] = (st_census_ct['Some_other'] * st_census_ct['DUE_MWh']) / st_census_ct['Total_popu']
    st_census_ct['TOM_DUE'] = (st_census_ct['two_more'] * st_census_ct['DUE_MWh']) / st_census_ct['Total_popu']
    st_census_ct['White_IUE'] = (st_census_ct['White'] * st_census_ct['IUE_MWh']) / st_census_ct['Total_popu']
    st_census_ct['Black_IUE'] = (st_census_ct['Black'] * st_census_ct['IUE_MWh']) / st_census_ct['Total_popu']
    st_census_ct['AmeIndian_IUE'] = (st_census_ct['Ame_indian'] * st_census_ct['IUE_MWh']) / st_census_ct['Total_popu']
    st_census_ct['Asian_IUE'] = (st_census_ct['Asian'] * st_census_ct['IUE_MWh']) / st_census_ct['Total_popu']
    st_census_ct['NatHawaii_IUE'] = (st_census_ct['Nat_Hawaii'] * st_census_ct['IUE_MWh']) / st_census_ct['Total_popu']
    st_census_ct['SO_IUE'] = (st_census_ct['Some_other'] * st_census_ct['IUE_MWh']) / st_census_ct['Total_popu']
    st_census_ct['TOM_IUE'] = (st_census_ct['two_more'] * st_census_ct['IUE_MWh']) / st_census_ct['Total_popu']
    
    ST_census_agg_TUE.append(st_census_ct)
    
#Adding geometry data
ST_TUE_geom = []

for df in ST_census_agg_TUE:
    st_ge = census_nc[['NAMELSAD','geometry']].merge(df, on='NAMELSAD', how='outer')
    st_ge = st_ge.to_crs('EPSG:4269')
    st_ge['centroid'] = st_ge.geometry.centroid
    st_ge['centroid_x'] = st_ge['centroid'].x
    st_ge['centroid_y'] = st_ge['centroid'].y
    ST_TUE_geom.append(st_ge)

# For the barplots

m2s_percen_a = []

for df in ST_TUE_geom:
    t = 0
    filter_PV = df.loc[(df['Time']==Captures[t])]
    TUE_pv = pd.pivot_table(filter_PV, values = ['White_TUE','Black_TUE','AmeIndian_TUE','Asian_TUE','NatHawaii_TUE','SO_TUE','TOM_TUE'], index = 'NAMELSAD', aggfunc = np.sum).sum()
    TUE_pv.index = TUE_pv.index.str.replace('_TUE', '')
    DUE_pv = pd.pivot_table(filter_PV, values = ['White_DUE','Black_DUE','AmeIndian_DUE','Asian_DUE','NatHawaii_DUE','SO_DUE','TOM_DUE'], index = 'NAMELSAD', aggfunc = np.sum).sum()
    DUE_pv.index = TUE_pv.index.str.replace('_TUE', '')
    IUE_pv = pd.pivot_table(filter_PV, values = ['White_IUE','Black_IUE','AmeIndian_IUE','Asian_IUE','NatHawaii_IUE','SO_IUE','TOM_IUE'], index = 'NAMELSAD', aggfunc = np.sum).sum()
    IUE_pv.index = TUE_pv.index.str.replace('_TUE', '')

    m2s_p = pd.DataFrame()
    m2s_p['Indirect: Load shedding'] = IUE_pv
    m2s_p['Direct: Impacts on Load'] = DUE_pv
    m2s_p['Total Unserved'] = TUE_pv
    m2s_p['% Indirect'] = (m2s_p['Indirect: Load shedding']/m2s_p['Total Unserved'])*100
    m2s_p['% Direct'] = (m2s_p['Direct: Impacts on Load']/m2s_p['Total Unserved'])*100
    m2s_percen_a.append(m2s_p)

m2s_percen_b = []

for df in ST_TUE_geom:
    t = 0
    filter_PV = df.loc[(df['Time']==Captures[t+1])]
    TUE_pv = pd.pivot_table(filter_PV, values = ['White_TUE','Black_TUE','AmeIndian_TUE','Asian_TUE','NatHawaii_TUE','SO_TUE','TOM_TUE'], index = 'NAMELSAD', aggfunc = np.sum).sum()
    TUE_pv.index = TUE_pv.index.str.replace('_TUE', '')
    DUE_pv = pd.pivot_table(filter_PV, values = ['White_DUE','Black_DUE','AmeIndian_DUE','Asian_DUE','NatHawaii_DUE','SO_DUE','TOM_DUE'], index = 'NAMELSAD', aggfunc = np.sum).sum()
    DUE_pv.index = TUE_pv.index.str.replace('_TUE', '')
    IUE_pv = pd.pivot_table(filter_PV, values = ['White_IUE','Black_IUE','AmeIndian_IUE','Asian_IUE','NatHawaii_IUE','SO_IUE','TOM_IUE'], index = 'NAMELSAD', aggfunc = np.sum).sum()
    IUE_pv.index = TUE_pv.index.str.replace('_TUE', '')

    m2s_p = pd.DataFrame()
    m2s_p['Indirect: Load shedding'] = IUE_pv
    m2s_p['Direct: Impacts on Load'] = DUE_pv
    m2s_p['Total Unserved'] = TUE_pv
    m2s_p['% Indirect'] = (m2s_p['Indirect: Load shedding']/m2s_p['Total Unserved'])*100
    m2s_p['% Direct'] = (m2s_p['Direct: Impacts on Load']/m2s_p['Total Unserved'])*100
    m2s_percen_b.append(m2s_p)

m2s_percen_c = []

for df in ST_TUE_geom:
    t = 0
    filter_PV = df.loc[(df['Time']==Captures[t+2])]
    TUE_pv = pd.pivot_table(filter_PV, values = ['White_TUE','Black_TUE','AmeIndian_TUE','Asian_TUE','NatHawaii_TUE','SO_TUE','TOM_TUE'], index = 'NAMELSAD', aggfunc = np.sum).sum()
    TUE_pv.index = TUE_pv.index.str.replace('_TUE', '')
    DUE_pv = pd.pivot_table(filter_PV, values = ['White_DUE','Black_DUE','AmeIndian_DUE','Asian_DUE','NatHawaii_DUE','SO_DUE','TOM_DUE'], index = 'NAMELSAD', aggfunc = np.sum).sum()
    DUE_pv.index = TUE_pv.index.str.replace('_TUE', '')
    IUE_pv = pd.pivot_table(filter_PV, values = ['White_IUE','Black_IUE','AmeIndian_IUE','Asian_IUE','NatHawaii_IUE','SO_IUE','TOM_IUE'], index = 'NAMELSAD', aggfunc = np.sum).sum()
    IUE_pv.index = TUE_pv.index.str.replace('_TUE', '')

    m2s_p = pd.DataFrame()
    m2s_p['Indirect: Load shedding'] = IUE_pv
    m2s_p['Direct: Impacts on Load'] = DUE_pv
    m2s_p['Total Unserved'] = TUE_pv
    m2s_p['% Indirect'] = (m2s_p['Indirect: Load shedding']/m2s_p['Total Unserved'])*100
    m2s_p['% Direct'] = (m2s_p['Direct: Impacts on Load']/m2s_p['Total Unserved'])*100
    m2s_percen_c.append(m2s_p)

m2s_percen_d = []

for df in ST_TUE_geom:
    t = 0
    filter_PV = df.loc[(df['Time']==Captures[t+3])]
    TUE_pv = pd.pivot_table(filter_PV, values = ['White_TUE','Black_TUE','AmeIndian_TUE','Asian_TUE','NatHawaii_TUE','SO_TUE','TOM_TUE'], index = 'NAMELSAD', aggfunc = np.sum).sum()
    TUE_pv.index = TUE_pv.index.str.replace('_TUE', '')
    DUE_pv = pd.pivot_table(filter_PV, values = ['White_DUE','Black_DUE','AmeIndian_DUE','Asian_DUE','NatHawaii_DUE','SO_DUE','TOM_DUE'], index = 'NAMELSAD', aggfunc = np.sum).sum()
    DUE_pv.index = TUE_pv.index.str.replace('_TUE', '')
    IUE_pv = pd.pivot_table(filter_PV, values = ['White_IUE','Black_IUE','AmeIndian_IUE','Asian_IUE','NatHawaii_IUE','SO_IUE','TOM_IUE'], index = 'NAMELSAD', aggfunc = np.sum).sum()
    IUE_pv.index = TUE_pv.index.str.replace('_TUE', '')

    m2s_p = pd.DataFrame()
    m2s_p['Indirect: Load shedding'] = IUE_pv
    m2s_p['Direct: Impacts on Load'] = DUE_pv
    m2s_p['Total Unserved'] = TUE_pv
    m2s_p['% Indirect'] = (m2s_p['Indirect: Load shedding']/m2s_p['Total Unserved'])*100
    m2s_p['% Direct'] = (m2s_p['Direct: Impacts on Load']/m2s_p['Total Unserved'])*100
    m2s_percen_d.append(m2s_p)

    
# We start to plot here
wspacing = 0.17
hspacing = 0.15

# Colormap
cmap_unserved = mpl.cm.coolwarm

# Plotting
fig, axs = plt.subplots(2,2,figsize=(100, 50),gridspec_kw={'wspace': wspacing, 'hspace': hspacing})

# Arial family for all the figures
mpl.rcParams['font.family'] = "Arial"

#------FIRST COLUMN-----#
###FIRST PANNEL
ax = axs[0,0]

t = 0
#Plot races - SLACK
ax.bar(Race[0], m2s_percen_c[t+2]['% Indirect'][0], color = 'turquoise') #American Indian
ax.bar(Race[1], m2s_percen_c[t+2]['% Indirect'][1], color = 'turquoise') #Asian
ax.bar(Race[2], m2s_percen_c[t+2]['% Indirect'][2], color = 'turquoise') #Black
ax.bar(Race[3], m2s_percen_c[t+2]['% Indirect'][3], color = 'turquoise') #Native Hawaiian
ax.bar(Race[4], m2s_percen_c[t+2]['% Indirect'][4], color = 'turquoise') #Some other race
ax.bar(Race[5], m2s_percen_c[t+2]['% Indirect'][5], color = 'turquoise') #Two or more
ax.bar(Race[6], m2s_percen_c[t+2]['% Indirect'][6], color = 'turquoise') #White

#Plot races - SUBSTATION LEVEL
ax.bar(Race[0], m2s_percen_c[t+2]['% Direct'][0], bottom = m2s_percen_c[t+2]['% Indirect'][0],color = 'gray') #American Indian
ax.bar(Race[1], m2s_percen_c[t+2]['% Direct'][1], bottom = m2s_percen_c[t+2]['% Indirect'][1],color = 'gray') #Asian
ax.bar(Race[2], m2s_percen_c[t+2]['% Direct'][2], bottom = m2s_percen_c[t+2]['% Indirect'][2],color = 'gray') #Black
ax.bar(Race[3], m2s_percen_c[t+2]['% Direct'][3], bottom = m2s_percen_c[t+2]['% Indirect'][3],color = 'gray') #Native Hawaiian
ax.bar(Race[4], m2s_percen_c[t+2]['% Direct'][4], bottom = m2s_percen_c[t+2]['% Indirect'][4],color = 'gray') #Some other race
ax.bar(Race[5], m2s_percen_c[t+2]['% Direct'][5], bottom = m2s_percen_c[t+2]['% Indirect'][5],color = 'gray') #Two or more
ax.bar(Race[6], m2s_percen_c[t+2]['% Direct'][6], bottom = m2s_percen_c[t+2]['% Indirect'][6],color = 'gray') #White

#Ticks
ticks = [0,125,500,875,1250,1625,2000]

#Axes parameters
ax.tick_params(axis= 'y', direction = 'out', length=30, labelsize=font_number)#, labelrotation = 90)
ax.tick_params(axis= 'x', direction = 'out', length=0, labelsize=0.1, labelrotation = 90)
ax.set_ylabel('Unserved energy (%)', fontsize=font_title, fontweight = 'bold')

#Manually create legend of the plot
#define handles and labels that will get added to legend
handles, labels = ax.get_legend_handles_labels()

#define patches and lines to add to legend
patch1 = mpatches.Patch(color='turquoise', label='IUE (%)')
patch2 = mpatches.Patch(color='gray', label='DUE (%)')   
line1 = Line2D([0], [0], label='TUE (MWh)', color='black', linewidth= 10, linestyle='--')

#add handles
handles.extend([patch1, patch2, line1])

ax.legend(handles = handles, loc = 'upper left', bbox_to_anchor = (0.1,1.2), ncol=3, fontsize = font_text)

#Adding total amount of unserved energy
ax2 = ax.twinx()

#Plotting the unserved energy
ax2.plot(ax.get_xticks(),m2s_percen_c[t+2]['Total Unserved'].values, linestyle = '--', marker = 'o', color = 'black', 
         linewidth = 10, markersize = marker_font)

#Aligning the y axis from both plots
ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks())))

#Axes parameters (Twin ax)
ax2.tick_params(axis= 'y', direction = 'out', length=30, labelsize=font_number)
ax2.set_ylabel('Unserved Energy (MWh)', fontsize=font_title, fontweight = 'bold', labelpad = 25)

#Editing the y axis
start, end = ax2.get_ylim()
ax2.yaxis.set_ticks(np.arange(0, 2000, 200))

ax2.set_ylim(0,2000)


###SECOND PANNEL################################################

ax = axs[1,0]

t = 0
#Plot races - SLACK
ax.bar(Race[0], m2s_percen_c[t+8]['% Indirect'][0], color = 'turquoise') #American Indian
ax.bar(Race[1], m2s_percen_c[t+8]['% Indirect'][1], color = 'turquoise') #Asian
ax.bar(Race[2], m2s_percen_c[t+8]['% Indirect'][2], color = 'turquoise') #Black
ax.bar(Race[3], m2s_percen_c[t+8]['% Indirect'][3], color = 'turquoise') #Native Hawaiian
ax.bar(Race[4], m2s_percen_c[t+8]['% Indirect'][4], color = 'turquoise') #Some other race
ax.bar(Race[5], m2s_percen_c[t+8]['% Indirect'][5], color = 'turquoise') #Two or more
ax.bar(Race[6], m2s_percen_c[t+8]['% Indirect'][6], color = 'turquoise') #White

#Plot races - SUBSTATION LEVEL
ax.bar(Race[0], m2s_percen_c[t+8]['% Direct'][0], bottom = m2s_percen_c[t+8]['% Indirect'][0],color = 'gray') #American Indian
ax.bar(Race[1], m2s_percen_c[t+8]['% Direct'][1], bottom = m2s_percen_c[t+8]['% Indirect'][1],color = 'gray') #Asian
ax.bar(Race[2], m2s_percen_c[t+8]['% Direct'][2], bottom = m2s_percen_c[t+8]['% Indirect'][2],color = 'gray') #Black
ax.bar(Race[3], m2s_percen_c[t+8]['% Direct'][3], bottom = m2s_percen_c[t+8]['% Indirect'][3],color = 'gray') #Native Hawaiian
ax.bar(Race[4], m2s_percen_c[t+8]['% Direct'][4], bottom = m2s_percen_c[t+8]['% Indirect'][4],color = 'gray') #Some other race
ax.bar(Race[5], m2s_percen_c[t+8]['% Direct'][5], bottom = m2s_percen_c[t+8]['% Indirect'][5],color = 'gray') #Two or more
ax.bar(Race[6], m2s_percen_c[t+8]['% Direct'][6], bottom = m2s_percen_c[t+8]['% Indirect'][6],color = 'gray') #White

#Ticks
ticks = [0,125,500,875,1250,1625,2000]

#Axes parameters
ax.tick_params(axis= 'y', direction = 'out', length=30, labelsize=font_number)#, labelrotation = 90)
ax.tick_params(axis= 'x', direction = 'out', length=30, labelsize=font_number, labelrotation = 90)
ax.set_ylabel('Unserved energy (%)', fontsize=font_title, fontweight = 'bold')


#Adding total amount of unserved energy
ax3 = ax.twinx()

#Plotting the unserved energy
ax3.plot(ax.get_xticks(),m2s_percen_c[t+8]['Total Unserved'].values, linestyle = '--', marker = 'o', color = 'black', 
         linewidth = 10, markersize = marker_font)

#Aligning the y axis from both plots
ax3.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks())))

#Axes parameters (Twin ax)
ax3.tick_params(axis= 'y', direction = 'out', length=30, labelsize=font_number)
ax3.set_ylabel('Unserved energy (MWh)', fontsize=font_title, fontweight = 'bold', labelpad = 25)

#Editing the y axis
start, end = ax2.get_ylim()
ax3.yaxis.set_ticks(np.arange(0, 2000, 200))

ax3.set_ylim(0,2000)

plt.savefig(folder_images+'MAIN_TUE_communities_barplots.png', dpi=150, bbox_inches='tight')
plt.show()
plt.clf()


####Other projection###################################
# We start to plot here
wspacing = 0.17
hspacing = 0.15

# Colormap
cmap_unserved = mpl.cm.coolwarm

TUE_2ft_fig = ST_TUE_geom[t+2].loc[(ST_TUE_geom[t+2]["Time"]==Captures[2])]
TUE_8ft_fig = ST_TUE_geom[t+8].loc[(ST_TUE_geom[t+8]["Time"]==Captures[2])]

fig, axs = plt.subplots(2,2,figsize=(100, 50),subplot_kw={"projection": ccrs.EqualEarth()}, gridspec_kw={'wspace': wspacing, 'hspace': hspacing})

###FIRST PANNEL
ax = axs[0,1]

# Adding ocean, states and limits
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES)
ax.set_extent([-83.76, -75.8, 33.7, 37], crs=ccrs.PlateCarree())

# Define the normalization
norm_census = mcolors.Normalize(vmin=census_nc['POC %'].min(), vmax=census_nc['POC %'].max())
norm_census_svi = mcolors.Normalize(vmin=census_svi['SVI_index'].min(), vmax=census_svi['SVI_index'].max())

# Plot with normalization
census_svi.plot(column='SVI_index', ax=ax, legend=False, cmap=cmap_unserved, norm=norm_census_svi, transform=ccrs.PlateCarree()) 

ax.annotate('b', xy = (0,0), xytext=(40,1140), 
             xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
             bbox = dict(boxstyle = "square", fc = "black"))

#Colorbar
sm_k1 = plt.cm.ScalarMappable(cmap=cmap_unserved, norm=norm_census_svi)
sm_k1.set_array([])
cb = plt.colorbar(sm_k1, location="right",fraction=0.02, pad=0.04,ax=ax)
cb.set_label("Social Vulnerability Index (SVI)", family = 'Arial', size = font_title, style = 'normal',fontweight = 'bold', labelpad = 35)
cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out', length = font_tick)

# Buble plot
scatt = ax.scatter(TUE_2ft_fig['centroid_x'], TUE_2ft_fig['centroid_y'], s=TUE_2ft_fig['TUE_MWh']*70 ,transform = ccrs.PlateCarree(),color = 'black', marker = 'o',edgecolors= 'black')

# Create legend manually
marker_sizes = [TUE_2ft_fig['TUE_MWh'].mean()*1, TUE_2ft_fig['TUE_MWh'].mean()*0.5, TUE_2ft_fig['TUE_MWh'].mean()*0.2]
legend_labels = ['537 MWh', ' ', '3 MWh']

# Create legend handles
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=size, label=label) for size, label in zip(marker_sizes, legend_labels)]

# Add the legend to the plot
ax.legend(handles=handles, loc='lower left', frameon=False,labelspacing=0.25, prop={'size': font_legend})


###SECOND PANNEL
ax = axs[1,1]

# Adding ocean, states and limits
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.STATES)
ax.set_extent([-83.76, -75.8, 33.7, 37], crs=ccrs.PlateCarree())

# Define the normalization
norm_census = mcolors.Normalize(vmin=census_nc['POC %'].min(), vmax=census_nc['POC %'].max())
norm_census_svi = mcolors.Normalize(vmin=census_svi['SVI_index'].min(), vmax=census_svi['SVI_index'].max())

# Plot with normalization
census_nc.plot(column='POC %', ax=ax, legend=False, cmap=cmap_unserved, norm=norm_census, transform=ccrs.PlateCarree()) 

ax.annotate('d', xy = (0,0), xytext=(40,1140), 
             xycoords = 'axes points', size = font_box, textcoords='offset points', color = 'white',
             bbox = dict(boxstyle = "square", fc = "black"))

#Colorbar
sm_k1 = plt.cm.ScalarMappable(cmap=cmap_unserved, norm=norm_census)
sm_k1.set_array([])
cb = plt.colorbar(sm_k1, location="right",fraction=0.02, pad=0.04,ax=ax)
cb.set_label("Percentage of people of color (%)", family = 'Arial', size = font_title, style = 'normal',fontweight = 'bold', labelpad = 35)
cb.ax.tick_params(axis = 'both',labelsize=font_number, direction = 'out', length = font_tick)

# Buble plot
scatt = ax.scatter(TUE_8ft_fig['centroid_x'], TUE_8ft_fig['centroid_y'], s=TUE_8ft_fig['TUE_MWh']*70 ,transform = ccrs.PlateCarree(),color = 'black', marker = 'o',edgecolors= 'black')

# Create legend manually
marker_sizes = [TUE_2ft_fig['TUE_MWh'].mean()*1, TUE_2ft_fig['TUE_MWh'].mean()*0.5, TUE_2ft_fig['TUE_MWh'].mean()*0.2]
legend_labels = ['537 MWh', ' ', '3 MWh']

# Create legend handles
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=size, label=label) for size, label in zip(marker_sizes, legend_labels)]

# Add the legend to the plot
ax.legend(handles=handles, loc='lower left', frameon=False,labelspacing=0.25, prop={'size': font_legend})

plt.savefig(folder_images+'MAIN_TUE_communities.png', dpi=150, bbox_inches='tight')
plt.show()
plt.clf()

