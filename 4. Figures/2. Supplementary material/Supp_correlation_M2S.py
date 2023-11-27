# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 19:46:19 2023

Hay que editar esto, falta mucho para llegar al producto que quiero!!!

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
from scipy.stats import pearsonr


#folder
folder_shp = 'D:/5. Jordan_1st_paper/7. Shapefiles/'
during = 'D:/2. Research/5. Latest data/17. Figures/M2S_network/during/'
folder_3 = 'D:/8. GitHub/M2S_line_outages/Outputs_2/'
folder_images = 'D:/5. Jordan_1st_paper/Images/Latest/' #final images are in this folder    
folder_inputs = 'D:/8. GitHub/M2S_line_outages/Inputs/'
folder_svi = 'D:/5. Jordan_1st_paper/13. SVI index/'
folder_format = 'D:/5. Jordan_1st_paper/1. Text/2. Paper_tables/'

#fonts
font_box = 160#88#88 #texto con letra
font_text = 140#70#60
font_number = 160#70#60
font_title = 162#85#80
font_scatter = 500#500
marker_font = 50
font_tick = 25
font_legend = 160#60

#Snapshots
Pannels = ['a','b','c','d']
Race_cate = ['American Indian', 'Asian', 'Black', 'Native Hawaiian', 'Some Other Race', 'Two or More Races', 'White']
Race_abr = ['AI','ASI','BL','NH','SOR','TOM','WH']
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
df_st_duals_2ft['LMP_2ft'] = df_st_duals_2ft['Value']
df_st_duals_8ft.to_csv(folder_format + "M2S_duals_8ft.csv")
df_st_duals_8ft['LMP_8ft'] = df_st_duals_8ft['Value']


# Extra columns
census_nc['Total_popu'][70] = 60203

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

existing_index = m2s_percen_a[6].index
m2s_percen_a[7] = pd.DataFrame(0, index=existing_index, columns=m2s_percen_a[6].columns)
m2s_percen_a[8] = pd.DataFrame(0, index=existing_index, columns=m2s_percen_a[6].columns)
m2s_percen_a[9] = pd.DataFrame(0, index=existing_index, columns=m2s_percen_a[6].columns)
m2s_percen_a[10] = pd.DataFrame(0, index=existing_index, columns=m2s_percen_a[6].columns)


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

existing_index = m2s_percen_b[7].index
m2s_percen_b[8] = pd.DataFrame(0, index=existing_index, columns=m2s_percen_b[7].columns)
m2s_percen_b[9] = pd.DataFrame(0, index=existing_index, columns=m2s_percen_b[7].columns)
m2s_percen_b[10] = pd.DataFrame(0, index=existing_index, columns=m2s_percen_b[7].columns)

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

existing_index = m2s_percen_c[8].index
m2s_percen_c[9] = pd.DataFrame(0, index=existing_index, columns=m2s_percen_c[8].columns)
m2s_percen_c[10] = pd.DataFrame(0, index=existing_index, columns=m2s_percen_c[8].columns)


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

existing_index = m2s_percen_d[8].index
m2s_percen_d[10] = pd.DataFrame(0, index=existing_index, columns=m2s_percen_d[8].columns)

#Data for the correlation

# Total unserved energy
TUE_2ft_fig = ST_TUE_geom[t+2].loc[(ST_TUE_geom[t+2]["Time"]==Captures[2])]
TUE_2ft_final = TUE_2ft_fig.merge(census_svi[['NAMELSAD','SVI_index']], on = "NAMELSAD")
TUE_8ft_fig = ST_TUE_geom[t+8].loc[(ST_TUE_geom[t+8]["Time"]==Captures[2])]
TUE_8ft_final = TUE_8ft_fig.merge(census_svi[['NAMELSAD','SVI_index']], on = "NAMELSAD")    

# Locational Marginal Prices (Shadow cost)
Duals_2ft_fig = df_st_duals_2ft.loc[(df_st_duals_2ft["Time"]==Captures[2])]
Duals_8ft_fig = df_st_duals_8ft.loc[(df_st_duals_8ft["Time"]==Captures[2])]
Duals_2ft =gpd.sjoin(Duals_2ft_fig[['Bus','LMP_2ft','geometry']],census_nc, how='left', op="within")
Duals_8ft =gpd.sjoin(Duals_8ft_fig[['Bus','LMP_8ft','geometry']],census_nc, how='left', op="within")

Duals_2ft_census = Duals_2ft.groupby(['NAMELSAD'])[['LMP_2ft','POC %', 'POW %']].mean().reset_index()
Duals_8ft_census = Duals_8ft.groupby(['NAMELSAD'])[['LMP_8ft','POC %', 'POW %']].mean().reset_index()

# All together
Data_2ft_final = TUE_2ft_final[['NAMELSAD','IUE_MWh','DUE_MWh', 'TUE_MWh', 'SVI_index']].merge(Duals_2ft_census, on = "NAMELSAD")
Data_2ft_final  = Data_2ft_final.rename(columns={'IUE_MWh': 'IUE_2ft', 'DUE_MWh': 'DUE_2ft',
                                                 'TUE_MWh': 'TUE_2ft','POC %': 'POC', 'POW %': 'POW'})
Data_8ft_final = TUE_8ft_final[['NAMELSAD','IUE_MWh','DUE_MWh', 'TUE_MWh', 'SVI_index']].merge(Duals_8ft_census, on = "NAMELSAD")
Data_8ft_final  = Data_8ft_final.rename(columns={'IUE_MWh': 'IUE_8ft', 'DUE_MWh': 'DUE_8ft',
                                                 'TUE_MWh': 'TUE_8ft','POC %': 'POC', 'POW %': 'POW'})
    
df_imp = pd.merge(Data_2ft_final, Data_8ft_final[['NAMELSAD', 'IUE_8ft', 'DUE_8ft', 'TUE_8ft', 'LMP_8ft']], on='NAMELSAD', how='left')
df_imp = df_imp.fillna(0)

# Calculate the correlation matrix
correlation_matrix = df_imp.corr()

# Create an empty DataFrame to store p-values
p_values = pd.DataFrame(index=correlation_matrix.index, columns=correlation_matrix.columns)

# Loop through the columns and rows of the correlation matrix
for row in correlation_matrix.index:
    for col in correlation_matrix.columns:
        # Get the correlation coefficient and p-value
        corr_coef, p_value = pearsonr(df_imp[row], df_imp[col])
        # Store the p-value in the p_values DataFrame
        p_values.loc[row, col] = p_value

# Print or use the p_values DataFrame
print(p_values)

# Create the heatmap
plt.figure(figsize=(8, 6))

# Arial family for all the figures
mpl.rcParams['font.family'] = "Arial"

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation between grid metrics at the county level', fontweight = 'bold')
plt.savefig(folder_images+'M2S_correlation.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

