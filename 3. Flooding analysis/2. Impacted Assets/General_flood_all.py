# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 09:20:03 2022

@author: lprieto
"""
import pandas as pd
import glob
import os
import geopandas as gpd
import numpy as np

#folders
Substations = 'D:/NC State/2. Research/Shapefiles/'
Solar = 'D:/15. Globus/RESULTS/All_solar/'
Cape_fear_st = 'D:/15. Globus/RESULTS/KCape/1. Substations/' 
Cape_fear_solar = 'D:/15. Globus/RESULTS/KCape/2. Solar pannels/' 
Lumber_st = 'D:/15. Globus/RESULTS/KLumber/1. Substations/' 
Lumber_solar = 'D:/15. Globus/RESULTS/KLumber/2. Solar pannels/' 
Neuse_st = 'D:/15. Globus/RESULTS/KNeuse/1. Substations/' 
Neuse_solar = 'D:/15. Globus/RESULTS/KNeuse/2. Solar pannels/' 
TarP_st = 'D:/15. Globus/RESULTS/KTarp/1. Substations/' 
TarP_solar = 'D:/15. Globus/RESULTS/KTarp/2. Solar pannels/' 
White_st = 'D:/15. Globus/RESULTS/KWhite/1. Substations/' 
White_solar = 'D:/15. Globus/RESULTS/KWhite/2. Solar pannels/' 
Yadkin_st = 'D:/15. Globus/RESULTS/KYadkin/1. Substations/' 
Yadkin_solar = 'D:/15. Globus/RESULTS/KYadkin/2. Solar pannels/'
Flood_st = 'D:/15. Globus/RESULTS/All_substations/'
Flood_solar = 'D:/15. Globus/RESULTS/All_solar/'
Input_model = 'D:/15. Globus/RESULTS/INPUT_model/'
folder_2 = 'D:/2. Research/5. Latest data/1. Balanciong Authorities/'
folder_3 = 'D:/15. Globus/Chrono depth/Inputs/'
folder_4 = 'D:/15. Globus/Chrono depth/Data_load/'
dates = 'D:/5. Jordan_1st_paper/5. Dates/'

#Substations all
ST = gpd.read_file(Substations + 'Substations_M2S_final.shp')
ST_all = ST.head(662)

#Solar all basins
Solar_df = pd.read_csv(Solar+'CF_solar_basins.csv')

#----------------------Substations---------------------------##
#Cape Fear
cape_st_df = pd.read_csv(Flood_st+'CF_depth_all.csv')
cape_st_df['Basin'] = ST_all['Basin']
cape_st_df2 = cape_st_df[(cape_st_df.Basin != "Neuse") & 
                        (cape_st_df.Basin != "Others") & 
                        (cape_st_df.Basin != "White Oak") & 
                        (cape_st_df.Basin != "Tar Pamlico") & 
                        (cape_st_df.Basin != "Lumber")]
#Lumber
lu_st_df = pd.read_csv(Flood_st+'LU_depth_all.csv')
lu_st_df['Basin'] = ST_all['Basin']
lu_st_df2 = lu_st_df[(lu_st_df.Basin != "Neuse") & 
                        (lu_st_df.Basin != "Others") & 
                        (lu_st_df.Basin != "White Oak") & 
                        (lu_st_df.Basin != "Tar Pamlico") & 
                        (lu_st_df.Basin != "Cape Fear")]

#Neuse
ne_st_df = pd.read_csv(Flood_st+'NE_depth_all.csv')
ne_st_df['Basin'] = ST_all['Basin']
ne_st_df2 = ne_st_df[(ne_st_df.Basin != "Lumber") & 
                        (ne_st_df.Basin != "Others") & 
                        (ne_st_df.Basin != "White Oak") & 
                        (ne_st_df.Basin != "Tar Pamlico") & 
                        (ne_st_df.Basin != "Cape Fear")]

#Tar Pamlinco
tp_st_df = pd.read_csv(Flood_st+'TP_depth_all.csv')
tp_st_df['Basin'] = ST_all['Basin']
tp_st_df2 = tp_st_df[(tp_st_df.Basin != "Neuse") & 
                        (tp_st_df.Basin != "Others") & 
                        (tp_st_df.Basin != "White Oak") & 
                        (tp_st_df.Basin != "Lumber") & 
                        (tp_st_df.Basin != "Cape Fear")]
#White Oak
wo_st_df = pd.read_csv(Flood_st+'WO_depth_all.csv')
wo_st_df['Basin'] = ST_all['Basin']
wo_st_df2 = wo_st_df[(wo_st_df.Basin != "Neuse") & 
                        (wo_st_df.Basin != "Others") & 
                        (wo_st_df.Basin != "Lumber") & 
                        (wo_st_df.Basin != "Tar Pamlico") & 
                        (wo_st_df.Basin != "Cape Fear")]

#Yadkin Pee Dee
ypd_st_df = pd.read_csv(Flood_st+'YPD_depth_all.csv')
ypd_st_df['Basin'] = ST_all['Basin']
ypd_st_df2 = ypd_st_df[(ypd_st_df.Basin != "Neuse") & 
                        (ypd_st_df.Basin != "White Oak") & 
                        (ypd_st_df.Basin != "Lumber") & 
                        (ypd_st_df.Basin != "Tar Pamlico") & 
                        (ypd_st_df.Basin != "Cape Fear")]

#All
all_basins = pd.concat([cape_st_df2, lu_st_df2, ne_st_df2, tp_st_df2, 
                        wo_st_df2, ypd_st_df2], axis=0)

all_basins.to_csv(Input_model+'ST_all_basins_flood.csv')


#----------------------Solar pannels---------------------------##
#Cape Fear
cape_so_df = pd.read_csv(Flood_solar+'CF_solar_all.csv')
cape_so_df['Basin'] = Solar_df['Basin']
cape_so_df['node'] = Solar_df['node']
cape_so_df2 = cape_so_df[(cape_so_df.Basin != "Neuse") & 
                        (cape_so_df.Basin != "Others") & 
                        (cape_so_df.Basin != "White Oak") & 
                        (cape_so_df.Basin != "Tar Pamlico") & 
                        (cape_so_df.Basin != "Lumber")]

#Lumber
lu_so_df = pd.read_csv(Flood_solar+'LU_solar_all.csv')
lu_so_df['Basin'] = Solar_df['Basin']
lu_so_df['node'] = Solar_df['node']
lu_so_df2 = lu_so_df[(lu_so_df.Basin != "Neuse") & 
                        (lu_so_df.Basin != "Others") & 
                        (lu_so_df.Basin != "White Oak") & 
                        (lu_so_df.Basin != "Tar Pamlico") & 
                        (lu_so_df.Basin != "Cape Fear")]

#Neuse
ne_so_df = pd.read_csv(Flood_solar+'NE_solar_all.csv')
ne_so_df['Basin'] = Solar_df['Basin']
ne_so_df['node'] = Solar_df['node']
ne_so_df2 = ne_so_df[(ne_so_df.Basin != "Lumber") & 
                        (ne_so_df.Basin != "Others") & 
                        (ne_so_df.Basin != "White Oak") & 
                        (ne_so_df.Basin != "Tar Pamlico") & 
                        (ne_so_df.Basin != "Cape Fear")]

#Tar Pamlinco
tp_so_df = pd.read_csv(Flood_solar+'TP_solar_all.csv')
tp_so_df['Basin'] = Solar_df['Basin']
tp_so_df['node'] = Solar_df['node']
tp_so_df2 = tp_so_df[(tp_so_df.Basin != "Neuse") & 
                        (tp_so_df.Basin != "Others") & 
                        (tp_so_df.Basin != "White Oak") & 
                        (tp_so_df.Basin != "Lumber") & 
                        (tp_so_df.Basin != "Cape Fear")]
#White Oak
wo_so_df = pd.read_csv(Flood_solar+'WO_solar_all.csv')
wo_so_df['Basin'] = Solar_df['Basin']
wo_so_df['node'] = Solar_df['node']
wo_so_df2 = wo_so_df[(wo_so_df.Basin != "Neuse") & 
                        (wo_so_df.Basin != "Others") & 
                        (wo_so_df.Basin != "Lumber") & 
                        (wo_so_df.Basin != "Tar Pamlico") & 
                        (wo_so_df.Basin != "Cape Fear")]

#Yadkin Pee Dee
ypd_so_df = pd.read_csv(Flood_solar+'YPD_solar_all.csv')
ypd_so_df['Basin'] = Solar_df['Basin']
ypd_so_df['node'] = Solar_df['node']
ypd_so_df2 = ypd_so_df[(ypd_so_df.Basin != "Neuse") & 
                        (ypd_so_df.Basin != "White Oak") & 
                        (ypd_so_df.Basin != "Lumber") & 
                        (ypd_so_df.Basin != "Tar Pamlico") & 
                        (ypd_so_df.Basin != "Cape Fear")]

#All
all_basins_so = pd.concat([cape_so_df2, lu_so_df2, ne_so_df2, tp_so_df2, 
                        wo_so_df2, ypd_so_df2], axis=0)
all_basins_so.to_csv(Input_model+'Solar_all_basins_flood.csv')

#-----------------Flooding analysis: SUBSTATIONS-----------------------#

#empty dataframe
df_st_0ft = all_basins[['name','Basin']]
df_st_2ft = all_basins[['name','Basin']]
df_st_5ft = all_basins[['name','Basin']]
df_st_10ft = all_basins[['name','Basin']]


Simulated_hours = 721
Number_substations = 662

#Identify the substation that gets flooded at >=0ft
z = np.zeros(662)

for j in range(Simulated_hours):
    col_name = "h_" + str(j)
    df_st_0ft[col_name] = z
    df_st_2ft[col_name] = z
    df_st_5ft[col_name] = z
    df_st_10ft[col_name] = z


#Flood of 0 ft
depth_st = 0 #ft
for k in range(Simulated_hours):
    for i in range (Number_substations):
        if all_basins['h_'+str(k)][i]>=depth_st:
            x = 0
        else:
            x = 1
        df_st_0ft[ "h_" + str(k)][i] = x   

#Flood of 2 ft
depth_st = 2 #ft
for k in range(Simulated_hours):
    for i in range (Number_substations):
        if all_basins['h_'+str(k)][i]>=depth_st:
            x = 0
        else:
            x = 1
        df_st_2ft[ "h_" + str(k)][i] = x   

#Flood of 5 ft
depth_st = 5 #ft
for k in range(Simulated_hours):
    for i in range (Number_substations):
        if all_basins['h_'+str(k)][i]>=depth_st:
            x = 0
        else:
            x = 1
        df_st_5ft[ "h_" + str(k)][i] = x   

#Flood of 10 ft
depth_st = 10 #ft
for k in range(Simulated_hours):
    for i in range (Number_substations):
        if all_basins['h_'+str(k)][i]>=depth_st:
            x = 0
        else:
            x = 1
        df_st_10ft[ "h_" + str(k)][i] = x   

##-------------------Substations-------------------------##
#----0ft-----#
#Transpose
df_1_0ft = df_st_0ft.set_index('name')
df_0ft = df_1_0ft.drop(columns=['Basin'])
df_0ft_ST = df_0ft.transpose()
df_0ft_ST['Date'] = pd.date_range(start='2018-09-10', end='2018-10-10', freq = "H")
df_0ft_substations = df_0ft_ST.set_index("Date")
df_0ft_substations.to_csv(Input_model+'Flood_st_0ft.csv')

#----2ft-----#
#Transpose
df_1_2ft = df_st_2ft.set_index('name')
df_2ft = df_1_2ft.drop(columns=['Basin'])
df_2ft_ST = df_2ft.transpose()
df_2ft_ST['Date'] = pd.date_range(start='2018-09-10', end='2018-10-10', freq = "H")
df_2ft_substations = df_2ft_ST.set_index("Date")
df_2ft_substations.to_csv(Input_model+'Flood_st_2ft.csv')

#----5ft-----#
#Transpose
df_1_5ft = df_st_5ft.set_index('name')
df_5ft = df_1_5ft.drop(columns=['Basin'])
df_5ft_ST = df_5ft.transpose()
df_5ft_ST['Date'] = pd.date_range(start='2018-09-10', end='2018-10-10', freq = "H")
df_5ft_substations = df_5ft_ST.set_index("Date")
df_5ft_substations.to_csv(Input_model+'Flood_st_5ft.csv')

#----10ft-----#
#Transpose
df_1_10ft = df_st_10ft.set_index('name')
df_10ft = df_1_10ft.drop(columns=['Basin'])
df_10ft_ST = df_10ft.transpose()
df_10ft_ST['Date'] = pd.date_range(start='2018-09-10', end='2018-10-10', freq = "H")
df_10ft_substations = df_10ft_ST.set_index("Date")
df_10ft_substations.to_csv(Input_model+'Flood_st_10ft.csv')

#-----------------Flooding analysis: SOLAR PANNELS-----------------------#
#empty dataframe
df_so_0ft = all_basins_so[['node','Basin']]
df_so_2ft = all_basins_so[['node','Basin']]
df_so_5ft = all_basins_so[['node','Basin']]
df_so_10ft = all_basins_so[['node','Basin']]

Simulated_hours = 721
Number_solar_pp = 669

#Identify the substation that gets flooded at >=0ft
z = np.zeros(669)

for j in range(Simulated_hours):
    col_name = "h_" + str(j)
    df_so_0ft[col_name] = z
    df_so_2ft[col_name] = z
    df_so_5ft[col_name] = z
    df_so_10ft[col_name] = z

#Flood of 0 ft
depth_so = 0 #ft
for k in range(Simulated_hours):
    for i in range (Number_solar_pp):
        if all_basins_so['h_'+str(k)][i]>=depth_so:
            x = 0
        else:
            x = 1
        df_so_0ft[ "h_" + str(k)][i] = x   

#Flood of 2 ft
depth_so = 2 #ft
for k in range(Simulated_hours):
    for i in range (Number_solar_pp):
        if all_basins_so['h_'+str(k)][i]>=depth_so:
            x = 0
        else:
            x = 1
        df_so_2ft[ "h_" + str(k)][i] = x   

#Flood of 5 ft
depth_so = 5 #ft
for k in range(Simulated_hours):
    for i in range (Number_solar_pp):
        if all_basins_so['h_'+str(k)][i]>=depth_so:
            x = 0
        else:
            x = 1
        df_so_5ft[ "h_" + str(k)][i] = x   

#Flood of 10 ft
depth_so = 10 #ft
for k in range(Simulated_hours):
    for i in range (Number_solar_pp):
        if all_basins_so['h_'+str(k)][i]>=depth_so:
            x = 0
        else:
            x = 1
        df_so_10ft[ "h_" + str(k)][i] = x   

##-------------------Solar pannels-------------------------##
#----0ft-----#
#Transpose
df_1_0ft_so = df_so_0ft.set_index('node')
df_0ft_so = df_1_0ft_so.drop(columns=['Basin'])
df_0ft_so = df_0ft_so.transpose()
df_0ft_so['Date'] = pd.date_range(start='2018-09-10', end='2018-10-10', freq = "H")
df_0ft_solar = df_0ft_so.set_index("Date")
df_0ft_solar.to_csv(Input_model+'Flood_solar_0ft.csv')

#----2ft-----#
#Transpose
df_1_2ft_so = df_so_2ft.set_index('node')
df_2ft_so = df_1_2ft_so.drop(columns=['Basin'])
df_2ft_so = df_2ft_so.transpose()
df_2ft_so['Date'] = pd.date_range(start='2018-09-10', end='2018-10-10', freq = "H")
df_2ft_solar = df_2ft_so.set_index("Date")
df_2ft_solar.to_csv(Input_model+'Flood_solar_2ft.csv')


#----5ft-----#
#Transpose
df_1_5ft_so = df_so_5ft.set_index('node')
df_5ft_so = df_1_5ft_so.drop(columns=['Basin'])
df_5ft_so = df_5ft_so.transpose()
df_5ft_so['Date'] = pd.date_range(start='2018-09-10', end='2018-10-10', freq = "H")
df_5ft_solar = df_5ft_so.set_index("Date")
df_5ft_solar.to_csv(Input_model+'Flood_solar_5ft.csv')


#----10ft-----#
#Transpose
df_1_10ft_so = df_so_10ft.set_index('node')
df_10ft_so = df_1_10ft_so.drop(columns=['Basin'])
df_10ft_so = df_10ft_so.transpose()
df_10ft_so['Date'] = pd.date_range(start='2018-09-10', end='2018-10-10', freq = "H")
df_10ft_solar = df_10ft_so.set_index("Date")
df_10ft_solar.to_csv(Input_model+'Flood_solar_10ft.csv')
