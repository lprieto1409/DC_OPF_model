# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 21:53:40 2022

@author: lprieto
"""
import pandas as pd
import glob
import os

#folders
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

#--------------Substations--------------------------
#--Cape Fear
for i in range(721):
    cape_st_df = pd.read_csv(Cape_fear_st+'CF_depth_h'+str(i)+'.csv')
    df = cape_st_df.rename(columns={"CF_flood_mean":"h_"+str(i)})
    df.to_csv(Cape_fear_st+'CF_depth_h'+str(i)+'.csv')

#Joining all files in one
all_files = glob.glob(os.path.join(Cape_fear_st, "*.csv"))

df_cf = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
df_cf.to_csv(Cape_fear_st+'CF_depth_all.csv')

#Lumber
for i in range(721):
    lu_st_df = pd.read_csv(Lumber_st+'LU_depth_h'+str(i)+'.csv')
    df = lu_st_df.rename(columns={"LU_flood_mean":"h_"+str(i)})
    df.to_csv(Lumber_st+'LU_depth_h'+str(i)+'.csv')

#Joining all files in one
all_files = glob.glob(os.path.join(Lumber_st, "*.csv"))

df_lu = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
df_lu.to_csv(Lumber_st+'LU_depth_all.csv')

#Neuse
for i in range(721):
    ne_st_df = pd.read_csv(Neuse_st+'NE_depth_h'+str(i)+'.csv')
    df = ne_st_df.rename(columns={"NE_flood_mean":"h_"+str(i)})
    df.to_csv(Neuse_st+'NE_depth_h'+str(i)+'.csv')

#Joining all files in one
all_files = glob.glob(os.path.join(Neuse_st, "*.csv"))

df_ne = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
df_ne.to_csv(Neuse_st+'NE_depth_all.csv')

#Tar Pamlico
for i in range(721):
    tp_st_df = pd.read_csv(TarP_st+'TP_depth_h'+str(i)+'.csv')
    df = tp_st_df.rename(columns={"TP_flood_mean":"h_"+str(i)})
    df.to_csv(TarP_st+'TP_depth_h'+str(i)+'.csv')

#Joining all files in one
all_files = glob.glob(os.path.join(TarP_st, "*.csv"))

df_tp = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
df_tp.to_csv(TarP_st+'TP_depth_all.csv')
    
#White Oak
for i in range(721):
    wo_st_df = pd.read_csv(White_st+'WO_depth_h'+str(i)+'.csv')
    df = wo_st_df.rename(columns={"WO_flood_mean":"h_"+str(i)})
    df.to_csv(White_st+'WO_depth_h'+str(i)+'.csv')

#Joining all files in one
all_files = glob.glob(os.path.join(White_st, "*.csv"))

df_wo = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
df_wo.to_csv(White_st+'WO_depth_all.csv')

#Yadkin
for i in range(0,257,1):
    ypd_st_df = pd.read_csv(Yadkin_st+'YPD_depth_ST_h'+str(i)+'.csv')
    df = ypd_st_df.rename(columns={"YPD_flood_mean":"h_"+str(i)})
    df.to_csv(Yadkin_st+'YPD_depth_ST_h'+str(i)+'.csv')

#Joining all files in one
all_files = glob.glob(os.path.join(Yadkin_st, "*.csv"))

df_ypd = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
df_ypd.to_csv(Yadkin_st+'WO_depth_all.csv')

#--------------Solar pannels--------------------------
#--Cape Fear
for i in range(721):
    cape_sol_df = pd.read_csv(Cape_fear_solar+'CF_solar_h'+str(i)+'.csv')
    df = cape_sol_df.rename(columns={"CF_flood_solar_mean":"h_"+str(i)})
    df.to_csv(Cape_fear_solar+'CF_solar_h'+str(i)+'.csv')

#Joining all files in one
all_files = glob.glob(os.path.join(Cape_fear_solar, "*.csv"))

df_cf_so = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
df_cf_so.to_csv(Cape_fear_solar+'CF_solar_all.csv')

#--Lumber
for i in range(721):
    lumber_sol_df = pd.read_csv(Lumber_solar+'LU_solar_h'+str(i)+'.csv')
    df = lumber_sol_df.rename(columns={"LU_flood_solar_mean":"h_"+str(i)})
    df.to_csv(Lumber_solar+'LU_solar_h'+str(i)+'.csv')

#Joining all files in one
all_files = glob.glob(os.path.join(Lumber_solar, "*.csv"))

df_lu_so = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
df_lu_so.to_csv(Lumber_solar+'LU_solar_all.csv')

#--Neuse
for i in range(721):
    neuse_sol_df = pd.read_csv(Neuse_solar+'NE_solar_h'+str(i)+'.csv')
    df = neuse_sol_df.rename(columns={"NE_flood_solar_mean":"h_"+str(i)})
    df.to_csv(Neuse_solar+'NE_solar_h'+str(i)+'.csv')

#Joining all files in one
all_files = glob.glob(os.path.join(Neuse_solar, "*.csv"))

df_ne_so = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
df_ne_so.to_csv(Neuse_solar+'NE_solar_all.csv')

#Tar Pamlico
for i in range(721):
    tp_sol_df = pd.read_csv(TarP_solar+'TP_solar_h'+str(i)+'.csv')
    df = tp_sol_df.rename(columns={"TP_flood_solar_mean":"h_"+str(i)})
    df.to_csv(TarP_solar+'TP_solar_h'+str(i)+'.csv')

#Joining all files in one
all_files = glob.glob(os.path.join(TarP_solar, "*.csv"))

df_tp_so = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
df_tp_so.to_csv(TarP_solar+'TP_solar_all.csv')

#White Oak
for i in range(721):
    tp_sol_df = pd.read_csv(White_solar+'WO_solar_h'+str(i)+'.csv')
    df = tp_sol_df.rename(columns={"WO_flood_solar_mean":"h_"+str(i)})
    df.to_csv(White_solar+'WO_solar_h'+str(i)+'.csv')

#Joining all files in one
all_files = glob.glob(os.path.join(White_solar, "*.csv"))

df_WO_so = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
df_WO_so.to_csv(White_solar+'WO_solar_all.csv')

#Yadkin
for i in range(601,721,1):
    tp_sol_df = pd.read_csv(Yadkin_solar+'YPD_solar_h'+str(i)+'.csv')
    df = tp_sol_df.rename(columns={"YPD_flood_solar_mean":"h_"+str(i)})
    df.to_csv(Yadkin_solar+'YPD_solar_h'+str(i)+'.csv')

#Joining all files in one
all_files = glob.glob(os.path.join(Yadkin_solar, "*.csv"))

df_YPD_so = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
df_YPD_so.to_csv(Yadkin_solar+'YPD_solar_all.csv')


