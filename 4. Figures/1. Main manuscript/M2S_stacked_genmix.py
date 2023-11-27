# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 19:00:17 2023

@author: lprieto
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Folder location
folder = 'D:/2. Research/5. Latest data/14. Working on the outputs from the model/365_newprices/'
folder_images = 'D:/5. Jordan_1st_paper/Images/Latest/'

# Generation mix
gen_mix = pd.read_excel(folder + 'Gen_mix_M2S_2.xls', sheet_name= 'Calibration')
gen_mix = gen_mix.drop([2, 3, 8]).reset_index(drop=True)

# Define the data
historical = gen_mix['Egrid_percen']
simulated = gen_mix['M2S_percen']
labels = gen_mix['Fuel types']
simulations = ['Historical', 'Simulated']

# Create the DataFrame
df = pd.DataFrame({'Fuel type': labels,'Historical': historical, 'Simulated': simulated})

# Create a figure and a set of subplots
fig, ax = plt.subplots(figsize=(7, 8))

# Arial family for all the figures
mpl.rcParams['font.family'] = "Arial"

# Set the bar width
bar_width = 0.4

# Define colors for each fuel type
colors = gen_mix['Color']

# Creating the bars

# Historical
for i in range(len(df['Fuel type'])):
    bottom = sum(df['Historical'][:i]) if i > 0 else 0
    ax.bar(simulations[0], df['Historical'][i], bar_width, bottom=bottom, color=colors[i])
    ax.text(simulations[0], bottom + df['Historical'][i]/2, f"{df['Historical'][i]:.2f}%", ha='center', va='center', color='black', fontweight='bold', fontsize=13)


# Simulated
for i in range(len(df['Fuel type'])):
    bottom = sum(df['Simulated'][:i]) if i > 0 else 0
    ax.bar(simulations[1], df['Simulated'][i], bar_width, bottom=bottom, color=colors[i])
    ax.text(simulations[1], bottom + df['Simulated'][i]/2, f"{df['Simulated'][i]:.2f}%", ha='center', va='center', color='black', fontweight='bold', fontsize=13)

# Adding the legend outside the plot
patches = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(df['Fuel type']))]
ax.legend(patches, df['Fuel type'].values, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

# Adding labels for x and y axis
plt.ylabel('Generation Mix (%)', fontsize=16, fontweight='bold')

# Increase font size of ticks
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(np.arange(0, 110, 10),fontsize=14)

# Show the plot
fig.tight_layout()
plt.savefig(folder_images+'M2S_genmix_cali.png', dpi=150, bbox_inches='tight')
plt.show()
plt.clf()
