#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 08:45:58 2024

@author: mario
"""

import json
import matplotlib.pyplot as plt

###############################################################################
######################## ENTER DATA FROM OTHER SCRIPT #########################
###############################################################################
# look at the filename

MOVIE_TITLE = 'Four Weddings and a Funeral' # 1415 or 7073
# MOVIE_TITLE = 'Lost in Translation' # 1221 or 6107
THRESHOLD = 0.7
NUMIMAGES = 1415

FILENAME = MOVIE_TITLE+'_numimages='+str(NUMIMAGES)+'_analysis_thresh='+str(THRESHOLD)+'_labels_dict.json'
with open(FILENAME) as f:
    labels_dict = json.load(f)
    
# print(labels_dict)

# choose some labels that are in the labels_occur_list
items_search_list = ['person', 'umbrella', 'tie', 'wine glass', 'bottle', 
                     'potted plant', 'car', 'bus', 'dining table']
items_num = len(items_search_list)

from itertools import cycle # we need to cycle through styles for plotting a few lines
# styles = ['b-', 'r-', 'g-', 'b-', 'c-', 'y-'] # for plot
colors = ['black', 'red', 'green', 'blue', 'cyan', 'yellow', 'magenta', 'royalblue',
          'orangered', 'gold'] # for bar
style_cycler = cycle(colors)

# let's plot a bar plot over all frames with the number of detected items per frame
for i, item in enumerate(items_search_list):
    data_x = []
    data_y = []
    offset = i / (items_num +1)
    for j in range(len(labels_dict)):
        labels_per_image = labels_dict[str(j)]
        item_num = labels_per_image.count(item)
        data_x.append(j + offset) # we need an offset in x otherwise we overwrite items with same occurences
        data_y.append(item_num)   # at the same frame number
    plt.bar(data_x, data_y, color = next(style_cycler), width = 0.5, label = item)
    data_x.clear()
    data_y.clear()
plt.xlabel('Frame number / 1')
plt.ylabel('Number of items / 1')
plt.legend() 
plt.title(MOVIE_TITLE)
#plt.savefig(MOVIE_TITLE+'_numimages='+str(len(labels_dict))+'_analysis_thresh='+str(THRESHOLD)+'.jpg', dpi=600)
#plt.show()

