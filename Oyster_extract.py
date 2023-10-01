#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extracts the trip length / size from
Oyster Card data given by the TFL API webpage

@author: jake
"""

"IMPORTS"
import pandas as pd
import networkx as nx
import string
import numpy as np
from file_io import save_obj

"LOAD AND CLEAN DATA"
#oyster data
oyster_data = pd.read_csv('data/Oyster_data/Nov09Export.csv')
oyster_data = oyster_data[oyster_data['StartStn'] != 'Unstarted']
oyster_data = oyster_data[oyster_data['StartStn'] != 'Bus']
oyster_data = oyster_data[oyster_data['EndStation'] != 'Unfinished']
oyster_data = oyster_data[oyster_data['EndStation'] != 'Bus']
#tube graph
G = nx.read_gpickle('data/tube_graph_processed.pkl')

"CLEAN NAMES ON GRAPH"
#get names from graph
stn_to_name = nx.get_node_attributes(G, 'name')

#delete unwanted characters
del_name = [' Underground Station',
            ' Rail Station',
            ' DLR Station']

replacements = [("Queen's Park", "Queens Park"),
                ("Regent's Park", "Regents Park"),
                ("St. James's Park", "St James's Park"),
                ("King's Cross & St Pancras International", "Kings Cross T"),
                ("Earl's Court", "Earls Court"),
                ("Edgware Road (Bakerloo)", "Edgware Road B"),
                ("Edgware Road (Circle Line)", "Edgware Road M"),
                ("Hammersmith", "Hammersmith D"),
                ("Harrow-on-the-Hill", "Harrow on the Hill"),
                ("Harrow & Wealdstone", "Harrow Wealdstone"),
                ("Heathrow Airport Terminal 5", "Heathrow Term 5"),
                ("Heathrow Terminals 2 & 3", "Heathrow Terms 123"),
                ("High Street Kensington", "High Street Kens")]

#remove excess strings ('Underground Station', etc.)
for i in range(len(del_name)):
    stn_to_name = {k:v.replace(del_name[i], '') for k,v in stn_to_name.items()}
#manually replace station names
for replacement in replacements:
    stn_to_name = {k:v.replace(replacement[0], replacement[1]) for k,v in stn_to_name.items()}

#get reverse lookup
name_to_stn = {v:k for k,v in stn_to_name.items()}

"COLLECT LENGTHS / SIZE OF TRIP"
mislabeled_stations = set()
trip_lengths = []
trip_sizes = []
edge_count = dict(G.edges())
edge_count = dict.fromkeys(edge_count, 0)
for row_id, row in oyster_data.iterrows():
    
    #check if start station name in graph
    if row['StartStn'] in name_to_stn.keys():
        
        #check if end station name in graph
        if row['EndStation'] in name_to_stn.keys():
            
            #if both names pass, get length (dist) & size (# stops) of journey
            path = nx.dijkstra_path(G, name_to_stn[row['StartStn']],
                                    name_to_stn[row['EndStation']], weight = 'dist')
            path_dist = nx.dijkstra_path_length(G, name_to_stn[row['StartStn']],
                                                name_to_stn[row['EndStation']], weight = 'dist')
        
            #count edges
            for i in range(len(path)-1):
                edge_count[(path[i], path[i+1])] += 1
                
            #append length / time
            trip_lengths.append(path_dist)
            trip_sizes.append(len(path))
        
        #else add name to be checked
        else:
            mislabeled_stations.add(row['EndStation'])
    
    #else add name to be checked
    else:
        mislabeled_stations.add(row['StartStn'])
        
"SAVE THE DATA"
save_obj('data/Oyster_data/output/TFL-edge_count.pkl', edge_count)
save_obj('data/Oyster_data/output/TFL-trip_lengths.pkl', trip_lengths)
save_obj('data/Oyster_data/output/TFL-trip_sizes.pkl', trip_sizes)