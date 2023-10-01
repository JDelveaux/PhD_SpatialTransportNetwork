#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extra functions / modifications for TFL graph

Includes:
    Connect walkable stations

@author: jake
"""

"IMPORTS"
from geopy import geodesic
import pandas as pd

"FUNCTIONS"
def connect_walkable_stations(G, threshold, penalty):
    """
    

    Parameters
    ----------
    G : NetworkX graph
        Graph of the TFL transport network.
    threshold : float
        max distance (in km) to connect by walking.
    penalty : float
        distance pentalty for walking.

    Returns
    -------
    G : The same graph with added walking edges

    """
    G_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    close_pairs = []
    for row_id, row in G_df.iterrows():
        for c_row_id, c_row in G_df.iterrows():
            dist = geodesic(row['pos'], c_row['pos']).km
            if dist < threshold and row_id != c_row_id:
                close_pairs.append([row_id, c_row_id, dist*penalty])
    
    for pair in close_pairs:
        G.add_edge(pair[0], pair[1], dist = pair[2], walk = True)
        G.add_edge(pair[1], pair[0], dist = pair[2], walk = True)
        
    return G