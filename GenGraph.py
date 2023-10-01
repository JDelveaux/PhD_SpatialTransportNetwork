#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds and cleans the TFL transport network from API data

Modes:
   Tube
   DLR
   TFLRail
   Overground
   
Node Attributes:
    pos #(position)
    nlc
    name #(common name)
    
Edge Attributes:
    dist #(length in km)
    line
    ll_am
    ll_mid
    ll_pm
    ll_tot #(line loads / # passengers from TFL data)

@author: jake
"""

"IMPORTS"
import geopandas as gpd
import pandas as pd
from ratelimit import limits
import networkx as nx
import time
from geopy.distance import geodesic
from shapely.geometry import Point
from GenGraph_lib import tfl_requests
import itertools
import numpy as np

"PARAMETERS"
modes = ['tube', 'dlr', 'overground', 'tflrail']

#TFL API ID and Key
app_id = '77ee1608'
app_key = '7234d332e4e5a671ce84f05251046ac4'

"LOAD DATA"
#NAPTAN -> NLC codes
nap_nlc = pd.read_csv('data/GenGraph_data/station_naptan_nlc.csv')
line_loads = pd.read_csv('data/GenGraph_data/2018MTT_Link_Load.csv')

#London Map
fp = 'data/GenGraph_data/LondonMap/London_Ward.shp'
map_df = gpd.read_file(fp)
map_df = map_df.to_crs("+init=epsg:4326")
#London Coords
cent_lat = 51.50052358379
cent_lon = -0.10938142185
cent_rad = 40000.0 #meters

"GEOGRAPHIC FUNCTIONS"
def in_london(stoppoint, london_gpdf):
    loc = Point(stoppoint['lon'], stoppoint['lat'])
    return any(london_gpdf.contains(loc))

def dist(sp1, sp2):
    return geodesic((sp1['lat'], sp1['lon']),(sp2['lat'], sp2['lon']))

"BUILD GRAPH"
@limits(calls=450, period=60) #limit to 450 hits/minute (max 500)
def get_tfl_network_from_api(modes, app_id, app_key):
    """
    

    Parameters
    ----------
    modes : list of str
        Transport modes to consider.
    app_id : str
        TFL API ID.
    app_key : str
        TFL API Key.

    Returns
    -------
    NetworkX graph with TFL stations as nodes and connections
    as edges.

    """
    
    #initialize directed network
    G = nx.DiGraph()
    
    #add stations one mode at a time
    for mode in modes:
        
        #print statement
        print(str(mode) + ' start')
        start = time.time()
        
        #get all lines in mode
        lines = tfl_requests.get_lines(mode, app_id = app_id, app_key = app_key)
        
        for line in lines:
            for direction in ['inbound', 'outbound']:
                
                #get all routes for each line
                routes = tfl_requests.get_routes(line['id'], direction, app_id = app_id, app_key = app_key)
                
                for route in routes['orderedLineRoutes']:
                    
                    #get all stations on route
                    for i in range(len(route['naptanIds']) - 1):
                        
                        #stoppoint information
                        s1 = tfl_requests.get_stoppoint_from_id(route['naptanIds'][i], app_id = app_id, app_key = app_key)
                        s2 = tfl_requests.get_stoppoint_from_id(route['naptanIds'][i+1], app_id = app_id, app_key = app_key)
                        
                        #identifier
                        n1 = route['naptanIds'][i]
                        n2 = route['naptanIds'][i+1]
                        
                        #add to graph
                        G.add_node(n1, pos = (s1['lat'], s1['lon']), 
                                   name = s1['commonName'])
                        G.add_node(n2, pos = (s2['lat'], s2['lon']),
                                   name = s2['commonName'])
                        G.add_edge(n1, n2, dist = dist(s1, s2).km, 
                                   line = line['id'])
                        
        print(str(mode) + ' end in %s'%(time.time()-start))
    
    return G

G = get_tfl_network_from_api(modes, app_id, app_key)
Nodes_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')

"MERGE STATIONS WITH 2+ NAMES"
dupe = []
for row_id, row in Nodes_df.iterrows():
    if len(Nodes_df[Nodes_df['pos'] == row['pos']]) > 1:
        d1 = [row['name']]
        for row_id_, row_ in Nodes_df[Nodes_df['pos'] == row['pos']].iterrows():
            d1.extend([row_id_])
        dupe.append(d1)
#remove duplicates from duplicate list
dupe.sort()
dupe = list(k for k,_ in itertools.groupby(dupe))

for dupe_nodes in dupe:
    for i in range(2, len(dupe_nodes)):
        try:
            G = nx.contracted_nodes(G, dupe_nodes[1], dupe_nodes[i], self_loops = False)
        except KeyError as e:
            print('KeyError on: %s'%e)
            

"APPEND NLCs TO STATIONS"
#create data [stn naptan --> stn nlc] for graph
nap_to_nlc = dict()
for row_id, row in nap_nlc.iterrows():
    nlc = str(row['nlc_id'])
    if nlc != np.nan and not nlc.isalpha():
        nlc = int(nlc)
    nap_to_nlc[row['naptan_id']] = row['nlc_id']
    
#apply nlc attribute to nodes
nx.set_node_attributes(G, nap_to_nlc, name='nlc')

#get dictionary for reverse lookup
stn_to_nlc = nx.get_node_attributes(G,'nlc')

#check if all stations have NLC code
if len(list(G.nodes())) != len(list(stn_to_nlc.keys())):
    print('WARNING: Some stations without NLC idenfiers')
    
"GET DIRECTED LINK LOAD"
#nlc from link load data to stn in graph
nlc_to_stn = {int(v):k for k,v in stn_to_nlc.items()}
#initialize arrays
ll_am = dict()
ll_pm = dict()
ll_mid = dict()
ll_tot = dict()
err = []
#parse and apply data
for row_id, row in line_loads.iterrows():
    try:
        ll_am[(nlc_to_stn[row['From_NLC']], nlc_to_stn[row['To_NLC']])] = {'ll_am' : row['AM']}
        ll_mid[(nlc_to_stn[row['From_NLC']], nlc_to_stn[row['To_NLC']])] = {'ll_mid' : row['Midday']}
        ll_pm[(nlc_to_stn[row['From_NLC']], nlc_to_stn[row['To_NLC']])] = {'ll_pm' : row['PM']}
        ll_tot[(nlc_to_stn[row['From_NLC']], nlc_to_stn[row['To_NLC']])] = {'ll_tot' : row['Total']}
    #if NLC / Station in Link Load data missing in graph
    except KeyError as e:
        err.append(e.args[0])

nx.set_edge_attributes(G, ll_am)
nx.set_edge_attributes(G, ll_mid)
nx.set_edge_attributes(G, ll_pm)
nx.set_edge_attributes(G, ll_tot)

"REMOVE ISOLATES AND SELF-LOOPS"
G.remove_nodes_from(nx.isolates(G))
G.remove_edges_from(nx.selfloop_edges(G))

"SAVE GRAPH"
nx.write_gpickle(G, "data/tube_graph_processed.pkl")