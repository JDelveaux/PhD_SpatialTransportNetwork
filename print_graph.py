#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization for Tube Graph from build_API_graph.py
and with additional data from simulations

@author: jake
"""

"IMPORTS"
import matplotlib.pyplot as plt
import geopandas as gpd
import networkx as nx
import matplotlib as mpl
import math
import numpy as np
import scipy.stats as sps
from mpl_toolkits import axes_grid1


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

"PRINT"
def print_graph(tube_graph, options):
    """
    Prints the graph from build_API_graph.py with additional
    data and visualization options

    Parameters
    ----------
    tube_graph : NetworkX Graph
        Graph of the London Underground/Overground
    options : Array [with_map(bool), save_graph, file_name, draw_param(dict)]
        Print options:
            0 - with_map = draw graph with London background
            1 - save_graph = save graph as pdf to images
            2 - file_name = if saving plot to file
            3 - draw_param:
                node_color
                node_size
                with_labels
                alpha (node transparency)
                width (of edges)
                edge_color (true or false)
                weight_param (which edge att. for weights)
                edge_list (if weight_param == 'ex' or 'external')
                
    Returns
    -------
    None

    """
    
    plt.rcParams["figure.figsize"] = [16,9]
    plt.rcParams["figure.dpi"] = 100
    
    dp = options[3]
    #Create figure and axis
    f, ax = plt.subplots()
    #Check to plot London background
    if options[0] == True:
        fp = 'data/Print_data/LondonMap/London_Ward.shp'
        map_df = gpd.read_file(fp)
        map_df = map_df.to_crs("+init=epsg:4326")
        map_df.plot(ax=ax)
    #Get node positions
    pos = nx.get_node_attributes(tube_graph, 'pos')
    #reverse since nx.draw does lat, long not long, lat
    pos = {k:(v[1],v[0]) for k,v in pos.items()}
    
    
    #Plot graph
    if dp['edge_color'] != True:
        nx.draw(tube_graph, pos=pos, node_color = dp['node_color'],
                node_size = dp['node_size'], with_labels = dp['with_labels'],
                alpha = dp['alpha'], width = dp['width'])
    else:
        if dp['weight_param'] == 'ex' or dp['weight_param'] == 'external':
            edges = list(dp['edge_list'].keys())
            weights = list(dp['edge_list'].values())
        else:
            edges,weights = zip(*nx.get_edge_attributes(tube_graph,dp['weight_param']).items()) 
        
        if dp['remove_unweighted'] == True:
            tube_graph.remove_edges_from(np.setdiff1d(list(tube_graph.edges), list(nx.get_edge_attributes(tube_graph, 'll_am').keys())))
            
        #Colorbar
        vmin = min(weights)
        vmax = max(weights)
        cmap = plt.cm.coolwarm
        #sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, fraction=0.02, pad=0.04)
        
        nx.draw(tube_graph, pos=pos, node_color=dp['node_color'], 
                edgelist=edges, edge_color=weights, 
                width=dp['width'], edge_cmap=cmap, vmin=vmin, 
                vmax = vmax, alpha = dp['alpha'], node_size = dp['node_size'],
                with_labels = dp['with_labels'])
        
    #Print graph
    plt.title(options[2])
        
    if options[1] == True:
        plt.savefig(options[2],bbox_inches="tight")
    else:
        plt.show()
    del f
    
    
"GET SPATIAL DISTRIBUTION"
#Get the spatial distribution for the population density moving through the graph
def midpoint(lat1, lon1, lat2, lon2):
#Input values as degrees
#Convert to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    
    bx = math.cos(lat2) * math.cos(lon2 - lon1)
    by = math.cos(lat2) * math.sin(lon2 - lon1)
    lat3 = math.atan2(math.sin(lat1) + math.sin(lat2), \
           math.sqrt((math.cos(lat1) + bx) * (math.cos(lat1) \
           + bx) + by**2))
    lon3 = lon1 + math.atan2(by, math.cos(lat1) + bx)

    return (round(math.degrees(lat3), 4), round(math.degrees(lon3), 4))

def density(ref_graph, edge_cnt, title):
    
    #Get London Map
    fp = 'data/Print_data/LondonMap/London_Ward.shp'
    map_df = gpd.read_file(fp)
    map_df = map_df.to_crs("+init=epsg:4326")
    
    #Get node positions
    pos = nx.get_node_attributes(ref_graph, 'pos')
    lats = [v[0] for v in pos.values()]
    lons = [v[1] for v in pos.values()]

    #Remove Isolated Nodes & Undirect Graph
    ref_graph = ref_graph.to_undirected()
    ref_graph.remove_nodes_from(list(nx.isolates(ref_graph)))
    
    #Get dict of locations / weights
    points = dict()
    for edge, weight in edge_cnt.items():
        lat1, lon1 = pos[edge[0]]
        lat2, lon2 = pos[edge[1]]
        points[midpoint(lat1, lon1, lat2, lon2)] = weight
    
    #Calculate KDE
    x = [xx for yy, xx in points.keys()] #i.e. longs
    y = [yy for yy, xx in points.keys()] #i.e. lats
    weights = list(points.values())
    
    bnds = map_df.total_bounds
    x_rng = [min(min(lons), bnds[0]), max(max(lons), bnds[2])]
    y_rng = [min(min(lats), bnds[1]), max(max(lats), bnds[3])]
    
    xx, yy = np.mgrid[x_rng[0]:x_rng[1]:100j, y_rng[0]:y_rng[1]:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = sps.gaussian_kde(values, weights = weights)
    f = np.reshape(kernel(positions).T, xx.shape)
    
    #Plot KDE + graph
    plt.figure()
        
    map_df.plot()
    
    #networkx plots (lat, lon) not (lon, lat)
    pos = {k:(v[1],v[0]) for k,v in pos.items()}
    
    nx.draw_networkx(ref_graph, pos=pos, with_labels = False, node_size = 10, alpha = 0.5)
    plt.contour(xx, yy, f, cmap = 'Reds', alpha = 0.75)
    plt.contourf(xx, yy, f, cmap = 'Reds', alpha = 0.5)
    
    #entire graph
    plt.xlim(x_rng[0], x_rng[1])
    plt.ylim(y_rng[0], y_rng[1])
    #or just london box
    plt.xlim(bnds[0], bnds[2])
    plt.ylim(bnds[1], bnds[3])
    
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    
    plt.title(title)
    
    plt.colorbar()


def plot_nodecolor(tube_graph,c,title,top=0):
    """
    Parameters
    ----------
    tube_graph : Networkx Network
        your tube graph.
    c : Dictionary
        a dict of weights (i.e. as returned by nx.betweenness_centrality(graph).
    title: string
        a title for the plot
    top: int
        label nodes with 'top' highest values
    
    Returns
    -------
    None.

    """
    plt.rcParams["figure.figsize"] = [16,9]
    plt.rcParams["figure.dpi"] = 100
    
    f, ax = plt.subplots()
    
    fp = 'data/Print_data/LondonMap/London_Ward.shp'
    map_df = gpd.read_file(fp)
    map_df = map_df.to_crs("+init=epsg:4326")
    map_df.plot(ax=ax, color='grey')
    
    nodelist = tube_graph.nodes()
    node_to_idx = {v:k for k,v in enumerate(nodelist)}
    nodelist = [node_to_idx[node] for node in nodelist]
    pos = nx.get_node_attributes(tube_graph,'pos')
    
    if top > 0:
        
        #get names
        names = nx.get_node_attributes(tube_graph, 'name')
        c_v = c.values()
        c_r = {v:k for k,v in c.items()}
        top_locs = [c_r.get(sorted(c_v)[-i]) for i in range(1,top+1)]
        top_locs_name = [names[c] for c in top_locs]
    
        for i in range(top):
            plt.scatter(pos[top_locs[i]][0], pos[top_locs[i]][1], 
                        label = top_locs_name[i] + ': %.3f'%c[top_locs[i]])
        plt.legend(loc = 'lower left')
            
    ne = nx.draw_networkx_edges(tube_graph, pos = pos, width = 0.2, arrows = False)
    pos = {node_to_idx[k]:v for k,v in pos.items()} #change to index number of node weight
    nc = nx.draw_networkx_nodes(tube_graph, pos, nodelist=nodelist, node_color=list(c.values()),
                                with_labels = False, node_size=10, cmap=plt.cm.Reds)
    
    plt.colorbar(nc)
    plt.axis('off')
    plt.title(title)
    
    #line_up, = plt.plot([1, 2, 3], label='Line 2')
    #line_down, = plt.plot([3, 2, 1], label='Line 1')
    #plt.legend([line_up, line_down], ['Line Up', 'Line Down'])
    
    #plt.legend(handles = list(labels.values()))
    
    plt.show()
    return None