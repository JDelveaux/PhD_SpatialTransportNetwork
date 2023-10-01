#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compares trip sizes / trip lengths / edge counts
from a model to the TFL oystercard data

@author: jake
"""

"IMPORTS"
from print_graph import density
import matplotlib.pyplot as plt
import pandas as pd
from file_io import load_obj
import glob
from helper import combine_dicts
import seaborn as sns
import networkx as nx

"FIGURE PARAMETERS"
plt.rcParams["figure.figsize"] = [16,9]
plt.rcParams["figure.dpi"] = 100

"LOAD DATA"
#which model to load from
model_name = 'RW'

#paths for model simulation results
lengths_paths = 'data/{0}_data/sim_results/trip_lengths-*.pkl'.format(model_name)
sizes_paths   = 'data/{0}_data/sim_results/trip_sizes-*.pkl'.format(model_name)
ec_paths      = 'data/{0}_data/sim_results/edge_count-*.pkl'.format(model_name)

#load the lengths (dist) data
sim_lengths = []
for f in glob.glob(lengths_paths):
    tmp = load_obj(f)
    sim_lengths.extend(tmp)

#load the sizes (# stops) data
sim_sizes = []
for f in glob.glob(sizes_paths):
    tmp = load_obj(f)
    sim_sizes.extend(tmp)
    
#load the edge count data
sim_ec = dict()
for f in glob.glob(ec_paths):
    tmp = load_obj(f)
    sim_ec = combine_dicts(sim_ec, tmp)

#load the TFL data
tfl_lengths = load_obj('data/Oyster_data/output/TFL-trip_lengths.pkl')
tfl_sizes   = load_obj('data/Oyster_data/output/TFL-trip_sizes.pkl')
tfl_ec      = load_obj('data/Oyster_data/output/TFL-edge_count.pkl')

#load the tube graph
tube_graph = nx.read_gpickle('data/tube_graph_processed.pkl')

"PLOT LENGTH / SIZE HISTOGRAMS"
#histogram parameters
N_bins = 100
normalize = True

#assume average transport speed of 50km/hr
speed_hr = 50.0
speed_min = speed_hr/60.0

#turn lengths into minutes
sim_lengths[:] = [x/speed_min for x in sim_lengths]
tfl_lengths[:] = [x/speed_min for x in tfl_lengths]

#first plot the lengths
plt.hist([sim_lengths, tfl_lengths], density = normalize, bins = N_bins,
         label = [model_name + ' model', 'Oyster data'])
sns.kdeplot(sim_lengths, color = 'blue')
sns.kdeplot(tfl_lengths, color = 'orange')
plt.title("Distribution of Travel Time / Journey")
plt.legend(loc=1)
plt.show()

#then plot the trip sizes
plt.hist([sim_sizes, tfl_sizes], density = normalize, bins = N_bins,
         label = [model_name + ' model', 'Oyster data'])
plt.title("Distribution of Number of Stops / Journey")
plt.legend(loc=1)
plt.show()

"PLOT SPATIAL DISTRIBUTIONS"
density(tube_graph, sim_ec, 'Spatial Density, %s'%model_name)
density(tube_graph, tfl_ec, 'Spatial Density, %s'%'TFL Oyster')