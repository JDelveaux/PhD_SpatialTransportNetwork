#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model:
    1. Agents randomly placed on network
    2. Assigned destination _randomly_ from TFL OD matrix
    3. Agent moves to destination via shortest path
    4. Repeat for N steps

Data Collected:
    1. Travel times
    2. Trip lengths
    3. Edge counts

@author: jake
"""

"IMPORTS"
import numpy as np
import numpy.random as rnd
import pandas as pd
import networkx as nx
import time
import multiprocessing as mp
from file_io import load_obj, save_obj, next_path
from helper import combine_dicts

"READ DATA"
#TFL OD Matrix
OD = pd.read_csv('data/OD_data/2018_OD_Network.csv')
#Network
tube_graph = nx.read_gpickle("data/tube_graph_processed.pkl")
#Table of OD Probabilities (pre-processed)
prob_table = np.load('data/OD_data/prob_table.npy', allow_pickle = True)
#index on probability table to nlc
prob_idx_to_nlc = load_obj('data/OD_data/prob_idx_to_nlc.pkl')

"PARAMETERS"
N_agents = 10000
N_iter   = int(1000/mp.cpu_count()) #per core!

def transport(output, G, prob_table, N_agents, N_iter):
    """
    
    Parameters
    ----------
    output : MP output
        for multiprocessing.
    G : networkX graph
        current tube graph (see gen_graph).
    prob_table : numpy D-array
        probability matrix for OD pairs.
    N_agents : int
        number of agents on network.
    N_iter : int
        number of iterations.

    Returns
    -------
    Array of edge_counts (dict), trip_lengths (list) and trip_size (list).

    """
    
    #extra setup
    print_iter = 1e16 #how often to print (1e16 to skip printing)
    start = time.time()
    
    #generate edge_count dict
    edge_count = dict(G.edges())
    edge_count = dict.fromkeys(edge_count, 0)
    
    #for trip dists (in km) and trip size (# of stations)
    trip_lengths = []
    trip_sizes = []
    
    #for station names, NLCs
    nodes = list(G.nodes())
    stn_to_nlc = nx.get_node_attributes(G, 'nlc')
    nlc_to_stn = {v:k for k,v in stn_to_nlc.items()}
    nlcs = [stn_to_nlc[node] for node in nodes]
    nlc_to_prob_idx = {v:k for k,v in prob_idx_to_nlc.items()}
    
    #place initial agents
    locations = rnd.choice(nodes, size = N_agents, replace=True)
    
    for _iter in range(N_iter):
        
        for agent in range(N_agents):
            
            #get origin nlc
            o_stn = locations[agent]
            o_nlc = int(stn_to_nlc[o_stn])
            o_idx = nlc_to_prob_idx[o_nlc]
            #get destination
            if sum(prob_table[:,o_idx]) != 0:
                d_nlc = rnd.choice(nlcs, p=prob_table[:,o_idx])
            else:
                d_nlc = rnd.choice(nlcs)
            d_stn = nlc_to_stn[d_nlc]
            
            #attempt to find a path
            try:
                path = nx.dijkstra_path(G, o_stn, d_stn, weight = 'dist')
                path_dist = nx.dijkstra_path_length(G, o_stn, d_stn, weight = 'dist')
            except:
                #if no path, move user to random location
                print('no path to dest: ' + str(d_stn) + ' from origin: ' + str(o_stn))
                locations[agent] = rnd.choice(nodes)
                continue
            
            #count edges
            for i in range(len(path)-1):
                edge_count[(path[i], path[i+1])] += 1
            #append length / time
            trip_lengths.append(path_dist)
            trip_sizes.append(len(path))
            
            #update agent location
            locations[agent] = d_stn
            
        if _iter%print_iter == print_iter - 1:
            print('%i: %i iterations in %.3f seconds'%(_iter + 1, print_iter, time.time() - start))
            start = time.time()
        
    output.put([edge_count, trip_lengths, trip_sizes])

"MULTIPROCESSING and RUN"
start_run = time.time()
# parameters
numCores = mp.cpu_count()

# initialize a output queue
coreOutput = mp.Queue()
out = []

# initialize the processes for each core
process = [mp.Process(target = transport,
                      args = (coreOutput, tube_graph, prob_table, N_agents, N_iter))
           for coreNbr in range(numCores)]

for p in process:
    print('starting')
    p.start()
    print('started')
for p in process:
    print('in')
    out.append(coreOutput.get())
    print('out')
for p in process:
    print('joining')
    p.join()
    print('joined')

end_run = time.time()
print('%i Agents for %i Iterations in %.3f seconds'%(N_agents,N_iter*mp.cpu_count(),end_run-start_run))
    
"ACCUMULATE RESULTS"
edge_count = dict()
trip_lengths = []
trip_sizes = []
for core in range(numCores):
    edge_count = combine_dicts(edge_count, out[core][0])
    trip_lengths.extend(out[core][1])
    trip_sizes.extend(out[core][2])
print('Results accumulated')

"SAVE RESULTS"
save_obj(next_path('data/OD_data/sim_results/edge_count-%s.pkl'), edge_count)
save_obj(next_path('data/OD_data/sim_results/trip_lengths-%s.pkl'), trip_lengths)
save_obj(next_path('data/OD_data/sim_results/trip_sizes-%s.pkl'), trip_sizes)
print('Results saved')