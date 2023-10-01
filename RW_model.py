#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Random Walk / Master Equation Model

A collection of agents randomly move around the
transportation network. They are randomly moved
by either:
    1. Uniform probability / degree-based
    2. # of (real) passengers / TFL Link Load

Input:
    Time of Day
    Weighted / Unweighted

Output:
    Edge Counts
    MFPT

@author: jake
"""

"IMPORTS"
import pandas as pd
import numpy as np
import numpy.random as rnd
import networkx as nx
import time
from msmtools.analysis import mfpt
import multiprocessing as mp
from file_io import save_obj, next_path
import scipy.linalg as la
from RW_lib import MarkovChainSummary as mcs #steady_state, fmpt, var_fmpt

"LOAD DATA"
tube_graph  = nx.read_gpickle("data/tube_graph_processed.pkl")
line_loads = pd.read_csv('data/RW_data/2018MTT_Link_Load.csv')
stn_to_nlc = nx.get_node_attributes(tube_graph, 'nlc')
#probability to travel from station O to D based on TFL OD matrix
    #note, obtained from OD_lib/gen_prob_table.py
prob_table =  np.load('data/RW_data/prob_table.npy', allow_pickle = True)

"PARAMETERS"
#optional random number for movement
rnd_seed = 100
if rnd_seed != 100:
    rnd.seed(rnd_seed)

#which time to consider for TFL data
time_day = 'tot'
#use TFL Link Load data?
weighted = True
#number of steps (per CPU)
N_steps = int(1e6)
#convergence limit for SS probabilites
conv = 1e-6

"GET THE TRANSITION MATRIX"
#get number of nodes
N = len(tube_graph.nodes())
#assign nodes a number
idx_to_node = {k:v for k,v in enumerate(tube_graph.nodes())}
node_to_idx = {v:k for k,v in idx_to_node.items()}
idx_list = np.arange(N)

#get the adjacency matrix
if weighted:
    A = nx.adjacency_matrix(tube_graph, weight = 'll_%s'%time_day)
else:
    A = nx.adjacency_matrix(tube_graph)
A = A.todense()
A = np.array(A, dtype = np.float64)

#evaluate degree matrix
D = np.diag(np.sum(A, axis=0))

#and transition matrix T = D^(-1)A
T = np.dot(np.linalg.inv(D),A).T
#and normalize
T = T/T.sum(axis=0, keepdims=1)

"RANDOMLY WALK"
#start walking from random node
def walk(output, N_steps, T, idx_list):
    visited = [rnd.choice(idx_list)]
    for k in range(N_steps):
        visited.append(rnd.choice(idx_list,
                                  p=T[:,visited[-1]]/sum(T[:,visited[-1]])))
    output.put(visited)

start_run = time.time()
# parameters
numCores = mp.cpu_count()

# initialize a output queue
coreOutput = mp.Queue()
out = []

# initialize the processes for each core
process = [mp.Process(target = walk,
                      args = (coreOutput, N_steps, T, idx_list))
           for coreNbr in range(numCores)]

for p in process:
    print('starting')
    p.start()
    print('started')
for p in process:
    print('in')
    out.extend(coreOutput.get())
    print('out')
for p in process:
    print('joining')
    p.join()
    print('joined')

end_run = time.time()
print('%i Agents for %i Iterations in %.3f seconds'%(numCores,N_steps,end_run-start_run))


"GET SS VECTOR"
# define the starting probability
ss_prob = rnd.randint(0, 1000, N)
ss_prob = ss_prob/ss_prob.sum()
p_old = np.zeros(N)
p_old[0] = 1
save_p = []

step = 0
while step < N_steps and any(abs(p_old-ss_prob)) > conv:
    
    #multiply prob vector by transmat
    p_old = ss_prob
    ss_prob = T@ss_prob
    #increment step
    step += 1
    
    if step%(int(N_steps/1000.0)) == 0:
        save_p.append(ss_prob)
    
"GET MFPT"
def MFPT_cnt(out, N, visited, node_start, node_end):
    """
    Calculates MFPT from path array
    """
    MT = np.zeros((N,N))
    errors = np.zeros((N,N))
    if node_end > N:
        node_end = N
    for MFPT_NODE in range(node_start, node_end):
        start_ = time.time()
        get_T = dict.fromkeys(np.arange(N), 0) #total time
        get_R = dict.fromkeys(np.arange(N), 0) #total runs
        active = dict.fromkeys(np.arange(N), False) #set all active to False
        for i in range(len(visited)):
            if not active[visited[i]]:
                active[visited[i]] = True #'start point' for starting node
            if visited[i] == MFPT_NODE:
                #if we hit the desired node for MFPT
                #add +1 run for all 'on'/'started' nodes and reset
                for n in range(N):
                    if active[n]:
                        get_R[n] += 1 #add 1 more 'run'
                        active[n] = False #reset
            for n in range(N):
                #for all active nodes, add 1 to time taken to reach 
                if active[n]== True:
                    get_T[n] += 1 #add one more step
        for n in range(N):
            #add results to matrix (MFPT j->i)
            try:
                MT[MFPT_NODE, n] = get_T[n]/get_R[n]
            except ZeroDivisionError:
                get_R[n] = 1
                MT[MFPT_NODE, n] = get_T[n]/get_R[n]
                errors[MFPT_NODE, n] += 1
        end_ = time.time()
        print('Node %i of %i complete (in %.3f seconds)'%(MFPT_NODE, N, end_-start_))
                
    out.put([MT, get_T, get_R])

#run mfpt by multiprocessing
start_MFPT = time.time()
# parameters
numCores = mp.cpu_count()
nodes_per_core = round(N/mp.cpu_count())

# initialize a output queue
coreOutput = mp.Queue()
mfpt_outs = []

# initialize the processes for each core
process = [mp.Process(target =  MFPT_cnt,
                      args = (coreOutput, N, out, coreNbr*nodes_per_core, (coreNbr+1)*nodes_per_core))
           for coreNbr in range(numCores)]

for p in process:
    print('starting')
    p.start()
    print('started')
for p in process:
    print('in')
    mfpt_outs.append(coreOutput.get())
    print('out')
for p in process:
    print('joining')
    p.join()
    print('joined')

end_MFPT = time.time()
print('Counted MFPT in %.3f seconds'%(end_MFPT-start_MFPT))

def MFPT_lin(N, T, ss_prob):
    #Calculates MFPT array from (actual or esimated) transition matrix
    MT = np.zeros((N,N))
    for n in range(N):
        MT[n,:] = mfpt(T, n)
    
    for n in range(N):
        MT[n,n] = 1.0/ss_prob[n]
    return MT

mt_lin = MFPT_lin(N,T, ss_prob)

def MFPT_eig(N, T, ss_prob):
    "From Eqn. 31 Correlation fxns and Kemney Constant"
    fp_eig = np.zeros((N,N))
    w, vl, vr = la.eig(T, left = True, right = True)
    sorted_idx = np.argsort(w)
    for j in range(N):
        for k in range(N):
            fp_eig[j,k] = 1.0/ss_prob[j]
            #skip first eigenvalue of 1
            factor = 1.0
            for l in sorted_idx[1:]:
                factor += (w[l]/(1.0-w[l]))*vr[j,l]*(vl[j,l]-vl[k,l])
            fp_eig[j,k] *= factor
            
    return fp_eig

mt_eig = MFPT_eig(N, T, ss_prob)
            

mfpt_start = time.time()
MT_lin = MFPT_lin(N, T)
print('MFPT in %.3f seconds'%(time.time()-mfpt_start))

"GET WEIGHTED MFPT"
w_fpt = np.zeros((N,N))
for i in range(N):
    for j in range(N):
         w_fpt[i,j] = prob_table[i,j] * mt_lin[i,j]

"GET EDGE COUNT"
edge_count = dict(tube_graph.edges())
edge_count = dict.fromkeys(edge_count, 0)
error = []
for i in range(2, len(out) - 1):
    #note that run to run the edge won't usually exist
    try:
        edge_count[(idx_to_node[out[i]], idx_to_node[out[i+1]])] += 1
    except KeyError as e:
        error.append(e)

"ESTIMATE TRIP SIZE / LENGTH FROM MFPT"
#for trip dists (in km) and trip size (# of stations)
trip_lengths = []
trip_sizes = []

"SAVE RESULTS"
save_obj(next_path('data/RW_data/sim_results/edge_count-%s.pkl'), edge_count)
save_obj(next_path('data/OD_data/sim_results/trip_lengths-%s.pkl'), trip_lengths)
save_obj(next_path('data/OD_data/sim_results/trip_sizes-%s.pkl'), trip_sizes)
print('Results saved')