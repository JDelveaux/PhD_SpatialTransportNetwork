# -*- coding: utf-8 -*-
"""
This function determines the probability to go from
origin 'o' to destination 'd' via the TFL OD matrix.

@author: K1898719
"""

"IMPORTS"
import numpy as np
import pandas as pd
import networkx as nx
import pickle
import time

"READ DATA"
OD = pd.read_csv('../data/OD_data/2018_OD_Network.csv')
G = nx.read_gpickle("../data/tube_graph_processed.pkl")

"GENERATE MOVEMENT PROBABILITY"
#start timer
start = time.time()

#get list of station NLCs present in graph
station_nlcs = [int(x) for x in list(nx.get_node_attributes(G,'nlc').values())]

#index position of all NLCS
prob_idx_to_nlc = {k:v for k,v in enumerate(station_nlcs)}
nlc_to_prob_idx = {v:k for k,v in prob_idx_to_nlc.items()}

#initialize N x N probability table
prob_table = np.zeros((len(station_nlcs), len(station_nlcs)))

#strip OD matrix of unused values
OD = OD[OD.End_NLC.isin(station_nlcs)]
OD = OD[OD.Start_NLC.isin(station_nlcs)]

#for every nlc origin
for nlc in station_nlcs:
    cnt = 0
    i_id = nlc_to_prob_idx[int(nlc)]
    #for all possibile desintations
    for line_id, line in OD.loc[OD['Start_NLC'] == nlc].iterrows():
        o_id = nlc_to_prob_idx[int(line['End_NLC'])]
        #get number of travelers for that desintation
        prob_table[o_id][i_id] = int(line['Total'])
        cnt += int(line['Total'])
    #divide by number of travellers for probabiltiy
    if cnt > 0:
        prob_table[:,i_id] /= cnt
        
"SAVE DATA"

def save_obj(file, obj):
    with open(file, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
#NOTE FORMAT: prob_table[i][j] is the probability to go TO 'i' FROM 'j'
np.save('../data/OD_data/prob_table.npy', prob_table, allow_pickle = True)
save_obj('../data/OD_data/prob_idx_to_nlc.pkl', prob_idx_to_nlc)

print('Finished probablity table in %.3f seconds'%(time.time()-start))