#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent based model for users on a transport network.
The agents can perform three actions after boarding
a specified station & line.
    1. Continue to the next station
    2. Switch lines
    3. Exit
    
This file runs the process

Input:
    - NetworkX graph of transport network

Output:
    - Complete journeys of all agents
        inc. stations visted and order
    
@author: jake
"""

"IMPORTS"
import networkx as nx
import time

"PARAMETERS"
N_people = 100
N_iter   = 100
time_day = 'AM'
#save the results?
save = False

"LOAD DATA"
tube_graph  = nx.read_gpickle("data/tube_graph_processed.pkl")

"DEFINE PROCESS"
def run():
    pass