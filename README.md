# Spatial Transport Analysis for TFL
Models the TFL (Transport for London) transportation network

## Sources
1. TFL’s Unified API ([api.tfl.gov.uk](api.tfl.gov.uk))
2. TFL’s NUMBAT Dataset (http://crowding.data.tfl.gov.uk)
3. External information (google, https://www.whatdotheyknow.com/, etc.)
4. TFL Oyster card data

## Models
1. Origin-Destination Assignment
2. Random Walk
3. Psychological/Agent choice

### OD ASSIGNMENT 
Model Algorithm:
1. Agents are randomly placed on the network
2. Each is assigned a random destination (described below)
3. Agents move from Origin to Destination via shortest path
4. Repeat for N steps
   
Data required:
- Station locations and connections
- Origin-Destination travel matrix Outputs:
- Trip times (km spent travelling)
- Trip lengths (number of stops)
- Edge counts (number of passengers who passed along a given edge)
  
Outputs:
- Trip times (km spent travelling)
- Trip lengths (number of stops)
- Edge counts (number of passengers who passed along a given edge)

### THE RANDOM WALK
Full details of this model can be found in "documents/MEM.pdf" 

Model Algorithm:
1. One agent is placed randomly on the network
2. At each timestep, the agent randomly moves to another station.
3. This process is repeated N times for M steps
   
Data required:
- Station locations and connections
- (Optional) Station-to-station flow data
Outputs:

- Edge counts
- Mean first passage times (MFPTs)

### THE PSYCHOLOGICAL MODEL
Full details of this model can be found in "documents/PM.pdf" 

Model Algorithm
1. Agents are placed randomly around the network
2. At each station, agents can perform three actions:
(a) Exit the station (and end the journey)
(b) Swap trains (e.g. Picadilly -> Central @ Holborn) (c) Continue to the next stop
3. When the agent exits, this records their journey and places them randomly on the map Repeat movement for N steps

This model has since been discontinued. It uses too much data and had wildly inaccurate results. I do think it is possible to have this model work if the dynamics are streamlined, therefore, I will say the mistakes I ran into and the user may attempt to recreate this model if (s)he wishes.

## Data Processing
There are two files used for data processing and visualization (outside of the helper files for the various models and creation of the graph described above). These are "print_graph.py" and "TFL_compare.py". We will start with the latter.

### TFL_COMPARE.PY
This file processes the outputs from the models described above, namely, travel time, travel length, and edge counts, and compares it with the actual TFL data. First, the program loads in all of the data from the simulation results folders. Then it loads data from the TFL oyster card sample. The data is compared first by making histograms and density plots of the travel time and travel length data. The travel times for simulations are obtained by taking the total distance (in km) and dividing by the average tube speed of 0.83 km/min (50 km/hr).

### PRINT_GRAPH.PY
This file is used for all of the graph visualizations and spatial density plots. More than anything, this is just a fancy plotting program with easily modifiable parameters, allowing one to plot networkX graphs with color bars, colored nodes, different line sizes, with a background map, etc. etc. The documentation can be found within the code.

Any and All
Questions should be directed to jake.delveaux@icloud.com
