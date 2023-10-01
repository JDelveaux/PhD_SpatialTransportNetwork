#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agents for A1 model
The agents can perform three actions once assigned
a station and line:
    1. Continue to next station
    2. Swap lines
    3. Exit

Once the agents exit, their journey is saved
and they appear as new agents at a new starting
location.

The agents move based on the weight of a particular
edge on the graph. E.g. distance or link load.

@author: jake
"""

import numpy.random as rnd

class Person():
    
    #time in minutes to swap lines
    swap_time = 5.0
    
    def __init__(self):
        #initialize agent with zero travel time
        self.travel_time = 0.0
        self.locations   = []
        self.journeys    = []
        #pre-allocate exit & swap probability
        self.exit_prob   = rnd.rand()
        self.swap_prob   = rnd.rand()
        
    def end_trip(self, entrance_popularity):
        #first save the journey
        self.journeys.append([self.locations, self.travel_time])
        #then enter a new station
        self.enter_station()
        
    def enter_station(self):
        #reset travel time
        self.travel_time = 0.0
        #randomly select station
        self.locations = ['insert choice function here']
    
    def exit_station(self):
        #test whether to exit
        test_probability = 0.0
        return test_probability > self.exit_prob
    
    def swap_train(self):
        #pick new line
        pass
        #add time penalty
        self.travel_time += self.swap_time
        #and next location
        self.locations.append()
        pass
    
    def next_stop(self):
        #append the next stop and add the travel time
        self.locations.append()
        self.travel_time += 0.0
    
    def run_process(self, N_iter):
        
        #set agent's initial locations
        self.enter_station()
        
        for _ in range(N_iter):
            
            #check if exiting
            if self.exit_station():
                self.end_trip()
            
            #otherwise
            else:
                
                #check if we swap
                stn_swap_chance = 0.0
                if stn_swap_chance > self.swap_prob():
                    self.swap_train()
                    
                #otherwise, move to next station on line
                else:
                    self.next_stop()