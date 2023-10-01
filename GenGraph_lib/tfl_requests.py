#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for requests on TFL API

@author: jake
"""

"IMPORTS"
import requests


"FUNCTIONS"
def get_lines(mode, app_id = None, app_key = None, base = "https://api.tfl.gov.uk/"):
    #Add credentials
    if app_id != None and app_key != None:
        cred = "?app_id={0}&app_key={1}".format(app_id, app_key)
    else:
        cred = ''
                
    #Pull from server
    req = requests.get((base + 'Line/Mode/{0}' + cred).format(mode))
    return req.json()

def get_routes(line_id, direction, app_id = None, app_key = None, base = "https://api.tfl.gov.uk/"):
    #Add credentials
    if app_id != None and app_key != None:
        cred = "&app_id={0}&app_key={1}".format(app_id, app_key)
    else:
        cred = ''
                
    #Pull from server
    req = requests.get((base + 'Line/{0}/Route/Sequence/{1}?serviceTypes=Regular&exclueCrowding=true' + cred).format(line_id, direction))
    return req.json()

def get_stoppoint_from_id(id, app_id = None, app_key = None, base = "https://api.tfl.gov.uk/"):
    #Add credentials
    if app_id != None and app_key != None:
        cred = "?app_id={0}&app_key={1}".format(app_id, app_key)
    else:
        cred = ''
        
    #Pull from server
    req = requests.get((base + 'StopPoint/{0}' + cred).format(id))
    return req.json()

def get_stoppoints_in_area(lat, lon, radius, stoptypes, app_id = None, app_key = None, base = "https://api.tfl.gov.uk/"):
    #Add credentials
    if app_id != None and app_key != None:
        cred = "?app_id={0}&app_key={1}".format(app_id, app_key)
    else:
        print('This function requires id and key')
        return None
        
    #Pull from server
    types = str()
    for st in stoptypes:
        types += st + ','
    req = requests.get((base + 'StopPoint' + cred + '&lat={0}&lon={1}&radius={2}&stoptypes={3}').format(lat, lon, radius, types))
    return req.json()