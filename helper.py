#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for general python

Includes:
    myround(x, base) : Round to nearest _
    combine_dicts(dict_a, dict_b) : Combine / add dictionaries

@author: jake
"""

def myround(x, base=5):
    return base * round(x/base)

def combine_dicts(a,b):
    return dict(list(a.items()) + list(b.items()) + 
                [(k, a[k] + b[k]) for k in set(b) & set(a)])