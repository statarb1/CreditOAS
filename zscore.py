# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 20:45:52 2020

@author: aggarp
"""

def zscore(x, window, expanding):
    if expanding == 0:
        r = x.rolling(window=window)
    else:
        r = x.expanding(1)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z
