# -*- coding: utf-8 -*-
def zscore(x, window, expanding):

    """
    @author: aggarp
    
    Summary: The function calculates a z score (normalisation) of the input data, 
    either using a rolling or an expanding window
    
    Inputs: 
        x - this is the input data to be normalised
        window - this is the window length to calculate rolling z scores
        expandingFlag - this is a boolean flag and takes a 0 or 1 value. If 
        expandingFlag is set to 1, then an expanding window z score is calculated, 
        else a rolling window z score is calcuated
    
    Output:
        z - this contains the normalised values of x
        
    """
    
    if expanding == 0:
        r = x.rolling(window=window)
    else:
        r = x.expanding(1)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z
