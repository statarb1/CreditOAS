# -*- coding: utf-8 -*-

def returns_analysis(OASCompAdjOAS, df_excess_ret_index, asset_class):

    """
    @author: Pranav
    
    Summary: 
        The function does the correlation analysis on the input series and plots 
        the relevant graphs
    
    Inputs: 
        OASCompAdjOAS - This is the unnormlised independent variable being tested
    
        df_excess_ret_index - this is the return for each asset
    
        asset_class - this is a string that contains the name of the asset class 
    
        that the analysis shall be performed on
    
    Outputs:
        OAS - the value of OASCompAdjOAS, extracted for the particular asset class
    
        score - the normalised value of OASCompAdjOAS, extracted for the particular 
        asset class
    
        cum_ret - This contains the cumulative return of the excess return over the
        specified window lengths (1-78 weeks)
    
        corr_full - Contains the correlation between the score and the cumulative
        return over each period. This calculates the correlations for each series
        over the entire dataset
    
        corr - Contains the correlation between the score and the cumulative
        return over each period. This differs from corr_full as it calculates the 
        correlations for each series for specific periods of time, rather than over
        the whole timeframe
        
    """
    
    import sys
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    import seaborn as sns
    import os
    from scipy.stats.stats import pearsonr
    from zscore import zscore
    
    asset_class_index = df_excess_ret_index[asset_class]
    OAS = OASCompAdjOAS[asset_class]
    score = zscore(OAS,1,1)
    
    # Calculate cumulative returns
    
    period = range(1,79)
    cum_ret = np.zeros([len(asset_class_index),len(period)])
    
    for period_length in period:
        cum_ret[:,period_length-1] = asset_class_index.shift(periods = -period_length)/asset_class_index - 1
        
    cum_ret = pd.DataFrame(cum_ret,columns = period, index = asset_class_index.index)
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    
    idx = score.index.intersection(cum_ret.index)
    cum_ret = cum_ret.loc[idx]
    score = score.loc[idx]
    OAS = OAS.loc[idx]
    
    plt.figure()
    plt.scatter(OAS, cum_ret[12]*100,facecolors='none', edgecolors='b')
    plt.title(asset_class + ' - Starting OAS and Next 13 Week Returns')
    #plt.legend(WDHYIndicesCurrentComposition.columns)
    plt.ylabel('Period Excess Returns (%)')
    plt.xlabel('Starting OAS (bps)')
    plt.show()
    
    plt.figure()
    plt.scatter(OAS, cum_ret[25]*100,facecolors='none', edgecolors='b')
    plt.title(asset_class + ' - Starting OAS and Next 26 Week Returns')
    #plt.legend(WDHYIndicesCurrentComposition.columns)
    plt.ylabel('Period Excess Returns (%)')
    plt.xlabel('Starting OAS (bps)')
    plt.show()
    
    plt.figure()
    plt.scatter(OAS, cum_ret[51]*100,facecolors='none', edgecolors='b')
    plt.title(asset_class + ' - Starting OAS and Next 52 Week Returns')
    #plt.legend(WDHYIndicesCurrentComposition.columns)
    plt.ylabel('Period Excess Returns (%)')
    plt.xlabel('Starting OAS (bps)')
    plt.show()
    
    #Calculate correlations over the entire period
    
    corr_full = np.zeros(cum_ret.shape[1])
    
    for lag in range(cum_ret.shape[1]):
        corr_full[lag] = OAS.corr(cum_ret.iloc[:,lag], method = 'pearson')
    
    plt.figure()
    plt.plot(list(range(1,79)),corr_full)
    plt.title('Correlation of ' + asset_class +' OAS With Next Period Returns')
    plt.xlabel('Period Length (weeks)')
    plt.ylabel('Correlation')
    plt.show()    
    
    # Break down the full time frame into discrete chunks and calculate the 
    # correlations over those periods
    
    stDt = ['1999-01-01','2002-01-01','2004-01-01','2006-01-01','2008-01-01'\
        ,'2010-01-01','2012-01-01','2014-01-01','2016-01-01','2018-01-01']
    endDt = ['2002-01-01','2004-01-01','2006-01-01','2008-01-01'\
        ,'2010-01-01','2012-01-01','2014-01-01','2016-01-01','2018-01-01','2020-04-10']
    
    corr = np.zeros([len(stDt),cum_ret.shape[1]])
    
    plt.figure()
    for i in range(len(stDt)):
        cum_ret_tmp = cum_ret.loc[stDt[i]:endDt[i]]
        score_tmp = score.loc[stDt[i]:endDt[i]]
        OAS_tmp = OAS.loc[stDt[i]:endDt[i]]
        
        for lag in range(cum_ret.shape[1]):
            corr[i, lag] = OAS_tmp.corr(cum_ret_tmp.iloc[:,lag], method = 'pearson')
    
        plt.plot(list(range(1,79)), corr[i,:], label = str(stDt[i])+ '-'+ str(endDt[i]))
        plt.title('Correlation of ' + asset_class + ' OAS With Next Period Returns')
        plt.xlabel('Period Length (weeks)')
        plt.ylabel('Correlation')
    plt.legend()
    plt.show()
    
    return OAS, score, cum_ret, corr_full, corr