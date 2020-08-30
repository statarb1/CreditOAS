# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# MATLAB command to write the txt files
#creditData_excess_ret = fts2ascii('creditData_excess_ret.csv', creditData.excessRet);
%matplotlib qt
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from zscore import zscore
from scipy.stats.stats import pearsonr

sys.path.append(r"\\client\U$\Systematic\mcode\matFiles")
sys.path.append(r"\\client\U$\AssetAllocation\Pranav\CPD\data_science_blogpost")

endDt = pd.to_datetime('20200612', format='%Y%m%d')
stDt = pd.to_datetime('20000101', format='%Y%m%d')

# Read in Data
df_excess_ret = pd.read_csv(r'\\client\U$\Systematic\mcode\matFiles\creditData_excess_ret.txt', sep = ' ')
df_excess_ret.columns = df_excess_ret.columns.str.replace('\t' , '')
df_excess_ret['dates'] = pd.to_datetime(df_excess_ret['dates'])
df_excess_ret = df_excess_ret.loc[(df_excess_ret['dates'] > stDt) & (df_excess_ret['dates']<= endDt)]
df_excess_ret.set_index('dates', inplace=True)
df_excess_ret = df_excess_ret.astype('float64')
df_excess_ret.info()

df_full_OAS = pd.read_csv(r'\\client\U$\Systematic\mcode\matFiles\creditData_full_OAS.txt', sep = ' ', skiprows=range(1))
df_full_OAS.columns = df_full_OAS.columns.str.replace("\t","")
df_full_OAS['dates'] = pd.to_datetime(df_full_OAS['dates'])
df_full_OAS = df_full_OAS.loc[(df_full_OAS['dates'] > stDt) & (df_full_OAS['dates']<= endDt)]
df_full_OAS.set_index('dates', inplace=True)
df_full_OAS = df_full_OAS.astype('float64')
df_full_OAS.info()

df_OAS = pd.read_csv(r'\\client\U$\Systematic\mcode\matFiles\creditData_OAS.txt', sep = '\s+', skiprows=range(1))
df_OAS.columns = df_OAS.columns.str.replace("\t","")
df_OAS['dates'] = pd.to_datetime(df_OAS['dates'])
df_OAS = df_OAS.loc[(df_OAS['dates'] > stDt) & (df_OAS['dates']<= endDt)]
df_OAS.set_index('dates', inplace=True)
df_OAS = df_OAS.astype('float64')
df_OAS.info()

df_full_MV = pd.read_csv(r'\\client\U$\Systematic\mcode\matFiles\creditData_full_MV.txt', sep = ' ', skiprows=range(1))
df_full_MV.columns = df_full_MV.columns.str.replace("\t","")
df_full_MV['dates'] = pd.to_datetime(df_full_MV['dates'])
df_full_MV = df_full_MV.loc[(df_full_MV['dates'] > stDt) & (df_full_MV['dates']<= endDt)]
df_full_MV.set_index('dates', inplace=True)
df_full_MV = df_full_MV.astype('float64')
df_full_MV.info()

colNames = [col for col in df_full_MV.columns if 'IGWD' in col]
WDIGIndicesMV = df_full_MV[colNames]
WDIGIndicesCurrentComposition=WDIGIndicesMV.iloc[-1,:].divide(WDIGIndicesMV.iloc[-1,:].sum(),axis=0)
WDIGIndicesComposition = WDIGIndicesMV.divide(WDIGIndicesMV.sum(axis=1),axis=0)
WDIGIndicesOAS = df_full_OAS[colNames]
WDIGCompositionAdjustedOAS = pd.DataFrame(np.sum(WDIGIndicesOAS*WDIGIndicesCurrentComposition,1), columns = ['IGWDAll'])

colNames = [col for col in df_full_MV.columns if 'IGUSD' in col]
USDIGIndicesMV = df_full_MV[colNames]
USDIGIndicesCurrentComposition=USDIGIndicesMV.iloc[-1,:].divide(USDIGIndicesMV.iloc[-1,:].sum(),axis=0)
USDIGIndicesComposition = USDIGIndicesMV.divide(USDIGIndicesMV.sum(axis=1),axis=0)
USDIGIndicesOAS = df_full_OAS[colNames]
USDIGCompositionAdjustedOAS = pd.DataFrame(np.sum(USDIGIndicesOAS*USDIGIndicesCurrentComposition,1), columns = ['IGUSDAll'])

colNames = [col for col in df_full_MV.columns if 'IGEUR' in col]
EURIGIndicesMV = df_full_MV[colNames]
EURIGIndicesCurrentComposition=EURIGIndicesMV.iloc[-1,:].divide(EURIGIndicesMV.iloc[-1,:].sum(),axis=0)
EURIGIndicesComposition = EURIGIndicesMV.divide(EURIGIndicesMV.sum(axis=1),axis=0)
EURIGIndicesOAS = df_full_OAS[colNames]
EURIGCompositionAdjustedOAS = pd.DataFrame(np.sum(EURIGIndicesOAS*EURIGIndicesCurrentComposition,1), columns = ['IGEURAll'])

colNames = [col for col in df_full_MV.columns if 'IGGBP' in col]
GBPIGIndicesMV = df_full_MV[colNames]
GBPIGIndicesCurrentComposition=GBPIGIndicesMV.iloc[-1,:].divide(GBPIGIndicesMV.iloc[-1,:].sum(),axis=0)
GBPIGIndicesComposition = GBPIGIndicesMV.divide(GBPIGIndicesMV.sum(axis=1),axis=0)
GBPIGIndicesOAS = df_full_OAS[colNames]
GBPIGCompositionAdjustedOAS = pd.DataFrame(np.sum(GBPIGIndicesOAS*GBPIGIndicesCurrentComposition,1), columns = ['IGGBPAll'])

colNames = [col for col in df_full_MV.columns if 'IGShortUSD' in col]
USDIGShortIndicesMV = df_full_MV[colNames]
USDIGShortIndicesCurrentComposition=USDIGShortIndicesMV.iloc[-1,:].divide(USDIGShortIndicesMV.iloc[-1,:].sum(),axis=0)
USDIGShortIndicesComposition = USDIGShortIndicesMV.divide(USDIGShortIndicesMV.sum(axis=1),axis=0)
USDIGShortIndicesOAS = df_full_OAS[colNames]
USDIGShortCompositionAdjustedOAS = pd.DataFrame(np.sum(USDIGShortIndicesOAS*USDIGShortIndicesCurrentComposition,1), columns = ['IGShortUSD'])

colNames = [col for col in df_full_MV.columns if 'IGShortEUR' in col]
EURIGShortIndicesMV = df_full_MV[colNames]
EURIGShortIndicesCurrentComposition=EURIGShortIndicesMV.iloc[-1,:].divide(EURIGShortIndicesMV.iloc[-1,:].sum(),axis=0)
EURIGShortIndicesComposition = EURIGShortIndicesMV.divide(EURIGShortIndicesMV.sum(axis=1),axis=0)
EURIGShortIndicesOAS = df_full_OAS[colNames]
EURIGShortCompositionAdjustedOAS = pd.DataFrame(np.sum(EURIGShortIndicesOAS*EURIGShortIndicesCurrentComposition,1), columns = ['IGShortEUR'])

colNames = [col for col in df_full_MV.columns if 'IGShortGBP' in col]
GBPIGShortIndicesMV = df_full_MV[colNames]
GBPIGShortIndicesCurrentComposition = GBPIGShortIndicesMV.iloc[-1,:].divide(GBPIGShortIndicesMV.iloc[-1,:].sum(),axis=0)
GBPIGShortIndicesComposition = GBPIGShortIndicesMV.divide(GBPIGShortIndicesMV.sum(axis=1),axis=0)
GBPIGShortIndicesOAS = df_full_OAS[colNames]
GBPIGShortCompositionAdjustedOAS = pd.DataFrame(np.sum(GBPIGShortIndicesOAS*GBPIGShortIndicesCurrentComposition,1), columns = ['IGShortGBP'])

colNames = [col for col in df_full_MV.columns if 'HYWD' in col]
WDHYIndicesMV = df_full_MV[colNames]
WDHYIndicesCurrentComposition=WDHYIndicesMV.iloc[-1,:].divide(WDHYIndicesMV.iloc[-1,:].sum(),axis=0)
WDHYIndicesComposition = WDHYIndicesMV.divide(WDHYIndicesMV.sum(axis=1),axis=0)
WDHYIndicesOAS = df_full_OAS[colNames]
WDHYCompositionAdjustedOAS = pd.DataFrame(np.sum(WDHYIndicesOAS*WDHYIndicesCurrentComposition,1), columns = ['HYWDAll'])

colNames = [col for col in df_full_MV.columns if 'HYUSD' in col]
USDHYIndicesMV = df_full_MV[colNames]
USDHYIndicesCurrentComposition=USDHYIndicesMV.iloc[-1,:].divide(USDHYIndicesMV.iloc[-1,:].sum(),axis=0)
USDHYIndicesComposition = USDHYIndicesMV.divide(WDHYIndicesMV.sum(axis=1),axis=0)
USDHYIndicesOAS = df_full_OAS[colNames]
USDHYCompositionAdjustedOAS = pd.DataFrame(np.sum(USDHYIndicesOAS*USDHYIndicesCurrentComposition,1), columns = ['HYUSDAll'])

colNames = [col for col in df_full_MV.columns if 'HYEUR' in col]
EURHYIndicesMV = df_full_MV[colNames]
EURHYIndicesCurrentComposition=EURHYIndicesMV.iloc[-1,:].divide(EURHYIndicesMV.iloc[-1,:].sum(),axis=0)
EURHYIndicesComposition = EURHYIndicesMV.divide(EURHYIndicesMV.sum(axis=1),axis=0)
EURHYIndicesOAS = df_full_OAS[colNames]
EURHYCompositionAdjustedOAS = pd.DataFrame(np.sum(EURHYIndicesOAS*EURHYIndicesCurrentComposition,1), columns = ['HYEURAll'])

OASCompAdjOAS = pd.concat([WDIGCompositionAdjustedOAS, USDIGCompositionAdjustedOAS,
                 GBPIGCompositionAdjustedOAS, EURIGCompositionAdjustedOAS, 
                 USDIGShortCompositionAdjustedOAS, GBPIGShortCompositionAdjustedOAS, EURIGShortCompositionAdjustedOAS,
                 WDHYCompositionAdjustedOAS, USDHYCompositionAdjustedOAS, EURHYCompositionAdjustedOAS], axis = 1, sort = False)

# calculate periodic returns
df_excess_ret_index = df_excess_ret.copy()
df_excess_ret_index += 1
df_excess_ret_index = df_excess_ret_index.cumprod(axis = 'index', skipna = True)

plt.plot(df_excess_ret_index)
plt.legend(df_excess_ret_index.columns)

asset_class = 'HYWDAll'
asset_class_index = df_excess_ret_index[asset_class]
OAS = OASCompAdjOAS[asset_class]
score = zscore(OAS,1,1)

#plt.plot(score)
#plt.plot(OASCompAdjOAS[asset_class]*0.02)
#plt.legend('Score','OAS')

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

corr = np.zeros(cum_ret.shape[1])

for lag in range(cum_ret.shape[1]):
    corr[lag] = OAS.corr(cum_ret.iloc[:,lag], method = 'pearson')

plt.plot(list(range(1,79)),corr)
plt.title('Correlation of IG OAS With Next Period Returns')
plt.xlabel('Period Length (weeks)')
plt.ylabel('Correlation')
plt.show()    

stDt = ['1999-01-01','2002-01-01','2004-01-01','2006-01-01','2008-01-01'\
    ,'2010-01-01','2012-01-01','2014-01-01','2016-01-01','2018-01-01']
endDt = ['2002-01-01','2004-01-01','2006-01-01','2008-01-01'\
    ,'2010-01-01','2012-01-01','2014-01-01','2016-01-01','2018-01-01','2020-04-10']

corr = np.zeros([len(stDt),cum_ret.shape[1]])

for i in range(len(stDt)):
    cum_ret_tmp = cum_ret.loc[stDt[i]:endDt[i]]
    score_tmp = score.loc[stDt[i]:endDt[i]]
    OAS_tmp = OAS.loc[stDt[i]:endDt[i]]
    
    for lag in range(cum_ret.shape[1]):
        corr[i, lag] = OAS_tmp.corr(cum_ret_tmp.iloc[:,lag], method = 'pearson')

    plt.plot(list(range(1,79)), corr[i,:], label = str(stDt[i])+ '-'+ str(endDt[i]))
    plt.title('Correlation of HY OAS With Next Period Returns')
    plt.xlabel('Period Length (weeks)')
    plt.ylabel('Correlation')
plt.legend()
plt.show()    
    
################### work from here onwards

# Graphs
# OAS
plt.plot(df_OAS['IGWDAll'])
plt.title('Investment Grade - Option Adjusted Spread')
plt.grid()
plt.ylabel('Basis Points (0.01%)')
plt.ylim([0, 500])
plt.axhline(np.mean(df_OAS['IGWDAll']),linestyle="dashed",linewidth= 1.5)

plt.plot(df_OAS['HYWDAll'])
plt.title('Global High Yield - Option Adjusted Spread')
plt.grid()
plt.ylabel('Basis Points (0.01%)')
plt.ylim([0, 2000])
plt.axhline(np.mean(df_OAS['HYWDAll']),linestyle="dashed",linewidth= 1.5)

# Composition
plt.plot(WDIGIndicesComposition*100)
plt.title('Global IG Split by Credit Rating')
#plt.legend(WDIGIndicesCurrentComposition.columns)
plt.legend(('AAA','AA','A','BBB'))
plt.ylabel('Percentage of Index at Each Rating (%)')
plt.show()

plt.plot(WDHYIndicesComposition*100)
plt.title('Global HY Split by Credit Rating')
#plt.legend(WDHYIndicesCurrentComposition.columns)
plt.legend(('BB','B','CCC'))
plt.ylabel('Percentage of Index at Each Rating (%)')
plt.show()

# Composition Adjusted OAS
plt.plot(WDIGCompositionAdjustedOAS)
plt.plot(df_OAS['IGWDAll'])
plt.ylabel('Basis Points (0.01%)')
plt.legend(('Composition Adjusted OAS','Index OAS (Non Composition Adjusted)'))
plt.title('Global Investment Grade')

plt.plot(WDHYCompositionAdjustedOAS)
plt.plot(df_OAS['HYWDAll'])
plt.ylabel('Basis Points (0.01%)')
plt.legend(('Composition Adjusted OAS','Index OAS (Non Composition Adjusted)'))
plt.title('Global High Yield')

# OAS vs Excess Returns Scatter Plots
plt.scatter(OAS, cum_ret[12]*100,facecolors='none', edgecolors='b')
plt.title('IG - Starting OAS and Next 13 Week Returns')
#plt.legend(WDHYIndicesCurrentComposition.columns)
plt.ylabel('Period Excess Returns (%)')
plt.xlabel('Starting OAS (bps)')
plt.show()

plt.scatter(OAS, cum_ret[25]*100,facecolors='none', edgecolors='b')
plt.title('IG - Starting OAS and Next 26 Week Returns')
#plt.legend(WDHYIndicesCurrentComposition.columns)
plt.ylabel('Period Excess Returns (%)')
plt.xlabel('Starting OAS (bps)')
plt.show()

plt.scatter(OAS, cum_ret[51]*100,facecolors='none', edgecolors='b')
plt.title('IG - Starting OAS and Next 52 Week Returns')
#plt.legend(WDHYIndicesCurrentComposition.columns)
plt.ylabel('Period Excess Returns (%)')
plt.xlabel('Starting OAS (bps)')
plt.show()







