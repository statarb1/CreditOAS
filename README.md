# CreditOAS

# Abstract (Summary)
Timing entry into different asset classes is the fundamental problem in tactical asset allocation. Investment grade and high yield bonds form a significant asset class with a particularly strong value anchor which makes timing the asset class a fruitful exercise. 

This article explores using the option adjusted spread (OAS) on credit indices as a timing signal into the asset class and as a predictor for future returns for both IG and HY indices.

Specifically, the three questions I would like to answer are:

1. Have credit indices changed in composition over time
2. Is it possible to get a historical option adjusted spread for investment grade and high yield bond indices if adjust for the change in composition? Does adjusting the indices for changing quality have an impact on the spread?
3. Can we use the option adjusted spread as a predictor for future returns?

# Conclusion (Summary of the Results)

1. The analysis clearly shows that the composition of the IG and HY indices has changed significantly over time. While the IG index quality has worsened (more BBB), the quality of the HY index has improved (more BB's)
2. The analysis clearly shows that if we hold the composition (quality) of the indices as constant, the OAS for both the IG and HY indices would have been different over time. This is especially true for IG, where the current worse credit quality would have led to much higher OAS at the height of the financial crisis in 2008-09.  
3. Finally the analysis shows that there is a clear positive correlation between the OAS and the future excess return of both the IG and HY indices. Indeed the correlation is much stronger than one find generally in financial markets. This leads us to conclude that OAS are indeed good predictors for future excess return. This predictive effect lasts over a fairly long time horizon with a significantly positive correlation at 78 weeks. 

# Getting Started
The OAS_script.py file has all the code required to run the analysis and produce all the graphs 
The OAS_script.ipynb file has the corresponding jupyter notebook

# Prerequisites
Python 3 required. The scripts have been written using Spyder 
The following libraries were used:

sys
Pandas
Numpy 
matplotlib
datetime 
Seaborn 
os
scipy

# Contributing
NA

# Versioning
This is version 1.0

# Authors
•	Pranav Aggawral - Coutts Asset Management

# License
This project is licensed under the MIT License - see the LICENSE.md file for details
