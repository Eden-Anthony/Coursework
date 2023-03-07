# -*- coding: utf-8 -*-
# %%
"""
Created on Tue Feb 28 10:57:34 2023

@author: TonyE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats
from math import floor

CID = 1853219

#Set seeds
np.random.seed(CID)
random.seed(CID)


# Read CSV
df = pd.read_csv('ae1220.csv')
    
# 1a) Plot histogram of wavelengths
def plotHist():
    hist = df['Y'].hist(bins = 30, rwidth=0.9)
    hist.set_xlabel('Wavelength [nm]')
    hist.set_ylabel('Frequency')
    hist.set_title('Distribution of Wavelengths')

# 1a) Plot boxplot of wavelengths
def plotBoxPlot():
    boxplotFig, boxplotAxis = plt.subplots()
    boxplot = df.boxplot(ax = boxplotAxis, column = ['Y'])
    boxplot.set_ylabel('Wavelength [nm]')
    boxplot.set_title('Distribution of Wavelengths')

# 1b) Caclulate summary statistics
median = df['Y'].median()
mean = df['Y'].mean()
trimmedMean = stats.trim_mean(df, 0.1)
std = df['Y'].std()
Q1 = df['Y'].quantile(0.25)
Q3 = df['Y'].quantile(0.75)
IQR = Q3 - Q1

# 1c) Plot scatter plot of wavelength vs time

scatterFig, scatterAxis = plt.subplots()
def plotScatter():
    df.plot.scatter(x = 'X', y = 'Y', ax = scatterAxis)
    scatterAxis.set_xlabel('Time')
    scatterAxis.set_ylabel('Wavelength')
    scatterAxis.set_title('Wavelength vs Time')

# 2a) Linear regression estimate
Beta, alpha, r, p, std_err = stats.linregress(df['X'], df['Y'])
linEst = df['X']*Beta + alpha
def plotScatterLinear():
    scatterAxis.plot(df['X'], linEst, label = 'Linear Estimate')

# 2b) Quadratic regression estimate

quadCoeffs = np.polyfit(df['X'], df['Y'], 2)
quadEst = quadCoeffs[0]*df['X']**2 + quadCoeffs[1]*df['X'] + quadCoeffs[2]
def plotScatterQuad():
    scatterAxis.plot(df['X'], quadEst, label = 'Quadratic Estimate')
    scatterAxis.legend()
# %%
# 2c) Higher order estimates, standardising X

df['X_Standardised'] = (df['X']-df['X'].mean())/df['X'].std()
scatterFig2, scatterAxis2 = plt.subplots()
nModels = 5
startingOrder =23
modelIncrements = 1
models = pd.DataFrame({'Order': np.arange(startingOrder,startingOrder+nModels*modelIncrements,modelIncrements), 
                       'Model': [None] * nModels,
                       'MaxLL': np.zeros(nModels),
                       'AIC': np.zeros(nModels),
                       'Residuals': [None] * nModels})


def getKEstFunc(k):
    kCoeffs = np.polyfit(df['X_Standardised'], df['Y'], k)
    return (lambda x: sum([kCoeffs[j]*x**(k-j) for j in range(k+1)]))
    
def plotScatterHighOrder():
    n = 0
    modelList = []
    for k in range (startingOrder,startingOrder + nModels*modelIncrements,modelIncrements):
        kCoeffs = np.polyfit(df['X_Standardised'], df['Y'], k)
        kEstFunc = getKEstFunc(k)
        modelList.append(kEstFunc)
        scatterAxis2.plot(df['X_Standardised'], kEstFunc(df['X_Standardised']), c = 'red')
        n+=1
    models['Model'] = modelList.copy()

    df.plot.scatter(x = 'X_Standardised', y = 'Y', ax = scatterAxis2)
    scatterAxis2.set_xlabel('Standardised Time')
    scatterAxis2.set_ylabel('Wavelength')
    scatterAxis2.set_title('Wavelength vs Time')
    #scatterAxis2.legend()

plotScatterHighOrder()

def maxll (modelFunc):
    n = df.shape[0]
    pi = np.pi
    errors = df['Y']-modelFunc(df['X_Standardised'])
    var = errors.var()
    logLikelihood = -(n/2) *np.log(2*pi*var) - (1/(2*var))*((df['Y']-modelFunc(df['X_Standardised']))**2).sum()
    return (logLikelihood)

# Populate max likelihood column in DF
for i in range (len(models)):
    models.at[i,'MaxLL'] = maxll(models.loc[i]['Model']) 
# %% 
# 2d) Calculate AIC values

# AIC is based on the principle of parsimony, which states that a model with fewer parameters
# is generally preferable to a model with more parameters, 
# all other things being equal. AIC penalizes models with more parameters, 
# to avoid overfitting and improve the generalizability of the model to new data.
def calcAIC():
    for i, model in models.iterrows():
        q = model['Order'] + 1 + 1 # q is number of estimated parameters, we estimate order + 1 coeffs, and the variance
        models.at[i,'AIC'] = 2*q - 2*model['MaxLL']
        
calcAIC()

# 2e) Identify lowest AIC and plot a qq  plot of its residuals to check for normality
# Do a qq plot instead ********
def calcResiduals ():
    residualsFig, residualsAxis = plt.subplots()
    bestFitIndex = models['AIC'].idxmin() # find index of lowest AIC
    bestFit = models.loc[bestFitIndex]
    residuals = df['Y']-bestFit['Model'](df['X_Standardised'])
    models.at[bestFitIndex,'Residuals'] = residuals
    order=bestFit['Order']
    
    #Histogram to check that residuals approximately follow a normal distribution
    stats.probplot(residuals, dist="norm", plot = residualsAxis)
    residualsAxis.set_xlabel('Theoretical Quantiles')
    residualsAxis.set_ylabel('Observed Values')
    residualsAxis.set_title(f'Distribution of Residuals at k = {order}')
    plt.legend()
    
    return models.loc[bestFitIndex]

bestFit = calcResiduals()
# %%
# 2f) Sample every 10 points

sampleDF = df[9::10]
sampleDF = sampleDF.reset_index()
k = bestFit['Order']
predictions = bestFit['Model'](sampleDF['X_Standardised'])
sampleFig, sampleAxis = plt.subplots()
sampleAxis.scatter(sampleDF['X_Standardised'], sampleDF['Y'], s = 5)
sampleAxis.set_xlabel('Standardised Time')
sampleAxis.set_ylabel('Wavelength')
sampleAxis.set_title('Sample Scatter Plot')
sampleAxis.plot(sampleDF['X_Standardised'], predictions, label = f'Model for k = {k}', c = 'red')
sampleAxis.legend()
    

# 3a) Bootstrapping

N = [1,10,30,100] # number of bootstrapped samples
sampleSize = sampleDF.shape[0]
confFig, confAx = plt.subplots(2,2, sharey = True)
sampleResiduals = sampleDF['Y'] - bestFit['Model'](sampleDF['X_Standardised'])
y_pred = bestFit['Model'](sampleDF['X_Standardised'])

for j,n in zip (range(0,4),N): # iterate through plots, current plot is plot_j
    samples = np.zeros((n, sampleSize)) # each row is a sample of 85 
    for i in range (0,n): # generate all n bootstrap samples
        bootstrappedResiduals = np.random.choice(sampleResiduals, size = sampleSize, replace = True)
        sampleY = y_pred + bootstrappedResiduals
        k = bestFit['Order']
        kCoeffs = np.polyfit(sampleDF['X_Standardised'], sampleY, k)
        kEstFunc = lambda x: sum([kCoeffs[h]*x**(k-h) for h in range(k+1)])
        samples[i] = kEstFunc(sampleDF['X_Standardised'])

    # Calculate Confidence Intervals
    quantiles = np.zeros((sampleSize, 2))
    for i in range (0,sampleSize):
        lb = np.quantile(samples[:,i], 0.025)
        ub = np.quantile(samples[:,i], 0.975)
        quantiles[i] = [lb,ub]
    
    row = 1 if j>1 else 0
    col = 0 if (j==0 or j == 2) else 1
    confAx[row,col].scatter(sampleDF['X_Standardised'], sampleDF['Y'], s = 5)
    confAx[row,col].set_xlabel('Standardised Time')
    confAx[row,col].set_ylabel('Wavelength')
    #confAx[row,col].plot(sampleDF['X_Standardised'], predictions, label = f'Model for k = {k}', c = 'red')
    #confAx[row,col].legend()

    # Plot CIs
    confAx[row,col].fill_between(sampleDF['X_Standardised'], quantiles[:,0],quantiles[:,1], alpha = 0.5, label = f'{n} Sample Bootstrapped 95% CI')
    #confAx[row,col].legend()

    #Plot Actual Confidence Band
    ssres = ((sampleDF['Y'] - y_pred) ** 2).sum()
    #dof = n(bestFit['Order']+1)
    n2 = len(sampleDF['Y'])
    SE = np.sqrt(ssres / (n2 - (bestFit['Order']+1)))
    # calculate the 95% confidence intervals
    z_val = stats.norm.ppf(0.975) # for 95% confidence interval
    LB  = y_pred - z_val * SE
    UB = y_pred + z_val * SE
    confAx[row,col].fill_between(sampleDF['X_Standardised'], LB,UB, alpha = 0.5, label = 'True 95% CI')
    #confAx[row,col].legend()
plt.show()

# %%
