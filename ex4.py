# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:04:06 2021

@author: s202442
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

def g(x, lambda_):
    return lambda_*np.exp(-lambda_*x)    

def estimate_probability(X,a):
    return np.sum(X>a)/len(X)

def estimate_probability_importance(X, a):
    X = X * (X>a)
    div = np.random.normal(a, 1, size=len(X))
    div = div * (div>0.2)+0.2
    theta = X/abs(np.random.normal(a, 1, size=len(X)))
    return(np.mean(theta))

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), ss.sem(a)
    h = se * ss.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h
#%%

n=100
samples = np.random.uniform(0,1,n)
samples10 = np.random.uniform(0,1,[10,n])
X = np.zeros(n)
Y = np.zeros(n)
Z = np.zeros(n)
W = np.zeros(n)

for i in range(n):
    X[i] = np.exp(samples[i])
    Y[i] = (X[i]+(np.exp(1)/X[i]))/2

c = -np.cov(X,Y)[0,1]/np.var(Y)


for i in range(n):
    Z[i] = X[i] + c*(samples[i]-0.5)
    
for i in range(n):
    for j in range(10):
        W[i] += np.exp(j/10 + samples10[j,i]/10)/10
        
print(mean_confidence_interval(X))
print(mean_confidence_interval(Y))
print(mean_confidence_interval(Z))
print(mean_confidence_interval(W))
#%%
lambdaa=0.5
lambdas =[]
thetas = []
for i in range(70):
    lambdaa+=0.05
    lambdas.append(lambdaa)
    x = np.random.exponential(1,10000)
    x = x * (x<1)
    theta = (np.exp(x)/g(x,lambdaa))
    thetas.append(np.mean(theta))
    #print(np.mean(theta))
plt.plot(lambdas, thetas)
plt.axhline(1.718, color='red')
plt.ylim(1.2,2.5)
plt.xlim(1,3)
plt.xlabel("$\lambda$ value")
plt.ylabel("integral approx value")
plt.legend(["$\lambda$ value", "True value"])
print(min(thetas))
#%%
ans = []
for i in range(1000):
    X = np.random.normal(0,1,1000)
    prob = estimate_probability(X, 2)
    prob_importance = estimate_probability_importance(X, 2)
    ans.append(prob_importance)
    #print(prob)
    #print(prob_importance)
plt.hist(ans,1000)