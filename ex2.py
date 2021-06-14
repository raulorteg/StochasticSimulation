# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 10:12:55 2021

@author: s202442
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

def geometric_distribution(p, size):
    U = np.random.uniform(0,1,size)
    return (np.log(U)/np.log(1-p))+1

def chi_square_test(n_observed, n_expected, num_classes):
	"""
	n_observed:
	n_expected:
	num_classes:
	"""
	temp = np.power((np.array(n_expected) - np.array(n_observed)),2)
	return np.sum(np.divide(temp, n_expected))


def direct_method(p, N, class_num):
    samples_U = np.random.uniform(0, 1, N)
    hist, _ = np.histogram(samples_U)
    #hist = hist/N
    cdf_dist = np.cumsum(p)
    new_dist = np.zeros(class_num)
    for k in range(class_num):
        for i in range(N):
            if k==0:
                if samples_U[i]<=cdf_dist[k]:
                    new_dist[k]+=1
            elif k==class_num:
                if samples_U[i]>cdf_dist[k]:
                    new_dist[k]+=1
            elif cdf_dist[k-1]<samples_U[i]<=cdf_dist[k]:
                new_dist[k]+=1
    return new_dist/N
                

def rejection_method(p, N, class_num, c):
    #np.random.seed(10)
    samples_U = np.random.uniform(0, 1, N)
    samples_U2 = np.random.uniform(0, 1, N)    
    hist, _ = np.histogram(samples_U)
    #hist = hist/N
    cdf_dist = np.cumsum(p)
    new_dist = np.zeros(class_num)
    for k in range(class_num):
        for i in range(N):
            if samples_U2[i]<p[k]/c:
                new_dist[k]+=1
    return new_dist/N

def FL(p):
    p = np.array(p)
    n = len(p) 
    L = np.arange(n) 
    F = n*p
    print(type(F))
    G = L[F>=1] 
    S = (L[F<=1]).tolist()
    while len(S) != 0:
        i = G[0]
        j = S[0]
        L[j] = i
        F[i] = F[i] - (1 - F[j])
        if F[i] < 1:
            G = G[1:]
            S.append(i)
        S = S[1:]
    return F,L

def Alias_method(p, size):  
    F,L = FL(p)
    obs = np.zeros(size) 
    for i in range(size):
        I = 1 + int(len(F)*np.random.uniform(size=1)) 
        if(np.random.uniform(size=1)<=F[I-1]):
            obs[i] = (I-1)
        else: 
            obs[i] = (L[I-1])
    return obs

def expotential(N, lambda_):
    samples = np.random.uniform(0,1,N)
    X = -np.log(samples)/lambda_
    return X

def normal_generation(N):
    samples_U = np.random.uniform(0, 1, N)
    samples_U2 = np.random.uniform(0, 1, N)
    Z1, Z2 = np.sqrt(-2*np.log(samples_U)) * np.array([np.cos(2*np.pi*samples_U2),
                                                       np.sin(2*np.pi*samples_U2)])
    return Z1, Z2

def pareto(N, beta, k):
    samples_U = np.random.uniform(0, 1, N)
    X = beta*(samples_U**(-1/k))
    return X
    
#%%
for i in [0.1, 0.3, 0.5, 0.7, 0.9]:
    geo = geometric_distribution(i, 10000)
    plt.hist(geo, 50, alpha=0.6)
    plt.xlim(0,50)
plt.legend(['p=0.1', 'p=0.3', 'p=0.5', 'p=0.7', 'p=0.9'])
plt.xlabel("value")
plt.ylabel("number of samples")
#%%
n_samp = 100
err_dir = []
err_rej = []
err_ali = []
for i in range(1000):
    samples_U = np.random.uniform(1, 7, 10000)
    #n, bins, patches = plt.hist(samples_U)
    #cdf_dist = np.cumsum(n)/10000
    p = [7/48, 5/48, 1/8, 1/16, 1/4, 5/16]
    aaa = direct_method(p, n_samp, 6)
    bbb = rejection_method(p, n_samp, 6, 1)
    ccc = np.histogram(Alias_method(p, n_samp),6)[0]/n_samp
    print(p)
    print(aaa)
    print(bbb)
    print(ccc)
    err_dir.append(np.sum((np.array(aaa) - np.array(p))**2))
    err_rej.append(np.sum((np.array(bbb) - np.array(p))**2))
    err_ali.append(np.sum((np.array(ccc) - np.array(p))**2))
    
print('direct mean MSE: ', np.mean(err_dir))
print('rejection mean MSE: ', np.mean(err_rej))
print('aliasing mean MSE: ', np.mean(err_ali))

aaa = direct_method(p, 1000000, 6)
bbb = rejection_method(p, 1000000, 6, 1)
ccc = np.histogram(Alias_method(p, 1000000),6)[0]/1000000
print("Chi2test direct", chi_square_test(aaa, p, 6))
print("Chi2test rejection", chi_square_test(bbb, p, 6))
print("Chi2test aliasing", chi_square_test(ccc, p, 6))

#%%
exp_dist = expotential(100000, 1)
#dist_exp_hist, _ = np.histogram(exp_dist, 100)
plt.figure()
plt.hist(exp_dist, 100)
plt.xlabel('value')
plt.ylabel('number of samples')
#plt.plot(dist_exp_hist)
#%%
norm_dist = normal_generation(100000)
norm_dist_hist, val = np.histogram(norm_dist[0], 100)
norm_dist_hist2, val2 = np.histogram(norm_dist[1], 100)
plt.plot(val[:-1],norm_dist_hist)
plt.plot(val2[:-1],norm_dist_hist2)
plt.figure()
plt.subplot(1,2,1)
plt.hist(exp_dist, 100)
plt.xlabel('value')
plt.ylabel('number of samples')
plt.title("Exponential, $\lambda$ = 1")
plt.subplot(1,2,2)
plt.hist(norm_dist[0], 100)
plt.xlabel('value')
plt.ylabel('number of samples')
plt.title("Normal")
#%%
k = [2.05, 2.5, 3, 4]
beta = 1
plt.figure()
for value in k:
    pareto_dist = pareto(10000, beta, value)
    pareto_hist, val = np.histogram(pareto_dist, 1000)
    plt.plot(val[:-1], pareto_hist)
    plt.hist(pareto_dist,1000, alpha=0.6)
    plt.xlim(0,6)
    plt.ylim(0,2500)
    
    EX = beta*value/(value-1)
    VarX = (beta**2) * (value / ((value-2)*((value-1)**2)))
    #print((beta**2)*(value/(((value-1)**2)*value-2)))
    
    print("k =", value)
    print("mean:", np.mean(pareto_dist))
    print('analytical mean:', EX)
    print('variance:', np.var(pareto_dist))
    print('analytical variance:', VarX)
plt.legend(["k = 2.05", "k = 2.5", "k = 3", "k = 4"])
plt.xlabel('value')
plt.ylabel('number of samples')
plt.title("Pareto distribution")
plt.show()
    
#%%
import scipy.stats as ss
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), ss.sem(a)
    h = se * ss.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

confidence_intervals = np.zeros([100,2])
for i in range(100):
    data = normal_generation(10)[0]
    confidence_intervals[i,:] = mean_confidence_interval(data, 0.95)[1:]

plt.plot(confidence_intervals[:,0])
plt.plot(confidence_intervals[:,1])

data = normal_generation(10000)[0]
conf_int = mean_confidence_interval(data, 0.95)[1:]

plt.xlabel('trial')
plt.ylabel('interval value')
#plt.axhline(conf_int[0], c='r')
#plt.axhline(conf_int[1], c='r')
