# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 12:03:24 2021

@author: s202442
"""
import numpy as np
import matplotlib.pyplot as plt

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

def state_change(P, states):
    P = np.cumsum(P, 1)
    roll = np.random.uniform(0,1,len(states))
    new_deaths = 0
    for i in range(len(states)):
        in_state = states[i]
        if roll[i]<P[states[i],states[i]]:
            states[i] = states[i]
        elif P[states[i],states[i]]<roll[i]<P[states[i],states[i]+1]:
            states[i] =  states[i]+1
        elif P[states[i],states[i]+1]<roll[i]<P[states[i],states[i]+2]:
            states[i] =  states[i]+2
        elif P[states[i],states[i]+2]<roll[i]<P[states[i],states[i]+3]:
            states[i] =  states[i]+3
        elif P[states[i],states[i]+3]<roll[i]<P[states[i],states[i]+4]:
            states[i] =  states[i]+4
            new_deaths+=1
        #if in_state!=states[i]:
            #print(in_state,"->",states[i])
    return states, new_deaths
#%%
#task 1
P = np.array([[0.9915, 0.005, 0.0025, 0, 0.001], 
              [0, 0.986, 0.005, 0.004, 0.005],
              [0, 0, 0.992, 0.003, 0.005],
              [0, 0, 0, 0.991, 0.009],
              [0, 0, 0, 0, 1]])

women_num = 1000

states = np.zeros(women_num, dtype=int)
deds = []
#for j in range(women_num):
j=0
old_states = 0
times = []
while sum(states==4)!=women_num:
    j+=1
    states, new_deaths = state_change(P, states)
    #deds[j] = sum(states==4)
    deds.append(sum(states==4))
    for deaths in range(new_deaths):
        times.append(j)
print('ded:', sum(states==4))
deds_pdf = np.diff(np.array(deds))
plt.plot(deds)
plt.figure()
plt.hist(np.array(times))
#%%
#task 2
theta = np.array([1000, 0, 0, 0])
Ps = P[:-1, :-1]
ps = P[:-1, -1]
dist = []
for i in range(1,1000):
    dist.append(theta@(Ps**i)@ps)
plt.plot(np.array(dist)*60)
#%%
#task 5
deaths_sim = np.zeros(100)
for i in range(100):
    states = np.zeros(350, dtype = int)
    for j in range(350):
        states, new_deaths = state_change(P, states)
        #deds[j] = sum(states==4)
        deds.append(sum(states==4))
        for deaths in range(new_deaths):
            times.append(j)
    print('ded:', sum(states==4))
    #print(i)
    deaths_sim[i] = sum(states==4)
print(np.mean(deaths_sim)/350)
