# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 12:03:24 2021

@author: s202442
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

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
            
        if in_state!=states[i]:
            #print(in_state,"->",states[i])
            if states[i]==4:
                new_deaths+=1
        
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
plt.figure()
plt.plot(deds)
plt.xlabel("Months")
plt.ylabel("Cumulative number of deaths")
print("last ded", times[-1])
plt.figure()
plt.hist(np.array(times), 10)
plt.xlabel("Months")
plt.ylabel("Number of deaths")
#%%
#%%
#task 2
women_num = 1000

states = np.zeros(women_num, dtype=int)
deds = []
#for j in range(women_num):
j=0
old_states = 0
times = []
healthy = []
for i in range(120):
    states, new_deaths = state_change(P, states)
    #deds[j] = sum(states==4)
    deds.append(sum(states==4))
    healthy.append(sum(states==0))
    for deaths in range(new_deaths):
        times.append(i)
print('ded:', sum(states==4))
print('healthy:', sum(states==0))
deds_pdf = np.diff(np.array(deds))
plt.figure()
plt.plot(deds)
plt.xlabel("Months")
plt.ylabel("Cumulative number of deaths")
print("last ded", times[-1])
plt.figure()
plt.hist(np.array(times), 10)
plt.xlabel("Months")
plt.ylabel("Number of deaths")

plt.figure()
hist_states = np.histogram(states, 5)
plt.bar([0,1,2,3,4], hist_states[0])
plt.xlabel("State")
plt.ylabel("Number of women")

#%%

theta = np.array([1000, 0, 0, 0])
Ps = P[:-1, :-1]
ps = P[:-1, -1]
dist = []
dist_mean = []
for i in range(2,120):
    dist.append(theta@(Ps**i)@ps)
mean_prob = theta@(np.linalg.inv(np.identity(Ps.shape[0])-Ps))@np.ones(Ps.shape)
plt.figure()
plt.plot(np.arange(2,120),np.array(dist))
#plt.plot(np.arange(2,120),np.array(dist_mean))
plt.xlabel('Months')
plt.ylabel('Rate of healthy women')
plt.plot(np.array(healthy)/1000)
plt.legend(['analytical', 'simulation'])
#%%
plt.figure()
plt.bar([0,1,2,3,4], np.array([1,0,0,0,0])@np.linalg.matrix_power(P, 120), alpha=0.5)
plt.bar([0,1,2,3,4], hist_states[0]/1000, alpha=0.5)
plt.legend(['Analytical distribution of states', 'Simulated distribution of states'])
plt.xlabel('State')
plt.ylabel('Rate of women in each state')
dist_test_states = ss.chisquare(hist_states[0]/1000, np.array([1,0,0,0,0])@np.linalg.matrix_power(P, 120))
print(dist_test_states)
#%%
dist_test = ss.chisquare(np.array(healthy[2:])/1000, dist)
print(dist_test)
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
