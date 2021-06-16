# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 12:12:38 2021

@author: s202442
"""
import numpy as np
import matplotlib.pyplot as plt

class Patient:
    def __init__(self, P, condition=0):
        self.P = P
        self.time = 0
        self.states = P.shape[0]
        self.current_state = 0
        self.P_c = np.cumsum(self.P, 1)
        self.death_time = None
    def check_change(self):
        self.time+=1
        P = self.P_c
        roll = np.random.uniform(0,1,1)[0]
        in_state = self.current_state
        if roll<P[self.current_state,self.current_state]:
            self.current_state = self.current_state
        elif P[self.current_state,self.current_state]<roll<P[self.current_state,self.current_state+1]:
            self.current_state =  self.current_state+1
        elif P[self.current_state,self.current_state+1]<roll<P[self.current_state,self.current_state+2]:
            self.current_state =  self.current_state+2
        elif P[self.current_state,self.current_state+2]<roll<P[self.current_state,self.current_state+3]:
            self.current_state =  self.current_state+3
        elif P[self.current_state,self.current_state+3]<roll<P[self.current_state,self.current_state+4]:
            self.current_state =  self.current_state+4
        if in_state!=self.current_state:
            print(in_state,"->",self.current_state)
            if self.current_state==4:
                self.death_time = self.time
        return self.current_state, self.time, self.death_time
#%%
#task 4
woman = []
P = np.array([[0.9915, 0.005, 0.0025, 0, 0.001], 
              [0, 0.986, 0.005, 0.004, 0.005],
              [0, 0, 0.992, 0.003, 0.005],
              [0, 0, 0, 0.991, 0.009],
              [0, 0, 0, 0, 1]])

statuses = np.zeros(1000)
"""
for i in range(1000):
    woman.append(Patient(P))    

for t in range(100):
    for i in range(1000):
        statuses[i] = woman[i].check_change()[0]
print(sum(statuses==4))
"""
while len(woman)<1000:
    test_woman = Patient(P)
    for i in range(12):
        test_woman.check_change()
    if test_woman.current_state>0 and test_woman.current_state<4:
        woman.append(test_woman)

dist = []
death_times = []

for i in range(1000):
    dist.append(woman[i].current_state)
    statuses[i] = woman[i].current_state
plt.hist(dist,3)


while sum(statuses==4)<1000:
    for i in range(1000):
        statuses[i] = woman[i].check_change()[0]

for i in range(1000):
    death_times.append(woman[i].death_time)
    
plt.hist(death_times,50)
