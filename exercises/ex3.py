# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 11:17:16 2021

@author: s202442
"""
import numpy as np
import math
import scipy.stats as ss
import matplotlib.pyplot as plt

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), ss.sem(a)
    h = se * ss.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


class Customer:
    def __init__(self, num, service_time, arrival_time, time):
        self.id = num
        self.service_time = service_time
        self.arrival_time = arrival_time
        self.time_in_process = 0
        self.start_time = time
        
    def check_if_ready(self):
        self.time_in_process += 0.1
        if self.time_in_process - self.service_time >= 0:
            return True
        else: return False
        
n_service_units = 10
mean_service_time = 8
mean_time_between_customers = 1
n_simulations = 10
n_customers = 10000
blocked_list = []

for num_sim in range(n_simulations):
    #arrivals = np.random.poisson(mean_time_between_customers, n_customers)
    #arrivals = np.round(ss.erlang.rvs(1, loc=1,scale=1,size=n_customers),1)
    arrivals = np.round(np.array([ss.expon.rvs(scale=1/0.8333) if u < 0.8 else ss.expon.rvs(scale=0.2) for u in list(np.random.random_sample(10000))]))
    cum_arrivals = np.cumsum(arrivals)
    #processing_time = np.round(np.random.exponential(mean_service_time, n_customers),1)
    #processing_time = np.ones(n_customers)*mean_service_time/2
    #processing_time = np.round(np.random.pareto(2.05, n_customers), 1)
    #processing_time = np.round(np.random.normal(1,1,n_customers),1)
    processing_time = np.round(np.random.geometric(1,n_customers),1)
    #processing_time[processing_time<0] = 0 #for gaussian
    customers_done = 0
    current_customers = 0
    customers = []
    time=0
    clients_in_progress = 0
    blocked = 0
    
    while n_customers - customers_done>0:
        #print(customers_done)
        num_of_new = len(np.where(np.isclose(cum_arrivals, time)==True)[0])
        for ind, cust in enumerate(customers):
            if cust.check_if_ready():
                del customers[ind]
                current_customers -= 1
        #print(num_of_new, current_customers)
        if num_of_new > 0 and current_customers < 10:
            #print(num_of_new, current_customers)
            for i in range(num_of_new):
                if customers_done==10000:
                    break
                #print(customers_done)
                if current_customers<10:
                    customers.append(Customer(customers_done, 
                                              processing_time[customers_done], 
                                              cum_arrivals[customers_done], time))
                    customers_done+=1
                    current_customers+=1
                elif current_customers==10:
                    blocked+=1
                    customers_done+=1
        elif num_of_new > 0 and current_customers == 10:
            blocked+=num_of_new
            customers_done+=num_of_new
        time = round(time + 0.1, 1)
    
    blocked_list.append(blocked/customers_done)
    print('Blocked:', blocked/customers_done)

print(mean_confidence_interval(blocked_list, 0.95))

suma=0
for i in range(10):
    suma+=8**i / math.factorial(i)
print("Erlangs B-formula:", (8**10/math.factorial(10))/suma)

#%%
