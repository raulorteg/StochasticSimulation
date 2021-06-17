# --------------------------------------- 
# Load libraries: 
# ---------------------------------------
""" IPython """
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

""" Data Handling """
import numpy as np
import matplotlib.pyplot as plt

""" Set seed """
np.random.seed(1992)

""" for warnings """
import warnings 
warnings.simplefilter("ignore")


# ---------------------------------------
# General Functions: 
# X: time -> state
# --------------------------------------- 

""" Density Histogram """
def density_Histogram(x, n_bins = 10, figsize=(10,6), title= "Density Histogram", xlabel= 'Classes', ylabel='Density', xticks=False):
    plt.figure(figsize=figsize)
    n_bins = n_bins
    bins = np.linspace(min(x), max(x), n_bins + 1)    
    weights = np.ones(len(x))/len(x)
    plt.hist(x, n_bins, weights=weights, ec='black') # ec short for "edgecolor"
    plt.title(title)
    plt.xlabel(xlabel)
    if xticks : plt.xticks(ticks=bins, labels=[str(xi) for xi in np.round(bins,2)])
    plt.ylabel(ylabel)
    plt.show()

""" (95%) Confidence Interval student-t distribution """
def Mean_confidence_intervals(x, alpha=0.05):
    import scipy.stats as stats
    mean = np.mean(x)
    std  = np.std(x,ddof=1)
    ddof = len(x)-1
    critical_value = stats.t.ppf(1-alpha/2, ddof)
    lower = mean - critical_value * (std/np.sqrt(len(x)))
    upper = mean + critical_value * (std/np.sqrt(len(x)))
    return {'LCI': lower, 'Mean': mean, 'UCI': upper}



"""  Continuous-Time Markov Chains """
def CTMC(Q, max_time):
    states = range(len(Q))
    states_times = {state:0 for state in states}  # Time spent on each state
    time = 0  # Init time tracker 
    current_state = 0  # First state after they had their breast tumor removed
    while time < max_time:
        # Sample states time to identify a transition to next state
        sojourns  = [np.random.exponential(scale=1/rate) for rate in Q[current_state][:current_state]]
        sojourns += [np.inf]  # An infinite sojourn to the same state
        sojourns += [np.random.exponential(scale=1/rate) for rate in Q[current_state][current_state + 1:]]
        # Identify the next state (closest in time)
        next_state = min(states, key=lambda sojourn: sojourns[sojourn])
        # Update
        sojourn    = sojourns[next_state]
        time      += sojourn        
        states_times[current_state] += sojourn        
        current_state = next_state  # Transition
    # Report: final state and final time needed to reach it
    final_state = max(range(len(Q)), key=lambda x: states_times[x])
    if final_state != 4: 
        print('No death and Exceeds the Max time = {time}, Trying again :)')
        return CTMC(Q, max_time)
    final_time = sum(list(states_times.values())[:-1]) 

    return {'final_time' : final_time, 'final_state' : final_state}



# ---------------------------------------
# Task 07: 
# --------------------------------------- 
"""  Transition Rate Matrix """
Q = np.array( [[ -0.0085,  0.005,  0.0025,     0,  0.001],
               [       0, -0.014,   0.005,  0.004, 0.005],
               [       0,      0,  -0.008,  0.003, 0.005],
               [       0,      0,       0, -0.009, 0.009],
               [       0,      0,       0,      0,     0]] )

n_women  = 10**3
max_time = 1500 # Max allowed Months (See: part 01 report)

""" test """
simulation = CTMC(Q = Q, max_time = 1500)
simulation['final_time']
simulation['final_state']

""" Simulate 10,000 women each over 1500 months """
months_until_death = np.array([])
for i in range(n_women):
    months_until_death = np.append(months_until_death, CTMC(Q = Q, max_time = 1500)['final_time'])

""" Histogram """
density_Histogram(months_until_death, n_bins = 10, figsize=(10,6), title= "Density Histogram CTMC", xlabel= 'Months', ylabel='Deaths (%)', xticks=True)

""" (95%) Mean_confidence_Intervals """
CI = Mean_confidence_intervals(months_until_death, alpha=0.05)
print(f"LCI = {np.round(CI['LCI'],3)}, \
        Mean = {np.round(CI['Mean'],3)}, \
        UCI = {np.round(CI['UCI'],3)}")
