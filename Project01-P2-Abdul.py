# --------------------------------------- 
# Load Libraries and Packages: 
# ---------------------------------------
""" IPython """
from math import exp
from IPython import get_ipython
from matplotlib import colors
get_ipython().run_line_magic('matplotlib', 'inline')

""" Data Handling """
import numpy as np
import matplotlib.pyplot as plt

""" Set seed """
np.random.seed(1992)

""" for warnings """
import warnings 
warnings.simplefilter("ignore")

""" load lifelines package: 
        needed for the to compare the survival functions of two samples.
""" 
!pip install lifelines

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

""" Chi-Square test """
def Chi2(obs,exp=None,n_bins=10):
    from scipy.stats import chisquare
    f_obs,_,_ = plt.hist(obs, n_bins)
    plt.close()
    if exp is None:
        f_exp = [np.round(len(obs) / n_bins) for i in range(n_bins)]  
    else:
        f_exp,_,_ = plt.hist(exp, n_bins)  
        plt.close()
    chi2_test_statistic, pvalue = chisquare(f_obs=f_obs, f_exp=f_exp)
    print(f'Chi-square test statistic =  {np.round(chi2_test_statistic,4)}\nchi-square test p-value = {np.round(pvalue,4)}')
    result = 'PASSED' if pvalue > 0.05 else 'NOT PASSED'
    print(f'Chi-square test is {result} under \u03B1 = 0.05')

""" (95%) Confidence Interval """
def Confidence_intervals(x, alpha=0.05, Std=False):
    import scipy.stats as stats
    mean = np.mean(x)
    std  = np.std(x,ddof=1)
    ddof = len(x)-1
    critical_value = stats.t.ppf(1-alpha/2, ddof)
    lower = mean - critical_value * (std/np.sqrt(len(x)))
    upper = mean + critical_value * (std/np.sqrt(len(x)))
    # Std CI
    if Std:
        lower = std * np.sqrt ( (len(x)-1) / stats.distributions.chi2.ppf(1-alpha/2, ddof))
        upper = std * np.sqrt ( (len(x)-1) / stats.distributions.chi2.ppf(alpha/2, ddof))
        return {'LCI': lower, 'Std': std, 'UCI': upper}
    # Mean CI
    return {'LCI': lower, 'Mean': mean, 'UCI': upper}

"""  Empirical Continuous-Time Markov Chains """
def CTMC(Q, max_time):
    Q = list(Q)  # matrix as list also works
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

    # Report: final state, time needed to reach it and states times
    final_state = max(range(len(Q)), key=lambda x: states_times[x])
    if final_state != 4: 
        print('No death and Exceeds the Max time = {time}, Trying again :)')
        return CTMC(Q, max_time)
    final_time = sum(list(states_times.values())[:-1])     
    return {'final_time' : final_time, 'final_state' : final_state, 'states_times': states_times}

""" Theoretical Continuous-Time Markov Chains """
def CTMC_theoretical(Q, p0, t):
    import scipy.linalg as linalg 
    Qs   = Q[:-1,:-1]
    return 1 - p0 @ linalg.expm(Qs*t) @ np.ones(len(Qs))

""" Log-Rank test """
def Log_rank(obs,exp):
    from lifelines.statistics import logrank_test
    results = logrank_test(ns0, ns1)
    print(f'log-rank test test-statistic =  {np.round(results.test_statistic,4)}')
    print(f'log-rank test p-value = {np.round(results.p_value,4)}')

# ---------------------------------------
# Task 07: 
# --------------------------------------- 
""" Parameters """
Q = np.array( [[ -0.0085,  0.005,  0.0025,     0,  0.001],
               [       0, -0.014,   0.005,  0.004, 0.005],
               [       0,      0,  -0.008,  0.003, 0.005],
               [       0,      0,       0, -0.009, 0.009],
               [       0,      0,       0,      0,     0]] ) # Transition Rate Matrix

n_women  = 10**3
max_time = 1500 # Max allowed Months (See: part 01 report)

""" Test: Continuous-Time Markov Chains """
simulation = CTMC(Q = Q, max_time = 1500)
simulation['final_time']
simulation['final_state']
simulation['states_times']

""" Simulate 10,000 women each over 1500 months """
months_until_death = np.array([])
states_times_list  = np.array([]) # needed for the proportion of women with cancer reappeared after 30.5 months 
for i in range(n_women):
    ctmc = CTMC(Q = Q, max_time = 1500)
    months_until_death = np.append(months_until_death,ctmc['final_time'])
    states_times_list = np.append(states_times_list,ctmc['states_times'])

""" Histogram """
density_Histogram(months_until_death, n_bins = 10, figsize=(10,6), title= "Density Histogram CTMC", xlabel= 'Months', ylabel='Deaths (%)', xticks=True)

""" (95%) Mean Confidence Intervals """
CI = Confidence_intervals(months_until_death, alpha=0.05)
print(f"LCI = {np.round(CI['LCI'],3)}, \
        Mean = {np.round(CI['Mean'],3)}, \
        UCI = {np.round(CI['UCI'],3)}")


""" (95%) Mean Confidence Intervals """
CI = Confidence_intervals(months_until_death, alpha=0.05, Std=True)
print(f"LCI = {np.round(CI['LCI'],3)}, \
        Std = {np.round(CI['Std'],3)}, \
        UCI = {np.round(CI['UCI'],3)}")

""" Proportion of women with cancer reappeared after 30.5 months """
counts = 0
for i in range(len(states_times_list)):
    ith_states_times = np.array(list(states_times_list[i].values()))
    ith_states_times = ith_states_times[(ith_states_times!= 0) & (ith_states_times!= np.inf)]
    if len(ith_states_times)!= 0:
        if(ith_states_times[0]>30.5):
            counts+=1
print(f'Proportion of women with cancer reappeared after 30.5 months = { counts / n_women}') 

# ---------------------------------------
# Task 08: 
# --------------------------------------- 
""" Parameters """
p0 = [1, 0, 0, 0]
max_time = 1500 # Max allowed Months (See: part 01 report)

""" Test: Theoretical Continuous-Time Markov Chains"""
CTMC_theoretical(Q=Q, p0=p0, t=max_time)

""" Theoretical (Expected) Sample """
times_linspace = np.linspace(start=0, stop=max_time, num=500)
expected_CDF = np.array([])
for time in times_linspace:
    expected_CDF = np.append(expected_CDF, CTMC_theoretical(Q=Q, p0=p0, t=time))

observed_CDF = np.array([])
for time in times_linspace:
    observed_CDF = np.append(observed_CDF, sum(months_until_death<=time)/len(months_until_death))

""" Histogram """
density_Histogram(observed_CDF, n_bins = 10, figsize=(10,6), title= "Empirical CTMC Density Histogram", xlabel= 'CDF', ylabel='Density', xticks=True)
density_Histogram(expected_CDF, n_bins = 10, figsize=(10,6), title= "Theoretical CTMC Density Histogram", xlabel= 'CDF', ylabel='Density', xticks=True)

""" Chi2 test """
Chi2(obs=observed_CDF, exp=expected_CDF)

# ---------------------------------------
# Task 09: 
# --------------------------------------- 
""" Parameters """
r00 = - sum([0.0025, 0.00125,      0, 0.001]) 
r11 = - sum([     0,       0,  0.002, 0.005]) 
r22 = - sum([     0,       0,  0.003, 0.005]) 
r33 = - sum([     0,       0,      0, 0.009]) 
r44 = - sum([     0,       0,      0,     0])

Q_treat = np.array([
    [    r00, 0.0025, 0.00125,      0, 0.001],
    [      0,    r11,       0,  0.002, 0.005], 
    [      0,      0,     r22,  0.003, 0.005], 
    [      0,      0,       0,    r33, 0.009], 
    [      0,      0,       0,      0,   r44] ]) # Transition Rate Matrix

n_women  = 10**3
max_time = 1500 # Max allowed Months (See: part 01 report)

""" Test: Continuous-Time Markov Chains """
simulation = CTMC(Q = Q_treat, max_time = max_time)
simulation['final_time']
simulation['final_state']
simulation['states_times']

""" Simulate 1000 women each over 1500 months """
months_until_death_treat = np.array([])
for i in range(n_women):
    months_until_death_treat = np.append(months_until_death_treat, CTMC(Q = Q_treat, max_time = max_time)['final_time'])

""" Histogram With Treatment vs Without Treatment """ 
n_bins = 10
plt.figure(figsize=(10,6))
weights = np.ones(len(months_until_death))/len(months_until_death)
ns, bins , _ = plt.hist([months_until_death,months_until_death_treat],weights=[weights,weights], bins=n_bins, ec='black', color= ['blue','green'])
plt.legend(['Without Treatment', 'With Treatment'])
plt.title('With Treatment vs Without Treatment - Density Histogram')
plt.xlabel('Months')
plt.xticks(ticks=bins, labels=[str(xi) for xi in np.round(bins,1)])
plt.ylabel('Death Density')
plt.show()

""" Kaplan Meier survival function estimate of the survival function vs survival function for women """
n_bins = 100
ns, bins , _ = plt.hist([months_until_death,months_until_death_treat], bins=n_bins, ec='black')
plt.close()
N = len(months_until_death)
ns0 = (N - np.array(ns[0])) / N # Calculate the survival function of Without Treatment 
ns1 = (N - np.array(ns[1])) / N # Calculate the survival function of With Treatment
plt.figure(figsize=(10,6))
plt.plot(np.linspace(min(bins), max(bins),n_bins),ns0,'blue',label='Without Treatment',linewidth=2)
plt.plot(np.linspace(min(bins), max(bins),n_bins),ns1,'green',label='With Treatment',linewidth=2)
plt.title('With Treatment vs Without Treatment - Kaplan-Meier Survival Function')
plt.xlabel('Months')
plt.ylabel('Survival Kaplan-Meier')
plt.legend()
plt.show()

""" Log Rank test for survival analysis """
Log_rank(obs=ns1, exp=ns0)