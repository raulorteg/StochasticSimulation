"""
Generate simulated values from the following distributions
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def exponential_distribution(lamda, N):
    u = np.random.uniform(low=0, high=1, size=N)
    return np.log(u)/lamda

def normal_distribution(N):
    u1 = np.random.uniform(low=0, high=1, size=N)
    u2 = np.random.uniform(low=0, high=1, size=N)
    Z1 = np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)
    Z2 = np.sqrt(-2*np.log(u1))*np.sin(2*np.pi*u2)
    return Z1, Z2

def pareto_distribution(k, beta, N):
    u = np.random.uniform(low=0, high=1, size=N)
    x = beta*(np.power(u, -1/k))
    z = np.bincount(np.array(x, dtype=int))
    return x, z

def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.show()

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

if __name__ == "__main__":
    """
    # exercise 1a. Exponential distribution
    x = exponential_distribution(lamda=-1, N=1000)
    z, _, _ = plt.hist(x, bins=10)
    plt.show()

    # exercise 1b. Normal distribution
    x1, x2 = normal_distribution(N=1000)
    plt.hist([x1,x2], alpha=0.5, label=["Z1","Z2"], color=["blue", "red"])
    plt.title("Normal distribution")
    plt.legend()
    plt.show()

    # exercise 1c. Pareto distribution
    sols_x, sols_z = [], []
    for k in [2.05, 2.5, 3, 4]:
        x, z = pareto_distribution(k, beta=1, N=1000)
        sols_x.append(x)
        sols_z.append(z)
        
    plt.hist(sols_x, alpha=0.5, label=["k=2.05","k=2.5","k=3","k=4"])
    plt.title("Pareto distribution")
    plt.legend()
    plt.show()

    # exercise 2
    rel_mean_errors, rel_var_errors = [], []
    k_array, beta_array = np.linspace(2,10,80), [1] #np.linspace(1,10,9)
    for k in k_array:
        for beta in beta_array:

            x, z = pareto_distribution(k, beta, N=1000)
            exp_mean, exp_var = x.mean(), x.var()

            th_mean = beta*k/(k-1)
            th_var = (beta**2)*(k/(((k-1)**2)*(k-2)))

            rel_err_mean = abs(th_mean-exp_mean)*100/th_mean
            rel_err_var = abs(th_var-exp_var)*100/th_var

            rel_mean_errors.append(rel_err_mean)
            rel_var_errors.append(rel_err_var)

            # print(f"Theoretical mean: {th_mean}, Simulated mean: {exp_mean}")
            # print(f"Theoretical var: {th_var}, Simulated Variance: {exp_var}")

    plt.plot(k_array, rel_mean_errors, label="relerr mean")
    plt.plot(k_array, rel_var_errors, label="relerr var")
    plt.xlabel("k")
    plt.ylabel("Relative error %")
    plt.legend()
    plt.show()
    """

    # exercise 3
    mean_interval, up_interval, down_interval, dist_interval = [], [], [], []
    for i in range(100):
        x1, x2 = normal_distribution(N=100)

        mean, down, up = mean_confidence_interval(x1, confidence=0.95)
        mean_interval.append(mean)
        dist_interval.append(up-down)
        up_interval.append(up)
        down_interval.append(down)

    plt.plot(mean_interval, label="mean")
    plt.plot(up_interval, label="upper bound")
    plt.plot(down_interval, label="lower bound")
    plt.legend()
    plt.show()

    plt.plot(dist_interval)
    plt.show()




