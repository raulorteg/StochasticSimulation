"""
In the excercise you can use a build in procedure for generating
random numbers. Compare the results obtained in simulations with
expected results. Use histograms (and tests).
"""

import numpy as np
import matplotlib.pyplot as plt


def geometric_distribution(p,N):
    # f(n) = P(X=n) = (1-p)^(n-1) * p
    u = np.random.uniform(low=0, high=1, size=N)
    x = np.floor(np.log(u)/np.log(1-p)) + 1
    z = np.bincount(np.array(x, dtype=int))
    return x, z

def driver_methods(p_vec,N,method,test):
    assert method in ["crude", "rejection", "alias"], "method not implemented. try: crude, rejection, alias"

    if method == "crude":
        x, z = crude_method(p_vec, N)
    elif method == "rejection":
        x, z = rejection_method(p_vec, N)
    else:
        x, z = alias_method(p_vec, N)

    if test:
        n_expected = np.floor(N*p_vec)
        n_observed = z
        chi_squared = chi_square_test(n_observed=z, n_expected=n_expected, num_classes=len(p_vec))
        return x, z, chi_squared
    return x, z

def crude_method(p_vec, N):
    u = np.random.uniform(low=0.0, high=1.0, size=N)
    z = np.zeros(len(p_vec))
    x = []
    cumsum_p = np.cumsum(p_vec)
    for u_i in u:
        for idx in range(len(cumsum_p)):
            if idx == 0:
                if (u_i <= cumsum_p[idx]):
                    z[idx] += 1
                    x.append(idx+1)
            else:
                if (cumsum_p[idx-1] < u_i) and (u_i <= cumsum_p[idx]):
                    z[idx] += 1
                    x.append(idx+1)
    return x, z

def rejection_method(p_vec, N):
    x = []
    z = np.zeros(len(p_vec))
    c = max(p_vec)
    u1 = np.random.uniform(low=0.0, high=1.0, size=N)
    I = (len(p_vec)*u1) + 1

    for _ in range(N):
        u2 = np.random.uniform(low=0.0, high=1.0, size=1)
        for I_i in I:
            idx = int(I_i-1)
            if (u2 <= p_vec[idx]/c):
                z[idx] += 1
                x.append(idx+1)
    return x, z

def alias_method(p_vec, N):
    #Generate F and L
    n = len(p_vec)
    F = n*p_vec
    L = list(range(1, n+1))
    G = [i for i in range(len(F)) if F[i] >= 1]
    S = [i for i in range(len(F)) if F[i] <= 1]
    while len(S) != 0:
        k = G[0]
        j = S[0]
        L[j] = k
        F[k] = F[k] - (1 - F[j])
        if F[k] < 1:
            G = G[1:]
            S.append(k)
        S = S[1:]
    
    x = []
    while len(x) < N:
        y = np.floor(n * np.random.uniform(0,1)) + 1
        u2 = np.random.uniform(0,1)
        if u2 < F[int(y-1)]:
            x.append(y)
        else:
            x.append(L[int(y-1)])

    z = np.bincount(np.array(x, dtype=int))
    return x, z

def histogram_comp_2d(x, y, title):
    n_bins = len(x)
    plt.hist([x,y], n_bins, alpha=0.5, label=["Simulated","Theoretical"], color=["blue", "red"])
    plt.title(title)
    plt.legend()
    plt.xlabel("Classes")
    plt.ylabel("p")
    plt.show()

def chi_square_test(n_observed, n_expected, num_classes):
    """
    n_observed:
    n_expected:
    num_classes:
    """
    temp = np.power((np.array(n_expected) - np.array(n_observed)),2)
    return np.sum(np.divide(temp, n_expected))


if __name__ == "__main__":
        
        # exercise 1. Geometric distribution
        x_theor = np.random.geometric(p=0.35, size=10000)
        z_theor = np.bincount(x_theor)
        x, z = geometric_distribution(p=0.35, N=10000)
        # histogram_comp_2d(x=z_theor/sum(z_theor), y=z/sum(z), title="Geometric distribution")

        # parameters
        p_vec = np.array([7/48, 5/48, 1/8, 1/16, 1/4, 5/16])

        # exercise 2a. Crude method
        print("Crude method:")
        x, z = crude_method(p_vec, N=1000)
        print(p_vec)
        print(z/sum(z))
        # histogram_comp_2d(x=p_vec, y=z/sum(z), title="Crude method")

        # exercise 2b. Rejection method
        print("Rejection method:")
        x, z = rejection_method(p_vec, N=1000) 
        # histogram_comp_2d(x=p_vec, y=z/sum(z), title="Rejection method")
        print(p_vec)
        print(z/sum(z))

        # exercise 2c. Alias method
        x, z = alias_method(p_vec, N=1000)
        # histogram_comp_2d(x=p_vec, y=z/sum(z), title="Alias method")
        print(p_vec)
        print(z/sum(z))

        # run chi-squared tests on the three methods
        for method in ["crude", "rejection", "alias"]:
            x, z, test = driver_methods(p_vec=p_vec, N=1000, method=method, test=True)
            print(f"Method: {method}, chi_square: {test}")
