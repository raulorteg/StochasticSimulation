"""
Exercise 8
Bootstrap methods.
@author Raul Ortega
Created 12/06/2021
"""
import numpy as np
import random
import matplotlib.pyplot as plt

def bootstrap_sampling_ex13(X, a, b, n_samples, iterations=1000):
	prob_array = np.zeros(iterations)
	ctr, mean = 0, np.mean(X)
	for iter_ in range(iterations):
		idxs = [random.randint(0,n_samples-1) for _ in range(n_samples)]
		samples = [X[idx] for idx in idxs]
		if ((sum(samples)/n_samples - mean) > a) and ((sum(samples)/n_samples - mean) < b):
			ctr += 1
		prob_array[iter_] = ctr/(iter_+1)

	prob = ctr/iterations
	return prob, prob_array

def bootstrap_sampling_variance(X, n_samples, iterations=1000):
	var_array = np.zeros(iterations)
	for iter_ in range(iterations):
		idxs = [random.randint(0,n_samples-1) for _ in range(n_samples)]
		samples = np.array([X[idx] for idx in idxs])
		variance = sum((samples-np.mean(samples))**2)/(n_samples-1)
		var_array[iter_] = variance

	return variance, var_array

def bootstrap_sampling_estimates(X, n_samples, replications, iterations=100):
	var_median_array, var_mean_array = np.zeros(iterations), np.zeros(iterations)
	for iter_ in range(iterations):
		median_array, mean_array = np.zeros(replications), np.zeros(replications)
		for r in range(replications):
			idxs = [random.randint(0,n_samples-1) for _ in range(n_samples)]
			samples = np.array([X[idx] for idx in idxs])
			median_array[r] = np.median(samples)
			mean_array[r] = np.mean(samples)

		variance_median = sum((median_array-np.mean(median_array))**2)/(replications-1)
		var_median_array[iter_] = variance_median

		variance_mean = sum((mean_array-np.mean(mean_array))**2)/(replications-1)
		var_mean_array[iter_] = variance_mean

	return variance_median, var_median_array, variance_mean, var_mean_array

def pareto_distribution(k, beta, N):
    u = np.random.uniform(low=0, high=1, size=N)
    x = beta*(np.power(u, -1/k))
    z = np.bincount(np.array(x, dtype=int))
    return x, z

if __name__ == "__main__":

	# exercise 13. Chapter 8
	X = [56, 101, 78, 67, 93, 87, 64, 72, 80, 69]
	a, b = -5, 5
	n_samples = 10
	prob, prob_array = bootstrap_sampling_ex13(X, a, b, n_samples)
	print(f"Estimated prob.: {prob}")
	plt.plot(prob_array)
	plt.ylabel("Estimated prob.")
	plt.xlabel("Iteration")
	plt.show()

	# exercise 15. Chapter 8
	X = [5, 4, 9, 6, 21, 17, 11, 20, 7, 10, 21, 15, 13, 16, 8]
	n_samples = 15
	variance, var_array = bootstrap_sampling_variance(X, n_samples)
	print(f"Estimated variance: {variance}. Real Variance: {np.var(np.array(X))}")
	plt.plot(var_array, label="Est. Variance")
	true_variance = np.var(np.array(X))*np.ones(len(var_array))
	plt.plot(true_variance, label="Variance")
	plt.legend()
	plt.ylabel("Estimated Variance.")
	plt.xlabel("Iteration")
	plt.show()

	# Compute variance of the median of 200 samples from Pareto distribution
	X, _ = pareto_distribution(k=1.05, beta=1, N=200)
	n_samples = 200
	variance_median, var_median_array, variance_mean, var_mean_array = bootstrap_sampling_estimates(X, n_samples, replications=100)
	print(f"Est. variance of the median: {variance_median}")
	print(f"Est. variance of the mean: {variance_mean}")
	plt.plot(var_median_array)
	plt.ylabel("Estimated Variance of median.")
	plt.xlabel("Iteration")
	plt.show()

	plt.plot(var_mean_array)
	plt.ylabel("Estimated Variance of mean.")
	plt.xlabel("Iteration")
	plt.show()

