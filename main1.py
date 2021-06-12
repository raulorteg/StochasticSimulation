"""
1. Write a program implementing a linear congruential generator
(LCG). Be sure that the program works correctly using only
integer representation
"""

import numpy as np
import matplotlib.pyplot as plt

def congruential_generator(m,a,c,x0,N):
	"""
	Parameters:
	m: modulus m > 0
	a: multiplier 0 < a < m
	c: increment 0 <= c < m
	x0: seed 0 <= x0 < m
	N: number of random numbers
	"""
	x = np.zeros(N+1)
	x[0] = x0;

	for i in range(N):
		x[i+1] = (a*x[i] + c) % m

	return x

def chi_square_test(n_observed, n_expected, num_classes):
	"""
	n_observed:
	n_expected:
	num_classes:
	"""
	temp = np.power((np.array(n_expected) - np.array(n_observed)),2)
	return np.sum(np.divide(temp, n_expected))

def kolmogorov_smirnov_test(x_observed):
    n = len(x_observed)
    empirical = sorted(x_observed)
    hypothized = np.linspace(0, 1, n)
    D = max(abs(np.array(empirical) - np.array(hypothized)))
    test_value =  (sqrt(n) + 0.12 + (0.11/sqrt(n))) * D
    return test_value

if __name__ == "__main__":

	# parameters
	num_classes = 16
	N = 10000
	m = 16
	a = 5
	c = 1
	x0 = 3

	x = congruential_generator(m=m,a=a,c=c,x0=x0,N=N)

	n_observed, bins, patches = plt.hist(x, bins=num_classes)
	plt.close()

	# Chi square test
	n_expected = (N/num_classes)*np.ones(num_classes)
	T = chi_square_test(n_observed, n_expected, num_classes)
	print(f"Chi-square test: {T}")

	# Kolmogorov test
	kolgomorov_test(x)

