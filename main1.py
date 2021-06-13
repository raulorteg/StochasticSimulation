"""
1. Write a program implementing a linear congruential generator
(LCG). Be sure that the program works correctly using only
integer representation
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

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
    empirical = sorted(x_observed)/max(x_observed)
    hypothized = np.linspace(0, 1, n)
    D = max(abs(np.array(empirical) - np.array(hypothized)))
    test_value =  (sqrt(n) + 0.12 + (0.11/sqrt(n))) * D
    return test_value


if __name__ == "__main__":


	"""
	Parts a, b.
	"""
	# parameters
	num_classes = 16
	N = 10000
	m = 16
	a = 5
	c = 1
	x0 = 3

	x = congruential_generator(m=m,a=a,c=c,x0=x0,N=N)
	n_observed, _ = np.histogram(x, bins=num_classes)

	# scatter plot
	plt.plot(x[1:,], x[:-1,], "+")
	plt.show()

	# histogram
	plt.hist(x, bins=num_classes)
	plt.show()

	# kolmogorov test graph
	plt.plot(sorted(x)/max(x))
	plt.plot(np.linspace(0,1,len(x)))
	plt.show()

	# Chi square test
	n_expected = (N/num_classes)*np.ones(num_classes)
	T = chi_square_test(n_observed, n_expected, num_classes)
	print(f"Chi-square test: {T}")

	# Kolmogorov test
	kolg_test = kolmogorov_smirnov_test(x)
	print(f"Kolmogorov-Smirnov test: {kolg_test}")

	"""
	experimenting for different values of m,a,c
	"""
	m_array = [14,16,18,20,22,24]
	a_array = [3,4,5,6,7,8,9]
	c_array = [1,2,3,4,5,6,7,8,9,10]

	chi_array = []
	kolm_array = []
	best_chi = np.inf
	best_comb_chi = None
	best_kol = np.inf
	best_comb_kol = None

	N = 1000
	for m in m_array:
		for a in a_array:
			for c in c_array:
				x = congruential_generator(m=m,a=a,c=c,x0=x0,N=N)
				n_observed, _ = np.histogram(x, bins=num_classes)
				n_expected = (N/num_classes)*np.ones(num_classes)
				T = chi_square_test(n_observed, n_expected, num_classes)
				chi_array.append(T)
				kolg_test = kolmogorov_smirnov_test(x)
				kolm_array.append(kolg_test)

				if kolg_test < best_kol:
					best_kol = kolg_test
					best_comb_kol = (m,a,c)

				if T < best_chi:
					best_chi = T
					best_comb_chi = (m,a,c)

	plt.plot(chi_array)
	plt.show()

	plt.plot(kolm_array)
	plt.show()

	print(f"Best Kolmogorob-Smirnov: {best_comb_kol} with {best_kol}")
	print(f"Best chi: {best_comb_chi} with {best_chi}")