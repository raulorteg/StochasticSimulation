"""
Exercise 6: Markov Chain Monte Carlo

1. The number of busy lines in a trunk group (Erlang system) is
given by a truncated Poisson distribution
Generate values from this distribution by applying the
Metropolis-Hastings algorithm, verify with a χ2-test. You can
use the parameter values from exercise 4
"""

import numpy as np
from math import log, floor, factorial, sqrt
from scipy import stats
import matplotlib.pyplot as plt

def analytical_truncated_poisson_1call(A, N):
    x = np.zeros(N)
    for idx in range(N):
        x[idx] = A**idx /factorial(idx)
    return x

def truncated_poisson_dist(A, x):
	return (A**x)/factorial(x)

def truncated_poisson_dist_2call(A_array, x, y):
	assert len(A_array) == 2, "A_array must include A1, A2"
	return ((A_array[0]**x)/factorial(x))*((A_array[1]**y)/factorial(y))

def metropolis_hastings_1call(N, fun):
	"""
	The Metropolis–Hastings algorithm can draw samples from any probability distribution P(x),
	provided that we know a function f(x) proportional to the density of P and the values of f(x) can be calculated.

	The requirement that f(x) must only be proportional to the density, rather than exactly
	equal to it, makes the Metropolis–Hastings algorithm particularly useful, because calculating the
	necessary normalization factor is often extremely difficult in practice
	"""
	x = np.zeros(N)
	for idx in range(N-1):
		x_prime = np.random.randint(low=0, high=11)
		fun_x_prim = fun(A=8, x=x_prime)

		fun_x = fun(A=8, x=x[idx])
		accept_prob = min(1, fun_x_prim/fun_x)

		u = np.random.uniform(low=0, high=1)
		if u <= accept_prob:
			x[idx+1] = x_prime
		else:
			x[idx+1] = x[idx]
	return x

def metropolis_hastings_2call(N, fun):
	"""
	The Metropolis–Hastings algorithm can draw samples from any probability distribution P(x),
	provided that we know a function f(x) proportional to the density of P and the values of f(x) can be calculated.

	The requirement that f(x) must only be proportional to the density, rather than exactly
	equal to it, makes the Metropolis–Hastings algorithm particularly useful, because calculating the
	necessary normalization factor is often extremely difficult in practice
	"""
	x, y = np.zeros(N), np.zeros(N)

	for idx_x in range(N-1):
		for idx_y in range(N-1):
			fun_x = fun(A_array=[4,4], x=x[idx_x], y=y[idx_y])

			x_prime, x_dprime = np.random.randint(0,11), np.random.randint(0,11)
			fun_x_prim = fun(A_array=[4,4], x=x_prime, y=x_dprime)
			accept_prob = min(1, fun_x_prim/fun_x)

			u = np.random.uniform(low=0, high=1)
			if u <= accept_prob:
				x[idx_x+1] = x_prime
				y[idx_y+1] = x_dprime

			else:
				x[idx_x+1] = x[idx_x]
				y[idx_y+1] = y[idx_y]
	return x, y

def metropolis_hastings_2call_coordwise(N, fun):
	"""
	The Metropolis–Hastings algorithm can draw samples from any probability distribution P(x),
	provided that we know a function f(x) proportional to the density of P and the values of f(x) can be calculated.

	The requirement that f(x) must only be proportional to the density, rather than exactly
	equal to it, makes the Metropolis–Hastings algorithm particularly useful, because calculating the
	necessary normalization factor is often extremely difficult in practice
	"""
	x, y = np.zeros(N), np.zeros(N)

	for idx_x in range(N-1):
		fun_x = fun(A_array=[4,4], x=x[idx_x], y=y[idx_x])

		x_prime, x_dprime = np.random.randint(0,11), np.random.randint(0,11)
		fun_x_prim = fun(A_array=[4,4], x=x_prime, y=x_dprime)
		accept_prob = min(1, fun_x_prim/fun_x)

		u = np.random.uniform(low=0, high=1)
		if u <= accept_prob:
			x[idx_x+1] = x_prime

		else:
			x[idx_x+1] = x[idx_x]

	for idx_y in range(N-1):
		fun_x = fun(A_array=[4,4], x=x[idx_y], y=y[idx_y])

		x_prime, x_dprime = np.random.randint(0,11), np.random.randint(0,11)
		fun_x_prim = fun(A_array=[4,4], x=x_prime, y=x_dprime)
		accept_prob = min(1, fun_x_prim/fun_x)

		u = np.random.uniform(low=0, high=1)
		if u <= accept_prob:
			y[idx_y+1] = x_dprime

		else:
			y[idx_y+1] = y[idx_y]

	return x, y


if __name__ == "__main__":

	# Exercise 1
	x_obs = metropolis_hastings_1call(N=1000, fun=truncated_poisson_dist)
	x_exp = analytical_truncated_poisson_1call(N=1000, A=8)
	
	plt.hist([x_exp])
	plt.show()

	z_obs, _ = np.histogram(x_obs, bins=10)
	z_exp, _ = np.histogram(x_exp, bins=10)

	chisq, p = stats.chisquare(f_obs=z_obs, f_exp=z_exp)
	print(f"Exercise1. chisq={chisq}, p={p}")


	# Exercise 2a
	x, y = metropolis_hastings_2call(N=1000, fun=truncated_poisson_dist_2call)
	plt.hist2d(x,y)
	plt.show()

	# Exercise 2b
	x, y = metropolis_hastings_2call_coordwise(N=1000, fun=truncated_poisson_dist_2call)
	plt.hist2d(x,y)
	plt.show()
