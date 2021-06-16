"""
Exercise 6: Markov Chain Monte Carlo
Created 13/06/2021

"""

import numpy as np
import random
from math import log, floor, factorial, sqrt
from scipy import stats
import matplotlib.pyplot as plt

def truncated_poisson_dist(A, x):
	return (A**x)/factorial(x)

def truncated_poisson_dist_2call(A_array, x, y):
	assert len(A_array) == 2, "A_array must include A1, A2"
	return ((A_array[0]**x)/factorial(x))*((A_array[1]**y)/factorial(y))

def metropolis_hastings_1call(N):
	"""
	The Metropolis–Hastings algorithm can draw samples from any probability distribution P(x),
	provided that we know a function f(x) proportional to the density of P and the values of f(x) can be calculated.

	The requirement that f(x) must only be proportional to the density, rather than exactly
	equal to it, makes the Metropolis–Hastings algorithm particularly useful, because calculating the
	necessary normalization factor is often extremely difficult in practice
	"""
	x = [int(random.randint(0,11)) for _ in range(N)]
	for idx in range(N-1):
		x_prime = int(random.randint(0, 11))
		fun_x_prim = (8**x_prime)/factorial(x_prime)

		fun_x = (8**x[idx])/factorial(x[idx])
		accept_prob = fun_x_prim/fun_x

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
	x = [int(random.randint(0,11)) for _ in range(N)]
	y = [int(random.randint(0,11)) for _ in range(N)]
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
	x = [int(random.randint(0,11)) for _ in range(N)]
	y = [int(random.randint(0,11)) for _ in range(N)]
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
	x_obs = metropolis_hastings_1call(N=10000)
	x_exp = []
	for _ in range(10000):
		variate = stats.poisson.rvs(8, size=1)[0]
		while variate > 11:
			variate = stats.poisson.rvs(8, size=1)[0]
		x_exp.append(variate)
	x_exp = np.array(x_exp)

	print(np.unique(np.array(x_obs)))
	print(np.unique(np.array(x_exp)))

	z_obs = np.bincount(sorted(x_obs))
	z_exp = np.bincount(sorted(x_exp))
	print(z_exp)
	print(z_obs)

	chisq, p = stats.chisquare(f_obs=z_obs, f_exp=z_exp)
	print(f"Exercise1. chisq={chisq}, p={p}")

	plt.hist([x_exp, x_obs])
	plt.show()
	

	# Exercise 2a
	N = 1000
	x_obs, y_obs = metropolis_hastings_2call(N=N, fun=truncated_poisson_dist_2call)
	plt.hist2d(x_obs,y_obs)
	plt.show()

	# Exercise 2b
	x, y = metropolis_hastings_2call_coordwise(N=N, fun=truncated_poisson_dist_2call)
	plt.hist2d(x,y)
	plt.show()