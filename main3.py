"""
Write a discrete event simulation program for a blocking system,
i.e. a system with m service units and no waiting room. The offered
traffic A is the product of the mean arrival rate and the mean
service time.
"""

import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

def compute_stats(customer, blocked_customers):
	arrival_array.append(customer.arrival_time)
	service_array.append(customer.service_time)
	delay_array.append(customer.delay)
	if customer.blocked:
		blocked_customers += 1

def generate_queue(num_customers):
	# generate the queue of customers
	queue = []
	arrival_time = 0
	for idx in range(num_customers):
		customer = Customer(idx)
		customer.arrival_time += arrival_time
		arrival_time = customer.arrival_time
		queue.append(customer)
	return queue

def plot_stats(arrival_array, service_array, delay_array, blocked_customers):
	# plot statistics
	plt.plot(arrival_array)
	plt.title("Arrival time (absolute)")
	plt.xlabel("Customer #number")
	plt.ylabel("Time units")
	plt.show()

	plt.plot(service_array)
	plt.title("Service time")
	plt.xlabel("Customer #number")
	plt.ylabel("Time units")
	plt.show()

	plt.hist(service_array, bins=20)
	plt.show()

	plt.plot(delay_array)
	plt.title("Delay time")
	plt.ylabel("Time units")
	plt.show()

	print(f"Fraction blocked customers: {blocked_customers*100/len(arrival_array)}")

class Customer:
	def __init__(self, id_):
		self.id = id_
		self.arrival_time = self.sample_arrival_time(loc=1)
		self.service_time = self.sample_service_time(loc=8)
		self.delay = None
		self.blocked = False

	def sample_arrival_time(self, loc):
		return stats.poisson.rvs(loc, size=1)[0] # poisson

	def sample_service_time(self, loc):
		return stats.expon.rvs(loc, size=1)[0] # exponential

	def compute_delay(self, time):
		self.delay = max(time-self.arrival_time, 0)
		if self.delay > 0:
			self.blocked = True

class Service_unit:
	def __init__(self, id_):
		self.id = id_
		self.countdown = 0.0

	def servicing_customer(self, service_time):
		self.countdown = service_time

	def update_time(self):
		self.countdown -= 1.0
		self.countdown = max(0.0, self.countdown)


if __name__ == "__main__":
	"""
	Exercise1.
	The arrival process is modelled as a Poisson process. Report the
	fraction of blocked customers, and a confidence interval for this
	fraction. Choose the service time distribution as exponential.
	Parameters: m = 10, mean service time = 8 time units, mean
	time between customers = 1 time unit (corresponding to an
	offered traffic of 8 erlang), 10 x 10.000 customers.
	"""

	num_customers = 100
	num_service_units = 10
	blocked_customers = 0
	delay_array, arrival_array, service_array = [], [], []

	# generate queu of customers
	queue = generate_queue(num_customers)

	# generate service units
	service_units = [Service_unit(idx) for idx in range(num_service_units)]

	# Start service, start stream of customers
	customer_id, time = 0,0
	while customer_id < (num_customers):

		# allocate all possible customers
		for idx in range(num_service_units):
			if service_units[idx].countdown <= 0.0:
				print(f"Welcome to our services! Customer: {customer_id}")

				customer = queue[customer_id]
				customer.compute_delay(time)
				compute_stats(customer, blocked_customers)

				service_units[idx].servicing_customer(customer.service_time)
				customer_id += 1

		time += 1

		# decrase countdown on service_units (time passes)
		for idx in range(num_service_units):
			service_units[idx].update_time()

	plot_stats(arrival_array, service_array, delay_array, blocked_customers)