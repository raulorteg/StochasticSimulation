"""
Project 1
"""
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.linalg import matrix_power

class Patient:
	def __init__(self, p, id_, x0=1):

		self.p = p
		self.id_ = id_
		self.x0 = x0 # initial state
		self.x = x0
		self.iter = 0
		self.alive = self.is_alive()
		self.history = []
		self.history.append(self.x)

	def is_alive(self):
		if self.x == len(p)-1:
			alive = False
		else:
			alive = True
		return alive

	def iterate(self):
		sample_list = list(range(len(self.p)))
		weights = tuple([self.p[self.x,idx] for idx in sample_list])
		self.x = random.choices(sample_list, weights=weights)[0]

		self.alive = self.is_alive()

		self.iter  += 1
		self.history.append(self.x)

	def get_history(self):
		return self.history

	def get_final_iter(self):
		return self.iter

class Experiment:
	def __init__(self, p, N=100):
		self.N = N
		self.p = p
		self.patients = []
		self.patients_histories = []
		self.final_iters = []

	def generate_patients(self):
		for id_ in range(self.N):
			patient = Patient(p=self.p, id_=id_)
			self.patients.append(patient)

	def run(self):
		self.generate_patients()
		for patient in self.patients:
			print(f"Patient {patient.id_}")
			while patient.is_alive():
				patient.iterate()

			self.patients_histories.append(patient.get_history())
			self.final_iters.append(patient.get_final_iter())

	def get_histories(self):
		return self.patients_histories

	def get_final_iters(self):
		final_iters = sorted(self.final_iters)
		counts = np.bincount(final_iters)
		return final_iters, counts

def theoretical_dist_states(p, p0, time):
	"""
	returns theoretical distribution following p matrix from
	initial state p0 at time "time": p_t = p0(P^t)
	"""
	return np.matmul(p0,matrix_power(p, time))

if __name__ == "__main__":
	
	"""
	TASK 1:
	"""
	p = np.array([[0.9915, 0.005, 0.0025, 0, 0.001], [0, 0.986, 0.005, 0.004, 0.005],
	[0, 0, 0.992, 0.003, 0.005], [0, 0, 0, 0.991, 0.009], [0,0,0,0,1]])

	# define the experiment
	N = 1000
	experiment = Experiment(p=p, N=N)
	experiment.run()

	print("--Experiment finished--")

	histories = experiment.get_histories()
	final_iters, counts = experiment.get_final_iters()

	# plot results
	plt.hist(final_iters)
	plt.title("Distribution of patients over number of final iteration")
	plt.xlabel("Final iteration number (death)")
	plt.ylabel("Number of patients")
	plt.show()

	plt.plot(N-np.array(final_iters))
	plt.title("Number of alive patients")
	plt.xlabel("Iteration")
	plt.ylabel("Number of alive patients")
	plt.show()

	# lets see now at t=120 (Simulation)
	histories_t120 = []
	for history in histories:
		try:
			histories_t120.append(history[119])
		except:
			histories_t120.append(len(p)-1)

	plt.hist(histories_t120)
	plt.title("Distribution of patients over states t=120")
	plt.xlabel("States X_i")
	plt.ylabel("Number of patients")
	plt.show()

	counts_t120 = np.bincount(sorted(histories_t120))

	"""
	TASK 2:
	lets compare the simulation at t=120 with the 
	theoretical distribution
	"""
	pt = theoretical_dist_states(p=p, p0=np.array([1,0,0,0,0]), time=119)
	print(counts_t120)
	print(pt)



