import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.linalg import expm

class Patient:
	def __init__(self, q, id_, x0=0):

		self.q = q
		self.id_ = id_
		self.x0 = x0 # initial state
		self.x = x0
		self.time = 0.0
		self.alive = self.is_alive()
		self.history = []
		self.history.append((self.x, self.time))

	def is_alive(self):
		if self.x == len(q)-1:
			alive = False
		else:
			alive = True
		return alive

	def iterate(self):
		self.lamda = q[self.x, self.x] # get the diagonal
		self.time += np.random.exponential(scale=-1/self.lamda)

		# now transition to new state
		sample_list = list(range(self.x+1, len(self.q)))
		weights = tuple([-self.q[self.x,idx]/self.lamda for idx in sample_list])
		self.x = random.choices(sample_list, weights=weights)[0]

		# check if alive
		self.alive = self.is_alive()

		# update iteration, append to history
		self.history.append((self.x, self.time))

	def get_time(self, time):
		prev_snap_state = 0
		for snaphistory in self.history:
			snap_state, snap_time = snaphistory
			if snap_time > time:
				return prev_snap_state
			elif snap_time >= self.get_final_time():
				return snap_state
			prev_snap_state = snap_state

	def get_history(self):
		return self.history

	def get_final_time(self):
		return self.time


class Experiment:
	def __init__(self, q, N=1000, target_time=30.5):
		self.N = N
		self.q = q
		self.patients = []
		self.patients_histories = []
		self.times = []
		self.state_at_time = []
		self.target_time = target_time

	def generate_patients(self):
		for id_ in range(self.N):
			patient = Patient(q=self.q, id_=id_)
			self.patients.append(patient)

	def run(self):
		self.generate_patients()
		for patient in self.patients:
			print(f"Welcome Patient: {patient.id_}!")
			while patient.is_alive():
				patient.iterate()

			self.patients_histories.append(patient.get_history())
			self.times.append(patient.get_final_time())
			self.state_at_time.append(patient.get_time(time=self.target_time))


	def get_histories(self):
		return self.patients_histories

	def get_final_times(self):
		times = sorted(self.times)
		counts = np.bincount(times)
		return times, counts

	def get_dist_at_fixed_time(self):
		times = sorted(self.state_at_time)
		counts = np.bincount(times)
		return times, counts


def theoretical_lifetime(q, p0, time):
	"""
	returns theoretical distribution following q matrix from
	initial state p0 at time "time": p_t = p0(P^t)
	"""
	q_cut = q[0:len(q)-1, 0:len(q)-1]
	return 1-np.sum(np.matmul(p0,expm(q_cut*time)))

if __name__ == "__main__":
	
	"""
	TASK 7:
	"""
	q = np.array([[-0.0085, 0.005, 0.0025, 0, 0.001],
		[0, -0.014, 0.005, 0.004, 0.005],[0, 0, -0.008, 0.003, 0.005],
		[0, 0, 0, -0.009, 0.009],[0, 0, 0, 0, 0]])

	# define the experiment
	N = 1000
	experiment = Experiment(q=q, N=N, target_time=30.5)
	experiment.run()

	print("--Experiment finished--")

	histories = experiment.get_histories()
	times, counts = experiment.get_final_times()
	dist_target_time, counts_target_time = experiment.get_dist_at_fixed_time()


	plt.hist(dist_target_time)
	plt.xlabel("Stages of illness")
	plt.ylabel("Number of patients")
	plt.title("Distribution of patients over stages of illness t=30.5 months")
	plt.show()

	# plot results
	plt.hist(times)
	plt.title("Distribution of patients over elapsed time until deceased")
	plt.xlabel("Total elapsed time")
	plt.ylabel("Number of patients")
	plt.show()

	plt.plot(times)
	plt.title("Distribution of elapsed time until deacesed")
	plt.show()

	# theoretical distribution of lifetime
	lifetime_t30_5 = theoretical_lifetime(q, p0=[1,0,0,0], time=30.5)
	print(f"lifetime at t=30.5: {lifetime_t30_5}")

	# plot of lifetime for all time.
	times_list = np.linspace(0, max(times), 200)
	lifetimes_list = []
	for time in times_list:
		lifetimes_list.append(theoretical_lifetime(q, p0=[1,0,0,0], time=time))

	plt.plot(times_list, lifetimes_list)
	plt.title("Lifetime F(t) vs t")
	plt.xlabel("time t")
	plt.ylabel("F(t)")
	plt.show()







