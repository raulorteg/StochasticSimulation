
import numpy as np
import matplotlib.pyplot as plt

class Citizen:
	def __init__(self, state):
		self.state = state # 0: susceptible S, 1:infected I, 2:Recover R, 3:Dead D
		self.internal_clock = 0
		self.start_time = 0

	def transition(self, infected_rate, recovered_rate, total_infected, total_population, time):
		if self.state == 0: # if susceptible
			self.transition_susceptible(infected_rate, recovered_rate, total_infected, total_population, time)

		elif self.state == 1: # if infected
			self.transition_infected(time)

		elif self.state == 2: # if recovered
			self.transition_recovered(time)
		
		else:
			pass

	def transition_susceptible(self, infected_rate, recovered_rate, total_infected, total_population, time):
		# citizen gets infected
		## if np.random.uniform() < (infected_rate/(recovered_rate+infected_rate)):
		if np.random.uniform() < (total_infected/total_population)*(infected_rate/(recovered_rate+infected_rate)):
			self.state = 1 # infected
			self.internal_clock = np.random.poisson(lam=15.0) # 10 days average time being sick
			self.start_time = time # time when got infected

	def transition_infected(self, time):
		"""
		first check if disease has ended, then if ended check if died or recovered
		"""
		if (time - self.start_time) >= self.internal_clock:

			# constant prob p=0.05 of dying
			if np.random.uniform() < 0.05:
				self.state = 3 # citizen dies
			else:
				self.state = 2 # citizen recovers
				self.internal_clock = np.random.poisson(lam=90.0)
				self.start_time = time

			# TODO: save time of transitions

	def transition_recovered(self, time):
		"""
		first check if citizen has lost inmunity
		"""
		if (time - self.start_time) >= self.internal_clock:
			self.state = 0 # citizen is again susceptible


	def get_state(self):
		return self.state

class Population:
	def __init__(self, S, I, R, beta, gamma, time_final):
		self.beta = beta
		self.gamma = gamma
		self.time = 0
		self.time_final = time_final

		self.S  = S 
		self.I = I
		self.R = R
		self.D = 0

		self.N = S + I + R + self.D # total population
		self.citizens = []

		self.S_history, self.I_history, self.R_history, self.D_history = [], [], [], []

		self.spawn()
		self.save_progress()

	def spawn(self):
		for _ in range(self.S):
			self.citizens.append(Citizen(state=0))

		for _ in range(self.I):
			self.citizens.append(Citizen(state=1))

		for _ in range(self.R):
			self.citizens.append(Citizen(state=2))

	def save_progress(self):
		self.S_history.append(self.S)
		self.I_history.append(self.I)
		self.R_history.append(self.R)
		self.D_history.append(self.D)

	def iterate(self):
		while (self.time < self.time_final) and (self.I > 0):

			temp = np.zeros(4) # S, I, R, D

			infected_rate = (self.beta * self.S * self.I)/self.N
			recovered_rate = (self.gamma * self.I)
			# print(f"rate: {self.I/self.N}")

			for citizen in self.citizens:
				citizen.transition(infected_rate, recovered_rate, self.I, self.N, self.time)
				temp[citizen.get_state()] += 1

			self.S = temp[0]
			self.I = temp[1]
			self.R = temp[2]
			self.D = temp[3]
			self.N = self.S + self.I + self.R

			self.save_progress()

			if self.time % 50:
				print(f"t={self.time}:  S={self.S}, I={self.I}, R={self.R}, D={self.D}")

			self.time += 1.0

	def get_histories(self):
		return self.S_history, self.I_history, self.R_history, self.D_history


if __name__ == "__main__":
	Copenhaguen = Population(S=10000, I=10, R=0, beta=1.5, gamma=1.0, time_final=500)
	Copenhaguen.iterate()

	# get results
	S_history, I_history, R_history, D_history = Copenhaguen.get_histories()

	plt.plot(S_history, label="Susceptible")
	plt.plot(I_history, label="Infected")
	plt.plot(R_history, label="Recovered")
	plt.plot(D_history, label="Dead")
	plt.legend()
	plt.show()




