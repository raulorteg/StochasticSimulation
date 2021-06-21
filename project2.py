
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Citizen:
	def __init__(self, age, state):
		self.state = state # 0: susceptible S, 1:infected I, 2:Recover R, 3:Dead D
		self.internal_clock = 0
		self.start_time = 0
		self.age = age

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
			self.internal_clock = np.random.beta(a=10, b=3)*18 + (0.3*self.age) # 10 days average time being sick
			self.start_time = time # time when got infected

	def transition_infected(self, time):
		"""
		first check if disease has ended, then if ended check if died or recovered
		"""
		if (time - self.start_time) >= self.internal_clock:

			# constant prob p=0.05 of dying
			if np.random.uniform() < 0.01*(self.age*0.1):
				self.state = 3 # citizen dies
			else:
				self.state = 2 # citizen recovers
				self.internal_clock = np.random.normal(loc=60.0-0.2*self.age, scale=15.0)
				self.start_time = time

		else:
			# constant prob p=0.05 of dying
			if np.random.uniform() < 0.01*(self.age*0.1):
				self.state = 3 # citizen dies


	def transition_recovered(self, time):
		"""
		first check if citizen has lost inmunity
		"""
		if (time - self.start_time) >= self.internal_clock:
			self.state = 0 # citizen is again susceptible


	def get_state(self):
		return self.state

class Population:
	def __init__(self, S, I, beta, gamma, time_final, age_groups, R=0):
		self.beta = beta
		self.gamma = gamma
		self.time = 0
		self.time_final = time_final
		self.age_groups = age_groups

		self.S  = S 
		self.I = I
		self.R = R
		self.D = 0

		self.N = S + I + R + self.D # total population
		self.citizens = []

		self.S_history, self.I_history, self.R_history, self.D_history = [], [], [], []
		self.age_dist = []

		self.spawn()
		self.save_progress()

	def spawn(self):

		# generate citizens from ages sampled from denmarks population distribution
		ages = list(self.age_groups.keys())
		counts = np.array(list(self.age_groups.values()))
		probs = counts/sum(counts)

		# spawn healthy citizens
		for _ in range(self.N):
			age = np.random.choice(a=ages, p=probs)
			self.citizens.append(Citizen(age=int(age), state=0))
			self.age_dist.append(int(age))

		# infect I number of citizens at random
		idxs = np.random.randint(0, high=self.N + 1, size=self.I, dtype=int)
		for idx in idxs:
			self.citizens[idx].state = 1

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

	# extract the dictionary of people per age group in Denmark
	# from: https://www.dst.dk/en/Statistik/emner/befolkning-og-valg/befolkning-og-befolkningsfremskrivning/folketal
	df = pd.read_csv("FOLK1A.csv", sep=";")
	df['ALDER'].replace( {" years" : '' }, inplace= True, regex = True)
	df['ALDER'].replace( {" year" : '' }, inplace= True, regex = True)
	df = df.groupby(['ALDER'])['INDHOLD'].agg('sum')
	age_groups = df.to_dict()

	Copenhaguen = Population(S=10000, I=10, R=0, beta=1.5, gamma=1.0, time_final=500, age_groups=age_groups)
	Copenhaguen.iterate()

	# get results
	S_history, I_history, R_history, D_history = Copenhaguen.get_histories()

	plt.plot(S_history, label="Susceptible")
	plt.plot(I_history, label="Infected")
	plt.plot(R_history, label="Recovered")
	plt.plot(D_history, label="Dead")
	plt.legend()
	plt.show()

	year = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2018]
	population_by_continent = {
    'africa': [228, 284, 365, 477, 631, 814, 1044, 1275],
    'americas': [340, 425, 519, 619, 727, 840, 943, 1006],
    'asia': [1394, 1686, 2120, 2625, 3202, 3714, 4169, 4560],
    'europe': [220, 253, 276, 295, 310, 303, 294, 293],
    'oceania': [12, 15, 19, 22, 26, 31, 36, 39],
	}

	time = list(range(len(S_history)))
	distribution_stages = {
    'Susceptible': S_history,
    'Infected': I_history,
    'Recovered': R_history,
    'Dead': D_history
	}

	fig, ax = plt.subplots()
	ax.stackplot(time, distribution_stages.values(),
             labels=distribution_stages.keys())
	ax.legend(loc='upper left')
	ax.set_title('World population')
	ax.set_xlabel('Year')
	ax.set_ylabel('Number of people (millions)')
	plt.show()



