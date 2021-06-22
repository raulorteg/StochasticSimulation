# 02443 Stochastic Simulation DTU

## About
Exercises and Projects 1 & 2 for DTU's Stochastic Simulation course.

### Project 2: Pandemic Simulation

We start implementing a variation of the SIR model for simulating a pandemic that can be found in `project2.py`: There are citizens and Populations.

* A citizen has certain age, and is in a certain state: S (susceptible), I (Infected), R (Recovered), D (Dead). The age of the spawned citizen is sampled from the distribution of ages from Denmark, the State is initially S (susceptible) or Infected (I). The citizen blueprint is the Citizen class.

* A population has a list of Citizens, it spawns the citizens and is the object responsible to update the states (S, I, R, D) of the citizen at every iteration and update the total number of Susceptible people, Infected, Recovered and Dead

A citizen can be in any of the S, I, R, D states, and the transition of this states follow the digram in the figure below. A susceptible Citizen can get infected thus transitioning into I state, an Infect citizen can die while ill or recover once the illness is over, and in order to simulate waves of the disease a recovered citizen after some time looses inmunity and becomes Susceptible again.

<p float="center">
  <img src="plots/foo" alt="cycle_simulation" height="300" />
</p>

### Variation 1: Including Lockdown

### Variation 2: Including Vaccination programme

### Variation 3: Simulating infection spread over populations

We can extend now the Citizen class so that now the citizen belongs to a city (Citizen.city) and we can extend the Population class to allow in/out traffic. Every day some number N of resident of the city will leave the city to work to other city, since they leave for work we make it so that only people with ages 18-65 travel and only if their state is S, I, R (only if alive). This people are removed from the list of citizens from their city but we know were the belong to because of the Citizen.city attribute. This is the outwards travel.

Every day people that had left town the previous day come back (we select them by the Citizen.city attribute) and people from other city come to the city to work, this is the inwards traffic and we append this new citizens to the population of the city.

The idea of this experiment is to simulate how the disease can spread from one city to the other. We simulate two cities (say Malmo & Copnehaguen). Now every day they evolve and interchange people that work. We initialize Malmo without any infectious people and Copenhaguen with 10 Infected people and let it run for some iterations (days). The results are shown in the animation below.

<p float="center">
  <img src="cities_30.gif" alt="cities gif" height="300" />
</p>

### Exercises
Example of Simulated Annealing algorithm on circle and random map with 20 cities.

<p float="center">
  <img src="plots/circle.gif" alt="circle map gif" height="300" />
  <img src="plots/random.gif" alt="random map gif" height="300"/>
</p>

## Requirements
```pip install requirements.txt -r```
