
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os
from PIL import Image

class Citizen:
    def __init__(self, age, state, city):
        self.state = state # 0: susceptible S, 1:infected I, 2:Recover R, 3:Dead D
        self.internal_clock = 0
        self.start_time = 0
        self.age = age
        self.city = city

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
        if total_infected:
            if np.random.uniform() < 0.5*(total_infected/total_population)*(infected_rate/(recovered_rate+infected_rate)):
                self.state = 1 # infected
                self.internal_clock = np.random.beta(a=10, b=3)*18 + (0.3*self.age) # 10 days average time being sick
                self.start_time = time # time when got infected

    def transition_infected(self, time):
        """
        first check if disease has ended, then if ended check if died or recovered
        """
        if (time - self.start_time) >= self.internal_clock:

            self.state = 2 # citizen recovers
            self.internal_clock = np.random.normal(loc=60.0-0.2*self.age, scale=15.0)
            self.start_time = time

        else:
            # constant prob p=0.05 of dying
            if np.random.uniform() < 0.001*(self.age*0.1):
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
    def __init__(self, S, I, beta, gamma, age_groups, city, R=0):
        self.beta = beta
        self.gamma = gamma
        self.time = 0
        self.age_groups = age_groups
        self.city = city

        self.S  = S 
        self.I = I
        self.R = R
        self.D = 0

        self.N = S + I + R # total population
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
            self.citizens.append(Citizen(age=int(age), state=0, city=self.city))
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
            print(f"({self.city}): t={self.time}, S={self.S}, I={self.I}, R={self.R}, D={self.D}")

        self.time += 1.0

    def out_travel(self, N):
        """
        1. Get the citizens that are not from this city, they now return home from work.
        to check if they are from the city check if citize.city == self.city

        2. Get N number of citizens from the population that are within ages 18-65
        these will go out of the city, so remove them from the population self.citizens
        """
        
        travellers = []

        # get the people returning home to the other town
        for idx, citizen in enumerate(self.citizens):
            if citizen.city != self.city:
                travellers.append(citizen)
                del self.citizens[idx]

        n = 0
        # get the workers that are leaving town
        while n < N:
            idx = np.random.randint(0,len(self.citizens), size=1)[0]
            if (self.citizens[idx].age >= 18) and (self.citizens[idx].age <= 65):
                travellers.append(self.citizens[idx])
                del self.citizens[idx]
                n += 1
        return travellers

    def in_travel(self, in_travellers):
        """
        This are new citizens that come to work. this are citizens from another town between the ages 18-65.
        They should be appended to the population of the city at self.citizens
        """
        for citizen in in_travellers:
            self.citizens.append(citizen)

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

    Copenhaguen = Population(S=10000, I=10, R=0, beta=1.5, gamma=1.0, age_groups=age_groups, city="Copenhaguen")
    Malmo = Population(S=10000, I=0, R=0, beta=1.5, gamma=1.0, age_groups=age_groups, city="Malmo")
    iter_ = 0
    while (Copenhaguen.time < 300) and (Copenhaguen.I + Malmo.I > 0):
        Copenhaguen.iterate()
        Malmo.iterate()

        # exchange citizens
        out_copenhaguen = Copenhaguen.out_travel(N=10)
        out_malmo = Malmo.out_travel(N=10)
        Copenhaguen.in_travel(out_malmo)
        Malmo.in_travel(out_copenhaguen)

        # get results
        S_history_cph, I_history_cph, R_history_cph, D_history_cph = Copenhaguen.get_histories()
        S_history_malmo, I_history_malmo, R_history_malmo, D_history_malmo = Malmo.get_histories()

        # save figs
        time = list(range(len(S_history_cph)))
        distribution_stages_cph = {
        'Susceptible': S_history_cph,
        'Infected': I_history_cph,
        'Recovered': R_history_cph,
        'Dead': D_history_cph
        }
        distribution_stages_malmo = {
        'Susceptible': S_history_malmo,
        'Infected': I_history_malmo,
        'Recovered': R_history_malmo,
        'Dead': D_history_malmo
        }

        fig, ax = plt.subplots(2,1)

        ax[0].stackplot(time, distribution_stages_cph.values(),
             labels=distribution_stages_cph.keys())
        ax[0].legend(loc='upper left')
        ax[0].set_title('Population of Copenhaguen')
        ax[0].set_ylabel('Number of people')

        ax[1].stackplot(time, distribution_stages_malmo.values(),
             labels=distribution_stages_malmo.keys())
        ax[1].legend(loc='upper left')
        ax[1].set_title('Population of Malmo')
        ax[1].set_ylabel('Number of people')
        plt.tight_layout()
        if iter_ < 10:
            plt.savefig("cities"+ f"00{iter_}.png", format="png")
            plt.close()
        elif iter_ < 100:
            plt.savefig("cities"+ f"0{iter_}.png", format="png")
            plt.close()
        else:
            plt.savefig("cities"+ f"{iter_}.png", format="png")
            plt.close()

        iter_+=1

    # get results
    S_history_cph, I_history_cph, R_history_cph, D_history_cph = Copenhaguen.get_histories()
    S_history_malmo, I_history_malmo, R_history_malmo, D_history_malmo = Malmo.get_histories()

    """
    plt.plot(S_history, label="Susceptible")
    plt.plot(I_history, label="Infected")
    plt.plot(R_history, label="Recovered")
    plt.plot(D_history, label="Dead")
    plt.legend()
    plt.show()
    """

    time = list(range(len(S_history_cph)))
    distribution_stages_cph = {
    'Susceptible': S_history_cph,
    'Infected': I_history_cph,
    'Recovered': R_history_cph,
    'Dead': D_history_cph
    }
    distribution_stages_malmo = {
    'Susceptible': S_history_malmo,
    'Infected': I_history_malmo,
    'Recovered': R_history_malmo,
    'Dead': D_history_malmo
    }

    fig, ax = plt.subplots(2,1)

    ax[0].stackplot(time, distribution_stages_cph.values(),
             labels=distribution_stages_cph.keys())
    ax[0].legend(loc='upper left')
    ax[0].set_title('Population of Copenhaguen')
    ax[0].set_ylabel('Number of people')

    ax[1].stackplot(time, distribution_stages_malmo.values(),
             labels=distribution_stages_malmo.keys())
    ax[1].legend(loc='upper left')
    ax[1].set_title('Population of Malmo')
    ax[1].set_ylabel('Number of people')
    plt.tight_layout()
    plt.show()

    frames = []
    imgs = glob.glob("cities"+"*.png")
    imgs = sorted(imgs)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)
 
    # Save into a GIF file that loops forever
    frames[0].save('cities_90.gif', format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=90, loop=0)
    frames[0].save('cities_60.gif', format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=60, loop=0)
    frames[0].save('cities_30.gif', format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=30, loop=0)

    # remove the images
    for img in imgs:
        os.remove(img)