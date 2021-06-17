import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.linalg import expm, norm
from scipy.stats import chisquare
import copy

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
        return np.array(times), counts

    def get_dist_at_fixed_time(self):
        times = sorted(self.state_at_time)
        counts = np.bincount(times)
        return times, counts

    def get_time_series(self):
        time_series_patients = []
        for patient in self.patients:
            time_series_individual_patient = []
            history = patient.get_history()

            time_checked = 0
            state = 0
            while state < 4:
                for snapshot in history:
                    
                    state, time = snapshot
                    if time >= time_checked:
                        time_series_individual_patient.append(state)
                        time_checked += 48

            time_series_individual_patient.append(4)
            time_series_patients.append(time_series_individual_patient)
        return time_series_patients

def compute_jumps(time_series_patients):
    """
    individual_time_series = [[0, 1, 1, 1, 2, 3, ..., 4], [0, 1, 1, 1, 2, 3, ..., 4], .....]
    possible transitions:
    0->1, 0->2, 0->3, 0->4
    1->2, 1->3, 1->4
    2->3, 2->4
    3->4
    """
    total_jumps = np.zeros(10) # [0->1, 0->2, 0->3, 0->4 1->2, 1->3, 1->4 2->3, 2->4 3->4]
    for series in time_series_patients:

        transitions = [[0,1], [0,2], [0,3], [0,4], [1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]
        for idx, transition in enumerate(transitions):
            for i, elem in enumerate(series):   
                try:
                    cond = (series[i] == transition[0]) and (series[i+1] == transition[1])
                    if cond:
                        total_jumps[idx] += 1
                except:
                    pass

    return total_jumps


def compute_time_spent(time_series_patients):
    """
    individual_time_series = [[0, 1, 1, 1, 2, 3, ..., 4], [0, 1, 1, 1, 2, 3, ..., 4], .....]
    """
    S = np.zeros(5) # [0,1,2,3,4]
    for series in time_series_patients:
        unique_values = np.unique(series)
        counts = np.bincount(series)
        for idx in unique_values:
            S[idx] += counts[idx]*48
    return S[:-1]

def compute_q_matrix(S, N):
    """
    Computes q-matrix given S and N as:
    q_ij = N_ij/Si for i != j
    q_ii = sum(-q_ij) with i!=j

    N = [0->1, 0->2, 0->3, 0->4 1->2, 1->3, 1->4 2->3, 2->4 3->4]
    S = [0,1,2,3]
    """

    q = np.zeros((len(S)+1,len(S)+1))

    # first compute non-diagonal part
    q[0,1] = N[0]/S[0]
    q[0,2] = N[1]/S[0]
    q[0,3] = N[2]/S[0]
    q[0,4] = N[3]/S[0]

    q[1,2] = N[4]/S[1]
    q[1,3] = N[5]/S[1]
    q[1,4] = N[6]/S[1]

    q[2,3] = N[7]/S[2]
    q[2,4] = N[8]/S[2]

    q[3,4] = N[9]/S[3]

    # lets fill in the diagonal
    q[0,0] = - (q[0,1] + q[0,2] + q[0,3] + q[0,4])
    q[1,1] = - (q[1,2] + q[1,3] + q[1,4])
    q[2,2] = - (q[2,3] + q[2,4])
    q[3,3] = -(q[3,4])

    return q

def custom_norm(q_prev, q, mode="abs"):
    if mode == "abs":
        return sum(sum(np.absolute(q_prev - q)))
    else:
        return sum(sum(np.square(q_prev - q)))

def Q_estimator(q0, N=1000, threshold=10e-3):

    # simulate
    q_prev = q0
    experiment = Experiment(q=q_prev)
    experiment.run()
    time_series_patients = experiment.get_time_series()

    # compute update
    N_ = compute_jumps(time_series_patients)
    S = compute_time_spent(time_series_patients)
    q = compute_q_matrix(S, N_)

    ctr = 0
    while (custom_norm(q_prev, q) > threshold):
        ctr += 1
        print(f"Iteration: {ctr}, norm {custom_norm(q_prev, q)}")
        # simulate
        q_prev = copy.deepcopy(q)
        experiment = Experiment(q=q_prev, N=N)
        experiment.run()
        time_series_patients = experiment.get_time_series()

        # compute update
        N_ = compute_jumps(time_series_patients)
        S = compute_time_spent(time_series_patients)
        q = compute_q_matrix(S, N_)
        print(q_prev)
        print(q)
    print("hurray")
    return q





if __name__ == "__main__":


    q = np.array([[-0.0085, 0.005, 0.0025, 0, 0.001],
        [0, -0.014, 0.005, 0.004, 0.005],[0, 0, -0.008, 0.003, 0.005],
        [0, 0, 0, -0.009, 0.009],[0, 0, 0, 0, 0]])

    # define the experiment
    N = 1000
    experiment = Experiment(q=q, N=N, target_time=30.5)
    experiment.run()

    histories = experiment.get_histories()
    times, counts = experiment.get_final_times()
    dist_target_time, counts_target_time = experiment.get_dist_at_fixed_time()
    time_series_patients = experiment.get_time_series()
    
    """
    Part 12
    """
    N = compute_jumps(time_series_patients)
    S = compute_time_spent(time_series_patients)
    print(compute_q_matrix(S, N))
    print(q)

    """
    Part 13
    """
    q_rand = np.random.uniform(size=(5,5))
    estimated_q = Q_estimator(q0=q_rand, N=100000, threshold=1e-5)
    print(estimated_q)
    print(q)





