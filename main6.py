"""
Exercise 7:

1. Implement simulated annealing for the travelling salesman. As
proposal, permute two random stations on the route.
"""

import pandas as pd
import numpy as np
from math import sqrt, log
import math, random
import matplotlib.pyplot as plt

class Simulated_Annealing:
    def __init__(self, cost_matrix, temp_type, T=1, max_iter=100, T_threshold=0.001):
        assert temp_type in ["log", "sqrt"], "temperature function type not implemented. Try <sqrt>, <log>"
        self.cost_matrix = cost_matrix
        self.num_nodes = len(cost_matrix)
        self.nodes = range(self.num_nodes)
        self.temp_type = temp_type
        self.T = T
        self.max_iter = max_iter
        self.T_threshold = T_threshold

    def compute_segment_cost(self, node_start, node_end):
        assert (node_start in self.nodes) and (node_end in self.nodes), 'selected nodes are not within the node list'
        return self.cost_matrix[node_start,node_end]

    def compute_cost(self, path):
        path_cost = 0
        for idx in range((len(path)-1)):
            path_cost += self.compute_segment_cost(path[idx], path[idx+1])
        return path_cost

    def p_accept(self, path_cost):
        """
        Probability of accepting if the candidate is worse than current.
        Depends on the current temperature and difference between candidate and current.
        """
        return math.exp(-abs(path_cost - self.path_cost) / self.T)

    def accept(self, path):
        """
        Accept with probability 1 if candidate is better than current.
        Accept with probabilty p_accept(..) if candidate is worse.
        """
        path_cost = self.compute_cost(path)
        if path_cost < self.path_cost:
            self.path_cost, self.path = path_cost, path
        else:
            if np.random.uniform(low=0.0, high=1.0) < self.p_accept(path_cost):
                self.path_cost, self.path = path_cost, path

    def update_T(self, iter_):
        if self.temp_type == "sqrt":
            self.T = 1/sqrt(1+iter_)

        elif self.temp_type == "log":
            self.T = -log(1+iter_)

        else:
            self.T = self.T**0.99

    def initial_state(self):
        """
        Generate an initial solution. Pick a random node,
        the the path of the salesman is from one node to the closest
        other non-visited node. Greedy initialization.
        """
        path, path_cost = [], 0
        curr_node = random.randint(0, self.num_nodes - 1)
        path.append(curr_node)

        while len(path) < self.num_nodes:
            best_candidate, cost_best_candidate = None, np.inf
            for candidate in self.nodes:
                if (candidate != curr_node) and (candidate not in path):
                    candidate_cost = self.compute_segment_cost(curr_node, candidate)
                    if candidate_cost < cost_best_candidate:
                        cost_best_candidate = candidate_cost
                        best_candidate = candidate
            path.append(best_candidate)
            path_cost += cost_best_candidate

        self.path = np.array(path)
        self.path_cost = path_cost

    def run(self):
        # annealing algorithm
        cost_list, iter_ = [], 0
        self.initial_state() # initial path, path_cost
        cost_list.append(self.path_cost)
        while (self.T >= self.T_threshold) and (iter_ < self.max_iter):
            path = list(self.path)
            delta_idx= random.randint(2, self.num_nodes-1)
            idx = random.randint(0, self.num_nodes-1)
            path[idx:(idx+delta_idx)] = reversed(path[idx:(idx+delta_idx)])
            self.accept(path)

            cost_list.append(self.path_cost)
            iter_ += 1
            self.update_T(iter_)

        return self.path,self.path_cost,cost_list


if __name__ == "__main__":

    fname = "cost.csv"
    cost_matrix = pd.read_csv(fname, dtype='int').to_numpy()

    for temp_type in ['sqrt', 'log']:
        simulation = Simulated_Annealing(cost_matrix=cost_matrix, temp_type=temp_type, max_iter=1000)
        path, path_cost, cost_list = simulation.run()
    
        plt.plot(cost_list, label=temp_type)
    plt.show()
    
