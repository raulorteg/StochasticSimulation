"""
Exercise 7
Simulated annealing for the travelling salesman.
@author Raul Ortega
Created 12/06/2021
"""

import pandas as pd
import numpy as np
from math import sqrt, log, cos, sin
import math, random
import matplotlib.pyplot as plt

from PIL import Image
import glob

class Simulated_Annealing:
    def __init__(self, cost_matrix=[], temp_type="sqrt", T=1, max_iter=100, T_threshold=0.001, init_method="greedy", alpha=0.99, coordinates=False):
        assert temp_type in ["log", "sqrt", "exp"], "temperature function type not implemented. Try <sqrt>, <log>, <exp>"
        assert init_method in ["greedy", "random"], "initialization method not implemented. Try <greedy>, <random>"
        
        self.cost_matrix = cost_matrix
        if coordinates:
            self.compute_cost_matrix(coordinates)

        self.num_nodes = len(cost_matrix)
        self.nodes = range(self.num_nodes)
        self.temp_type = temp_type
        self.T = T
        self.max_iter = max_iter
        self.T_threshold = T_threshold
        self.init_method = init_method
        self.alpha = alpha

    def compute_cost_matrix(self, coordinates):
        """
        Fill in the matrix of distances, Compute the distance (euclidean) 
        between all pairs of cities.
        Note: Since distance from A to B is the same as B to A, the number of operations
        done here could be cut in half, but given the low number of cities its ok to repeat calculations.
        """
        for city in coordinates:
            for next_city in coordinates:
                distance = self.euclidean_distance(city, next_city)
                self.cost_matrix.append(distance)

        self.cost_matrix = np.array(self.cost_matrix).reshape((self.num_nodes, self.num_nodes))

    def euclidean_distance(self, origin, dest):
        x0, y0 = origin
        x1, y1 = dest
        return math.sqrt((x0-x1)**2 + (y0-y1)**2)


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
            self.T = -log(iter_/self.max_iter)

        else: # exponential
            self.T = self.T**self.alpha

    def initialize(self):
        """
        Generate an initial solution. Pick a random node,
        the the path of the salesman is from one node to the closest
        other non-visited node. Greedy initialization.

        If init method is random, then get a random permutation of the
        nodes to be the path and compute its cost
        """
        if self.init_method == "greedy":
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
        
        else: # random initialization
            self.path = list(self.nodes)
            random.shuffle(self.path)
            self.path_cost = self.compute_cost(self.path)


    def run(self, map=None):
        # annealing algorithm
        cost_list, iter_ = [], 0
        self.initialize() # initial path, path_cost
        cost_list.append(self.path_cost)
        while (self.T >= self.T_threshold) and (iter_ < self.max_iter):
            path = list(self.path)
            delta_idx= random.randint(2, self.num_nodes-1)
            idx = random.randint(0, self.num_nodes-1)
            path[idx:(idx+delta_idx)] = reversed(path[idx:(idx+delta_idx)])
            self.accept(path)

            cost_list.append(self.path_cost)

            if map: # to save plots for animation
                map.save_progress(path=self.path, iter_=iter_)

            iter_ += 1
            self.update_T(iter_)

        if map:
            map.animate()
        return self.path, self.path_cost, cost_list

class Circle_Map:
    def __init__(self, num_cities=20, radius=200, equispaced=True):
        self.num_cities = num_cities
        self.radius = radius
        self.equispaced = equispaced
        self.cities = []
        self.distance_matrix = []
        self.create_map()
        self.compute_distances()

    def create_map(self):
        if self.equispaced:
            delta_theta = 2*np.pi/self.num_cities
            thetas = np.cumsum(delta_theta*np.ones(self.num_cities))

            for theta in thetas:
                x, y = self.radius*math.cos(theta), self.radius*math.sin(theta)
                self.cities.append((x,y))

    def compute_distances(self):
        """
        Fill in the matrix of distances, Compute the distance (euclidean) 
        between all pairs of cities.
        Note: Since distance from A to B is the same as B to A, the number of operations
        done here could be cut in half, but given the low number of cities its ok to repeat calculations.
        """
        for city in self.cities:
            for next_city in self.cities:
                distance = self.euclidean_distance(city, next_city)
                self.distance_matrix.append(distance)

        self.distance_matrix = np.array(self.distance_matrix).reshape((self.num_cities, self.num_cities))

    def euclidean_distance(self, origin, dest):
        x0, y0 = origin
        x1, y1 = dest
        return math.sqrt((x0-x1)**2 + (y0-y1)**2)

    def plot(self, path=None):
        """
        Given a path plot the route of the salesman and the cities.
        """
        coord_path_x, coord_path_y = [], []
        for city in self.cities:
            plt.plot(*city, "xr") # plot the cities

        if path:
            past_city = self.cities[path[0]]
            plt.plot(*past_city, "og") # plot the starting point
            for i in range(1,self.num_cities):
                city_idx = path[i]
                coord_path_x.append([past_city[0], self.cities[city_idx][0]])
                coord_path_y.append([past_city[1], self.cities[city_idx][1]])
                past_city = self.cities[city_idx]

            for idx in range(len(coord_path_x)):
                plt.plot(coord_path_x[idx], coord_path_y[idx], "b")

        plt.title("Route of the Salesman on Circle Map")
        plt.show()

    def get_cities(self): # if want to work with the coordinates of cities
        return self.cities

    def get_distances(self): # if want to work with the computed distances
        return self.distance_matrix

class Random_Map:
    def __init__(self, max_x=100, max_y=100, num_cities=20, dir_plots="plots"):
        self.max_x = max_x
        self.max_y = max_y
        self.num_cities = num_cities
        self.cities = []
        self.distance_matrix = []
        self.create_map()
        self.compute_distances()
        self.dir_plots = dir_plots

    def create_map(self):
        u1 = [np.random.randint(0,self.max_x) for _ in range(self.num_cities)]
        u2 = [np.random.randint(0,self.max_y) for _ in range(self.num_cities)]
        self.cities = [city for city in zip(u1, u2)]

    def compute_distances(self):
        """
        Fill in the matrix of distances, Compute the distance (euclidean) 
        between all pairs of cities.
        Note: Since distance from A to B is the same as B to A, the number of operations
        done here could be cut in half, but given the low number of cities its ok to repeat calculations.
        """
        for city in self.cities:
            for next_city in self.cities:
                distance = self.euclidean_distance(city, next_city)
                self.distance_matrix.append(distance)

        self.distance_matrix = np.array(self.distance_matrix).reshape((self.num_cities, self.num_cities))

    def euclidean_distance(self, origin, dest):
        x0, y0 = origin
        x1, y1 = dest
        return math.sqrt((x0-x1)**2 + (y0-y1)**2)

    def plot(self, path=None):
        """
        Given a path plot the route of the salesman and the cities.
        """
        coord_path_x, coord_path_y = [], []
        for city in self.cities:
            plt.plot(*city, "xr") # plot the cities

        if path:
            past_city = self.cities[path[0]]
            plt.plot(*past_city, "og") # plot the starting point
            for i in range(1,self.num_cities):
                city_idx = path[i]
                coord_path_x.append([past_city[0], self.cities[city_idx][0]])
                coord_path_y.append([past_city[1], self.cities[city_idx][1]])
                past_city = self.cities[city_idx]

            for idx in range(len(coord_path_x)):
                plt.plot(coord_path_x[idx], coord_path_y[idx], "b")

        plt.title("Route of the Salesman on Random Map")
        plt.show()

    def save_progress(self, iter_, path=None):
        """
        Given a path plot the route of the salesman and the cities.
        """
        coord_path_x, coord_path_y = [], []
        for city in self.cities:
            plt.plot(*city, "xr") # plot the cities

        if path:
            past_city = self.cities[path[0]]
            plt.plot(*past_city, "og") # plot the starting point
            for i in range(1,self.num_cities):
                city_idx = path[i]
                coord_path_x.append([past_city[0], self.cities[city_idx][0]])
                coord_path_y.append([past_city[1], self.cities[city_idx][1]])
                past_city = self.cities[city_idx]

            for idx in range(len(coord_path_x)):
                plt.plot(coord_path_x[idx], coord_path_y[idx], "b")

            plt.title(f"Route of the Salesman on Random Map, iter={iter_}")
            if iter_ < 10:
                plt.savefig("plots/"+ f"0{iter_}.png", format="png")
                plt.close()
            else:
                plt.savefig("plots/"+ f"{iter_}.png", format="png")
                plt.close()

    def animate(self):
        # Create the frames
        frames = []
        imgs = glob.glob("plots/"+"*.png")
        imgs = sorted(imgs)
        for i in imgs:
            new_frame = Image.open(i)
            frames.append(new_frame)
 
        # Save into a GIF file that loops forever
        frames[0].save("plots"+'/png_to_gif.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=30, loop=0)

    def get_cities(self): # if want to work with the coordinates of cities
        return self.cities

    def get_distances(self): # if want to work with the computed distances
        return self.distance_matrix

if __name__ == "__main__":

    fname = "cost.csv"
    cost_matrix = pd.read_csv(fname, dtype='int').to_numpy()
    """
    # run variations of the method to compare initializations and T laws
    for init_method in ['greedy', 'random']:
        for temp_type in ['sqrt', 'log', 'exp']:
            simulation = Simulated_Annealing(cost_matrix=cost_matrix, temp_type=temp_type, init_method=init_method, max_iter=1000)
            path, path_cost, cost_list = simulation.run()
    
            plt.plot(cost_list, label=f"({init_method}, {temp_type})")
    plt.legend()
    plt.ylabel("Cost of path (distance)")
    plt.xlabel("Iterations")
    plt.show()

    # Lets test it in the Circle map
    circle_map = Circle_Map(num_cities=20)
    cost_matrix = circle_map.get_distances()

    simulation = Simulated_Annealing(cost_matrix=cost_matrix, temp_type='sqrt', init_method='random', max_iter=1000)
    path, path_cost, cost_list = simulation.run()
    circle_map.plot(path)
    """
    # Lets test it in the Random map
    random_map = Random_Map(num_cities=20)
    cost_matrix = random_map.get_distances()

    simulation = Simulated_Annealing(cost_matrix=cost_matrix, temp_type='sqrt', init_method='random', max_iter=1000)
    path, path_cost, cost_list = simulation.run(map=random_map)
    random_map.plot(path)