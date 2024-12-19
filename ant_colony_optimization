import numpy as np
import random

class AntColony:
    def __init__(self, distance_matrix, num_ants, num_iterations, alpha, beta, rho, q):
        self.distance_matrix = distance_matrix
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.num_cities = distance_matrix.shape[0]
        self.pheromones = np.ones((self.num_cities, self.num_cities))  # Initial pheromone levels

    def run(self):
        best_length = float('inf')
        best_tour = None

        for iteration in range(self.num_iterations):
            all_tours = []
            all_lengths = []

            for _ in range(self.num_ants):
                tour = self.construct_tour()
                length = self.calculate_tour_length(tour)
                all_tours.append(tour)
                all_lengths.append(length)

                # Update best tour
                if length < best_length:
                    best_length = length
                    best_tour = tour

            self.update_pheromones(all_tours, all_lengths)

        return best_tour, best_length

    def construct_tour(self):
        tour = []
        visited = set()
        current_city = random.randint(0, self.num_cities - 1)
        tour.append(current_city)
        visited.add(current_city)

        while len(tour) < self.num_cities:
            probabilities = self.calculate_probabilities(current_city, visited)
            next_city = self.roulette_wheel_selection(probabilities)
            tour.append(next_city)
            visited.add(next_city)
            current_city = next_city

        tour.append(tour[0])  # Return to the starting city
        return tour

    def calculate_probabilities(self, current_city, visited):
        pheromone = self.pheromones[current_city]
        distances = self.distance_matrix[current_city]

        # Calculate probabilities based on pheromone and heuristic information
        probabilities = []
        for city in range(self.num_cities):
            if city not in visited:
                prob = (pheromone[city] ** self.alpha) * ((1.0 / distances[city]) ** self.beta)
                probabilities.append(prob)
            else:
                probabilities.append(0)

        # Normalize probabilities
        total = sum(probabilities)
        return [p / total if total > 0 else 0 for p in probabilities]

    def roulette_wheel_selection(self, probabilities):
        r = random.random()
        cumulative_sum = 0
        for i, prob in enumerate(probabilities):
            cumulative_sum += prob
            if r < cumulative_sum:
                return i
        return len(probabilities) - 1  # Return the last index if no other selected

    def update_pheromones(self, all_tours, all_lengths):
        # Evaporate pheromones
        self.pheromones *= (1 - self.rho)

        # Deposit new pheromones
        for tour, length in zip(all_tours, all_lengths):
            for i in range(len(tour) - 1):
                self.pheromones[tour[i], tour[i + 1]] += self.q / length

    def calculate_tour_length(self, tour):
        length = 0
        for i in range(len(tour) - 1):
            length += self.distance_matrix[tour[i], tour[i + 1]]
        return length

# Example usage
if __name__ == "__main__":
    # Distance matrix for a TSP problem (symmetric)
    distance_matrix = np.array([
        [0, 4, 15, 1],
        [4, 0, 5, 8],
        [15, 5, 0, 4],
        [1, 8, 4, 0]
    ])

    num_ants = 10
    num_iterations = 100
    alpha = 1.0       # Influence of pheromone
    beta = 2.0        # Influence of heuristic
    rho = 0.5         # Pheromone evaporation rate
    q = 100           # Pheromone intensity

    aco = AntColony(distance_matrix, num_ants, num_iterations, alpha, beta, rho, q)
    best_tour, best_length = aco.run()

    print("Best Tour:", best_tour)
    print("Best Length:", best_length)
