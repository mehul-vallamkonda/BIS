import numpy as np

# Problem Definition
def fitness_function(x):
    return x**2 - 4*x + 4

# Parameters
grid_size = 10  # 10x10 grid
num_iterations = 100
search_space_min = -10
search_space_max = 10

# Initialize Population
grid = np.random.uniform(search_space_min, search_space_max, (grid_size, grid_size))
print("Initial Grid")
print(grid)
# Neighborhood Definition (Moore Neighborhood)
def get_neighbors(grid, i, j):
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if not (di == 0 and dj == 0):  # Exclude the cell itself
                ni, nj = i + di, j + dj
                if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                    neighbors.append(grid[ni, nj])
    return neighbors

# Iterative Update Process
best_solution = None
best_fitness = float('inf')

for iteration in range(num_iterations):
    new_grid = grid.copy()
    for i in range(grid_size):
        for j in range(grid_size):
            neighbors = get_neighbors(grid, i, j)
            if neighbors:
                # Update state: average of neighbors
                new_grid[i, j] = np.mean(neighbors)
            # Evaluate fitness
            fitness = fitness_function(new_grid[i, j])
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = new_grid[i, j]
                print('Best Solution : ',best_solution,'Best Fitness:', best_fitness,'Iteration:',iteration)
    grid = new_grid

# Output Best Solution
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")
