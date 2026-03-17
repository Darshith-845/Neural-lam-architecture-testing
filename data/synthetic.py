import numpy as np
import matplotlib.pyplot as plt
GRID_SIZE = 10

def step(grid):
    return (
        grid +
        0.1 * (
            np.roll(grid, 1, axis=0) +
            np.roll(grid, -1, axis=0) +
            np.roll(grid, 1, axis=1) +
            np.roll(grid, -1, axis=1)
            - 4 * grid
        )
    )

def generate_sequence_physics(initial_grid, steps):
    grids = [initial_grid]
    current = initial_grid.copy()

    for _ in range(steps):
        current = step(current)
        grids.append(current)

    return grids

def generate_initial_grid():
    return np.random.rand(GRID_SIZE, GRID_SIZE)

def visualize(sequence):
    for t in range(len(sequence)):
        plt.imshow(sequence[t], cmap='viridis')
        plt.title(f"Step {t}")
        plt.colorbar()
        plt.pause(0.1)
        plt.clf()

# seq = generate_sequence()
# visualize(seq)

