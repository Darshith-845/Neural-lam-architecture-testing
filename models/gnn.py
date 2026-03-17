import numpy as np

from data.synthetic import (
    generate_initial_grid,
    generate_sequence_physics,
    GRID_SIZE
)

def get_node_id(i, j):
    return i * GRID_SIZE + j

def grid_to_nodes(grid):
    return grid.flatten()

def nodes_to_grid(nodes):
    return nodes.reshape(GRID_SIZE, GRID_SIZE)

def build_edges():
    edges = []

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            current = get_node_id(i, j)

            if i > 0:
                edges.append((current, get_node_id(i - 1, j)))
            if i < GRID_SIZE - 1:
                edges.append((current, get_node_id(i + 1, j)))
            if j > 0:
                edges.append((current, get_node_id(i, j - 1)))
            if j < GRID_SIZE - 1:
                edges.append((current, get_node_id(i, j + 1)))

    return edges

class SimpleGNN:
    def __init__(self):
        self.W_self = 0.6
        self.W_neigh = 0.4

    def forward(self, node_features, edges):
        num_nodes = len(node_features)

        neighbor_sum = np.zeros_like(node_features)
        neighbor_count = np.zeros(num_nodes)

        for src, dst in edges:
            neighbor_sum[dst] += node_features[src]
            neighbor_count[dst] += 1

        neighbor_count[neighbor_count == 0] = 1
        neighbor_mean = neighbor_sum / neighbor_count

        return (
            self.W_self * node_features +
            self.W_neigh * neighbor_mean
        )

def generate_sequence_gnn(initial_grid, steps, gnn, edges):
    grids = [initial_grid]
    current_nodes = grid_to_nodes(initial_grid)

    for _ in range(steps):
        current_nodes = gnn.forward(current_nodes, edges)
        grids.append(nodes_to_grid(current_nodes))

    return grids

if __name__ == "__main__":
    grid = generate_initial_grid()

    edges = build_edges()
    gnn = SimpleGNN()

    physics_seq = generate_sequence_physics(grid, steps=5)
    gnn_seq = generate_sequence_gnn(grid, steps=5, gnn=gnn, edges=edges)

    print("Physics Output:\n", physics_seq[-1])
    print("\nGNN Output:\n", gnn_seq[-1])

    diff = np.abs(physics_seq[-1] - gnn_seq[-1]).mean()
    print("\nMean Difference:", diff)