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

def compute_loss(pred, target):
    return np.mean((pred - target) ** 2)

def train_step(gnn, input_nodes, target_nodes, edges, lr=0.01):
    pred = gnn.forward(input_nodes, edges)

    error = pred - target_nodes
    loss = np.mean(error ** 2)

    dW_self = 2 * np.mean(error * gnn.last_self)
    dW_neigh = 2 * np.mean(error * gnn.last_neigh)

    gnn.W_self -= lr * dW_self
    gnn.W_neigh -= lr * dW_neigh

    return loss

class SimpleGNN:
    def __init__(self):
        self.W_self = np.random.randn()
        self.W_neigh = np.random.randn()

    def forward(self, node_features, edges):
        num_nodes = len(node_features)

        neighbor_sum = np.zeros_like(node_features)
        neighbor_count = np.zeros(num_nodes)

        for src, dst in edges:
            neighbor_sum[dst] += node_features[src]
            neighbor_count[dst] += 1

        neighbor_count[neighbor_count == 0] = 1
        neighbor_mean = neighbor_sum / neighbor_count

        self.last_self = node_features
        self.last_neigh = neighbor_mean

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

    physics_seq = generate_sequence_physics(grid, steps=20)

    for epoch in range(200):
        total_loss = 0

        for t in range(len(physics_seq) - 1):
            input_nodes = grid_to_nodes(physics_seq[t])
            target_nodes = grid_to_nodes(physics_seq[t + 1])

            loss = train_step(gnn, input_nodes, target_nodes, edges)
            total_loss += loss

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.6f}")

    print("\nLearned Weights:")
    print("W_self:", gnn.W_self)
    print("W_neigh:", gnn.W_neigh)