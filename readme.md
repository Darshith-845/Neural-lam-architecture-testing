# Neural-LAM Mini Prototype 🚀

## Overview

This project is a **miniature implementation of a Graph Neural Network (GNN)** designed to simulate physical dynamics, inspired by the **Neural-LAM architecture** used in neural weather prediction.

The goal is to:

* Understand how GNNs can model physical systems
* Recreate diffusion-like dynamics using message passing
* Build a foundation for **probabilistic weather forecasting models**

---

## 🧠 Core Idea

We simulate a 2D grid where each cell represents a physical quantity (like temperature).

Two systems are implemented:

1. **Physics-based simulator**

   * Uses a diffusion equation
   * Represents real-world physical evolution

2. **GNN-based simulator**

   * Learns dynamics using message passing
   * Approximates the physics without explicit equations

---

## 🏗️ Project Structure

```
Neural-lam-architecture-testing/
│
├── data/
│   └── synthetic.py        # Physics-based data generation
│
├── models/
│   └── gnn.py              # GNN implementation + training
│
└── README.md
```

---

## ⚙️ How It Works

### 1. Physics Simulation

We use a diffusion-like update rule:

```
new_value = current + 0.1 * (neighbors - 4 * current)
```

This simulates:

* Heat diffusion
* Smoothing over space
* Local interactions

---

### 2. Graph Representation

* Each grid cell → **node**
* Each neighbor connection → **edge**

Edges connect:

* Up
* Down
* Left
* Right

---

### 3. GNN Message Passing

Each node updates using:

```
new = W_self * current + W_neigh * neighbor_mean
```

This is the learned equivalent of the physics rule.

---

### 4. Training Objective

We train the GNN to minimize:

```
MSE(predicted_next_state, true_next_state)
```

---

### 5. Multi-Step Rollout

After training, we test:

```
Model prediction → fed back as input → repeated
```

This simulates **time evolution**.

---

## 📈 Results

### Learned Weights

```
W_self  ≈ 0.59
W_neigh ≈ 0.40
```

These closely match the true physics equation:

```
W_self  ≈ 0.6
W_neigh ≈ 0.4
```

---

### Rollout Stability

```
Step 0  → Loss: 0.000000
Step 5  → Loss: ~0.0015
Step 10 → Loss: ~0.0018
```

✅ Stable predictions
✅ No divergence
✅ Accurate long-term behavior

---

## 🧩 Key Concepts Learned

* Graph Neural Networks (GNNs)
* Message Passing
* Diffusion Processes
* Dynamical Systems
* Auto-regressive Rollouts
* Error Accumulation

---

## ⚠️ Challenges Faced & Solutions

### 1. Import Errors (matplotlib issue)

**Problem:**

```
ImportError: cannot import name '_c_internal_utils'
```

**Solution:**

* Removed unnecessary matplotlib dependency
* Avoided circular imports

---

### 2. Incorrect Training with Softmax

**Problem:**

* Used softmax on weights
* Gradients became incorrect

**Solution:**

* Removed softmax during training
* Used raw weights for proper gradient updates

---

### 3. Model Instability

**Problem:**

* Weights diverging during training

**Solution:**

```
np.clip(weights, -2, 2)
```

---

### 4. Overfitting to Single Sequence

**Problem:**

* Model memorized one trajectory

**Solution:**

* Trained on multiple random initial grids

---

### 5. Rollout Error Accumulation

**Problem:**

* Predictions degrade over time

**Solution:**

* Evaluated multi-step rollout
* Verified stability instead of just one-step accuracy

---

## 🔍 What This Proves

This project demonstrates that:

* GNNs can **learn physical laws**
* Message passing approximates **spatial interactions**
* Learned models can simulate **time-evolving systems**

---

## 🚀 Next Steps

* Add **probabilistic outputs (mean + uncertainty)**
* Implement **negative log-likelihood loss**
* Extend to **multi-feature nodes**
* Move to **PyTorch for scalability**
* Build toward full **Neural-LAM-style architecture**

---

## 🎯 Goal

To evolve this into a **probabilistic neural weather prediction model** aligned with the Neural-LAM project for GSoC.

---

## 🧑‍💻 Author

Darshith Shetty

---

## ⭐ Final Note

This is not just a toy model — it is a **conceptual bridge** between:

* Physics-based simulation
* Graph Neural Networks
* Probabilistic forecasting systems

---
