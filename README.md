# stnco

# Tensor Network Contraction Optimization
This project provides an optimized approach to find tensor network contraction paths using graph-based algorithms. It contains three Python files:

1. graph_basic.py
2. tensor_network_graph.py
3. tensor_net_opt_einsum.py

## Overview
### graph_basic.py
This file contains the implementation of a basic undirected graph data structure. The Graph class provides methods for adding nodes, adding edges, and finding the minimum spanning tree of the graph using either Kruskal's or Prim's algorithms.

### tensor_network_graph.py
This file contains the implementation of the TensorNetworkGraphBasic class, which inherits from the Graph class. It is a base class for other Tensor Network Graph classes, providing a method to find a contraction path in the graph using the specified method (Kruskal's or Prim's algorithms). Two subclasses are defined: MaximumSpanTNG and MinimalCETNG. These classes build a Tensor Network Graph based on tensor indices and output indices, with edge weights determined by the shared dimension size or the computational expense, respectively.

### tensor_net_opt_einsum.py
This file contains the implementation of two custom optimizers for the opt_einsum library, `MaxSpanOptimizer` and `MinCEOptimizer`. These optimizers work with the `opt_einsum.contract()` function and find the contraction path by utilizing the `MaximumSpanTNG` and `MinimalCETNG` classes from tensor_network_graph.py, respectively.

## Usage
The custom optimizers can be used with the opt_einsum library to find the optimal tensor network contraction paths. Here's an example of how to use the custom optimizers:
```
from tensor_net_opt_einsum import MaxSpanOptimizer, MinCEOptimizer
import numpy as np
import opt_einsum as oe

# Define tensors
A = np.random.rand(2, 3, 4)
B = np.random.rand(4, 3, 5)
C = np.random.rand(5, 2, 6)

# Create an optimizer instance
optimizer = MaxSpanOptimizer()

# Perform the contraction using the custom optimizer
result = oe.contract('ijk, jkl, lmi -> im', A, B, C, backend='numpy', optimizer=optimizer)
```
You can switch between the `MaxSpanOptimizer` and the `MinCEOptimizer` depending on your requirements.
