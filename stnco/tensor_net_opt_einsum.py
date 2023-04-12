from tensor_network_graph import MaximumSpanTNG, MinimalCETNG
import opt_einsum as oe


"""
The module defines two optimizers for usage as a
custom optimizer for opt_einsum.contract(..., optimizer=CustomOptimizer)
"""

class MaxSpanOptimizer(oe.paths.PathOptimizer):
    """
    The optimizer finds the contraction path as the maximal
    spanning tree in the graph with edge weights equal to 
    the shared dimension size
    """
    def __call__(self, inputs, output, size_dict, memory_limit=None):
        tng = MaximumSpanTNG(inputs, output, size_dict)
        return tng.contract_path()


class MinCEOptimizer(oe.paths.PathOptimizer):
    """
    The optimizer finds the contraction path as the minimal
    spanninng tree in the graph with edge weights equal to 
    the computational expense of the pairwise contraction
    """
    def __call__(self, inputs, output, size_dict, memory_limit=None):
        tng = MinimalCETNG(inputs, output, size_dict)
        return tng.contract_path()
