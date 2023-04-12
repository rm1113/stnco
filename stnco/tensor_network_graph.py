from graph_basic import Graph
import math as m


class TensorNetworkGraphBasic(Graph):
    """
    The basic class for Tensor network contraction search classes based on span tree search in the graph.
    """
    def __init__(self):
        # A tuple containing the possible values for the contract path method
        self._methods = ('auto', 'kruskal', 'prim')
        # A list containing the start node index and its value for Prim algorithm
        self.start_node = [-1, None]
        super(TensorNetworkGraphBasic, self).__init__()

    def contract_path(self, method='auto', **kwargs):
        """
        Class method that contracts a path in the graph using the specified method
        :param method: str, optional. The spanning tree search method. Could be prim, kruskal or auto.
        :param kwargs: optional. Keyword arguments to passing throw span tree search method.
        :return: List(tuple), contraction path in a linear format
        """
        # If the method is set to auto, choose either kruskal or prim based on the size of the graph
        if method == 'auto':
            e, v = self._n_edges, self._n_vertices
            method = 'kruskal' if (e + v * m.log(v)) > e * m.log(e) else 'prim'

        # If the method is set to prim, call find_minimal_span with the start node
        if method == 'prim':
            path = self.find_minimal_span(method=method, start=self.start_node[0], **kwargs)

        # If the method is set to kruskal, call find_minimal_span without the start node
        elif method == 'kruskal':
            path = self.find_minimal_span(method=method, **kwargs)

        # Convert the path to a linear form
        return self._ssa_to_linear(path, self._n_vertices)

    @staticmethod
    def _ssa_to_linear(order, number_of_elements):
        """
        Method that converts the order of a path to a linear form
        :param order: path in the static single assignment form
        :param number_of_elements: the total number of elements in the path
        :return: the path converted to the linear form
        """
        cur_len = number_of_elements    # Store the number of elements
        indexes = [*range(number_of_elements)]    # Create a list of indices from 0 to number_of_elements

        # Create a list of groups, where each element belongs to a group
        groups = [*range(number_of_elements)]

        # A dictionary that maps each group to a set of elements
        groups_to_elements = {i: set([i]) for i in range(number_of_elements)}

        new_group = number_of_elements - 1  # Initialize the new group
        path = []                           # Create an empty path

        # Loop through the pairs in the initial path
        for pair in order:
            x, y = groups[pair[0]], groups[pair[1]]     # Get the group of each element in the pair
            path.append((indexes[x], indexes[y]))       # Add the pair to the path

            new_group += 1                              # Increment id of the new group

            # Merge the sets of elements in the two groups
            groups_to_elements[new_group] = groups_to_elements[x] | groups_to_elements[y]

            # Update the groups of the elements in the merged group
            for i in groups_to_elements[new_group]:
                groups[i] = new_group

            indexes.append(cur_len)     # Insert merged group as the new item in the end of list

            x, y = sorted([x, y])    # Sort x and y in the ascending  order
            for i in range(x, len(indexes)):    # For each element to the right of x
                if i > y:
                    indexes[i] -= 2     # Shifts element 2 positions left If the element to the right of y
                else:
                    indexes[i] -= 1     # Shifts element 1 positions left If the element between x and y
            cur_len -= 1                # Decrease the number of active elements.
        return path


class MaximumSpanTNG(TensorNetworkGraphBasic):
    """
    Class for building a Tensor Network Graph for the optimal
     tensor network contraction search using Maximum Spanning Tree technique.
    Arguments:
    isets -- List of tensor indices sets
    output -- List of output indices
    idx_sizes -- List of index sizes
    The arguments are compatible for opt_einsum custom optimizer.
    """
    def __init__(self, isets, output, idx_sizes):
        """
        Initialize MinimalCETNG with tensor indices sets, output indices, and index sizes.
        """
        # Call the constructor of the parent class
        super(MaximumSpanTNG, self).__init__()

        # Initialize a dictionary to store the mapping from dimensions to tensors
        i_to_tensor = dict()

        # Set the start node to None and the most heavy index to -1
        self.start_node = [None, -1]

        for k, tensor_shape in enumerate(isets):        # Iterate over the shapes of all tensors
            for dim in tensor_shape:                    # For each dimension of the tensor

                # Create an empty list in the dictionary for the current dimension if it doesn't already exist
                i_to_tensor.setdefault(dim, [])
                # Append the current tensor's index to the list in the dictionary for the current dimension
                i_to_tensor[dim].append(k)
                # If the current dimension's size is larger than the most heavy index, update the start node
                if idx_sizes[dim] > self.start_node[1]:
                    self.start_node = [k, idx_sizes[dim]]
            self._add_node()    # Add a new node to the graph

        for i in i_to_tensor.keys():    # For each dimension in the mapping from dimensions to tensors
            if i not in output:         # If the current dimension is not in the output indices
                a, b = i_to_tensor[i]   # Get the two tensors sharing the current dimension
                # Add an edge between the two tensors with weight equals to shared index
                # Invert the weight to find the maximal ST using an algorithm for the minimal ST
                self._add_edge(from_=a, to_=b, weight_=-idx_sizes[i])


class MinimalCETNG(TensorNetworkGraphBasic):
    """
    Class for building a Tensor Network Graph for the optimal
     tensor network contraction search using Minimal CE (computational expense) technique.
    Arguments:
    isets -- List of tensor indices sets
    output -- List of output indices
    idx_sizes -- List of index sizes
    The arguments are compatible for opt_einsum custom optimizer.
    """
    def __init__(self, isets, output, idx_sizes):
        """
        Initialize MinimalCETNG with tensor indices sets, output indices, and index sizes.
        """
        # Initialize TensorNetworkGraphBasic parent class
        super(MinimalCETNG, self).__init__()

        # Set the start node as the node index with the smallest tensor size
        self.start_node = [None, float('inf')]

        i_to_tensor = dict()        # Create a dictionary to map indices to the tensors they belong to
        tensor_sizes = [1 for _ in range(len(isets))]   # Initialize size of every tensor as 1

        # Calculate the product of indices for each tensor
        for k, tensor in enumerate(isets):          # For each tensor
            for dim in tensor:                      # For each tensor dimension
                i_to_tensor.setdefault(dim, [])     # Initialize the empty list of tensor sharing given index
                i_to_tensor[dim].append(k)          # Add the tensor to the list of tensors sharing the index
                tensor_sizes[k] *= idx_sizes[dim]   # Calculate the product of all indices of the tensor

            # Check if the current tensor is smaller than the current start node
            if tensor_sizes[k] < self.start_node[1]:
                self.start_node = [k, tensor_sizes[k]]
            self._add_node()    # Add node to the graph

        # For each index that is not in the output list, add an edge between the two tensors it belongs to
        for i in i_to_tensor.keys():
            if i not in output:
                a, b = i_to_tensor[i]
                w = tensor_sizes[a] * tensor_sizes[b] // idx_sizes[i]   # CE of the contraction of the pair (a,b)
                self._add_edge(from_=a, to_=b, weight_=w)               # Add the edge with weight equal ot CE


if __name__ == "__main__":
    # gr = TensorNetworkGraph(None, [2, 3, 4], None, [3, 5, 7, 8], None, [4, 5, 6], None, [6, 7])
    # gr = TensorNetworkGraph([set('abd'), set('ac'), set('bdc')], {'a': 1, 'b':2, 'c':3, 'd':4})
    gr = MinimalCETNG(isets=[set('sjr'), set('ijkm'), set('rkd'), set('dm')],
                      output=set('si'),
                      idx_sizes={'s': 2, 'j': 3, 'i': 8, 'r': 4, 'k': 5, 'm': 7, 'd': 6})
    print(gr)
    print(gr.contract_path(method='prim'))
