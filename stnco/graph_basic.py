from random import randint


class EdgeNode:
    """
    The class is edge implementation for usage in Graph class
    """
    def __init__(self,
                 start: int,  # start-point of the edge
                 to: int,  # end-point of the edge
                 weight: int = 1,  # weight of the edge
                 next_edge: 'EdgeNode' = None  # pointer to next edge in the list
                 ) -> None:
        """
        :param start: int,  source node of the edge
        :param to: int, destination node of the edge
        :param weight: int, default = 1, weight of the edge
        :param next_edge: EdgeNode, default None, pointer to next edge in the graph
        """
        self.start = start
        self.to = to
        self.weight = weight
        self.next_edge = next_edge


class UnionFind:
    def __init__(self, n):
        self.parents = [i for i in range(n)]  # list of pointers to parent nodes
        self.sizes = [1] * n  # number of elements in i-th subtree
        self.n = n  # number of elements in the set

    def _find(self, x):
        if self.parents[x] == x:
            return x
        return self._find(self.parents[x])

    def union_sets(self, s1, s2):
        root1 = self._find(s1)
        root2 = self._find(s2)

        if root1 == root2: return

        if self.sizes[root1] >= self.sizes[root2]:
            self.sizes[root1] += self.sizes[root2]
            self.parents[root2] = root1
        else:
            self.sizes[root2] += self.sizes[root1]
            self.parents[root1] = root2

    def same_component(self, s1, s2):
        return self._find(s1) == self._find(s2)


class MaxInt(int):
    def __init__(self):
        super(MaxInt, self).__init__()

    def __eq__(self, other):
        return isinstance(other, MaxInt)

    def __ne__(self, other):
        """MaxInt != other"""
        return not isinstance(other, MaxInt)

    def __lt__(self, other):
        """MaxInt < other"""
        return False

    def __le__(self, other):
        """MaxInt <= other"""
        if self.__eq__(other):
            return True
        return False

    def __gt__(self, other):
        """
        MaxInt > other
        MaxInt greater than everything except other MaxInt
        """
        return not isinstance(other, MaxInt)

    def __ge__(self, other):
        return True


class Graph:
    """
    This code defines a class called "Graph" which represents a graph data structure.
    The class has several methods and attributes to allow for the representation of edges and vertices in the graph.
    """
    def __init__(self, *args, directed: bool = False) -> None:
        """
        The __init__ method sets up the basic structure for the graph,
        by initializing the _edgenodes and _degrees lists,
        as well as the _n_vertices, _n_edges, and _directed attributes.
        The _min_span_mets attribute is a dictionary
        that maps the names of the minimum spanning tree algorithms to the corresponding methods in the class.
        :param args: useless parameters to avoid errors
        :param directed: bool, default is False, if True graph will be directed,
                                otherwise the graph undirected and new edges will be duplicated
        """
        self._edgenodes = []        # adjacency List
        self._degrees = []          # degree of every vertex
        self._n_vertices = 0        # total number of vertices
        self._n_edges = 0           # total number of edges
        self._directed = directed   # is the graph directed?
        self._min_span_mets = {'kruskal': self._kruskal,
                               'prim': self._prim}

    def _add_node(self, *args):
        """
        The method is used to add new nodes
        """
        self._edgenodes.append([])
        self._degrees.append(0)
        self._n_vertices += 1

    def _add_edge(self, from_: int, to_: int, weight_: int = 1) -> None:
        """
        The method is used to add new edge.
        The method also use another class called <EdgeNode> to represent the edges in the graph.

        :param from_: int, the index of the source node
        :param to_: int, the index of the destination node
        :param weight_: int, weight of the edge, default value is 1.
        :return: None
        """
        direct_edge = EdgeNode(start=from_, to=to_, weight=weight_)
        if self._edgenodes[from_]: self._edgenodes[from_][-1].next_edge = direct_edge
        self._edgenodes[from_].append(direct_edge)

        if not self._directed:
            opposite_edge = EdgeNode(start=to_, to=from_, weight=weight_)
            if self._edgenodes[to_]: self._edgenodes[to_][-1].next_edge = opposite_edge
            self._edgenodes[to_].append(opposite_edge)

        self._n_edges += 1

    def __str__(self):
        """
        The method allows for the graph to be printed in a human-readable format,
        by iterating over the edges and nodes of the graph.

        :return: str, the resulting string representation of the graph
        """
        if not self._edgenodes:
            return "Grapth is empty"

        to_return = ""
        for i, connections in enumerate(self._edgenodes):
            to_return += f"Node {i} ->:\n"
            for w in self._edgenodes[i]:
                to_return += f"\t -> node {w.to} weight={w.weight}\n"
        return to_return

    def find_minimal_span(self, method: str ='kruskal', **kwargs):
        """
        The find_minimal_span method is used to find the minimal spanning tree of the graph,
        using one of two algorithms: 'kruskal' or 'prim'.
        The method takes an optional method parameter which specifies the algorithm to use.

        :param method: str, algorithms for minimum spanning tree search. Must be 'kruskal' or 'prim'.
        :param kwargs: Additional parameters to pass to the chosen method.
        :return: List[tuple(int, int)], return the way that leads to minimum spanning tree of the graph.
        """

        if method not in self._min_span_mets.keys():
            raise ValueError(f"'method' must be in {list(self._min_span_mets.keys())}, but {method} given")
        return self._min_span_mets[method](**kwargs)

    def _prim(self, start: int = -1,
              return_weight: bool = False) -> 'List[tuple[int, int]]':
        """
        The method is an implementation of Prim' algorithm for finding the minimal spanning tree of the graph

        :param start: int, The start node in Prim' algorithm
        :param return_weight: bool, default = False, if False only the path will be returned.
                Otherwise, the total weight of the spanning tree will be returned also.
        :return: List[tuple[int, int]] or List[tuple[int, int]], int. The minimum spanning tree in form of path.
                                                        The total weight of the tree could be added optionally.
        """
        if start < 0 or start >= self._n_vertices:      # If start index is out of graph
            start = randint(0, self._n_vertices - 1)    # Set randomly start point

        in_tree = [False] * self._n_vertices
        distances = [MaxInt()] * self._n_vertices
        parent = [-1] * self._n_vertices

        cur_node = start
        distances[cur_node] = 0
        dist = 0
        weight = 0

        path = []

        while not in_tree[cur_node]:
            in_tree[cur_node] = True
            if cur_node != start:
                path.append((parent[cur_node], cur_node))
                weight += dist
            if self._edgenodes[cur_node]:
                temp_edge = self._edgenodes[cur_node][0]
                while temp_edge is not None:
                    next_node = temp_edge.to
                    if (distances[next_node] > temp_edge.weight) & (not in_tree[next_node]):
                        distances[next_node] = temp_edge.weight
                        parent[next_node] = cur_node
                    temp_edge = temp_edge.next_edge

            dist = MaxInt()
            for i in range(self._n_vertices):
                if (not in_tree[i]) & (dist > distances[i]):
                    dist = distances[i]
                    cur_node = i
        if return_weight:
            return path, weight
        return path

    def _kruskal(self, max_span: bool = False,
                 return_weight: bool = False) -> 'List[tuple[int, int]]':
        """
        The method is an implementation of Kruskal algorithm for finding the minimal spanning tree of the graph

        :param max_span: bool, default False, if True the method finds maximal spanning tree and minimal otherwise
        :param return_weight: bool, default = False, if False only the path will be returned.
                                Otherwise, the total weight of the spanning tree will be returned also.
        :return: List[tuple[int, int]] or List[tuple[int, int]], int. The minimum spanning tree in form of path.
                                                        The total weight of the tree could be added optionally.
        """
        weight = 0

        union_find = UnionFind(self._n_vertices)
        sorted_edges = sorted([item for sublist in self._edgenodes for item in sublist],
                              key=lambda x: x.weight, reverse=max_span)
        path = []

        for i in range(2 * self._n_edges):
            if not union_find.same_component(sorted_edges[i].start, sorted_edges[i].to):
                weight += sorted_edges[i].weight
                path.append((sorted_edges[i].start, sorted_edges[i].to))
                union_find.union_sets(sorted_edges[i].start, sorted_edges[i].to)
        if return_weight:
            return path, weight
        return path


if __name__ == "__main__":
    g = Graph(0)
    [g._add_node() for _ in range(7)]

    g._add_edge(0, 1, 5)
    g._add_edge(0, 2, 7)
    g._add_edge(0, 3, 12)

    g._add_edge(1, 2, 9)
    g._add_edge(1, 4, 7)

    g._add_edge(2, 3, 4)
    g._add_edge(2, 4, 4)
    g._add_edge(2, 5, 3)

    g._add_edge(3, 5, 7)

    g._add_edge(4, 5, 2)
    g._add_edge(4, 6, 5)

    g._add_edge(5, 6, 2)
    print(g)
    # print("Find the max span subtree by Prim method:")
    # print(g.find_maximal_span(method='prim', start=1, return_weight=True))
    # print("Find the max span subtree by Kruskal method:")
    # print(g.find_maximal_span(method='kruskal', return_weight=True))
    print("Find the min span subtree by Kruskal method:")
    print(g.find_minimal_span(method='kruskal', return_weight=True))
    print("Find the min span subtree by Prim method:")
    print(g.find_minimal_span(method='prim', return_weight=True, start=0))

    g = Graph()
    [g._add_node() for _ in range(4)]
    g._add_edge(0, 1, 6720)
    g._add_edge(0, 2, 240)
    g._add_edge(1, 2, 20160)
    g._add_edge(1, 3, 5040)
    g._add_edge(2, 3, 840)
    print(g.find_minimal_span())
