import networkx as nx
import random


def real_valued_2d_basic(x):
    return abs(x[0] - 3) + abs(x[1] - 5)


def knapsack(x):
    max_weight = 10
    items = [(10.0, 3.3), (1, .001), (1, .005), (1, 5.0), (3.0, .01), (5.0, 3.0)]
    profit = -1 * sum([x[i] * items[i][0] for i in range(len(items))])
    weight = sum([x[i]*items[i][1] for i in range(len(items))])
    if weight > max_weight or x[-1] < 0.05:
        return 1e12
    return profit/x[-1]


class KnapsackProblem:

    def __init__(self, max_weight, values, weights):
        self._max_weight = max_weight
        self._values = values
        self._weights = weights

    def get_profit(self, x):
        weight = sum([x[i] * self._weights[i] for i in range(len(self._weights))])
        if weight > self._max_weight:
            return 1e12
        profit = -1 * sum([x[i] * self._values[i] for i in range(len(self._values))])
        return profit


class GraphColoring:

    def __init__(self, colors, num_nodes=5):
        self._graph = nx.petersen_graph()
        self._dimenions = num_nodes
        self._colors = colors
        self._color_map = { node : random.choice(self._colors) for node in self._graph.nodes }

    def assign_color_map(self, proposed_colors):
        for i, key in enumerate(self._color_map.keys()):
            self._color_map[key] = proposed_colors[i]

    def _verify_graph(self):
        for node in self._graph.nodes:
            for neighbor in self._graph.neighbors(node):
                if self._color_map[node] == self._color_map[neighbor]:
                    return 0
        return -1

    def solve(self, proposed_colors):
        self.assign_color_map(proposed_colors)
        return self._verify_graph()

    def print_colors(self):
        for node in self._graph.nodes:
            print("Node:")
            print("\t[{0}] -> [{1}]".format(node, self._color_map[node]))
            print("Neighbors:")
            for neighbor in self._graph.neighbors(node):
                print("\t[{0}] -> [{1}]".format(neighbor, self._color_map[neighbor]))


if __name__ == "__main__":
    graph_coloring = GraphColoring(range(5))
    print(graph_coloring.solve([1, 2, 3, 4, 5]))
