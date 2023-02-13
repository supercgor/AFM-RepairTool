import networkx as nx
import numpy as np
from src.tools import indexGenerator, cdist
import matplotlib.pyplot as plt
import matplotlib


class Graph(nx.Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = indexGenerator()

    def addAtoms(self, elem, position):
        lst = list(
            (next(self.index), {"elem": elem, "position": pos_i}) for pos_i in position)
        self.add_nodes_from(lst)

    def link_nodes_by_dist(self, nodes_a, nodes_b, radius=5):
        index_a = np.asarray(list(index for index, _ in nodes_a))
        index_b = np.asarray(list(index for index, _ in nodes_b))
        pos_a = np.asarray(list(pos['position'] for _, pos in nodes_a))
        pos_b = np.asarray(list(pos['position'] for _, pos in nodes_b))
        dis_mat = cdist(pos_a, pos_b) < radius
        pair = np.nonzero(dis_mat)
        index_a = index_a[pair[0]]
        index_b = index_b[pair[1]]
        for ia, ib in zip(index_a, index_b):
            if ia != ib:
                self.add_edge(ia,ib)

    def get_nodes_by_attributes(self, attribute, key):
        return [x for x, y in self.nodes(data=True) if key(y[attribute])]


if __name__ == "__main__":
    g = Graph()
    g.addAtoms("O", np.random.uniform(0, 5, (10, 3)))
    g.link_nodes_by_dist(g.nodes(data=True),g.nodes(data=True))
    nx.draw(g)
    plt.show()
