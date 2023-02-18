import networkx as nx
import numpy as np
from src.tools import indexGenerator, cdist
import matplotlib.pyplot as plt
from src.const import bound_dict, atom_info
import cv2


class Graph(nx.Graph):
    def __init__(self, poscarinfo, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = indexGenerator()
        self.ainfo = atom_info
        self.binfo = bound_dict
        self.pinfo = poscarinfo
        self.lattice = self.pinfo['lattice']
        
    def addAtoms(self, elem, position):
        """_summary_

        Args:
            elem (_type_): "H" or "O"
            position (_type_): real position
        """
        lst = list(
            (next(self.index), {"elem": elem, "position": pos_i, "color": self.ainfo[elem]['color'], "radius": self.ainfo[elem]['radius']}) for pos_i in position)
        self.add_nodes_from(lst)

    def link_nodes_by_dist(self, nodes_a, nodes_b, radius=5.0):
        """link two sets of nodes if the distance is smaller than radius.

        Args:
            nodes_a (_type_): node1
            nodes_b (_type_): node2
            radius (float, optional): radius. Defaults to 5.0.
        """
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
    
    def link_nodes_by_bounds(self, nodes_a, nodes_b):
        index_a = np.asarray(list(index for index, _ in nodes_a))
        index_b = np.asarray(list(index for index, _ in nodes_b))
        pos_a = np.asarray(list(pos['position'] for _, pos in nodes_a))
        pos_b = np.asarray(list(pos['position'] for _, pos in nodes_b))
        dis_mat = cdist(pos_a, pos_b)
        pair = {nodes_a[0][1]['elem'], nodes_b[0][1]['elem']}
        if pair == {"O"}:
            dis_mat = np.logical_and(bound_dict[('O','O')]['lower'] <= dis_mat, dis_mat <= bound_dict[('O','O')]['upper'])
        elif pair == {"O", "H"}:
            dis_mat = np.logical_and(bound_dict[('O','H')]['lower'] <= dis_mat, dis_mat <= bound_dict[('O','H')]['upper'])
        elif pair == {"H"}:
            dis_mat = np.logical_and(bound_dict[('H','H')]['lower'] <= dis_mat, dis_mat <= bound_dict[('H','H')]['upper'])
        
        pair = np.nonzero(dis_mat)
        index_a = index_a[pair[0]]
        index_b = index_b[pair[1]]
        for ia, ib in zip(index_a, index_b):
            if ia != ib:
                self.add_edge(ia,ib)

    def get_nodes_by_attributes(self, attribute, key):
        return [x for x, y in self.nodes(data=True) if key(y[attribute])]
    
    def getAtomsPos(self):
        out = {}
        nodes = self.nodes(data=True)
        for i, node in nodes:
            if node['elem'] in out:
                out[node['elem']].append(node['position'])
            else:
                out[node['elem']]= [node['position']]
        return {elem: np.asarray(out[elem]) for elem in out}
    
    def getAtomsElem(self):
        out = {}
        nodes = self.nodes(data=True)
        for i, node in nodes:
            if node['elem'] in out:
                out[node['elem']].append((i, node))
            else:
                out[node['elem']]= [(i, node)]

        return out

    def plotAtoms(self, img):
        img = edgePlot(img, self)
        img = circPlot(img, self, reverse=False)
        return img

def circPlot(img, graph: Graph, mirror = True, reverse = True):
    """_summary_

    Args:
        img (_type_): _description_
        pos (_type_): _description_
        info (dict, optional): _description_. Defaults to {"color": (255, 255, 255), "radius": 0.0296}.
        O radius = 0.0296, H radius = 0.0184
    Returns:
        _type_: _description_
    """
    if mirror:
        img = cv2.flip(img, 0)
        
    resolution = np.diag(img.shape[:2]/ graph.pinfo['lattice'].diagonal()[:2])
    pos = graph.getAtomsPos()
    order = []
    for elem in pos:
        for pos_i in pos[elem]:
            order.append((elem,pos_i))
    order = sorted(order, key=lambda x: x[1][2], reverse=reverse)
    for elem, pos_i in order:
        pos_i = pos_i[:2] @ resolution
        pos_i = pos_i.astype(int)
        radi = graph.ainfo[elem]['radius'] * np.min(resolution.diagonal())
        radi = radi.astype(int)
        color = graph.ainfo[elem]['color']
        cv2.circle(img, pos_i, radi, color, -1)
        cv2.circle(img, pos_i, radi, (0, 0, 0), 1)
        
    if mirror:
        img = cv2.flip(img, 0)
    return img

def edgePlot(img, graph: Graph, mirror=True):
    if mirror:
        img = cv2.flip(img, 0)
    
    resolution = np.diag(img.shape[:2]/ graph.pinfo['lattice'].diagonal()[:2])
    
    edges = graph.edges()
    for i, j in edges:
        pos = nx.get_node_attributes(graph, "position")
        elem = nx.get_node_attributes(graph, "elem")
        pos_i = pos[i][:2] @ resolution
        pos_j = pos[j][:2] @ resolution
        pos_i = pos_i.astype(int)
        pos_j = pos_j.astype(int)
        color = graph.binfo[(elem[i],elem[j])]['color']
        thickness = graph.binfo[(elem[i],elem[j])]['thickness']
        cv2.line(img,pos_i,pos_j, color, thickness)
    if mirror:
        img = cv2.flip(img, 0)
    return img






if __name__ == "__main__":
    g = Graph()
    g.addAtoms("O", np.random.uniform(0, 5, (10, 3)))
    g.link_nodes_by_dist(g.nodes(data=True),g.nodes(data=True))
    nx.draw(g)
    plt.show()
