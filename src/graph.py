import networkx as nx
import numpy as np
from src.tools import indexGenerator, cdist
import matplotlib.pyplot as plt
from src.const import bound_dict, atom_info
import cv2
import src.seelib.npmath as npmath
import src.seelib.cv2eff as cve


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
        lst = [(next(self.index), {"elem": elem, "position": pos_i, "color": self.ainfo[elem]
                ['color'], "radius": self.ainfo[elem]['radius']}) for pos_i in position]
        self.add_nodes_from(lst)
    
    def link_nodes_by_dist(self, nodes_a, nodes_b, radius=(1.6, 5.0), edgecolor=None, edgethickness=None, edgeprior=None, tagFather=False):
        """link two sets of nodes if the distance is smaller than radius.

        Args:
            nodes_a (_type_): node1
            nodes_b (_type_): node2
            radius (float, optional): radius. Defaults to 5.0.
        """
        pos_a = np.asarray(list(self.nodes[i]['position'] for i in nodes_a))
        pos_b = np.asarray(list(self.nodes[i]['position'] for i in nodes_b))
        dis_mat = cdist(pos_a, pos_b)
        pair = npmath.mask_argmin(dis_mat, np.logical_and(
            dis_mat > radius[0], dis_mat < radius[1]), axis=1)
         
        pair = [(nodes_a[i], nodes_b[j]) for i, j in enumerate(pair)]
        for ia, ib in pair:
            if ia != ib:
                node_a, node_b = self.nodes[ia], self.nodes[ib]
                if edgecolor is None:
                    edgecolor = bound_dict[(
                        node_a["elem"], node_b["elem"])]['color']

                if edgethickness is None:
                    edgethickness = bound_dict[(
                        node_a["elem"], node_b["elem"])]['thickness']

                if edgeprior is None:
                    edgeprior = 1
                
                if tagFather:
                    self.nodes[ia]['father'] = ib
                    if 'son' not in self.nodes[ib]:
                        self.nodes[ib]['son'] = [ia]
                    else:
                        self.nodes[ib]['son'].append(ia)
                    
                self.add_edge(ia, ib, color=edgecolor,
                              thickness=edgethickness, prior=edgeprior, u=self.nodes[ia], v=self.nodes[ib])

    def link_nodes_by_bounds(self, nodes_a, nodes_b):
        pos_a = np.asarray(list(self.nodes[i]['position'] for i in nodes_a))
        pos_b = np.asarray(list(self.nodes[i]['position'] for i in nodes_b))
        dis_mat = cdist(pos_a, pos_b)
        pair = {self.nodes[nodes_a[0]]['elem'], self.nodes[nodes_b[0]]['elem']}
        if pair == {"O"}:
            dis_mat = np.logical_and(bound_dict[(
                'O', 'O')]['lower'] <= dis_mat, dis_mat <= bound_dict[('O', 'O')]['upper'])
        elif pair == {"O", "H"}:
            dis_mat = np.logical_and(bound_dict[(
                'O', 'H')]['lower'] <= dis_mat, dis_mat <= bound_dict[('O', 'H')]['upper'])
        elif pair == {"H"}:
            dis_mat = np.logical_and(bound_dict[(
                'H', 'H')]['lower'] <= dis_mat, dis_mat <= bound_dict[('H', 'H')]['upper'])

        pair = np.nonzero(dis_mat)
        pair = list(zip(np.asarray(nodes_a)[pair[0]],np.asarray(nodes_b)[pair[1]]))
        
        for ia, ib in pair:
            if ia != ib:
                node_a, node_b = self.nodes[ia], self.nodes[ib]
                color = bound_dict[(node_a["elem"], node_b["elem"])]['color']
                thickness = bound_dict[(
                    node_a["elem"], node_b["elem"])]['thickness']
                self.add_edge(ia, ib, color=color,
                              thickness=thickness, prior=1, u=self.nodes[ia], v=self.nodes[ib])

    def get_nodes_by_attributes(self, attribute, keyword):
        return [x for x, y in self.nodes(data=attribute) if y == keyword]

    def getAtomsPos(self):
        out = {}
        nodes = self.nodes(data=True)
        for i, node in nodes:
            if node['elem'] in out:
                out[node['elem']].append(node['position'])
            else:
                out[node['elem']] = [node['position']]
        return {elem: np.asarray(out[elem]) for elem in out}

    def getAtomsElem(self):
        out = {}
        nodes = self.nodes(data=True)
        for i, node in nodes:
            if node['elem'] in out:
                out[node['elem']].append((i, node))
            else:
                out[node['elem']] = [(i, node)]

        return out

    def edgesImgStat(self, edges, img, mirror=True, contrast=150):
        if mirror:
            img = cv2.flip(img, 0)

        img = cve.contrast(img, contrast)

        resolution = np.diag(
            img.shape[:2] / self.pinfo['lattice'].diagonal()[:2])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255

        bg_img = np.zeros(img.shape, dtype=np.uint8)
        out = {}

        for i in edges:
            edge = edges[i]
            u, v = (edge['u']['position'][:2] @ resolution).astype(
                int), (edge['v']['position'][:2] @ resolution).astype(int)
            mask = cv2.line(bg_img, u, v, 1, 1)
            m = npmath.mask_mean(img, mask)
            std = npmath.mask_std(img, mask)
            mini = npmath.mask_min(img, mask)
            out[i] = (m, std, mini)

        return out

    def plotNodes(self, img, nodes=None, mirror=True, resize=(256, 256), text=True):
        img = cv2.resize(img, resize)
        if mirror:
            img = cv2.flip(img, 0)

        resolution = np.diag(
            img.shape[:2] / self.pinfo['lattice'].diagonal()[:2])
        if nodes is None:
            nodes = self.nodes(data=True)

        order = [node[0] for node in nodes]
        order.sort(key=lambda index: - self.nodes[index]['position'][2])

        for i in order:
            pos_i = self.nodes[i]['position'][:2] @ resolution
            pos_i = pos_i.astype(int)
            radi = self.nodes[i]['radius'] * np.min(resolution.diagonal())
            radi = radi.astype(int)
            color = self.nodes[i]['color']
            cv2.circle(img, pos_i, radi, color, -1)
            cv2.circle(img, pos_i, radi, (0, 0, 0), 1)
            if text:
                txt = cve.genText(i, flip = 0)
                img = cve.transBind(img, txt, pos_i, "center")

        if mirror:
            img = cv2.flip(img, 0)
        return img

    def plotEdges(self, img, edges=None, mirror=True, transparency=0.5, resize=(256, 256)):
        img = cv2.resize(img, resize)

        if edges is None:
            edges = self.edges
        if mirror:
            img = cv2.flip(img, 0)

        draw = img.copy()

        resolution = np.diag(
            img.shape[:2] / self.pinfo['lattice'].diagonal()[:2])

        edges = sorted(edges, key=lambda x: self.edges[x]["prior"])

        for e in edges:
            data = self.edges[e]
            pos = nx.get_node_attributes(self, "position")
            pos_i = pos[e[0]][:2] @ resolution
            pos_j = pos[e[1]][:2] @ resolution
            pos_i = pos_i.astype(int)
            pos_j = pos_j.astype(int)
            color = data['color']
            thickness = data['thickness']
            cv2.line(draw, pos_i, pos_j, color, thickness)

        img = cv2.addWeighted(draw, transparency, img, 1 - transparency, 0)
        if mirror:
            img = cv2.flip(img, 0)
        return img
