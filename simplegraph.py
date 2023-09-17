#%%
import networkx as nx
import numpy as np
from src.seelib import npmath

def cdist(mata: np.ndarray, matb: np.ndarray, diag=None):
    if mata.ndim == 1:
        mat_a = mata.reshape(1, -1)
    else:
        mat_a = mata
    if matb.ndim == 1:
        mat_b = matb.reshape(1, -1)
    else:
        mat_b = matb
    x2 = np.sum(mat_a ** 2, axis=1)
    y2 = np.sum(mat_b ** 2, axis=1)
    xy = mat_a @ mat_b.T
    x2 = x2.reshape(-1, 1)
    out = x2 - 2*xy + y2
    out = out.astype(np.float32)
    out = np.sqrt(out)
    if diag is not None:
        np.fill_diagonal(out, diag)
    if mata.ndim == 1:
        out = out[0]
    if matb.ndim == 1:
        out = out[...,0]
    return out

def v_angle(vec_a, vec_b, mode="rad"):
    out = np.arccos((vec_a @ vec_b) /np.linalg.norm(vec_a)/ np.linalg.norm(vec_b))
    if np.isnan(out):
        out = 0
    if mode == "deg":
        out = out * 180 / np.pi
    return out

def p_angle(point_a, center, point_b, mode="rad"):
    return v_angle(point_a - center, point_b - center, mode)

class Graph(nx.Graph):
    def __init__(self, points_dict: dict[str, np.ndarray]):
        super().__init__()
        self.ind = 0
        
        self.addAtoms("O", points_dict["O"])
        self.addAtoms("H", points_dict["H"])
                
        hinds = self.get_nodes_by_attributes("elem", "H")

        # 这里是将H拉近到去最近的氧，然后建立连结
        for hindex in hinds:
            nearOs = self.nearNodes(hindex, elem = "O")
            oindex, odis = nearOs[0]
            if odis > 1.3:
                self.remove_node(hindex)
                continue
            hs = self.nodes[oindex]['bond']
            self.ppNode(oindex, hindex)
            if all(np.linalg.norm(self.nodes[i]['position'] - self.nodes[hindex]['position']) > 0.9 for i in hs):
                self.linkAtom(oindex,hindex)
            else:
                self.remove_node(hindex)    

    def addAtom(self, elem, position, radius=None):
        node = {"elem": None, "position": None, "radius": radius, "bond": []}
        node['elem'] = elem
        node['position'] = position

        if radius is None:
            if elem == "O":
                node["radius"] = 0.74
            elif elem == "H":
                node["radius"] = 0.46

        ind = self.ind
        self.add_node(ind, **node)
        self.ind += 1
        return ind

    def rmAtom(self, ind_a):
        self.remove_node(ind_a)

    def addAtoms(self, elem, position):
        if isinstance(elem, str):
            if position.ndim == 1:
                self.addAtom(elem, position)
            else:
                for posi in position:
                    self.addAtom(elem, posi)
        else:
            for e, p in zip(*elem, *position):
                self.addAtom(e, p)
    
    def linkAtom(self, ind_a: int, ind_b: int):
        out = self.detectLink(ind_a, ind_b)
        if out['type'] is not None:
            if ind_b not in self.nodes[ind_a][out['type']]:
                self.nodes[ind_a][out['type']].append(ind_b)
            if ind_a not in self.nodes[ind_b][out['type']]:
                self.nodes[ind_b][out['type']].append(ind_a)
            self.add_edge(ind_a, ind_b, **out)

    def detectLink(self, ind_a, ind_b):
        if (ind_a, ind_b) in self.edges:
            return self.edges[ind_a, ind_b]
        nodea = self.nodes[ind_a]
        nodeb = self.nodes[ind_b]
        elemi, elemj = nodea['elem'], nodeb['elem']
        if elemi == "H" and elemj == "O":
            return self.detectLink(ind_b, ind_a)

        edge = {"type": None, "r": None,
                "u": None, "v": None,
                "unode": {}, "vnode": {}}

        posi, posj = nodea['position'], nodeb['position']
        r = np.linalg.norm(posi - posj)

        edge['r'] = r
        edge['u'] = ind_a
        edge['v'] = ind_b
        if ind_a != ind_b:
            if {elemi, elemj} == {"O", "H"}:
                if r < 1.2:
                    edge['type'] = "bond"

                elif r < 2.4:
                    if len(nodea['Hbond']) <= 1 and nodeb['Hbond'] == []:
                        edge['type'] = "Hbond"

                elif r < 4.5:
                    if len(nodea['Hbond']) + len(nodea['LHbond']) <= 1 and nodeb['Hbond'] == [] and nodeb['LHbond'] == []:
                        edge['type'] = "LHbond"

            elif {elemi, elemj} == {"O"}:
                if r < 3.4:
                    if len(nodea['OObond']) <= 4 and len(nodeb['OObond']) <= 4:
                        edge['type'] = "OObond"

                elif 3.4 <= r <= 6:
                    if len(nodea['OObond']) + len(nodea['LOObond']) <= 4 and len(nodeb['OObond']) + len(nodeb['LOObond']) <= 4:
                        ang_det = all(p_angle(self.nodes[anext]['position'], posi, posj, mode="deg") >= 60 for anext in nodea['OObond']) and \
                            all(p_angle(self.nodes[bnext]['position'], posj, posi, mode="deg") >= 60 for bnext in nodeb['OObond']) and \
                            all(p_angle(self.nodes[anext]['position'], posi, posj, mode="deg") >= 50 for anext in nodea['LOObond']) and \
                            all(p_angle(self.nodes[bnext]['position'], posj, posi, mode="deg") >= 50 for bnext in nodeb['LOObond'])
                        if ang_det:
                            edge['type'] = "LOObond"

            elif {elemi, elemj} == {"H"}:
                pass

        return edge
    
    def nearNodes(self, key, elem=None):
        """list all the nodes according to the distance

        Args:
            key (_type_): an index or a position
            elem (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: [( index, distance ), ...]
        """
        if elem == None:
            indices = [i for i in self.nodes()]
        else:
            indices = self.get_nodes_by_attributes('elem', elem)

        indices = np.asarray(indices)
        if isinstance(key, int):
            pos = np.asarray([self.nodes[i]['position']
                            for i in indices if i != key])
            posi = self.nodes[key]['position']

        else:
            pos = np.asarray([self.nodes[i]['position'] for i in indices])
            posi = key
        dis_mat = cdist(posi, pos)
        order = np.argsort(dis_mat)
        return np.array([indices[order], dis_mat[order]]).T

    def get_nodes_by_indices(self, indices, attribute=False, asarray=False):
        out = [self.nodes(data=attribute)[i] for i in indices]
        if asarray:
            return np.asarray(out)
        else:
            return out

    def get_edges_by_attributes(self, attribute=False, keyword=None):
        if keyword is not None:
            return [(x, y) for x, y, z in self.edges(data=attribute) if z == keyword]
        else:
            return [x for x in self.edges(data=attribute)]

    def get_nodes_by_attributes(self, attribute=False, keyword=None):
        if keyword is not None:
            return [x for x, y in self.nodes(data=attribute) if y == keyword]
        else:
            return [x for x in self.nodes(data=attribute)]
    
    def ppNode(self, index, target, r= 0.9572):
        posi = self.nodes[index]['position']
        posj = self.nodes[target]['position']
        delta = posj - posi
        delta = delta / np.linalg.norm(delta) * r
        self.nodes[target]['position'] = posi + delta
    
    def ppNode_angle(self, hind1, oind, hind2, angle = 1.82422, r = 0.9572):
        hpos1 = self.nodes[hind1]['position']
        hpos2 = self.nodes[hind2]['position']
        opos = self.nodes[oind]['position']
        if p_angle(hpos1, opos, hpos2) != angle:
            i = hpos1 + hpos2 - 2 * opos
            i_norm = np.linalg.norm(i)
            if i_norm != 0:
                i = i / i_norm
            else:
                i += np.random.uniform(0,0.01, 3)
                i = i / np.linalg.norm(i)
            j = (hpos2 - opos) - ((hpos2- opos) @ i) * i
            j = j / np.linalg.norm(j)
            theta = angle/2
            r1 = opos + r * (np.cos(theta) * i - np.sin(theta) * j)
            r2 = opos + r * (np.cos(theta) * i + np.sin(theta) * j)
            self.nodes[hind1]['position'] = r1
            self.nodes[hind2]['position'] = r2
    
    @property
    def pos_dict(self):
        out = {}
        for ind, pos in self.get_nodes_by_attributes("position"):
            if self.nodes[ind]['elem'] not in out:
                out[self.nodes[ind]['elem']] = []
            out[self.nodes[ind]['elem']].append(pos)
        for key in out:
            out[key] = np.asarray(out[key])[...,(2,0,1)] # z, x, y
        return out

#%%
if __name__ == "__main__":
    pos = {"O": np.array([[0,0,0], [2,2,2]]), "H": np.array([[0,0,1],[0,1,0], [0, 1, 1], [2,1,0]])}
    g = Graph(pos)
    print(g)

#%%
print(g.nodes)
print(g.edges)
print(g.nodes[1])