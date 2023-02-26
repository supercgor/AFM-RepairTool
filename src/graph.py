import networkx as nx
import numpy as np
from src.tools import indexGenerator
import cv2
import src.seelib.npmath as npmath
import src.seelib.cv2eff as cve
from src.tools import poscar

class Graph(nx.Graph):
    def __init__(self, sample, resolution = (1024,1024), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = sample['name']
        self.image = sample['image']
        self.ref_img = self.image[3]
        self.img_reso = np.asarray(self.image[0].shape)
        self.box_reso = np.diag(sample['info']['lattice'])
        self.out_reso = np.asarray(resolution)
        
        self.position = {key: sample['position'][key] @ np.diag(self.box_reso) for key in sample['position']}
        self._img2out = np.diag(self.out_reso/self.img_reso)
        self._img2box = np.diag(self.box_reso[:2]/self.img_reso)
        self._box2out = np.diag(self.out_reso/self.box_reso[:2])
        self._box2img = np.diag(self.img_reso/self.box_reso[:2])
        
        self.index = indexGenerator()
    
    def addAtom(self, elem, position, color = None, radius = None, Hup = False, isGen = False):
        node = {"elem": None,"position": None,"color": color, "radius": radius, "bond": [], "Hbond": [], "LHbond": [], "OObond": [], "LOObond": [], "onLink": None, "Hup": False, "isGen": isGen}
        node['elem'] = elem
        node['position'] = position
        node['Hup'] = Hup
        
        if color is None:
            if elem == "O":
                node["color"] = (13, 13, 255)
            elif elem == "H":
                node["color"] = (255, 255, 255)
                
        if radius is None:
            if elem == "O":
                node["radius"] = 0.74
            elif elem == "H":
                node["radius"] = 0.46
        ind = next(self.index)
        self.add_node(ind, **node)
        
        return ind
    
    def rmAtom(self, ind_a):
        oind = self.nodes[ind_a]["bond"][0]
        ono = self.nodes[oind]
        for ooind in ono['OObond'] + ono['LOObond']:
            e = self.edges[(oind, ooind)]
            if ind_a in e['unode']:
                e['unode'] = {}
            if ind_a in e['vnode']:
                e['vnode'] = {}
                
        for ty in ['bond', 'Hbond', 'LHbond', 'OObond', 'LOObond']:
            for ind_b in self.nodes[ind_a][ty]:
                self.rmLink(ind_a, ind_b)
        self.remove_node(ind_a)
        
    def addAtoms(self, elem, position):
        if isinstance(elem,str):
            if position.ndim == 1:
                self.addAtom(elem, position)
            else:
                for posi in position:
                    self.addAtom(elem, posi)
        else:
            for e,p in zip(*elem, *position):
                self.addAtom(e, p)
    
    def detectLink(self, ind_a, ind_b):
        if (ind_a,ind_b) in self.edges:
            return self.edges[ind_a,ind_b]
        nodea = self.nodes[ind_a]
        nodeb = self.nodes[ind_b]
        elemi, elemj = nodea['elem'], nodeb['elem']
        if elemi == "H" and elemj == "O":
            return self.detectLink(ind_b, ind_a)
        
        edge = {"type": None,"r": None,"color": None,"thickness": None,"u": None,"v": None,"prior": None, "unode": {},"vnode": {}}
        
        posi,posj = nodea['position'], nodeb['position']
        r = np.linalg.norm(posi - posj)

        edge['r'] = r
        edge['u'] = ind_a
        edge['v'] = ind_b
        if ind_a != ind_b:
            if {elemi, elemj} == {"O","H"}:
                if r < 1.2:
                    edge['color'] = (0,0,0)
                    edge['thickness'] = 10
                    edge['type'] = "bond"
                        
                elif r < 2.4:
                    if len(nodea['Hbond']) <= 1 and nodeb['Hbond'] == []:
                        edge['color'] = (255, 160, 122)
                        edge['thickness'] = 3
                        edge['type'] = "Hbond"
                    
                elif r < 4.5:
                    if len(nodea['Hbond']) + len(nodea['LHbond']) <= 1 and nodeb['Hbond'] == [] and nodeb['LHbond'] == []:
                        edge['color'] = (255, 229, 204)
                        edge['thickness'] = 3
                        edge['type'] = "LHbond"
                    
            elif {elemi, elemj} == {"O"}:
                if r < 3.4:
                    if len(nodea['OObond']) <= 4 and len(nodeb['OObond']) <= 4:
                        edge['color'] = (80, 160, 122)
                        edge['thickness'] = 9
                        edge['type'] = "OObond"
                    
                elif 3.4 <= r <= 6:
                    if len(nodea['OObond']) + len(nodea['LOObond']) <= 4 and len(nodeb['OObond']) + len(nodeb['LOObond']) <= 4:
                        ang_det = all(npmath.p_angle(self.nodes[anext]['position'], posi, posj, mode= "deg") >= 60 for anext in nodea['OObond']) and \
                            all(npmath.p_angle(self.nodes[bnext]['position'], posj, posi, mode= "deg") >= 60 for bnext in nodeb['OObond']) and \
                            all(npmath.p_angle(self.nodes[anext]['position'], posi, posj, mode= "deg") >= 50 for anext in nodea['LOObond']) and \
                            all(npmath.p_angle(self.nodes[bnext]['position'], posj, posi, mode= "deg") >= 50 for bnext in nodeb['LOObond'])
                        if ang_det:
                            edge['color'] = (204, 204, 255)
                            edge['thickness'] = 9
                            edge['type'] = "LOObond"
                    
            elif {elemi, elemj} == {"H"}:
                pass
                # if nodea['bond'] != [] and nodeb['bond'] != []:
                #     if nodea['bond'] == nodeb['bond']:
                #         edge['color'] = (255, 255, 255)
                #         edge['thickness'] = 0
                #         edge['type'] = "bond"
        mask = np.zeros(self.img_reso)
        mask = cv2.line(mask, (posi[:2] @ self._box2img).astype(int) - 2, (posj[:2] @ self._box2img).astype(int) - 2, color = 1, thickness= 1)
        prior = (npmath.mask_mean(self.ref_img, mask), npmath.mask_std(self.ref_img, mask))
        edge['prior'] = prior
        return edge
    
    def detectH2B(self, hind):
        indd = self.nodes[hind]['onLink']
        if indd is not None:
            return {"link": indd, "info": self.edges[indd]['info']}
        else:
            hno = self.nodes[hind]
            oind = hno['bond'][0]
            ono = self.nodes[oind]
            score = np.inf
            mdis = np.inf
            for ty in ["OObond", "LOObond"]:
                for ooind in ono[ty]:
                    oono = self.nodes[ooind]
                    ang = npmath.p_angle(hno['position'],ono['position'], oono['position'], mode="deg")
                    oh = hno['position'] - ono['position']
                    oo = oono['position'] - ono['position']
                    vec = oh - oh @ (oo / np.linalg.norm(oo))
                    dis = np.linalg.norm(vec)
                    if (ang< 40) or (dis < 0.75 and ang < 55) or (dis < 0.5 and ang < 65) or (dis < 0.3 and ang < 80):
                        new_score = ang/45 + dis
                        if new_score < score:
                            score = new_score
                            indd = (oind, ooind)
                            mdis = dis
                            
            return {"link": indd, "info": (mdis, score)}
        
    def linkH2B(self, hind):
        out = self.detectH2B(hind)
        indd = out['link']
        if indd is not None:
            e = self.edges[indd]
            if e['u'] == indd[0]:
                if len(e['unode']) == 0:
                    e['unode'][hind] = out['info']
            else:
                if len(e['vnode']) == 0:
                    e['vnode'][hind] = out['info']
            self.nodes[hind]['onLink'] = indd
    
    def linkAtom(self, ind_a: int, ind_b: int):
        out = self.detectLink(ind_a,ind_b)
        if out['type'] is not None:
            if ind_b not in self.nodes[ind_a][out['type']]:
                self.nodes[ind_a][out['type']].append(ind_b)
            if ind_a not in self.nodes[ind_b][out['type']]:
                self.nodes[ind_b][out['type']].append(ind_a)
            self.add_edge(ind_a,ind_b, **out)
    
    def linkAtoms(self, nodes_a: int | list, nodes_b: int | list):
        if isinstance(nodes_a, int):
            nodes_a = [nodes_a]
        if isinstance(nodes_b, int):
            nodes_b = [nodes_b]
        posa = self.get_nodes_by_indices(nodes_a, attribute = 'position', asarray=True)
        posb = self.get_nodes_by_indices(nodes_b, attribute = 'position', asarray=True)
        dis_mat = npmath.cdist(posa,posb)
        order = np.unravel_index(np.argsort(dis_mat, axis=None),dis_mat.shape)
        for a, b in zip(*order):
            ind_a, ind_b = nodes_a[a], nodes_b[b]
            self.linkAtom(ind_a,ind_b)
    
    def rmLink(self, ind_a, ind_b):
        e = self.edges[(ind_a,ind_b)]
        ty = e['type']
        self.remove_edge(ind_a, ind_b)
        if ind_b in self.nodes[ind_a][ty]:
            self.nodes[ind_a][ty].remove(ind_b)
        if ind_a in self.nodes[ind_b][ty]:
            self.nodes[ind_b][ty].remove(ind_a)
    
    def nearNodes(self, index, elem = None):
        if elem == None:
            indices = [i for i in self.nodes()]
        else:
            indices = self.get_nodes_by_attributes('elem', elem)
            
        indices = np.asarray(indices)
        pos = np.asarray([self.nodes[i]['position'] for i in indices if i != index])
        posi = self.nodes[index]['position']

        dis_mat = npmath.cdist(posi, pos)
        order = np.argsort(dis_mat)
        return np.array([indices[order], dis_mat[order]]).T
    
    def ppNode(self, index, target, r):
        posi = self.nodes[index]['position']
        posj = self.nodes[target]['position']
        delta = posj - posi
        delta = delta/ np.linalg.norm(delta) * r
        self.nodes[target]['position'] = posi + delta

    def get_nodes_by_indices(self, indices, attribute = False, asarray = False):
        out = [self.nodes(data = attribute)[i] for i in indices]
        if asarray:
            return np.asarray(out)
        else:
            out

    def get_edges_by_attributes(self, attribute = False, keyword = None):
        if keyword is not None:
            return [(x, y) for x, y, z in self.edges(data = attribute) if z == keyword]
        else:
            return [x for x in self.edges(data = attribute)]
    
    def get_nodes_by_attributes(self, attribute = False, keyword = None):
        if keyword is not None:
            return [x for x, y in self.nodes(data = attribute) if y == keyword]
        else:
            return [x for x in self.nodes(data = attribute)]

    def plotNodes(self, img = 0, nodes=None, mirror=True, text=True):
        if isinstance(img, int):
            img = self.image[img].copy()
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, self.out_reso)
        if mirror:
            img = cv2.flip(img, 0)

        reso = self._box2out
        if nodes is None:
            nodes = self.nodes(data=True)

        order = [node[0] for node in nodes]
        order.sort(key=lambda index: - self.nodes[index]['position'][2])

        for i in order:
            target = img.copy()
            pos_i = self.nodes[i]['position'][:2] @ reso
            pos_i = pos_i.astype(int)
            radi = self.nodes[i]['radius'] * np.min(reso.diagonal())
            radi = radi.astype(int)
            color = self.nodes[i]['color']
            cv2.circle(target, pos_i, radi, color, -1)
            if self.nodes[i]['elem'] == "H":
                img = cv2.addWeighted(target, 0.5, img, 0.5, 0)
            else:
                img = cv2.addWeighted(target, 1, img, 0, 0)
            cv2.circle(img, pos_i, radi, (0, 0, 0), 1)
            if text and self.nodes[i]['elem'] == "O":
                txt = cve.genText(i, flip = 0)
                img = cve.transBind(img, txt, pos_i, "center")

        if mirror:
            img = cv2.flip(img, 0)
        return img

    def plotEdges(self, img = 0, edges=None, mirror=True, transparency=0.5, text = True):
        if isinstance(img, int):
            img = self.image[img].copy()
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, self.out_reso)
        if edges is None:
            edges = self.edges
        if mirror:
            img = cv2.flip(img, 0)

        draw = img.copy()

        reso = self._box2out

        edges = sorted(edges, key=lambda x: self.edges[x]["prior"])

        for e in edges:
            data = self.edges[e]
            thickness = data['thickness']
            if thickness <= 0:
                continue
            pos = nx.get_node_attributes(self, "position")
            pos_i = pos[e[0]][:2] @ reso
            pos_j = pos[e[1]][:2] @ reso
            pos_i = pos_i.astype(int)
            pos_j = pos_j.astype(int)
            color = data['color']
            cv2.line(draw, pos_i, pos_j, color, thickness)
            if text:
                txt = cve.genText(f"{int(self.edges[e]['prior'][0])},{int(self.edges[e]['prior'][1])}", flip = 0)
                img = cve.transBind(img, txt, (pos_i + pos_j)//2, "center")

        img = cv2.addWeighted(draw, transparency, img, 1 - transparency, 0)
        if mirror:
            img = cv2.flip(img, 0)
        return img

    def save(self, path):
        p = poscar()
        out = {"O":[],"H":[]}
        for ind, pos in self.get_nodes_by_attributes("position"):
            out[self.nodes[ind]['elem']].append(pos)
        for key in out:
            out[key] = np.asarray(out[key])
        p.generate_poscar(out)
        p.save(path)