import networkx as nx
import numpy as np
from .tools import indexGenerator
import cv2
from .seelib import npmath
from .seelib import cv2eff as cve
from .tools import poscar


class Graph(nx.Graph):
    def __init__(self, name, imgs, pl, res = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.image = imgs
        self.ref_img = self.image[3]
        self.img_reso = np.asarray(self.image[0].shape[-1::-1])
        self.box_reso = pl.lattice
        self.out_reso = self.img_reso * res
        self.position = pl.pos
        self._img2out = self.out_reso/self.img_reso
        self._img2box = self.box_reso[:2]/self.img_reso
        self._box2out = self.out_reso/self.box_reso[:2]
        self._box2img = self.img_reso/self.box_reso[:2]

        self.index = indexGenerator()
        
        self.addAtoms("O", pl.pos["O"])
        self.addAtoms("H", pl.pos["H"])
        
        self.pl = pl
        
        hinds = self.get_nodes_by_attributes("elem", "H")

        # 這裡是將H拉近到去最近的氧，然後建立連結
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
    
    def make_edges(self):
        # 將OO之間的關系找出來
        oinds = self.get_nodes_by_attributes("elem", "O")
        self.linkAtoms(oinds, oinds)
        
        # 將H放入OO的關系中
        hinds = self.get_nodes_by_attributes("elem", "H")
        for hind in hinds:
            self.linkH2B(hind)
        

    def addAtom(self, elem, position, color=None, radius=None, Hup=False, isGen=False):
        node = {"elem": None, "position": None, "color": color, "radius": radius, "bond": [], "Hbond": [
        ], "LHbond": [], "OObond": [], "LOObond": [], "onLink": None, "Hup": False, "isGen": isGen}
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
        if isinstance(elem, str):
            if position.ndim == 1:
                self.addAtom(elem, position)
            else:
                for posi in position:
                    self.addAtom(elem, posi)
        else:
            for e, p in zip(*elem, *position):
                self.addAtom(e, p)

    def detectLink(self, ind_a, ind_b):
        if (ind_a, ind_b) in self.edges:
            return self.edges[ind_a, ind_b]
        nodea = self.nodes[ind_a]
        nodeb = self.nodes[ind_b]
        elemi, elemj = nodea['elem'], nodeb['elem']
        if elemi == "H" and elemj == "O":
            return self.detectLink(ind_b, ind_a)

        edge = {"type": None, "r": None, "color": None, "thickness": None,
                "u": None, "v": None, "prior": None, "unode": {}, "vnode": {}}

        posi, posj = nodea['position'], nodeb['position']
        r = np.linalg.norm(posi - posj)

        edge['r'] = r
        edge['u'] = ind_a
        edge['v'] = ind_b
        if ind_a != ind_b:
            if {elemi, elemj} == {"O", "H"}:
                if r < 1.2:
                    edge['color'] = (0, 0, 0)
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
                        ang_det = all(npmath.p_angle(self.nodes[anext]['position'], posi, posj, mode="deg") >= 60 for anext in nodea['OObond']) and \
                            all(npmath.p_angle(self.nodes[bnext]['position'], posj, posi, mode="deg") >= 60 for bnext in nodeb['OObond']) and \
                            all(npmath.p_angle(self.nodes[anext]['position'], posi, posj, mode="deg") >= 50 for anext in nodea['LOObond']) and \
                            all(npmath.p_angle(
                                self.nodes[bnext]['position'], posj, posi, mode="deg") >= 50 for bnext in nodeb['LOObond'])
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
        mask = cv2.line(mask, (posi[:2] * self._box2img).astype(int) - 2,
                        (posj[:2] * self._box2img).astype(int) - 2, color=1, thickness=1)
        prior = 0
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
                    ang = npmath.p_angle(
                        hno['position'], ono['position'], oono['position'], mode="deg")
                    oh = hno['position'] - ono['position']
                    oo = oono['position'] - ono['position']
                    vec = oh - oh @ (oo / np.linalg.norm(oo))
                    dis = np.linalg.norm(vec)
                    if (ang < 40) or (dis < 0.75 and ang < 55) or (dis < 0.5 and ang < 65) or (dis < 0.3 and ang < 80):
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
        out = self.detectLink(ind_a, ind_b)
        if out['type'] is not None:
            if ind_b not in self.nodes[ind_a][out['type']]:
                self.nodes[ind_a][out['type']].append(ind_b)
            if ind_a not in self.nodes[ind_b][out['type']]:
                self.nodes[ind_b][out['type']].append(ind_a)
            self.add_edge(ind_a, ind_b, **out)

    def linkAtoms(self, nodes_a: int | list, nodes_b: int | list):
        if isinstance(nodes_a, int):
            nodes_a = [nodes_a]
        if isinstance(nodes_b, int):
            nodes_b = [nodes_b]
        posa = self.get_nodes_by_indices(
            nodes_a, attribute='position', asarray=True)
        posb = self.get_nodes_by_indices(
            nodes_b, attribute='position', asarray=True)
        dis_mat = npmath.cdist(posa, posb)
        order = np.unravel_index(np.argsort(dis_mat, axis=None), dis_mat.shape)
        for a, b in zip(*order):
            ind_a, ind_b = nodes_a[a], nodes_b[b]
            self.linkAtom(ind_a, ind_b)

    def rmLink(self, ind_a, ind_b):
        e = self.edges[(ind_a, ind_b)]
        ty = e['type']
        self.remove_edge(ind_a, ind_b)
        if ind_b in self.nodes[ind_a][ty]:
            self.nodes[ind_a][ty].remove(ind_b)
        if ind_a in self.nodes[ind_b][ty]:
            self.nodes[ind_b][ty].remove(ind_a)

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
        dis_mat = npmath.cdist(posi, pos)
        order = np.argsort(dis_mat)
        return np.array([indices[order], dis_mat[order]]).T

    def get_nodes_by_indices(self, indices, attribute=False, asarray=False):
        out = [self.nodes(data=attribute)[i] for i in indices]
        if asarray:
            return np.asarray(out)
        else:
            out

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
        if npmath.p_angle(hpos1, opos, hpos2) != angle:
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
    
    def pp_all_nodes(self):
        os = self.get_nodes_by_attributes("elem", "O")
        for oind in os:
            hs = self.nodes[oind]['bond']
            if len(hs) != 2:
                raise f"There are some atoms do not have 2H: {oind}: {hs}"
            else:
                self.ppNode_angle(hs[0], oind, hs[1])
        
    def plotNodes(self, img=0, nodes=None, mirror=True, text=True, reverse = True):
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
        order.sort(key=lambda index: - self.nodes[index]['position'][2], reverse=reverse)

        for i in order:
            target = img.copy()
            pos_i = self.nodes[i]['position'][:2] * reso
            pos_i = pos_i.astype(int)
            radi = self.nodes[i]['radius'] * np.min(reso)
            radi = radi.astype(int)
            color = self.nodes[i]['color']
            if self.nodes[i]['Hup']:
                color = (0, 150, 255)
            cv2.circle(target, pos_i, radi, color, -1)
            if self.nodes[i]['elem'] == "H":
                img = cv2.addWeighted(target, 0.5, img, 0.5, 0)
            else:
                img = cv2.addWeighted(target, 1, img, 0, 0)
            cv2.circle(img, pos_i, radi, (0, 0, 0), 1)
            if text and self.nodes[i]['elem'] == "O":
                txt = cve.genText(i, flip=0)
                img = cve.transBind(img, txt, pos_i, "center")

        if mirror:
            img = cv2.flip(img, 0)
        return img    
                
    def plotEdges(self, img=0, edges=None, mirror=True, transparency=0.5, text=True):
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
            pos_i = pos[e[0]][:2] * reso
            pos_j = pos[e[1]][:2] * reso
            pos_i = pos_i.astype(int)
            pos_j = pos_j.astype(int)
            color = data['color']
            cv2.line(draw, pos_i, pos_j, color, thickness)
            if text:
                txt = cve.genText(
                    f"{int(self.edges[e]['prior'][0])},{int(self.edges[e]['prior'][1])}", flip=0)
                img = cve.transBind(img, txt, (pos_i + pos_j)//2, "center")

        img = cv2.addWeighted(draw, transparency, img, 1 - transparency, 0)
        if mirror:
            img = cv2.flip(img, 0)
        return img

    def save(self, save_dir, name = None, mode = "poscar"):
        """轉化為 .poscar 或是 .data 文件

        Args:
            save_dir (_type_): 儲存位置
            name (str, optional): 名稱, 若為None, 則用poscarLoader給出的名稱
            mode (str, optional): 可以選擇: [data, poscar]
        """
        if name is None:
            name = self.name
        
        out = {"O": [], "H": []}
        for ind, pos in self.get_nodes_by_attributes("position"):
            out[self.nodes[ind]['elem']].append(pos)
        for key in out:
            out[key] = np.asarray(out[key])
        self.pl.upload_pos(out)
        self.pl.save(name, save_dir)

    def detect_Hup(self, pix_Hup, spawn_O = True, O_height = 1.5, spawn_H = True):
        pos_O = pix_Hup.copy()
        pos_O[...,:2] =  pos_O[...,:2] * self._img2box[:2]
        pos_O[...,2] = O_height
        for pos in pos_O:
            nearOs = self.nearNodes(pos, "O")
            if nearOs[0][1] < 2.0:
                oind, opos = nearOs[0][0], nearOs[0][1]
                self.nodes[oind]['Hup'] = True
                if spawn_H:
                    newHind = self.addAtom("H", pos + [0,0,1.5], Hup = True, isGen = True)
                    self.ppNode(oind, newHind)
                    self.linkAtom(oind, newHind)
                    nearHs = self.nearNodes(newHind, elem = "H")
                    for hind, hpos in nearHs:
                        if hpos > 1.1:
                            break
                        else:
                            self.rmAtom(hind)
                    
            elif spawn_O:
                newind = self.addAtom("O", pos, Hup = True, isGen = True)
                # for oind, opos in nearOs[1:]:
                #     if opos > 6.0:
                #         break
                #     else:
                #         self.linkAtom(newind, oind)
                if spawn_H:
                    newHind = self.addAtom("H", pos + [0,0,1.5], Hup = True, isGen = True)
                    self.ppNode(newind, newHind)
                    self.linkAtom(newind, newHind)
            
        
    def hasHup(self, oind):
        bondH = self.nodes[oind]['bond']
        for hind in bondH:
            hpos = self.nodes[hind]['position']
            opos = self.nodes[oind]['position']
            if (hpos - opos)[2] > 0.7:
                return True
        else:
            return False
        
    def wrtie_data(self):
        os = self.get_nodes_by_attributes("elem", "O")
        hs = self.get_nodes_by_attributes("elem", "H")
        part1 = f"""
{len(self.nodes)} atoms
{len(hs)} bonds
{len(os)} angles
2 atom types
1 bond types
1 angle types
-25.0 {self.box_reso[0] + 25:.1f} xlo xhi
-25.0 {self.box_reso[1] + 25:.1f} ylo yhi
-10.0 {self.box_reso[2] + 10:.1f} zlo zhi

Masses

1 15.9994
2 1.008

Pair Coeffs

1 0.21084 3.1668
2 0 0

Bond Coeffs

1 10000 0.9572

Angle Coeffs

1 10000 104.52

Atoms

"""
        fix = []
        part2 = ""
        
        a_num = 1
        for i, oind in enumerate(os):
            opos = self.nodes[oind]['position']
            str_pos = f"{opos[0]:.4f} {opos[1]:.4f} {opos[2]:.4f}"
            part2 += f"{a_num} {i} 1 -1.1794 {str_pos} 0 0 0\n"
            fix.append(a_num)
            a_num += 1
            for hind in self.nodes[oind]['bond']:
                hpos = self.nodes[hind]['position']
                str_pos = f"{hpos[0]:.4f} {hpos[1]:.4f} {hpos[2]:.4f}"
                part2 += f"{a_num} {i+1} 2 0.5897 {str_pos} 0 0 0\n"
                if self.nodes[hind]['Hup']:
                    fix.append(a_num)
                a_num += 1
        a_num -= 1
        part3 = """
Bonds

"""
        
        for i in range(len(hs)):
            part3 += f"{i + 1} 1 {1 + (i//2 * 3)} {(i//2) * 3 + 2 + i % 2}\n"
            
        part4 = """
Angles

"""
        
        for i in range(len(os)):
            part4 += f"{i+1} 1 {i*3 + 2} {i*3 +1} {i*3 + 3}\n"
            
        return part1 + part2 + part3 + part4, fix
    
        