from src.graph import Graph
import random
import numpy as np
np.random.seed(520)
random.seed(520)

class graphSolver():
    def __init__(self, g: Graph):
        self.g = g
        self.e = self.g.edges
        self.n = self.g.nodes

    def switch_solver(self):
        """若只有一粒的原子，四周能轉，則轉他

        Args:
            repeat (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        os = self.g.get_nodes_by_attributes("elem", "O")
        
        for oind in os:
            es = self.os(oind, mode = 1)
            es = [e for e in es if self.canswitch(e)]
            if len(es) >= 1:
                oos = [o for o in es if self.e[o]['type'] == "OObond"] # OO軌道
                if len(oos) == 0:
                    e = random.choice(es)
                else:
                    e = random.choice(oos)
                self.switch(e)
                    
                

    def edge_solver(self):
        """如果只有一條邊不滿足，則滿足那粒原子的那條邊

        Args:
            repeat (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        os = self.select_nodes_by_num(0, 2)
        random.shuffle(os)
        for oind in os:
            m = self.empty_edges(oind, mode = 1)
            if len(m) == 1:
                direction = self.pos(m[0])
                self.addH(oind,direction)

    def tri_solver(self):
        """刪去有三個氫的氧的某幾個氫
        """
        os = self.select_nodes_by_num(3, 100)
        for oind in os:
            n = self.os(oind, mode = 1) # 所有軌道
            hs = self.hs(oind)
            hs = [hind for hind in hs if not self.n[hind]['Hup']]
            os = self.empty_edges(oind, mode = 1) #未佔有的軌道
            m = self.select_node_if_linked(oind, inverse=True) #未成鍵氫
            m = [hind for hind in m if not self.n[hind]['Hup']]
            i_oos = [o for o in n if self.e[o]['type'] == "OObond" and o not in os and any(self.ainb(h, o) for h in hs)]# OO軌道
            i_loos = [o for o in n if self.e[o]['type'] == "LOObond" and o not in os and any(self.ainb(h, o) for h in hs)] # LOO軌道
            if len(m)>=1:
                self.g.rmAtom(random.choice(m))
            elif len(i_loos) >= 1:
                self.g.rmAtom(self.getH(oind, random.choice(i_loos)))
            elif len(i_oos) >= 1:
                self.g.rmAtom(self.getH(oind, random.choice(i_oos)))
            else:
                self.g.rmAtom(random.choice(hs))
            

    def adder_solver(self):
        """用以解決邊界上只有1-2個鍵的氧，而且填不滿氫

        """
        oss = self.select_nodes_by_bonds(0, 3)

        for oind in oss:
            if oind == 21:
                pass
            if len(self.hs(oind)) >= 2:
                    continue
            n = self.os(oind, mode = 1) # 所有軌道
            os = self.empty_edges(oind, mode = 1) #未佔有的軌道
            m = self.select_node_if_linked(oind) #成鍵氫
            i_oos = [o for o in n if self.e[o]['type'] == "OObond" and o not in os]# OO軌道
            i_loos = [o for o in n if self.e[o]['type'] == "LOObond" and o not in os] # LOO軌道
            oos = [o for o in os if self.e[o]['type'] == "OObond"]# OO空軌道
            loos = [o for o in os if self.e[o]['type'] == "LOObond"] # LOO空軌道
            if len(n) == 2: # 兩個鍵
                if len(self.hs(oind)) == 1: # 一個氫
                    if len(os) == 0: # 無未佔有軌道
                        e = self.os(oind, mode = 1)
                        if len(m) == 0:
                            direction = (self.pos(e[0]) + self.pos(e[1]))/2
                            if (direction - self.pos(oind)) @ (self.hs(oind)[0]-self.pos(oind)) < 0:
                                direction = self.pos(oind) *2 - direction
                        else:
                            e = e[0] if self.haveH(oind, e[1]) else e[1]
                            b = self.pos(e) - self.pos(oind)
                            a = self.pos(m[0]) - self.pos(oind)
                            b = b / np.linalg.norm(b)
                            a = a / np.linalg.norm(a)
                            j = b - (b @ a) * a
                            j = j / np.linalg.norm(j)
                            direction = -0.32 * a - 0.95 * j + self.pos(oind)
                        self.addH(oind, direction)
                    elif len(os) == 1: # 有一未佔有軌道
                        if len(m) == 0: # 沒有成鍵氫
                            self.addH(oind, self.pos(os[0]))
                        elif len(m) == 1: #有一個未成鍵氫
                            self.addH(oind, self.pos(m[0]))
                    elif len(os) >= 1: # 有2未佔有軌道
                        if len(oos) == 0:
                            indd = random.choice(os)
                            self.addH(oind, self.pos(indd))
                        elif len(oos) == 1:
                            indd = oos[0]
                            self.addH(oind, self.pos(indd))
                        elif len(oos) >= 2:
                            indd = random.choice(oos)
                            self.addH(oind, self.pos(indd))
                elif len(self.hs(oind)) == 0: # 沒有氫
                    if len(os)== 0: # 沒有空軌道
                        if len(i_oos) == 0: # 沒有一個OO 選一個最大的生成對氫
                            # 第一生成
                            s = random.sample(n, 2)
                            e, c, d = self.pos(oind),self.pos(s[0]), self.pos(s[1])
                            a_pal = 3 * e -c -d
                            nind = self.addH(oind, a_pal)
                            # 第二生成
                            b = np.asarray([0,0,-1])
                            a = self.pos(nind) - self.pos(oind)
                            b, a = b / np.linalg.norm(b), a / np.linalg.norm(a)
                            j = b - (b @ a) * a
                            j = j / np.linalg.norm(j)
                            direction = -0.32 * a + 0.95 * j + self.pos(oind)
                            self.addH(oind, direction)
                        elif len(i_oos) == 1: # 有一個OObond, 但被佔用了
                            # 第一生成
                            b = self.pos(random.choice(i_loos))
                            a = self.pos(i_oos[0]) - self.pos(oind)
                            b, a = b / np.linalg.norm(b), a / np.linalg.norm(a)
                            j = b - (b @ a) * a
                            j = j / np.linalg.norm(j)
                            direction = -0.32 * a + 0.95 * j + self.pos(oind)
                            nind = self.addH(oind, direction)
                            # 第二生成
                            b = np.asarray([0,0,-1])
                            a = self.pos(nind) - self.pos(oind)
                            b, a = b / np.linalg.norm(b), a / np.linalg.norm(a)
                            j = b - (b @ a) * a
                            j = j / np.linalg.norm(j)
                            direction = -0.32 * a + 0.95 * j + self.pos(oind)
                            self.addH(oind, direction)
                        elif len(i_oos) >= 2: # 有兩個OObond, 但被佔用了
                            # 第一生成
                            s = random.sample(i_oos, 2)
                            e, c, d = self.pos(oind),self.pos(s[0]), self.pos(s[1])
                            a_pal = 3 * e -c -d
                            nind = self.addH(oind, a_pal)
                            # 第二生成
                            b = np.asarray([0,0,-1])
                            a = self.pos(nind) - self.pos(oind)
                            b, a = b / np.linalg.norm(b), a / np.linalg.norm(a)
                            j = b - (b @ a) * a
                            j = j / np.linalg.norm(j)
                            direction = -0.32 * a + 0.95 * j + self.pos(oind)
                    elif len(os) == 1: #有一空軌
                        # 第一生成
                        indd = os[0]
                        nind = self.addH(oind, self.pos(indd))
                        # 第二生成
                        b = np.asarray([0,0,-1])
                        a = self.pos(nind) - self.pos(oind)
                        b, a = b / np.linalg.norm(b), a / np.linalg.norm(a)
                        j = b - (b @ a) * a
                        j = j / np.linalg.norm(j)
                        direction = -0.32 * a + 0.95 * j + self.pos(oind)
                        self.addH(oind, direction)
                    elif len(os) > 1: #有兩空軌
                        if len(oos) == 0:
                            indd = random.sample(os, 2)
                            for i in indd:
                                self.addH(oind, self.pos(i))
                        elif len(oos) == 1:
                            self.addH(oind, self.pos(oos[0]))
                            self.addH(oind, self.pos(loos[0]))
                        elif len(oos) >= 2:
                            indd = random.sample(oos, 2)
                            for i in indd:
                                self.addH(oind, self.pos(i))
            elif len(n) == 1: # 一個鍵
                if len(self.hs(oind)) == 1:
                    if len(m) == 1:
                        b = np.random.uniform(-1, 1, 3)
                        a = self.pos(m[0]) - self.pos(oind)
                        b, a = b / np.linalg.norm(b), a / np.linalg.norm(a)
                        j = b - (b @ a) * a
                        j = j / np.linalg.norm(j)
                        direction = -0.32 * a + 0.95 * j + self.pos(oind)
                        self.addH(oind, direction)
                    elif len(m) == 0:
                        indd = self.os(oind, mode=1)[0]
                        direction = self.pos(indd)
                        self.addH(oind, direction)
                elif len(self.hs(oind)) == 0:
                    if len(os) == 0:
                        # 第一生成
                        b = np.random.uniform(-1, 1, 3)
                        b[2] = 0
                        a = self.pos(n[0]) - self.pos(oind)
                        b, a = b / np.linalg.norm(b), a / np.linalg.norm(a)
                        j = b - (b @ a) * a
                        j = j / np.linalg.norm(j)
                        direction = -0.32 * a + 0.95 * j + self.pos(oind)
                        newind = self.addH(oind, direction)
                        # 第二生成
                        b = np.asarray([0,0,-1])
                        a = self.pos(newind) - self.pos(oind)
                        b, a = b / np.linalg.norm(b), a / np.linalg.norm(a)
                        j = b - (b @ a) * a
                        j = j / np.linalg.norm(j)
                        direction = -0.32 * a + 0.95 * j + self.pos(oind)
                        self.addH(oind, direction)
                    elif len(os) == 1:
                        # 第一生成
                        newind = self.addH(oind, self.pos(os[0]))
                        # 第二生成
                        b = np.asarray([0,0,-1])
                        a = self.pos(newind) - self.pos(oind)
                        b, a = b / np.linalg.norm(b), a / np.linalg.norm(a)
                        j = b - (b @ a) * a
                        j = j / np.linalg.norm(j)
                        direction = -0.32 * a + 0.95 * j + self.pos(oind)
                        self.addH(oind, direction)
            elif len(n) == 0:
                if len(os)== 0: # 沒有空軌道
                    if len(i_oos) == 0: # 沒有一個OO 選一個最大的生成對氫
                        # 第一生成
                        s = random.sample(n, 2)
                        e, c, d = self.pos(oind),self.pos(s[0]), self.pos(s[1])
                        a_pal = 3 * e -c -d
                        nind = self.addH(oind, a_pal)
                        # 第二生成
                        b = np.asarray([0,0,-1])
                        a = self.pos(nind) - self.pos(oind)
                        b, a = b / np.linalg.norm(b), a / np.linalg.norm(a)
                        j = b - (b @ a) * a
                        j = j / np.linalg.norm(j)
                        direction = -0.32 * a + 0.95 * j + self.pos(oind)
                        self.addH(oind, direction)
                    elif len(i_oos) == 1: # 有一個OObond, 但被佔用了
                        # 第一生成
                        b = self.pos(random.choice(i_loos))
                        a = self.pos(i_oos[0]) - self.pos(oind)
                        b, a = b / np.linalg.norm(b), a / np.linalg.norm(a)
                        j = b - (b @ a) * a
                        j = j / np.linalg.norm(j)
                        direction = -0.32 * a + 0.95 * j + self.pos(oind)
                        nind = self.addH(oind, direction)
                        # 第二生成
                        b = np.asarray([0,0,-1])
                        a = self.pos(nind) - self.pos(oind)
                        b, a = b / np.linalg.norm(b), a / np.linalg.norm(a)
                        j = b - (b @ a) * a
                        j = j / np.linalg.norm(j)
                        direction = -0.32 * a + 0.95 * j + self.pos(oind)
                        self.addH(oind, direction)
                    elif len(i_oos) >= 2: # 有兩個OObond, 但被佔用了
                        # 第一生成
                        s = random.sample(i_oos, 2)
                        e, c, d = self.pos(oind),self.pos(s[0]), self.pos(s[1])
                        a_pal = 3 * e -c -d
                        nind = self.addH(oind, a_pal)
                        # 第二生成
                        b = np.asarray([0,0,-1])
                        a = self.pos(nind) - self.pos(oind)
                        b, a = b / np.linalg.norm(b), a / np.linalg.norm(a)
                        j = b - (b @ a) * a
                        j = j / np.linalg.norm(j)
                        direction = -0.32 * a + 0.95 * j + self.pos(oind)

    def remain_solver(self):
        """用於解決3以上連結的粒子
        """
        oss = self.select_nodes_by_bonds(3, 10)
        for oind in oss:
            n = self.os(oind, mode = 1) # 所有軌道
            os = self.empty_edges(oind, mode=1) #未佔有的軌道
            m = self.select_node_if_linked(oind) #成鍵氫
            i_oos = [o for o in n if self.e[o]['type'] == "OObond" and o not in os]# OO軌道
            i_loos = [o for o in n if self.e[o]['type'] == "LOObond" and o not in os] # LOO軌道
            oos = [o for o in os if self.e[o]['type'] == "OObond"]# OO空軌道
            loos = [o for o in os if self.e[o]['type'] == "LOObond"] # LOO空軌道
            if len(self.hs(oind)) == 2: # 含有兩個氫
                continue
            elif len(self.hs(oind)) == 1: #只有一個氫
                if len(os) == 0: #沒有空軌
                    if len(m) == 0: #唯一的氫沒成鍵
                        if len(i_oos) == 0: # 沒有一個OO 選一個最大的生成對氫
                        # 第一生成
                            s = random.sample(n, 2)
                            e, c, d = self.pos(oind),self.pos(s[0]), self.pos(s[1])
                            a_pal = 3 * e -c -d
                            nind = self.addH(oind, a_pal)
                        elif len(i_oos) == 1: # 有一個OObond, 但被佔用了
                            # 第一生成
                            b = self.pos(random.choice(i_loos))
                            a = self.pos(i_oos[0]) - self.pos(oind)
                            b, a = b / np.linalg.norm(b), a / np.linalg.norm(a)
                            j = b - (b @ a) * a
                            j = j / np.linalg.norm(j)
                            direction = -0.32 * a + 0.95 * j + self.pos(oind)
                            nind = self.addH(oind, direction)
                        elif len(i_oos) >= 2: # 有兩個OObond, 但被佔用了
                            # 第一生成
                            c = max(i_oos, key = lambda x: self.e[x]['r'])
                            i_oos.remove(c)
                            d = max(i_oos, key = lambda x: self.e[x]['r'])
                            nind = self.addH(oind, (self.pos(c) + self.pos(d))/2)
                    elif len(m) == 1: #唯一的氫成了鍵
                        b = np.asarray([0,0,-1])
                        a = self.pos(m[0]) - self.pos(oind)
                        b, a = b / np.linalg.norm(b), a / np.linalg.norm(a)
                        j = b - (b @ a) * a
                        j = j / np.linalg.norm(j)
                        direction = -0.32 * a + 0.95 * j + self.pos(oind)
                        self.addH(oind, direction)  
                elif len(os) == 1: #有一空軌
                    indd = os[0]
                    self.addH(oind, self.pos(indd))
                elif len(os) > 1: #有兩空軌
                    if len(oos) == 0:
                        indd = random.choice(os)
                        self.addH(oind, self.pos(indd))
                    elif len(oos) == 1:
                        indd = oos[0]
                        self.addH(oind, self.pos(indd))
                    elif len(oos) >= 2:
                        indd = random.choice(oos)
                        self.addH(oind, self.pos(indd))
            elif len(self.hs(oind)) == 0: #沒有氫
                if len(os)== 0: # 沒有空軌道
                    if len(i_oos) == 0: # 沒有一個OO 選一個最大的生成對氫
                        # 第一生成
                        s = random.sample(n, 2)
                        e, c, d = self.pos(oind),self.pos(s[0]), self.pos(s[1])
                        a_pal = 3 * e -c -d
                        nind = self.addH(oind, a_pal)
                        # 第二生成
                        b = np.asarray([0,0,-1])
                        a = self.pos(nind) - self.pos(oind)
                        b, a = b / np.linalg.norm(b), a / np.linalg.norm(a)
                        j = b - (b @ a) * a
                        j = j / np.linalg.norm(j)
                        direction = -0.32 * a + 0.95 * j + self.pos(oind)
                        self.addH(oind, direction)
                    elif len(i_oos) == 1: # 有一個OObond, 但被佔用了
                        # 第一生成
                        b = self.pos(random.choice(i_loos))
                        a = self.pos(i_oos[0]) - self.pos(oind)
                        b, a = b / np.linalg.norm(b), a / np.linalg.norm(a)
                        j = b - (b @ a) * a
                        j = j / np.linalg.norm(j)
                        direction = -0.32 * a + 0.95 * j + self.pos(oind)
                        nind = self.addH(oind, direction)
                        # 第二生成
                        b = np.asarray([0,0,-1])
                        a = self.pos(nind) - self.pos(oind)
                        b, a = b / np.linalg.norm(b), a / np.linalg.norm(a)
                        j = b - (b @ a) * a
                        j = j / np.linalg.norm(j)
                        direction = -0.32 * a + 0.95 * j + self.pos(oind)
                        self.addH(oind, direction)
                    elif len(i_oos) >= 2: # 有兩個OObond, 但被佔用了
                        # 第一生成
                        s = random.sample(i_oos, 2)
                        e, c, d = self.pos(oind),self.pos(s[0]), self.pos(s[1])
                        a_pal = 3 * e -c -d
                        nind = self.addH(oind, a_pal)
                        # 第二生成
                        b = np.asarray([0,0,-1])
                        a = self.pos(nind) - self.pos(oind)
                        b, a = b / np.linalg.norm(b), a / np.linalg.norm(a)
                        j = b - (b @ a) * a
                        j = j / np.linalg.norm(j)
                        direction = -0.32 * a + 0.95 * j + self.pos(oind)
                elif len(os) == 1: #有一空軌
                    # 第一生成
                    indd = os[0]
                    nind = self.addH(oind, self.pos(indd))
                    # 第二生成
                    b = np.asarray([0,0,-1])
                    a = self.pos(nind) - self.pos(oind)
                    b, a = b / np.linalg.norm(b), a / np.linalg.norm(a)
                    j = b - (b @ a) * a
                    j = j / np.linalg.norm(j)
                    direction = -0.32 * a + 0.95 * j + self.pos(oind)
                    self.addH(oind, direction)
                elif len(os) > 1: #有兩空軌
                    if len(oos) == 0:
                        indd = random.sample(os, 2)
                        for i in indd:
                            self.addH(oind, self.pos(i))
                    elif len(oos) == 1:
                        self.addH(oind, self.pos(oos[0]))
                        self.addH(oind, self.pos(loos[0]))
                    elif len(oos) >= 2:
                        indd = random.sample(oos, 2)
                        for i in indd:
                            self.addH(oind, self.pos(i))

    def HH_destroyer(self):
        es = self.select_edge_by_num(2)
        for e in es:
            for ind in e:
                hind = self.getH(ind, e)
                u = self.empty_edges(ind, mode = 1)
                u = [i for i in u if self.e[i]['type'] == "OObond"]
                if self.n[hind]['Hup']:
                    continue
                elif len(u) >= 1:
                    self.g.rmAtom(hind)
                    break
                elif self.n[hind]['isGen'] == True:
                    self.g.rmAtom(hind)
                    break
            else:
                ind = random.choice(e)
                ind = self.getH(ind, e)
                self.g.rmAtom(ind)

    def addH(self, oind, direction):
        newind = self.g.addAtom("H", direction, (229, 255, 204), isGen = True)
        self.g.ppNode(oind, newind, 0.9584)
        self.g.linkAtom(oind, newind)
        self.g.linkH2B(newind)
        return newind

    def pos(self, ind):
        if isinstance(ind, int) or isinstance(ind, float):
            return self.n[ind]['position']
        else:
            return (self.n[self.e[ind]['u']]['position'] + self.n[self.e[ind]['v']]['position'])/2

    def isRational(self, oind):
        es = self.oobonds(oind)
        if len(self.n[oind]['bond']) != 2:
            return False
        for e in es:
            if self.numatom(e) != 1:
                return False
        return True

    def hs(self, oind, mode=0):
        out = self.n[oind]['bond']
        if mode == 0:
            return out
        else:
            return [(oind, hind) for hind in out]

    def empty_edges(self, oind, mode=0):
        """給出沒有H的連結

        Args:
            oind (_type_): _description_
            mode (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        os = self.os(oind, mode=1)
        out = []
        for o in os:
            if self.len(o) == 0:
                out.append(o)
        if mode == 1:
            return out
        else:
            return [o[0] if o[0] != oind else o[1] for o in out]
    
    def os(self, oind, mode=0):
        """
            mode 0: give index
            mode 1: give links
        """
        out = self.n[oind]['OObond'] + self.n[oind]['LOObond']
        if mode == 0:
            return out
        else:
            return [(oind, i) for i in out]

    def getH(self, oind, indd):
        e = self.e[indd]
        hind = e['unode'] if e['u'] == oind else e['vnode']
        if hind == {}:
            raise IndexError("No such hydrogen there")
        else:
            return list(hind.keys())[0]

    def getO(self, hind, indd):
        e = self.e[indd]
        oind = e['u'] if hind in e['unode'] else e['v']
        return oind

    def haveH(self, oind, indd):
        e = self.e[indd]
        hind = e['unode'] if e['u'] == oind else e['vnode']
        if hind == {}:
            return False
        hind = list(hind.keys())[0]
        if hind in self.n[oind]["bond"]:
            return True
        else:
            return False

    def select_nodes_by_bonds(self, start, stop=None):
        """按bond的數目給出nodes
        """
        out = self.g.get_nodes_by_attributes("elem", "O")
        if stop is None:
            stop = start + 1
        return [i for i in out if start <= self.len(i) < stop]

    def select_nodes_by_num(self, start, stop=None):
        """按atoms的數目給出nodes
        """
        out = self.g.get_nodes_by_attributes("elem", "O")
        if stop is None:
            stop = start + 1
        return [i for i in out if start <= len(self.hs(i)) < stop]

    def select_node_if_linked(self, oind, inverse=False):
        hinds = self.n[oind]['bond']
        if inverse:
            hinds = [hind for hind in hinds if self.n[hind]['onLink'] is None]
        else:
            hinds = [hind for hind in hinds if self.n[hind]['onLink'] is not None]
        return hinds
    
    def select_edge_by_num(self, start, stop=None):
        out = list(self.e)
        if stop is None:
            stop = start + 1
        return [e for e in out if self.e[e]['type'] in ["OObond", "LOObond"] and start <= self.len(e) < stop]
    
    def bond_hs(self, indd):
        out = []
        e = self.e[indd]
        if e['unode'] != {}:
            out.append(list(e['unode'].keys())[0])
        if e['vnode'] != {}:
            out.append(list(e['vnode'].keys())[0])
        return out

    def canswitch(self, indd):
        """2bond連結的switch 用以解決部分對着的H

        Args:
            oind (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.len(indd) != 1:
            return False
        ind_a, ind_b = indd[0], indd[1]
        max_num, min_num = max(len(self.hs(ind_a)), len(self.hs(ind_b))), min(len(self.hs(ind_a)), len(self.hs(ind_b)))
        if max_num > 2 and min_num < 2:
            return True
        else:
            return False

    def cfs(self, oind, indd):
        """判定能否switch(來)

        Args:
            oind (_type_): _description_
            indd (_type_): _description_

        Returns:
            _type_: _description_
        """
        ooinds = self.oobonds(oind)
        hinds = self.n[oind]['bond']
        bond_list = []
        for hind in hinds:
            for bond in ooinds:
                if indd == bond or indd == bond[::-1]:
                    continue
                if self.e[bond]['type'] != "OObond" and self.ainb(hind, bond):
                    bond_list.append(bond)
                    break

        return bond_list

    def find_pull_fallback(self, oind, indd=None, fall=1):
        try_list = []
        try_indd = None
        if indd is not None:
            try_list = self.cfs(oind, indd)
        if try_list != []:
            return [try_list[0]]
        if fall == 0:
            return try_list
        else:
            ooinds = self.oobonds(oind)
            if indd is None:
                ooinds = [i for i in ooinds if self.numatom(i) == 0]
            else:
                if indd in ooinds:
                    ooinds.remove(indd)
                else:
                    ooinds.remove(indd[::-1])
            if len(ooinds) >= 2:

                for try_indd in ooinds:
                    other_o_ind = try_indd[0] if try_indd[0] != oind else try_indd[1]
                    try_list = self.find_pull_fallback(
                        other_o_ind, try_indd, fall=fall - 1)
                    if try_list != []:
                        break
                else:
                    return try_list

                try_list.append(try_indd)
                return try_list

            # 一般不會發生, 因為只在 周圍有2個空位時才調用函數
            return []

    def cfps(self, oind, indd):
        """判定能否switch(去)

        Args:
            oind (_type_): _description_
            indd (_type_): _description_

        Returns:
            _type_: _description_
        """
        ooinds = self.oobonds(oind)
        bond_list = []
        for bond in ooinds:
            if indd == bond or indd == bond[::-1]:
                continue
            if self.numatom(bond) == 0 or (self.numatom(bond) == 1 and self.e[bond]['type'] == "LOObond"):
                other_ind = self.e[bond]['unode'] if self.e[bond]['u'] == oind else self.e[bond]['vnode']
                if other_ind != {}:
                    other_ind = list(other_ind.keys())[0]
                if other_ind in self.n[oind]['bond']:
                    continue
                bond_list.append(bond)
                break

        return bond_list

    def find_push_fallback(self, oind, indd=None, fall=1):
        try_list = []
        try_indd = None
        if indd is not None:
            try_list = self.cfps(oind, indd)
        if try_list != []:
            return [try_list[0]]
        if fall == 0:
            return try_list
        else:
            ooinds = self.oobonds(oind)
            if indd is not None:
                if indd in ooinds:
                    ooinds.remove(indd)
                else:
                    ooinds.remove(indd[::-1])

            hinds = self.n[oind]['bond']
            filter_ooinds = []
            for bond in ooinds:
                if self.numatom(bond) == 1 and not any(self.ainb(hind, bond) for hind in hinds):
                    filter_ooinds.append(bond)

            for try_indd in filter_ooinds:
                other_o_ind = try_indd[0] if try_indd[0] != oind else try_indd[1]
                try_list = self.find_push_fallback(
                    other_o_ind, try_indd, fall=fall - 1)
                if try_list != []:
                    break
            else:
                return try_list

            try_list = [try_indd] + try_list
            return try_list

    def fallback_solver(self, indd=None, fall=1):
        remain = self.get_num_atom(1)
        for ind in remain:
            path = self.find_pull_fallback(ind, indd=indd, fall=fall)
            if path == []:
                path = self.find_push_fallback(ind, indd=indd, fall=fall)
                if path == []:
                    continue
            i = 0
            while i < len(path) - 1:
                indd1 = path[i]
                indd2 = path[i+1]
                self.indd2indd(indd1, indd2)
                i += 1

            self.point_solver(repeat=0)

    def indd2indd(self, indd1, indd2):
        """_summary_

        Args:
            indd1 (_type_): del
            indd2 (_type_): gen
        """
        for i in indd1:
            if i in indd2:
                break

        j = indd1[0] if indd1[1] == i else indd1[1]
        k = indd2[0] if indd2[1] == i else indd2[1]
        e1 = self.e[indd1]
        rm = e1['unode'] if e1['u'] == i else e1['vnode']
        pos = (self.n[i]['position'] + self.n[k]['position']) / 2
        if rm != {}:
            rm = list(rm.keys())[0]
            pos[2] = self.n[rm]['position'][2]
            self.g.rmAtom(rm)

        newind = self.g.addAtom("H", pos)
        self.g.ppNode(i, newind, 0.9584)
        self.g.linkAtom(i, newind)
        self.g.linkH2B(newind)



    def switch(self, indd):
        if self.canswitch(indd):
            hind = self.bond_hs(indd)[0]
            oind = self.getO(hind, indd)
            oind = indd[0] if oind != indd[0] else indd[1]
            self.g.rmAtom(hind)
            self.addH(oind, self.pos(indd))
            

    def inverbond(self, indd):
        e = self.e[indd]

        if len(e['unode']) != 0:
            rmind = list(e['unode'].keys())[0]
            addind = e['v']
        else:
            rmind = list(e['vnode'].keys())[0]
            addind = e['u']

        pos = self.n[rmind]['position']
        self.g.rmAtom(rmind)
        newind = self.g.addAtom("H", pos, color=(229, 255, 204))
        self.g.ppNode(addind, newind, 0.9584)
        self.g.linkAtom(addind, newind)
        self.g.linkH2B(newind)

    def numatom(self, indd):
        return len(self.e[indd]['unode']) + len(self.e[indd]['vnode'])

    def len(self, ind):
        if isinstance(ind, int) or isinstance(ind, float) or isinstance(ind, np.ndarray):
            return len(self.n[ind]['OObond']) + len(self.n[ind]['LOObond'])
        elif isinstance(ind, tuple):
            if ind not in self.e:
                return 0
            return len(self.e[ind]['unode']) + len(self.e[ind]['vnode'])

    def oobonds(self, ind):
        node = self.n[ind]
        out = []
        for oind in node['OObond'] + node['LOObond']:
            out.append((ind, oind))
        return out

    def get_num_atom(self, num):
        out = self.g.get_nodes_by_attributes("elem", "O")
        return [i for i in out if len(self.n[i]['bond']) == num]

    def ainb(self, aind, indd):
        e = self.e[indd]
        if aind in e['unode'] or aind in e['vnode']:
            return True
        else:
            return False

    def solve(self):
        self.switch_solver()
        self.edge_solver()
        self.switch_solver()
        self.edge_solver()
        self.switch_solver()
        self.edge_solver()
        self.switch_solver()
        self.tri_solver()
        self.tri_solver()
        self.adder_solver()
        self.remain_solver()
        self.HH_destroyer()
        self.adder_solver()
        self.remain_solver()
        self.adder_solver()
        self.remain_solver()
        return