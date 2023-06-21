import os
import cv2
import numpy as np
import torch
from collections import OrderedDict
from .poscar import poscar
import json

class Sampler():
    def __init__(self, path = "../data/bulkexp", modelname = "tune_UNet_strong_baseline_withup", use_poscar = False):
        self.path = path
        self.modelname = modelname
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Not such dataset in {self.path}")
        self.use_poscar = use_poscar
        self.afm = os.listdir(f"{self.path}/combine_afm")
        self.label = os.listdir(f"{self.path}/combine/{modelname}")

    def __getitem__(self, index):
        name = self.afm[index]
        imgs = []
        pths = os.listdir(f"{self.path}/combine_afm/{name}")
        pths = sorted([p for p in pths if p.endswith(".png")], key=lambda x: int(x.split(".")[0]))
        readme = json.load(open(f"{self.path}/combine_afm/{name}/readme.json"))
        for path in pths:
            img = cv2.imread(f"{self.path}/combine_afm/{name}/{path}", cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, readme["reso"])
            img = np.flip(img.T, axis = 1)
            img = img[None, ...]
            imgs.append(img)
            
        imgs = np.stack(imgs) / 255
        imgs = torch.from_numpy(imgs).float()
        
        if self.use_poscar:
            dic = poscar._load_poscar(f"{self.path}/combine/{self.modelname}/{name}.poscar")
        else:
            dic = {}
            box = poscar._load_npy(f"{self.path}/combine/{self.modelname}/{name}.npy") # D H W E C
            dic["real_size"] = (3.0, readme["size"][0] * 10, readme["size"][1] * 10)
            dic["elem"] = ("O", "H")
            dic["scale"] = 1.0
            pos = poscar.box2pos(box, real_size = dic["real_size"],threshold = 0.5, nms = True)
            Z, X, Y = box.shape[:3]
            dic['pos'] = pos
        
        

        return name, imgs, dic

    def get(self, name):
        index = self.afm.index(name)
        return self.__getitem__(index)

    def get_npy(self, index):
        name = self.label[index].split(".poscar")[0] if self.use_poscar else self.npy[index].split(".npy")[0]
        loc = f"{self.path}/npy/{self.modelname}/{name}.npy"
        pred = np.load(loc)
        return pred

    def __len__(self):
        if self.use_poscar:
            return len(self.label)
        else:
            return len(self.npy)

    def __next__(self):
        for i in range(self.__len__):
            return self.__getitem__(i)

class dataLoader():
    def __init__(self, path, g):
        pass

class poscarLoader():
    def __init__(self, path, model_name = None, lattice=(25, 25, 3), out_size=(32, 32, 4), elem=("O", "H"), cutoff=OrderedDict(O=2.2, H=0.8), scale = 1.0):
        self._path = path
        self.name = ""
        self.model_name = model_name
        self._lattice = np.asarray(lattice)
        self.out_size = np.asarray(out_size)
        self._elem = elem
        self._elemnum = {}
        self._cutoff = cutoff
        self._zoom = [i/j for i, j in zip(lattice, out_size)]
        self._pos = {}
        self._scale = scale

    def load(self, name, NMS=True):
        """Load the poscar file or npy file. For npy file the Tensor should have the shape of ( B, X, Y, Z, 8).

        Args:
            name (str): file name

        Returns:
            info: dict with keys: 'scale': 1.0, 'lattice': diag_matrix, 'elem_num': 2, 'ele_name': ('O', 'H'), 'comment'
        """
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"No such directory: {self._path}")

        if name.split(".")[1] == "npy":
            self._load_npy(name, NMS=NMS)
        
        else:
            with open(f"{self._path}/{name}") as fr:
                comment = fr.readline().split("\x00")[0]
                line = fr.readline()
                self._scale = float(_clean(line)[0])
                lattice = []
                for _ in range(3):
                    lattice.append(_clean(fr.readline()).astype(float))
                self._lattice = np.diag(np.asarray(lattice))
                self._elem = _clean(fr.readline())
                self.elemnum = _clean(fr.readline()).astype(int)
                fr.readline()
                fr.readline()
                for ele, num in zip(self._elem, self.elemnum):
                    position = []
                    for _ in range(num):
                        line = _clean(fr.readline())
                        position.append(line[:3].astype(float) * self._lattice)
                    self._pos[ele] = np.asarray(position)
            if NMS:
                for ele, pos in self._pos.items():
                    self._pos[ele] = self.nms(pos, self._cutoff[ele])
        self.name = name

    def _load_npy(self, name, NMS=True, conf=0.5):
        pred = np.load(f"{self._path}/{name}")  # ( X, Y, Z, 8 )
        self.npy2pos(pred, NMS=NMS, conf=conf)

    def npy2pos(self, pred, NMS=True, conf=0.5):
        self._lattice = pred.shape[:3]
        ind = np.indices(self._lattice)
        ind = np.transpose(ind, (1, 2, 3, 0))
        pred = pred.cpu().numpy()
        pred = pred.reshape((*self._lattice, 2, 4))
        pred = np.transpose(pred, (3, 0, 1, 2, 4))
        pred[..., :3] = (pred[..., :3] + ind) * self._zoom
        for elem, submat in zip(self._cutoff, pred):
            select = submat[..., 3] > conf
            offset = submat[select]
            offset = offset[np.argsort(offset[..., 3])][::-1]
            if NMS:
                offset = self.nms(offset, self._cutoff[elem])
            self._pos[elem] = offset[..., :3]

    @staticmethod
    def nms(pos, cutoff):
        reduced_index = np.full(pos.shape[0], True)
        dis_mat = cdist(pos[..., :3], pos[..., :3]) < cutoff
        dis_mat = np.triu(dis_mat, k=1)
        trues = dis_mat.nonzero()
        for a, b in zip(*trues):
            if reduced_index[a]:
                reduced_index[b] = False
        return pos[reduced_index]

    def save(self, name, save_dir = None):
        output = ""
        output += f"{' '.join(self._elem)}\n"
        output += f"{1:3.1f}" + "\n"
        output += f"\t{self._lattice[0]:.8f} {0:.8f} {0:.8f}\n"
        output += f"\t{0:.8f} {self._lattice[1]:.8f} {0:.8f}\n"
        output += f"\t{0:.8f} {0:.8f} {self._lattice[2]:.8f}\n"
        output += f"\t{' '.join([str(ele) for ele in self._pos])}\n"
        output += f"\t{' '.join([str(self._pos[ele].shape[0]) for ele in self._pos])}\n"
        output += f"Selective dynamics\n"
        output += f"Direct\n"
        for ele in self.pos:
            p = self._pos[ele]
            for a in p:
                output += f" {a[0]/self._lattice[0]:.8f} {a[1]/self._lattice[1]:.8f} {a[2]/self._lattice[2]:.8f} T T T\n"

        if save_dir is None:
            path = f"{self._path}/result/{self.model_name}"
        else:
            path = f"{save_dir}"
            
        if not os.path.exists(path):
            os.mkdir(path)

        with open(f"{path}/{name}.poscar", 'w') as f:
            f.write(output)
        return

    def save4npy(self, name, pred, NMS=True, conf=0.7):
        return self.save(name, self.npy2pos(pred, NMS=NMS, conf=conf))

    def upload_pos(self, pos):
        self._pos = pos
            
    @property
    def pos(self):
        return self._pos
    
    @property
    def elem(self):
        return self._elem
    
    @property
    def lattice(self):
        return self._lattice

def _clean(line, splitter=' '):
    """
    clean the one line by splitter
    all the data need to do format convert
    ""splitter:: splitter in the line
    """
    data0 = []
    line = line.strip().replace('\t', ' ').replace('\x00', '')
    list2 = line.split(splitter)
    for i in list2:
        if i != '':
            data0.append(i)
    temp = np.array(data0)
    return temp


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
        out = out[..., 0]
    return out
