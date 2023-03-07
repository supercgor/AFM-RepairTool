import os
import cv2
import numpy as np
from collections import OrderedDict

class Sampler():
    def __init__(self, name, path="/home/supercgor/gitfile/data"):
        self.abs_path = f"{path}/{name}"
        if not os.path.exists(self.abs_path):
            raise FileNotFoundError(f"Not such dataset in {self.abs_path}")
        self.datalist = os.listdir(f"{self.abs_path}/afm")

    def __getitem__(self, index):
        img_path = f"{self.abs_path}/afm/{self.datalist[index]}"
        pl = poscarLoader(f"{self.abs_path}/label")
        pl.load(f"{self.datalist[index]}.poscar")
        images = []
        for path in sorted(os.listdir(img_path), key=lambda x: int(x.split(".")[0])):
            img = cv2.imread(f"{img_path}/{path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)

        return self.datalist[index], images, pl

    def get(self, name):
        index = self.datalist.index(name)
        return self.__getitem__(index)

    def get_npy(self, index):
        loc = f"{self.abs_path}/npy/{self.datalist[index]}.npy"
        pred = np.load(loc)
        return pred

    def __len__(self):
        return len(self.datalist)

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
