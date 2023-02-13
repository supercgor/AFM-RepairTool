import os
import json
import torch
from torch import nn
import cv2
import numpy as np
from torch.utils.data.dataset import Dataset
from src.tools import randomStep

class modelLoader():
    def __init__(self, name, path = "/home/supercgor/gitfile/data/model"):
        self.abs_path = f"{path}/{name}"
        self.new = False
        # check
        if not os.path.exists(self.abs_path):
            os.mkdir(self.abs_path)
            self.new = True 
        # load 
        self._load()
    
    def _load(self):
        if self.new:
            # create new model
            #TODO
            pass
        else:
            # load old model
            with open(f'{self.abs_path}/info.json') as f:
                self.config = json.load(f)
            
            if self.config['network'] == "unet":
                from network.unet3d_model import UNet3D as model
            
            self._model = model(1, self.config['channel'], self.config['Z'])
            
            self._model.load_state_dict(torch.load(f"{self.abs_path}/{self.config['name']}"))
    
    def cuda(self, parallel=False):
        self._model = self._model.cuda()
        if parallel:
            device_ids = list(map(int,os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
            self._model = nn.DataParallel(self._model, device_ids=device_ids)
    
    @property
    def model(self):
        return self._model
    
class sampler():
    def __init__(self, name, path ="/home/supercgor/gitfile/data"):
        self.abs_path = f"{path}/{name}"
        if not os.path.exists(self.abs_path):
            raise FileNotFoundError(f"Not such dataset in {self.abs_path}")
        self.datalist = os.listdir(f"{self.abs_path}/afm")
    
    def __getitem__(self, index):
        img_path = f"{self.abs_path}/afm/{self.datalist[index]}"
        pl = poscarLoader(f"{self.abs_path}/label")
        info, positions = pl.load(f"{self.datalist[index]}.poscar")
        images = []
        for path in sorted(os.listdir(img_path), key=lambda x: int(x.split(".")[0])):
            images.append(cv2.imread(f"{img_path}/{path}"))
        
        return {"info": info, "image": images, "position": positions}
    
    def __len__(self):
        return len(self.datalist)
    
class poscarLoader():
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not such label file in {path}")
        self.path = path
    
    def load(self, name):
        """
        read the POSCAR or CONTCAR of VASP FILE
        and return the data position
        """
        abs_path = f"{self.path}/{name}"
        with open(abs_path) as fr:
            comment = fr.readline().split("\x00")[0]
            line = fr.readline()
            scale_length = float(_clean(line)[0])
            lattice = []
            for i in range(3):
                lattice.append(_clean(fr.readline()).astype(float))
            lattice = np.array(lattice)
            ele_name = _clean(fr.readline())
            counts = _clean(fr.readline()).astype(int)
            ele_num = dict(zip(ele_name, counts))
            fr.readline()
            fr.readline()
            positions = {}
            for ele in ele_name:
                position = []
                for _ in range(ele_num[ele]):
                    line = _clean(fr.readline())
                    position.append(line[:3].astype(float))
                positions[ele] = np.asarray(position)
        info = {'scale': scale_length, 
                'lattice': lattice, 
                'ele_num': ele_num,
                'ele_name': tuple(ele_name),
                'comment': comment}
        return info, positions
    
    def save(self, info):
        pass
    
#TODO  
# class dataloader(Dataset):
#     def __init__(self, name, path ="/home/supercgor/gitfile/data", mode="train"):
#         self.abs_path = f"{path}/{name}"
#         if not os.path.exists(self.abs_path):
#             raise FileNotFoundError(f"Not such dataset in {self.abs_path}")
#         self.datalist = os.listdir(f"{self.abs_path}/afm")
    
#     def __getitem__(self, index):
#         img_path = f"{self.abs_path}/afm/{self.datalist[index]}"
#         pl = poscarLoader(f"{self.abs_path}/label")
#         info, positions = pl.load(f"{self.datalist[index]}.poscar")
#         images = []
#         for path in os.listdir(img_path):
#             images.append(cv2.imread(f"{img_path}/{path}"))
        
#         return {"info": info, "image": images, "position": positions}
    
#     def __len__(self):
#         return len(self.datalist)

    
def _clean(line, splitter=' '):
    """
    clean the one line by splitter
    all the data need to do format convert
    ""splitter:: splitter in the line
    """
    data0 = []
    line = line.strip().replace('\t', ' ').replace('\x00','')
    list2 = line.split(splitter)
    for i in list2:
        if i != '':
            data0.append(i)
    temp = np.array(data0)
    return temp
