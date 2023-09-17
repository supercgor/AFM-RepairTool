# %%
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# %%
from src.graph import Graph
from src.solver import graphSolver
from src.poscar import poscar

# model = "unet_tune_v1"
model = "unet_v0"
repair_dir = "../data/ice_cluster/npy"
save_dir = "../data/ice_cluster/result"
# %%
files = os.listdir(f"{repair_dir}/{model}")
for file in files:
    print(file)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(20, 10)
    grid = poscar._load_npy(f"{repair_dir}/{model}/{file}")
    pos_dict = poscar.box2pos(grid, threshold= 0.5)
    g = Graph("", np.ones((3, 128,128), dtype = np.uint8) * 128,pos_dict, res = 10)
    g.make_edges()
    img = g.plotNodes(0)
    # img = g.plotEdges(img, text = False, mirror= True, thickness_mul = 0.2)
    ax1.imshow(img)
    gs = graphSolver(g)
    gs.solve()
    g.pp_all_nodes()
    img = g.plotNodes(0)
    img = g.plotEdges(img, text = False, mirror= True, thickness_mul = 0.4)
    ax2.imshow(img)
    plt.show()
    os.makedirs(f"{save_dir}/{model}_repair", exist_ok=True)
    g.save(save_dir = f"{save_dir}/{model}_repair/{file.split('.')[0]}_fix.poscar")
    cv2.imwrite(f"{save_dir}/{model}_repair/{file.split('.')[0]}.png", img)
# %%
