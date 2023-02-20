import random
import numpy as np
from itertools import accumulate
import cv2
import matplotlib.pyplot as plt
from src.const import bound_dict, findPeakResolution
from findpeaks import findpeaks


seed = None
if seed is not None:
    random.seed(seed)


class randomStep():
    def __init__(self, max_value=20, sample_number=10):
        self.sn = sample_number
        self.mv = max_value
        self.steps = np.arange(
            0, max(max_value // sample_number, 1)) + 1  # e.g. [1,2]

        self.stepweights = np.exp(-0.5 * self.steps)
        # self.stepweights = np.full(len(self.steps),1, dtype=np.int32)

    def __next__(self):
        steps = [0] + random.choices(self.steps,
                                     weights=self.stepweights, k=self.sn-1)
        indices = np.asarray(list(accumulate(steps)))
        start = np.arange(0, self.mv - indices[-1])
        start = random.choices(start, weights=np.exp(-1 * start))
        return indices + start


class transform():
    def __init__(self, max_shift=5, noice=0.1, rec_size=0.01, rec_num=3):
        self.max_shift = max_shift
        self.noice = noice
        self.rec_size = rec_size
        self.rec_num = rec_num

    def act(self, img_list):
        for i in range(len(img_list)):
            shift = np.random.randint(-1 * self.max_shift, self.max_shift, 2)
            img_list[i] = pixel_shift(img_list[i], shift)
            img_list[i] = cutout(
                img_list[i], rec_size=self.rec_size, max_cut=self.rec_num)
            img_list[i] = add_noise(self.noice)


def pixel_shift(img, shift):
    translation_matrix = np.asarray(
        [[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float32)
    return cv2.warpAffine(img, translation_matrix, img.shape)

# a randomly rectangular place will be cut to zero
# the h, w of the rec box is the normal distribution of the square root(rec_size)
# the initial point of rec is uniform, if the rec touch the boundary, the rec wll be cut.


def cutout(img, rec_size=0.01, max_cut=3):
    height = img.shape[0]
    width = img.shape[1]
    for _ in range(random.randint(0, max_cut)):
        change_pixel = img.mean() * abs(np.random.normal(1, 0.1))
        change_pixel = np.clip(np.int8(change_pixel), 0, 255)

        rec_size = np.random.uniform(0, rec_size)
        max_size = int(height * width * rec_size)

        new_height = np.clip(abs(np.random.normal(
            np.sqrt(max_size), np.sqrt(max_size)*0.4)), 1, max_size)
        new_weight = max_size//new_height

        y = np.random.randint(0, height)
        x = np.random.randint(0, width)

        m_y, m_x = np.indices((new_height, new_weight))
        m_y = np.clip(m_y + y, 0, height - 1)
        m_x = np.clip(m_x + x, 0, width - 1)
        img[m_y, m_x] = change_pixel
    return img


def add_noise(img, c=0.1):
    noisemode, addmode = np.random.randint(0, [1, 2])
    if noisemode == 0:
        noise = np.random.normal(loc=0, scale=1, size=img.shape)
    elif noisemode == 1:
        noise = 0.5 * c * (img.max() - img.min())
        noise = np.random.uniform(-1 * noise, noise, img.shape)

    if addmode == 0:  # noise overlaid over image
        noisy = np.clip((img + noise * c * 255), 0, 255)
    elif addmode == 1:  # noise multiplied by image
        noisy = np.clip((img * (1 + noise * c)), 0, 255)
    elif addmode == 2:  # noise multiplied by bottom and top half images
        img2 = img / 255 * 2
        noisy = np.clip(np.where(img2 <= 1, (img2 * (1 + noise * c)),
                        (1 - img2 + 1) * (1 + noise * c) * -1 + 2) / 2, 0, 1)
        noisy = noisy * 255
    return noisy


def indexGenerator():
    i = 0
    while True:
        yield i
        i += 1


def cdist(mat_a: np.ndarray, mat_b: np.ndarray, diag=None):
    """_summary_
    Args:
        mat_a (np.ndarray): n * k
        mat_b (np.ndarray): m * k
        diag (bool, optional): _description_. Defaults to False.
    """
    x2 = np.sum(mat_a ** 2, axis=1)
    y2 = np.sum(mat_b ** 2, axis=1)
    xy = mat_a @ mat_b.T
    x2 = x2.reshape(-1, 1)
    out = x2 - 2*xy + y2
    out = out.astype(np.float32)
    out = np.sqrt(out)
    if diag is not None:
        np.fill_diagonal(out, diag)
    return out


def dist(pos_i, pos_j):
    return np.linalg.norm(pos_i - pos_j)


def isBound(node_i, node_j):
    pair = {node_i[1]['elem'], node_j[1]['elem']}
    pos_i = node_i[1]['position']
    pos_j = node_j[1]['position']
    if pair == {"O"}:
        return bound_dict[{'O', 'O'}]['lower'] <= dist(pos_i, pos_j) <= bound_dict[{'O', 'O'}]['upper']
    elif pair == {"O", "H"}:
        return bound_dict[{'O', 'H'}]['lower'] <= dist(pos_i, pos_j) <= bound_dict[{'O', 'H'}]['upper']
    elif pair == {"H"}:
        return bound_dict[{'H', 'H'}]['lower'] <= dist(pos_i, pos_j) <= bound_dict[{'H', 'H'}]['upper']


def imgsPeak(img: np.ndarray | list, fp: findpeaks | list, inverse=False, border=1, cluster_threshold = 3):
    out = []
    if isinstance(img, list) and isinstance(fp, findpeaks):
        for i in img:
            out.append(imgPeak(i, fp, inverse, border))
    elif isinstance(img, list) and isinstance(fp, list):
        for i, f in zip(img, fp):
            out.append(imgPeak(i, f, inverse, border))
    elif isinstance(img, np.ndarray) and isinstance(fp, list):
        for f in fp:
            out.append(imgPeak(img, f, inverse, border))
    else:
        out.append(imgPeak(img, fp, inverse, border))

    out = np.concatenate(out, axis=0)
    order = np.argsort(out[..., 2])[::-1]
    out = out[order]
    disMat = cdist(out[..., :2], out[..., :2]) < cluster_threshold
    disMat = np.triu(disMat, k=1)
    mem = {}
    for i, j in zip(*disMat.nonzero()):
        if i in mem:
            mem[j] = mem[i]
        else:
            mem[j] = i

    match = {}
    for i in range(len(out)):
        if i in mem:
            match[mem[i]].append(i)
        else:
            match[i] = [i]

    points = []
    for i in match:
        points.append(np.average(out[match[i]], axis=0))
    points = np.asarray(points)

    return points


def arrayReshape(inp, inpsize: tuple | np.ndarray, outsize: tuple | np.ndarray, dtype = None):
    print(outsize,inpsize)
    zoom = np.asarray(outsize) / np.asarray(inpsize)
    zoom = np.diag(zoom)
    if inp.shape[1] == 2:
        out = inp @ zoom[:2]
    elif inp.shape[1] == 3:
        out = inp @ zoom[:3]
    if dtype is not None:
        out.astype(dtype)
    return out

def imgPeak(img: np.ndarray, fp, inverse=False, border=1):
    if inverse:
        img = 255 - img
    resolution = img.shape[:2]
    out = fp.fit(img)
    out = out['persistence'].loc[:, ["x", "y", "score"]].to_numpy()
    if border != 0:
        de = np.logical_or(np.logical_or((out[..., 0] + 1 - border) <= 0, (out[..., 1] + 1 - border) <= 0),
                           (out[..., 0] - 1 + border) >= resolution[0], (out[..., 1] - 1 + border) >= resolution[1])
        de = de.nonzero()
        out = np.delete(out, de, axis=0)
    return out

def drawFindPeaksResult(img, pos, size= 3, color = (255,153,51), mirror=False):
    if mirror:
        img = cv2.flip(img, 0)
    
    resolution =  np.asarray(img.shape[:2]) / findPeakResolution
    resolution = np.diag(resolution)
    startPoints = pos - (size - 1)/2
    endPoints = pos + (size - 1)/2
    startPoints = (startPoints[...,:2] @ resolution).astype(int)
    endPoints = (endPoints[...,:2] @ resolution).astype(int)
    for sp, ep in zip(startPoints, endPoints):
        img = cv2.rectangle(img, sp, ep, color, -1)
        
    if mirror:
        img = cv2.flip(img, 0)
    return img

