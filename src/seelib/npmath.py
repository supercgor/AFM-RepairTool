import numpy as np

def dist(pos_i, pos_j):
    return np.linalg.norm(pos_i - pos_j)

def nans(*args, **kwargs):
    out = np.full(*args, **kwargs)
    out[:] = np.nan
    return out
    
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

def mask_sum(mat: np.ndarray, mask: np.ndarray):
    value = mat[mask.nonzero()]
    return np.sum(value)

def mask_mean(mat: np.ndarray, mask: np.ndarray):
    value = mat[mask.nonzero()]
    return np.mean(value)

def mask_std(mat: np.ndarray, mask: np.ndarray):
    value = mat[mask.nonzero()]
    return np.std(value)

def mask_max(mat: np.ndarray, mask: np.ndarray, axis = None):
    mat[np.where(mask == 0)] = - np.inf
    return np.max(mat, axis=axis)

def mask_min(mat: np.ndarray, mask: np.ndarray, axis = None):
    mat[np.where(mask == 0)] = np.inf
    return np.min(mat, axis=axis)

def mask_argmax(mat: np.ndarray, mask: np.ndarray, axis = None):
    mat[np.where(mask == 0)] = - np.inf
    return np.argmax(mat, axis=axis)

def mask_argmin(mat: np.ndarray, mask: np.ndarray, axis = None):
    mat[np.where(mask == 0)] = np.inf
    return np.argmin(mat, axis=axis)

def v_angle(vec_a, vec_b, mode="rad"):
    """_summary_

    Args:
        vec_a (_type_): _description_
        vec_b (_type_): _description_
        mode (str, optional): rad or deg. Defaults to "rad".
    """
    out = np.arccos((vec_a @ vec_b) /np.linalg.norm(vec_a)/ np.linalg.norm(vec_b))
    if np.isnan(out):
        out = 0
    if mode == "deg":
        out = out * 180 / np.pi
    return out

def p_angle(point_a, center, point_b, mode="rad"):
    """_summary_

    Args:
        vec_a (_type_): _description_
        vec_b (_type_): _description_
        mode (str, optional): rad or deg. Defaults to "rad".
    """
    return v_angle(point_a - center, point_b - center, mode)