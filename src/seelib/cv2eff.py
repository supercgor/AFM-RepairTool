import numpy as np
import cv2
import src.seelib.npmath as npmath
import matplotlib.pyplot as plt


def bnc(img, brightness, contrast):
    output = img * (contrast/127 + 1) - contrast + brightness
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output


def brightness(img, b):
    return bnc(img, b, 0)


def contrast(img, c):
    return bnc(img, 0, c)

def drawPoints(img, pos: np.ndarray, size = 1, color = (255,255,255), flip = None):
    """pos has to be strictly between 0 to 1 and 2d
    """
    if flip is not None:
        pos[...,flip] = 1 - pos[...,flip]
        
    pos = pos * img.shape
    startPoints = pos - (size - 1)/2
    endPoints = pos + (size - 1)/2
    startPoints = (startPoints[...,:2]).astype(int)
    endPoints = (endPoints[...,:2]).astype(int)
    for sp, ep in zip(startPoints, endPoints):
        # 這裡需要反向因為為numpy 是用 h, w format的, 而cv2是用 w, h format
        img = cv2.rectangle(img, sp[::-1], ep[::-1], color, -1)
        
    return img

def genText(text, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
            fontScale=1,
            thickness=1,
            color=(0, 0, 0),
            flip=None,
            background=np.nan):
    size, _ = cv2.getTextSize(str(text), fontFace, fontScale, thickness)
    size = (size[1], size[0], len(color))
    out = np.full(size, background)
    if not np.isnan(background):
        out = out.astype(np.uint8)
    cv2.putText(out, str(text), (0, size[0] - 1),
                fontFace, fontScale, color, thickness=thickness)
    if flip is not None:
        out = cv2.flip(out, flip)
    return out


def transBind(background, img, pos=(0, 0), mode="topleft"):
    bg = background.copy()

    h_bg, w_bg = bg.shape[0], bg.shape[1]

    h, w = img.shape[0], img.shape[1]

    x = pos[0]
    y = pos[1]

    if mode == "center":
        x = x - int(w/2)
        y = y - int(h/2)

    mask = ~np.isnan(img)

    if x >= 0 and y >= 0:

        h_part = h - max(0, y+h-h_bg)
        w_part = w - max(0, x+w-w_bg)

        mask = mask[0:h_part, 0:w_part, ...]

        bg[y:y+h_part, x:x+w_part, ...][mask] = img[0:h_part, 0:w_part, ...][mask]

    elif x < 0 and y < 0:

        h_part = h + y
        w_part = w + x

        mask = mask[h-h_part:h, w-w_part:w, ...]
        bg[0:0+h_part, 0:0+w_part, ...][mask] = img[h -
                                                    h_part:h, w-w_part:w, ...][mask]

    elif x < 0 and y >= 0:

        h_part = h - max(0, y+h-h_bg)
        w_part = w + x

        mask = mask[0:h_part, w-w_part:w, ...]
        bg[y:y+h_part, 0:0+w_part, ...][mask] = img[0:h_part, w-w_part:w, ...][mask]

    elif x >= 0 and y < 0:

        h_part = h + y
        w_part = w - max(0, x+w-w_bg)

        mask = mask[h-h_part:h, 0:w_part, ...]
        bg[0:0+h_part, x:x+w_part, ...][mask] = img[h-h_part:h, 0:w_part, ...][mask]

    return bg


def order_circ_mask(radius: int, pos: tuple = (0,0), org=None, repeat = False):
    if org is None:
        mask = np.zeros((radius * 2 + 1,radius * 2 + 1))
        pos = (pos[0] + radius, pos[1] + radius)
    else:
        mask = np.zeros(org.shape[:2])
    mask = cv2.circle(mask, pos, radius, 1, 1)
    order = mask.nonzero()
    pos_order = np.asarray(order).T
    pos_order = pos_order - pos
    l1 = pos_order[np.where(np.logical_and(
        pos_order[..., 1] >= 0, pos_order[..., 0] < 0))]
    l2 = pos_order[np.where(np.logical_and(
        pos_order[..., 1] >= 0, pos_order[..., 0] >= 0))]
    l2 = l2[np.argsort(l2[..., 1])][::-1]
    l2 = l2[np.argsort(l2[..., 0])]
    l3 = pos_order[np.where(np.logical_and(
        pos_order[..., 1] < 0, pos_order[..., 0] >= 0))][::-1]
    l4 = pos_order[np.where(np.logical_and(
        pos_order[..., 1] < 0, pos_order[..., 0] < 0))]
    l4 = l4[np.argsort(l4[..., 0])][::-1]
    l4 = l4[np.argsort(l4[..., 1])]
    a = np.concatenate([l1, l2, l3, l4])
    if org is not None:
       a = a + pos
    if repeat:
        a = np.asarray([a[-1], *a, a[0]])
    a = (a[..., 0], a[..., 1])
    
    return a


if __name__ == "__main__":
    a = npmath.nans((10, 10))
    a[0, 0] = 1
    a[5, 5] = 2
    b = np.zeros((10, 10))
    c = transBind(b, a, mode="topleft")
    print(c)
    d = genText("abc", background=255)
    cv2.imshow("123", d)
    cv2.waitKey(0)
