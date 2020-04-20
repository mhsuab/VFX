import cv2
import numpy as np
import sys
from tqdm import tqdm

def PhotographicGlobal(hdr, d, a):
    Lm = (a/np.exp(np.mean(np.log(d + hdr)))) * hdr
    Lm_max = np.max(Lm)
    Ld = (Lm * (1 + (Lm/(Lm_max**2)))) / (1 + Lm)
    return np.clip(np.array(Ld * 255), 0, 255).astype(np.uint8)

def gaussian_blurs(img, smax = 25, a = 1.0, fi = 8.0, epsilon = 0.01):
    h, w = img.shape
    blur_prev = img
    num_s = int((smax + 1)/2)
    blur_list = np.zeros(img.shape + (num_s,))
    vs_list = np.zeros(img.shape + (num_s, ))
    for i, s in enumerate(range(1, smax+1, 2)):
        blur = cv2.GaussianBlur(img, (s, s), 0)
        vs = np.abs((blur - blur_prev)/(2**fi*a/s**2 + blur_prev))
        blur_list[:, :, i] = sblur
        vs_list[:, :, i] = vs
    smax = np.argmax(vs_list>epsilon, axis = 2)
    smax[np.where(smax==0)] = 1
    smax -= 1
    I, J = np.ogrid[:h, :w]
    return blur_list[I, J, smax]

def PhotographicLocal(hdr, d = 1e-6, a = 0.5, method = 0):
    result = np.zeros_like(hdr, dtype = np.float32)
    weights = [0.065, 0.67, 0.265]
    Lw_ave = np.exp(np.mean(np.log(d + hdr)))
    for c in range(3):
        Lw = hdr[:, :, c]
        Lm = (a/Lw_ave) * Lw
        Ls = gaussian_blurs(Lm)
        Ld = Lm/(1 + Ls)
        result[:, :, c] = np.clip(np.array(Ld * 255), 0, 255).astype(np.uint8)
    return result

def gammatonemap(img, g):
    return cv2.pow(img/255., 1.0/g)