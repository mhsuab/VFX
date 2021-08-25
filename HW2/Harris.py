import os
import cv2
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters

TEXT_FACE = 0
TEXT_SCALE = 10
TEXT_THICKNESS = 3
TEXT = "x"
TEXT_COLOR = (0, 0, 255)

text_size, _ = cv2.getTextSize(TEXT, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
marked = lambda img, cor: cv2.putText(img, TEXT, (cor[0] - text_size[0]//2, cor[1] + text_size[1]//2), TEXT_FACE, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest = 'read', help = 'True to load images directly from "images.npy"', type = bool, default = False)
    parser.add_argument('-d', dest = 'directory', help = 'directory of the input images', type = str, default = 'raw_image')
    parser.add_argument('-s', dest = 'suffix', help = 'file type of image', type = str, default = '.JPG')
    args = parser.parse_args()
    return args

args = init_arg()

def plotImages(images, cmap, filename):
    print ('SAVE {}......'.format(filename))
    fig, axes = plt.subplots(2, 4, figsize = (12, 8))
    for i in range(len(images)):
        r = int(i/4)
        c = int(i%4)
        axes[r][c].imshow(cv2.resize(images[i], (0,0), fx = 0.3, fy = 0.3), cmap=cmap)
    plt.savefig('{}.png'.format(filename))
    plt.close()

def readImage(dir, extension):
    print ('READ images......')
    files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(extension)]
    files.sort(key = lambda x: int(x[-8:-4]))
    print (files)
    return np.array([cv2.imread(i) for i in files])

def CornerResponse(img, sigma = 3):
    print ('COMPute corner response R......')
    # Implement Harris Detection
    Ix = filters.gaussian_filter(img, (sigma, sigma), (0, 1))
    Iy = filters.gaussian_filter(img, (sigma, sigma), (1, 0))
    Sxx = filters.gaussian_filter(Ix*Ix, sigma)
    Sxy = filters.gaussian_filter(Ix*Iy, sigma)
    Syy = filters.gaussian_filter(Iy*Iy, sigma)
    
    det = Sxx * Syy - Sxy ** 2
    trace = Sxx + Syy
    # trace = Sxx + Syy + 1e-12
    
    k = 0.04 #0.04 - 0.06
    R = det - k * (trace ** 2)
    # R = det/trace
    return R

CornerDescription = lambda img, cor: img[cor[1]-1:cor[1]+2, cor[0]-1:cor[0]+2].flatten()

def CornerDescriptions(img, cors):
    return [CornerDescription(img, cor) for cor in cors]
    return list(map(CornerDescription, img, cors))
    # y, x = cor[0], cor[1]
    # if x >= 1 and y >= 1 and x < (img.shape[0] - 1) and y < (img.shape[1] - 1):
    #     return img[x-1:x+2, y-1:y+2]
    # print ('DAMN')

def CornerDetector(R, g, f, window=100, thres=100):
    print ('DETEct corner......')
    # print (R.dtype, np.max(R), np.min(R), np.mean(R))
    w = R.shape[1]

    R[:window, :] = 0
    R[-window:, :] = 0
    R[:, :window] = 0
    R[:, -window:] = 0
    
    LocalMax = filters.maximum_filter(R, (window, window))
    R = R * (R == LocalMax)

    idxs = np.argsort(R.flatten())[::-1][:thres]
    y = idxs // w
    x = idxs % w

    cor = np.vstack((x, y)).T
    return cor
    '''
    for i in coords:
        marked(g, (int(i[0]), int(i[1])))
    cv2.imwrite('tmp{}.jpg'.format(f), g)
    '''
    return {list(CornerDescription(g, (i[0], i[1]))): (i[0], i[1]) for i in cor}

def CornerMatching(img1, img2, d1, d2):
    pass

if __name__ == '__main__':
# input images
    # cv2 image : b g r
    if args.read:
        print ('LOAD images.......')
        images = np.load(os.path.join(args.directory, 'images.npy'))
        grayImages = np.load(os.path.join(args.directory, 'gray_images.npy'))
    else:
        images = readImage(args.directory, args.suffix)
        print ('SAVE images npy......')
        np.save(os.path.join(args.directory, 'images'), images)
        grayscale = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grayImages = np.array(list(map(grayscale, images)))
        np.save(os.path.join(args.directory, 'gray_images'), grayImages)
# plot gray scale images
    # plotImages(grayImages, 'gray', 'grayImages')

# corner detection
    R = list(map(CornerResponse, grayImages))
    cors = list(map(CornerDetector, R, images, [i for i in range(len(R))]))
    np.save('cors', np.array(cors))
    Desp = list(map(CornerDescriptions, images, cors))
    np.save('desp1', np.array(Desp[3]))
    np.save('desp2', np.array(Desp[4]))
    cors = np.load('cors.npy')

    from scipy.spatial import cKDTree
    tree = cKDTree(np.load('desp1.npy'))
    dist, idx = tree.query(np.load('desp2.npy'))

    for i in idx:
        marked(images[3], (int(cors[3][i][0]), int(cors[3][i][1])))
    cv2.imwrite('marked.jpg', images[3])

    concateImg = np.concatenate((images[4], images[5]), axis=1)
    
    concateImg = cv2.cvtColor(concateImg, cv2.COLOR_BGR2RGB)

    cols1 = images[4].shape[1]
    print (cols1)
    plt.figure(figsize=(30,20))
    plt.imshow(concateImg)
    for i, m in enumerate(idx):
        x1, y1 = cors[4][m]
        x2, y2 = cors[5][i]
        plt.plot([x1, x2+cols1], [y1, y2], c=[np.random.random(), np.random.random(), np.random.random()])
    plt.axis('off')
    plt.savefig('concate')