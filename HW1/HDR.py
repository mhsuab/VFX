import os
import cv2
import math
import argparse
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from tqdm import tqdm

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest = 'directory', help = 'directory of the input images', type = str, default = 'original_images')
    parser.add_argument('-p', dest = 'prefix', help = 'prefix of the images', type = str, default = 'IMG_')
    parser.add_argument('-n', dest = 'number', help = 'index of first image', type = int, default = 6570)
    parser.add_argument('-i', dest = 'index', help = 'total number of images', type = int, default = 9)
    parser.add_argument('-s', dest = 'suffix', help = 'file type of image', type = str, default = '.JPG')
    parser.add_argument('-f', dest = 'filename', help = 'filename of the output numpy array', type = str, default = 'raw_rgb')
    parser.add_argument('-a', dest = 'array', help = 'exist saved np files for images and enter its prefix', type = str, default = '')
    args = parser.parse_args()
    return args

args = init_arg()

N = 50
P = args.index

color = ['red','green','blue']

#exposure time in sec
exposure_time = [1/4, 1/2, 1, 2, 4, 8, 16, 32, 64]
log_exposure_time = [math.log(i) for i in exposure_time]

#weighted function
shift = 10
Zmin, Zmax = 0, 255
weightedFunc = lambda z: (z - Zmin + shift) if z <= (Zmin + Zmax)/2 else (Zmax - z + shift)
# weightedFunc = lambda z: z
w = np.array([weightedFunc(i) for i in range(256)])

_lambda = 100
lambda_str = 'lambda({})'.format(_lambda)

dir_path = 'result'

def show_raw(images):
    fig, axes = plt.subplots(3, 3, figsize = (12, 8))
    for i in range(len(images)):
        r = int(i/3)
        c = int(i%3)
        axes[r][c].imshow(cv2.resize(cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR), (0,0), fx = 0.3, fy = 0.3))

    plt.savefig(os.path.join(dir_path, 'allImages.png'))
    #plt.show()

def read_img(directory, prefix, number, index, suffix, filename):
    images = []
    images_rgb = [[], [], []]
    for i in tqdm(range(index)):
        images.append(cv2.imread(os.path.join(directory, prefix + str(number + i) + suffix)))
    
    #image alignment
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)

    for img in images:
        b, g, r = cv2.split(img)
        images_rgb[0].append(r)
        images_rgb[1].append(g)
        images_rgb[2].append(b)

    images_rgb = np.array(images_rgb)
    np.save(os.path.join(dir_path, filename), images_rgb)
    np.save(os.path.join(dir_path, 'images'), images)

    # show_raw(images)

    return images, images_rgb

def getCurve(z, curveColor):
    n = 254
    
    A = np.zeros((np.size(z, axis=0) * np.size(z,axis=1) + 1 + n, 256 + np.size(z, axis=0)))
    b = np.zeros((np.size(A,0),1))
    
    k = 0
    for i in range(np.size(z,0)):
        for j in range(np.size(z,1)):
            A[k][z[i][j]] = w[z[i][j]]
            A[k][Zmax + i + 1] = - w[z[i][j]]
            b[k][0] = log_exposure_time[j] * w[z[i][j]]
            k += 1
    
    A[np.size(z, axis=0) * np.size(z,axis=1)][127] = 1
    
    for i in range(n):
        A[np.size(z, axis=0) * np.size(z,axis=1) + i + 1][i] = _lambda * w[i + 1]
        A[np.size(z, axis=0) * np.size(z,axis=1) + i + 1][i + 1] = _lambda * (-2) * w[i + 1]
        A[np.size(z, axis=0) * np.size(z,axis=1) + i + 1][i + 2] = _lambda * w[i + 1]

    pseudoInvA = np.linalg.pinv(A)
    
    x = np.dot(pseudoInvA,b)
    g = x[0:256]
    
    
    plt.plot(g, color = curveColor)
    plt.savefig(os.path.join(dir_path, 'response_curve.png'))
    
    return g
    #plt.show()

def createHDRImage(rgb_images, g, color):
    lnE_r = np.zeros(rgb_images[0].shape[1:]).astype(np.float32)
    lnE_g = np.zeros(rgb_images[0].shape[1:]).astype(np.float32)
    lnE_b = np.zeros(rgb_images[0].shape[1:]).astype(np.float32)
    for x in tqdm(range(len(lnE_r))):
        for y in range(len(lnE_r[0])):
            de_r = 0.
            de_g = 0.
            de_b = 0.
            for j in range(len(rgb_images[0])):
                lnE_r[x][y] += w[rgb_images[0][j][x][y]] * (g[0][rgb_images[0][j][x][y]] - log_exposure_time[j])
                lnE_g[x][y] += w[rgb_images[1][j][x][y]] * (g[1][rgb_images[1][j][x][y]] - log_exposure_time[j])
                lnE_b[x][y] += w[rgb_images[2][j][x][y]] * (g[2][rgb_images[2][j][x][y]] - log_exposure_time[j])
                de_r += w[rgb_images[0][j][x][y]]
                de_g += w[rgb_images[1][j][x][y]]
                de_b += w[rgb_images[2][j][x][y]]
            lnE_r[x][y] /= de_r if de_r > 0 else 1.0
            lnE_g[x][y] /= de_g if de_g > 0 else 1.0
            lnE_b[x][y] /= de_b if de_b > 0 else 1.0
    lnE = np.array([lnE_r, lnE_g, lnE_b])
    return np.exp(lnE).astype(np.float32), lnE.astype(np.float32)

def plotRAD(rad, f, title):
    plt.imshow(rad, origin = 'lower', cmap = 'rainbow', interpolation = 'nearest') 
    plt.gca().invert_yaxis() 
    plt.colorbar() 
    plt.title(title) 
    plt.savefig(f) 
    plt.close()

if __name__=='__main__':
    print ('read images')
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    if args.array:
        images, rgb_images = np.load(os.path.join(dir_path, 'images.npy')), np.load(os.path.join(dir_path, args.array + '.npy'))
        #print (args.array)
    else:
        images, rgb_images = read_img(args.directory, args.prefix, args.number, args.index, args.suffix, args.filename)
        #print (args.directory, args.prefix, args.number, args.index)

    #get funcG
    print ('compute response curve')
    z = np.zeros((3,N,args.index),dtype = 'uint8') #[channel, pixel_num, image_num]
    for i in range(N):
        # random select pixel
        x = rd.randint(0,np.shape(rgb_images)[2] - 1) 
        y = rd.randint(0,np.shape(rgb_images)[3] - 1)
        for j in range(args.index):
            z[0][i][j] = rgb_images[0][j][x][y]
            z[1][i][j] = rgb_images[1][j][x][y]
            z[2][i][j] = rgb_images[2][j][x][y]
    
    g = []
    color = ['red','green','blue']
    for i in range(3):
        g.append(getCurve(z[i], color[i]))
    g = np.array(g).reshape(3, -1)

    # get radianceMap, final result
    print ('construct radiance map')
    radianceImg, lnE = createHDRImage(rgb_images, g, color)
    np.save(os.path.join(dir_path, 'RadMap'), radianceImg)
    np.save(os.path.join(dir_path, 'lnE'), lnE)
    cv2.imwrite(os.path.join(dir_path, 'HDR.hdr'), cv2.merge(radianceImg[::-1]))


