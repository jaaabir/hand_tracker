import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os,re,glob
import cv2 as cv
from tqdm import tqdm
import matplotlib.colors as mcolors

K_depth = np.array([
      [475.62, 0, 311.125],
      [0, 475.62, 245.965],
      [0, 0, 1]
])

K_color = np.array([
    [617.173, 0, 315.453],
    [0, 617.173, 242.259],
    [0, 0, 1]
])

E_depth = np.eye(4)

E_color = np.array([
    [1, 0, 0, 24.7],
    [0, 1, 0, -0.0471401],
    [0, 0, 1, 3.72045],
    [0, 0, 0, 1]
])

def get_ipath(root, pattern = '*on_depth.png'):
    images = glob.glob(os.path.join(root, '*object', '*', '*', '*', pattern))
    return images

def get_joints(jname):
    with open(jname, 'r') as file:
        kpts = file.read()
    kpts = re.split(r',|\n', kpts)[:-1]
    kpts = np.array(list(map(float, kpts))).astype(np.float32).reshape(-1, 3)
    ones = np.ones((kpts.shape[0], 1))
    kpts = np.hstack((kpts, ones))
    return kpts

def convert_3d_to_2d(kpts):
    x_depth = np.dot(E_depth, kpts.T).T
    u_depth = (K_depth[0, 0] * x_depth[:, 0] / x_depth[:, 2]) + K_depth[0, 2]
    v_depth = (K_depth[1, 1] * x_depth[:, 1] / x_depth[:, 2]) + K_depth[1, 2]
    uv_depth = np.hstack((u_depth.reshape(-1, 1), v_depth.reshape(-1, 1))).astype(np.uint32)
    return uv_depth

def renderPose(img, uv, return_lbl = False):
    colors = ['black'] + ['blue'] * 4 + ['orange'] * 4 + ['lime'] * 4 + ['yellow'] * 4 +  ['cyan'] * 4
    joints_pos = [
    'W', 
    'T0', 'T1', 'T2', 'T3', 
    'I0', 'I1', 'I2', 'I3', 
    'M0', 'M1', 'M2', 'M3', 
    'R0', 'R1', 'R2', 'R3', 
    'L0', 'L1', 'L2', 'L3'
    ]
    
    if type(uv) == list:
        uv = np.array(uv)
    connections = [[0,1], [1,2], [2,3], [3,4], 
                   [0,5], [5,6], [6,7], [7,8], 
                   [0,9], [9,10],[10,11], [11,12],
                   [0,13],[13,14], [14,15], [15,16], 
                   [0,17],[17,18], [18,19], [19, 20],
                   [1,5], [5,9], [9,13], [13,17]
                  ]
    
    for c in connections:
        a,b = uv[c[0]].astype(int),uv[c[1]].astype(int)
        img = cv.line(img, a, b, (255,0,0), 1)
        
    for ind, point in enumerate(uv):
        
        color = colors[ind]
        rgb = tuple(map(lambda x : x * 255, mcolors.to_rgb(color)))
        img = cv.circle(img, point.astype(int), 1, rgb, 2)
        
    if return_lbl:
        return img, labels
        
    return img

def show_img_from_path(path, uv = None):
    img = plt.imread(path)
    show_img(img, uv)
    
def show_img(img, uv = None):
    plt.title(img.shape)
    plt.imshow(img if uv is None else renderPose(img, uv))
    plt.axis(False)
    
import albumentations as A

def get_crop_uv(kpts, hw = (480, 640), thresh = 10):
    h,w = hw
    x1, x2 = np.min(kpts[:, 0]), np.max(kpts[:, 0])
    y1, y2 = np.min(kpts[:, 1]), np.max(kpts[:, 1])
    x1, x2 = max(0, x1 - thresh), min(x2 + thresh, w) 
    y1, y2 = max(0, y1 - thresh), min(y2 + thresh, h) 
    return x1,y1,x2,y2

def crop_and_resize(img, kpts, box, hw = (224, 224)):
    x1,y1,x2,y2 = box
    h, w = hw
    transform = A.Compose([
        A.Crop(x_min = x1, x_max = x2, y_min = y1, y_max = y2, always_apply= True, p=1),
        A.Resize(height = h, width = w, always_apply=True, p=1)
    ], keypoint_params = A.KeypointParams(format="xy", remove_invisible = False))
    transformed = transform(
          image=img,
          keypoints=kpts,
        )
    img  = transformed['image']
    kpts = transformed['keypoints']
    return img, kpts



def remove_useless_images(root, pattern):
    images = get_ipath(root, pattern)
    images = [file for file in images if '_on_' not in file]
    print(' Images based on given pattern '.center(80, '='))
    for i in images[:3]:
        print(f' {i} '.center(80, '='))
    print('...')
    print('-'.center(80, '='))

    choice = input('Are you sure, you want to remove these images (y/n) : ')

    if choice == 'y':
        for num, image in tqdm(enumerate(images)):
            print(f'Removing {image}')
            os.remove(image)
        print()
        print(f'Removed {num + 1} images')

    print(' DONE '.center(80, '='))

