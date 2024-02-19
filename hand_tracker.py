import cv2 as cv 
import numpy as np 
import torch 
import os 
from time import time 
import albumentations as A
import pyautogui as pg
from Mobnet import MobNet as Model, predict_
from ultralytics import YOLO
from torchvision.utils import draw_bounding_boxes as dbb
import matplotlib.colors as mcolors
import pafy

def renderPose(img, uv, return_lbl = False):
    colors = ['blue', 'orange', 'lime', 'yellow', 'cyan', 'black']
    root_lbls = ['index', 'middle', 'pinky', 'ring', 'thumb', 'center']
    
    labels = ['index3', 'index4', 'index2', 'index1', 
              'middle3', 'middle4', 'middle2', 'middle1',
              'pinky3', 'pinky4', 'pinky2', 'pinky1',
              'ring3', 'ring4', 'ring2', 'ring1',
              'thumb3', 'thumb4', 'thumb2', 'thumb1'
              'center'
             ]
    if type(uv) == list:
        uv = np.array(uv)
    connections = [[0,1], [2,0], [3,2], [4,5], 
                   [6,4], [7,6], [6,7], [8,9], 
                   [11,10], [10,8],[12,13], [14,12],
                   [15,14],[16,17], [18,16], [19,18], 
                   [20,1],[20,5], [20,9], [20,13], [20, 17]]
    iterr = -1
    for c in connections:
        a,b = uv[c[0]].astype(int),uv[c[1]].astype(int)
        img = cv.line(img, a, b, (255,0,0), 1)
        
    for ind, point in enumerate(uv):
        if ind % 4 == 0:
            iterr += 1
        rgb = tuple(map(lambda x : x * 255, mcolors.to_rgb(colors[iterr])))
        img = cv.circle(img, point.astype(int), 1, rgb, -1)
        
    if return_lbl:
        return img, labels
        
    return img

def renderBox(img, box):
    img = torch.from_numpy(img).permute(-1, 0, 1)
    boxes = torch.unsqueeze(box, 0)
    return dbb(img, boxes).permute(1, -1, 0).numpy()

def load_model(fname, root = 'models'):
    print(f'Loading model {fname} ...')
    path = os.path.join(root, fname)
    return torch.load(path)

def show_img(image, lable = None):
    if lable is not None:
        x,y = 10, 10
        color = (255,0,0)
        cv.putText(image, str(lable), (x,y), cv.FONT_HERSHEY_PLAIN, 1, color, 1, cv.LINE_AA)
    cv.imshow('hand tracker', image)
    
def flipImage(image):
    transform = A.Compose([A.HorizontalFlip(always_apply=True, p=1.0)])
    img = transform(image = image)['image']
    return img
    
def control_mouse(x, y, duration = 0.1):
    x, y = int(x), int(y)
    pg.moveTo(x, y, duration = duration)
    
def close_cam(cap):
    cap.release()
    cv.destroyAllWindows()
    
def final_transform(kpts, img_size = [480, 640]):
    image = np.zeros(shape=(224, 224))
    Transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
    ], 
    keypoint_params = A.KeypointParams(format="xy", remove_invisible = False)
    )
    
    transformed = Transform(
          image=image,
          keypoints=kpts,
        )
    kpts = transformed['keypoints']
    kpts = np.array(kpts)
    return kpts
    
def main(vc = 0, mode = 'detect', move_mouse = False, device = 'cpu'):
    # mode : [ detect, keypoints, both ]
    choice = input('Caputure from cam or yt (c/y) : ')
    if choice == 'y':
        video = pafy.new(input('yt link : '))
        best = video.getbest(preftype="webm")
        cap = cv.VideoCapture(best.url)
    else:
        cap = cv.VideoCapture(vc, cv.CAP_DSHOW) 
    old_fps, new_fps = 0, 0
    while cap.isOpened():
        new_fps = time()
        _, frame = cap.read()
        
        fps  = 1/(new_fps - old_fps)
        image = np.array(frame)
        # image = imutils.resize(image, width=1920, height=1080)
        box = []
        bb_img = None
        
        if mode == 'detect' or mode == 'both':
            result = YMODEL.predict(image.copy())[0]
            conf = result.boxes.conf.cpu().numpy()
            if conf.shape[0] > 0:
                top_conf = np.argmax(conf)
                box = result.boxes.xyxy[top_conf]
                bb_img = renderBox(image.copy(), box)
                
        kp_img = None
        index_finger = None
            
        if (mode == 'keypoints' or mode == 'both') and len(box) > 0:
            img, kpts  = predict_(MODEL, image.copy(), box, device = device)
            mkpt  = final_transform(kpts, img_size = (1080, 1920))
            index_finger = mkpt[3]
            kp_img   = renderPose(img.copy(), kpts)

        
        
        if kp_img is not None:
            show_img(kp_img, fps)
        
        print(index_finger)
        if move_mouse and index_finger is not None:
            x, y = index_finger
            control_mouse(x, y)
        
        old_fps = new_fps

        if cv.waitKey(10) == ord('q'):
            close_cam(cap)
            break
            
            
if __name__ == '__main__':
    IMG_SIZE = 224
    device = 'cuda'
    MODEL = Model(k = 21 * 2)
    MODEL.load_state_dict(load_model('mobnet3.pt'))
    YMODEL = YOLO(os.path.join('models','best.pt'))
    MODEL.to(device)
    YMODEL.to(device)
    print(YMODEL)
    print(f'GPU : {torch.cuda.is_available()}')
    main(vc = 0, mode='both', move_mouse=True, device = device)