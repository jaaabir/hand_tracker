import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import PIL
import urllib
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from tqdm import tqdm 
from time import time
import matplotlib.colors as mcolors
import cv2 as cv

class Config:
    def __init__(self, cls = "Human hand", subset = 'train', root_path = "/content/drive/MyDrive/"):
        self.path = {
            'obj_code'   : '/m/',
            'class'      : cls,
            'ROOT'       : root_path,
            'images'     : root_path + cls + '/{}/data',
            'labels'     : root_path + cls + '/{}/labels/detections.csv',
            'metadata'   : root_path + cls + '/{}/metadata/image_ids.csv',
        }
        
def validate_file(fname, ftype):
  leng = len(ftype)
  if fname[-leng:] != ftype:
    fname = f'{fname}.{ftype}'
  return fname

def read_json(fname):
  fname = validate_file(fname, ftype = 'json')
  with open(fname, 'r') as file:
    data = json.load(file)
  file.close()
  return data

def timeit(func):
  def wrapper(*args, **kwargs):
    st = time()
    res = func(*args, **kwargs)
    ed = time()
    print(f'Finished in {round(ed - st)} sec(s) ...')
    return res

  return wrapper

@timeit
def num_of_files(path):
  return len(os.listdir(path))

def get_subset_leng(config : dict, verbose : bool = True):
  train = config['images'].format('train')
  test  = config['images'].format('test')
  validation = config['images'].format('validation')
  train = num_of_files(train)
  validation = num_of_files(validation)
  test = num_of_files(test)
  if verbose:
    print(f'Number of files in train : {train}')
    print(f'Number of files in validation : {validation}')
    print(f'Number of files in test : {test}')
  return train, validation, test

@timeit
def read_csvs(paths):
  return [pd.read_csv(i) for i in paths]

def voc_to_yolo(box):
  x1, y1, x2, y2 = box

  # from pascal voc - yolo
  x = (x2 + x1) / 2
  y = (y2 + y1) / 2
  w = (x2 - x1)
  h = (y2 - y1)

  return x, y, w, h

def xy_to_wh(bbox):
  x1, y1, x2, y2 = bbox
  w = x2 - x1
  h = y2 - y1

  return w, h

def cxcy_to_xy(bbox, height, width):
  x_center, y_center, w, h = bbox
  w = w * width
  h = h * height
  x1 = ((2 * x_center * width) - w) // 2
  y1 = ((2 * y_center * height) - h) // 2

  return x1, y1

def plot_img_bbox(ax, image, target, show_labels = True, normalized = True, bbox_format = 'pascal_voc'):
    img = image.copy()
    bboxes = target['boxes']
    try:
      height, width, _ = img.shape
    except:
      height, width = img.shape
    color = 'red'
    for ind, bbox in enumerate(bboxes):

      if bbox_format == 'pascal_voc':
        if normalized:
          bbox = [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]
        w, h = xy_to_wh(bbox)

      else:
        x1, y1 = cxcy_to_xy(bbox, height, width)
        bbox = [x1, y1, bbox[2] * width, bbox[3] * height]
        w, h = bbox[2], bbox[3]

      rect = Rectangle((bbox[0], bbox[1]), w, h, linewidth = 1, edgecolor = color, facecolor='none')
      if show_labels:
          ax.text(bbox[0] + 0.05, bbox[1] + 0.05, 'Human hand', fontsize = 'medium', backgroundcolor = color)
      ax.add_patch(rect)

    ax.imshow(img)
    ax.axis(False);

def get_boxes(fname, df):
  idf = df[df.ImageID == fname]
  boxes = []
  for ind, rdf in idf.iterrows():
    coords = [rdf.XMin , rdf.YMin , rdf.XMax, rdf.YMax]
    boxes.append(coords)
  return {'boxes' : boxes}

def read_img(path):
  if path[: 8] == 'https://':
     return np.array(PIL.Image.open(urllib.request.urlopen(path)))
  path = validate_file(path, 'jpg')
  return plt.imread(path)
    
def get_rows(n, col):
  if n % col == 0:
    return n // col
  return (n // col) + 1

def show_images(config, path, df, n = 10, col = 5, subset = 'train', show_labels = False):
  row = get_rows(n, col)
  if type(path) == str:
    path = config['images'].format(subset)
    images = os.listdir(path) if type(path) == str else path
    timgs = config[f'{subset}_length']
  else:
    images = path
    timgs = len(images)
  indexes = np.random.randint(0, timgs, n)
  fig, ax = plt.subplots(row, col, figsize = (col * 5, row * 5))
  ind = 0
  for r in range(row):
    for c in range(col):
      image = images[indexes[ind]]
      name  = image.split('.')[0]
      image = os.path.join(config['images'].format(subset), image)
      image = read_img(image)
      if show_labels:
        target = get_boxes(name, df)
        if row == 1:
            plot_img_bbox(ax[c], image, target, show_labels = False)
        else:
          plot_img_bbox(ax[r, c], image, target, show_labels = False)
      else:
        plt.figure(figsize = (col * 5, row * 5))
        plt.tight_layout()
        plt.subplot(row, col, ind + 1)
        plt.title(f'shape : {image.shape}')
        plt.imshow(image)
        plt.axis(False)
      ind += 1
      
def compute_area(boxes, h, w, normalized = True):
  xmin, xmax, ymin, ymax = boxes
  if normalized:
    xmin, xmax = xmin * w, xmax * w
    ymin, ymax = ymin * h, ymax * h
  width = xmax - xmin + 1
  height = ymax - ymin + 1
  area = width * height
  return area

@timeit
def prepare_df_subset(config, subset):
  labels_path   = config['labels'].format(subset)
  images_path   = config['images'].format(subset)
  labels        = pd.read_csv(labels_path) 
  images        = os.listdir(images_path)
  indexes       = [i.split('.')[0] for i in images]
  df            = labels.copy()
  df.index      = df.ImageID
  df            = df.loc[indexes, :].reset_index(drop = True)
  df            = df[df.LabelName == config['obj_code']].reset_index(drop = True)
  width         = df.XMax - df.XMin + 1 
  height        = df.YMax - df.YMin + 1
  areas         = width * height 
  q1, q3        = np.percentile(areas, q = [25, 75])
  df['area']    = areas
  new_df        = df[df.IsGroupOf == 0]
  return new_df


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
  
