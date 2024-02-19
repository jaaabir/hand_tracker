import numpy as np
from torch.utils.data import Dataset
from hutils import *
import torch

class TrainValidIndex:
    def __init__(self, n, test_size, seed):
        idx = np.arange(n).tolist()
        np.random.seed(seed)
        self.trainIdx = np.random.choice(idx, size = int(n * (1 - test_size)), replace = False).tolist()
        self.testIdx = list(set(idx).difference(set(self.trainIdx)))
        print(f'Length of training data : {len(self.trainIdx)}')
        print(f'Length of testing data : {len(self.testIdx)}')

class Loader(Dataset):
    def __init__(self, df, img_size, transform = None, subset = 'train', val_split = 0.1, seed = 0):
        splitIdx = TrainValidIndex(df.shape[0], val_split, seed)
        if subset == 'train':
            self.data = df.values[splitIdx.trainIdx]
        else:
            self.data = df.values[splitIdx.testIdx]
        
        self.transform = transform
        self.img_size  = img_size
        
    def __getitem__(self, index):
        ipath, iid, ab_path, _ = self.data[index]
        iid = ipath.split('/')[-1].split('_')[0]
        image = plt.imread(ipath)
        jpath = os.path.join(ab_path, f'{iid}_joint_pos.txt')
        joints = get_joints(jpath)
        kpts = convert_3d_to_2d(joints).astype(int)
        h,w,_ = image.shape
        bbox = get_crop_uv(kpts, hw = (h, w), thresh = 30)
        if self.transform:
            image, kpts = self.transform(image, kpts, bbox, self.img_size)
        
        ### normalization
#       iage = image.astype(np.float32) / 255
        kpts = np.array(kpts).astype(np.uint32)
        kpts = (kpts.astype(np.float32) / self.img_size).flatten()
        
        ### np - torch
        image = torch.from_numpy(image).permute(-1, 0, 1)
        kpts = torch.from_numpy(kpts).to(torch.float32)
        
        return image, kpts
        
    
    def __len__(self):
        return self.data.shape[0]