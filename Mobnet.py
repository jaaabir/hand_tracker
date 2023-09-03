from torchvision import models 
import torch
import pytorch_lightning as pl 
import torch.nn as nn
import albumentations as A
import numpy as np 

class MobNet(pl.LightningModule):
    def __init__(self, k = 21*2, test_size = 0.2, verbose = False):
        super(MobNet, self).__init__()
        torch.cuda.empty_cache()
        self.test_size = test_size
        self.verbose = verbose
        self.criterion = nn.MSELoss().cuda()
        self.model = models.mobilenet_v2(pretrained=True, progress = True)
        num_features = self.model.classifier[1].in_features
        if verbose:
            print(f'Number of features in the last layer : {num_features}')
            print(f'Number of keypoints to detect : {k}')
        self.model.classifier = nn.Sequential(
             self.model.classifier[0],
             nn.Linear(num_features, num_features // 2),
             nn.ReLU(inplace=True),
             nn.Linear(num_features // 2, 128),
             nn.ReLU(inplace=True),
             nn.Linear(128, k),
             nn.ReLU(inplace=True)
            )
        print(self.model)
    
    def forward(self, image):
        return self.model(image)
    
    def training_step(self, batch, batch_idx):
        loss, _ = self.tr_process_(batch, batch_idx)
        self.log('train_loss', loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, preds = self.tr_process_(batch, batch_idx)
        self.log('val_loss', loss, sync_dist=True)
        return loss
    
    def tr_process_(self, batch, batch_idx):
        imgs, kpts = batch 
        kpts = torch.autograd.Variable(kpts)
        preds = self.forward(imgs)
        loss = self.criterion(preds, kpts) 
        return loss, preds
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.01)
        scheduler = {
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose = True),
        'monitor': 'val_loss',
        'interval': 'epoch',
        'frequency': 2
        }
        return [optimizer], [scheduler]
    
    
def predict_(model, img, bbox, IMG_SIZE = 224):
    x1,y1,x2,y2 = list(map(int, bbox))
    transformer = A.Compose([
        A.Crop(x_min = x1, y_min = y1, x_max = x2, y_max = y2, always_apply=True, p=1.0), 
        A.Resize(IMG_SIZE, IMG_SIZE)])
    img = (transformer(image = img)['image'] / 255).astype(np.float32) 
    img = torch.from_numpy(img).permute(-1, 0, 1) 
    img = torch.unsqueeze(img, dim = 0)                   
    model.eval()
    with torch.no_grad():
        preds = torch.squeeze(model(img))
    return torch.squeeze(img.detach()).permute(1,-1,0).numpy().copy(), (preds.reshape(-1, 2).numpy() * IMG_SIZE).astype(np.uint32)
    
