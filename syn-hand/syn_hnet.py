import torchvision.models as models
import pytorch_lightning as pl
import torch.nn as nn
import os
import torch
from hutils import crop_and_resize
from data_loader import Loader
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('medium')
class HNet(pl.LightningModule):
    def __init__(self, df = None, k = 21*2, img_size = (299,299), test_size = 0.2, verbose = False, seed = 0, batch = 64):
        super(HNet, self).__init__()
        torch.cuda.empty_cache()
        self.test_size = test_size
        self.verbose = verbose
        self.df = df
        self.SEED = seed
        self.BATCH = batch
        self.img_size = img_size
        self.criterion = nn.MSELoss().cuda()
        self.pt_path = os.path.join('models', 'inception_v3_google-0cc3c7bd.pth')
        self.model = models.inception_v3(weights= models.Inception_V3_Weights.DEFAULT, progress = True)
        num_features = self.model.fc.in_features
        if verbose:
            print(f'Number of features in the last layer : {num_features}')
            print(f'Number of keypoints to detect : {k}')
        self.model.fc = nn.Sequential(
             nn.Linear(num_features, num_features // 2, bias = False),
             nn.ReLU(inplace=True),
             nn.Dropout(p = .2),
             nn.Linear(num_features // 2, 256, bias = False),
             nn.ReLU(inplace=True),
             nn.Linear(256, k, bias = False),
             nn.ReLU(inplace=True),
            )
        if verbose:
            print(self.model)
    
    def forward(self, image):
        return self.model(image)
    
    def training_step(self, batch, batch_idx):
        loss, _ = self.tr_process_(batch, batch_idx, 'train')
        self.log('loss', loss, sync_dist=True, prog_bar=True, logger=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, preds = self.tr_process_(batch, batch_idx, 'val')
        self.log('val_loss', loss, sync_dist=True, prog_bar=True, logger=True, on_epoch=True)
        return loss
    
    def tr_process_(self, batch, batch_idx, subset):
        
        imgs, kpts = batch 
        kpts = torch.autograd.Variable(kpts)
        preds = self.forward(imgs)
        
        if subset == 'train':
            preds = preds.logits
            
        loss = self.criterion(preds, kpts) 
        
        return loss, preds
    
    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr = 0.01)
        scheduler = {
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose = True),
        'monitor': 'val_loss',
        'interval': 'epoch',
        'frequency': 2
        }
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        train_loader = Loader(self.df, self.img_size, transform = crop_and_resize, seed = self.SEED, subset = 'train', val_split = self.test_size)
        train_data   = DataLoader(train_loader, shuffle = True, batch_size = self.BATCH, num_workers = 16)
        return train_data
    
    def val_dataloader(self):
        val_loader   = Loader(self.df, self.img_size, transform = crop_and_resize, seed = self.SEED, subset = 'val', val_split = self.test_size)
        val_data     = DataLoader(val_loader, shuffle = False, batch_size = self.BATCH, num_workers = 16)
        return val_data
    
    
def save_model(model, mname):
    save_path       = os.path.join('models', mname)
    torch.save(model.state_dict(), save_path)
    
    
def load_model(mname, root = 'models'):
    model      = HNet(k = 21 * 2)
    path       = os.path.join(root, f'{mname}')
    model.load_state_dict(torch.load(path))
    return model