import torch
import pytorch_lightning as pl 
import torch.nn as nn
import albumentations as A
import numpy as np 
from hutils import crop_and_resize
from data_loader import Loader
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

torch.set_float32_matmul_precision('medium')
class HandNet(pl.LightningModule):
    def __init__(self, df = None, k = 21*2, img_size = (224,224), transform = None, test_size = 0.2, 
                 verbose = False, seed = 0, batch = 64, base_model = None, num_features = 1024, mname = None):
        super(HandNet, self).__init__()
        self.test_size = test_size
        self.verbose = verbose
        self.df = df
        self.transform = transform
        # self.scaler = GradScaler()
        self.SEED = seed
        self.BATCH = batch
        self.n_classes = k
        self.history = {'loss' : [], 'val_loss' : []}
        self.img_size = img_size
        self.criterion = nn.SmoothL1Loss().cuda()
        self.model = base_model
        sequence = nn.Sequential(
            nn.Linear(num_features, num_features // 2, bias = False),
            nn.ReLU(inplace = True),
            nn.Dropout(p = .2),
            nn.Linear(num_features // 2, 256, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(256, k, bias = True)
        )

        if mname != 'vit':
            try:
                self.model.fc = sequence
            except:
                self.model.classifier = sequence
        else:
            self.model.head = nn.Linear(768, k, bias = True)

        if verbose:
            print(self.model)
    
    def forward(self, image):
        return self.model(image)
    
    def training_step(self, batch, batch_idx):
        loss, _ = self.tr_process_(batch, batch_idx, 'train')
        # scaled_loss = self.scaler.scale(loss)
        self.log('loss', loss, sync_dist=False, prog_bar=True, logger=True, on_epoch=True)
        self.history['loss'].append(loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, preds = self.tr_process_(batch, batch_idx, 'val')
        self.log('val_loss', loss, sync_dist=False, prog_bar=True, logger=True, on_epoch=True)
        self.history['val_loss'].append(loss)
        return loss
    
    def tr_process_(self, batch, batch_idx, subset):
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
    
    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
    #     optimizer.zero_grad()
    #     self.scaler.step(optimizer)
    #     self.scaler.update()
        
    
    def train_dataloader(self):
        train_loader = Loader(self.df, self.img_size, transform = crop_and_resize, seed = self.SEED, subset = 'train', val_split = self.test_size)
        train_data   = DataLoader(train_loader, shuffle = True, batch_size = self.BATCH, num_workers = 4, persistent_workers=True, pin_memory=True)
        return train_data
    
    def val_dataloader(self):
        val_loader   = Loader(self.df, self.img_size, transform = crop_and_resize, seed = self.SEED, subset = 'val', val_split = self.test_size)
        val_data     = DataLoader(val_loader, shuffle = False, batch_size = self.BATCH, num_workers = 4, persistent_workers=True, pin_memory=True)
        return val_data
    
    
def predict_(model, img, bbox, IMG_SIZE = 224, device = 'cpu'):
    x1,y1,x2,y2 = list(map(int, bbox))
    transformer = A.Compose([
        A.Crop(x_min = x1, y_min = y1, x_max = x2, y_max = y2, always_apply=True, p=1.0), 
        A.Resize(IMG_SIZE, IMG_SIZE)])
    img = (transformer(image = img)['image'] / 255).astype(np.float32) 
    img = torch.from_numpy(img).permute(-1, 0, 1) 
    img = torch.unsqueeze(img, dim = 0).to(device)             
    model.eval()
    with torch.no_grad():
        preds = torch.squeeze(model(img))
    img = img.cpu()
    preds = preds.cpu()
    return torch.squeeze(img.detach()).permute(1,-1,0).numpy().copy(), (preds.reshape(-1, 2).numpy() * IMG_SIZE).astype(np.uint32)
    
