import numpy as np
import pandas as pd 
import os, gc
from pytorch_lightning import Trainer  
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch 
# from syn_hnet import HNet
from torchvision import models 
from handnet import HandNet
from warnings import filterwarnings
from memory_profiler import profile
import timm

@profile
def main():
    filterwarnings('ignore', category=DeprecationWarning)
    df = pd.read_csv('synthetic_hand.csv')

    

    ### configs 
    kpts        = 42
    epochs      = int(input('No. of Epochs to train (1<=(x)<=10): '))
    device      = torch.cuda.device_count()
    sample_run  = epochs <= 1
    mname       = input('Model name (inet/mnet/resnet/vit): ')
    logger_name = f'{mname}_results'
    df          = pd.read_csv('synthetic_hand.csv')
    logger      = TensorBoardLogger("lightning_logs", name = logger_name)
    batch_size  = 16
    test_size   = 0.1
    dirpath     = os.path.join('models', 'checkpoints') 
    pt_sname    = f'{mname}_ptr.pt'

    if mname == 'resnet':
        base_model = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V2)
        num_features = 2048
        img_size = (224, 224)
    elif mname == 'mnet':
        base_model = models.mobilenet_v3_large()
        num_features = 960
        img_size = (224, 224)
    elif mname == 'inet':
        base_model = models.inception_v3(weights= models.Inception_V3_Weights.DEFAULT, progress = True)
        num_features = base_model.fc.in_features
        img_size = (299, 299)
    elif mname == 'vit':
        base_mname = 'vit_base_patch16_224'
        base_model = timm.create_model(base_mname, pretrained= True)
        num_features = 768
        img_size = (224, 224)

    # ----------------------------------------------------------------------------------------------
    
    checkpoint_callback = ModelCheckpoint(
        dirpath    = dirpath,
        filename   = mname + '-{epoch:02d}-{val_loss:.4f}',
        save_top_k = 1,  
        monitor    = 'val_loss',
        mode       = 'min',
    )

    early_stopping = EarlyStopping(
        monitor   = 'val_loss',
        patience  = 3,
        verbose   = True,
        mode      = 'min'  ,
    )

    
    trainer     = Trainer(fast_dev_run = sample_run, log_every_n_steps = 1, accelerator='gpu' if device == 1 else 'cpu',
                        max_epochs = epochs, logger = logger, callbacks=[early_stopping])
    model       = HandNet(df, kpts, img_size, test_size = test_size, seed = 123, batch = batch_size, base_model = base_model, 
                          num_features = num_features, mname = mname)
    if pt_sname in os.listdir('models'):
        print(f'Pre-Trained model found {pt_sname}...')
        model.load_state_dict(torch.load(os.path.join('models', pt_sname)))

        
    print(f'Training {mname} for {epochs} epochs')
    print(f'DEV run : {sample_run}')
    print(f"Accelerator : {'gpu' if device == 1 else 'cpu'}")
    print(torch.cuda.get_device_name())

    del df, device, logger_name, batch_size, base_model
    gc.collect()

    print(model)

    torch.cuda.empty_cache()
    try:
        trainer.fit(model)
    except RuntimeError as err:
        print(err)
        print('Saving the trained weights ...')

    if not sample_run:
        sname = os.path.join('models', pt_sname)
        torch.save(model.state_dict(), sname)
        print(f'Saved the model weights to {sname}')


if __name__ == "__main__":
    main()





'''
if mname == 'hnet':
        img_size   = 299
        model  = HNet(df, k = kpts , verbose = True, img_size = (img_size, img_size), test_size = 0.1, seed = 123, batch = batch_size)
        model.load_state_dict(torch.load(pt_path))
    else:
        # prev_ckpt = os.path.join('models', 'mnet_sd2.pt')
        # model  = MobNet(df, k = kpts, img_size = (img_size, img_size), test_size = 0.1, seed = 123, 
        #                 batch=batch_size, verbose = True, prev_ckpt = prev_ckpt)
        img_size   = 224
        prev_model = os.path.join('models', 'mnet_tr.pt')
        model  = torch.load(prev_model)
'''