from comet_ml import Experiment
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from random import randint
from unet import SeismicData
from unet import UNet   
from unet import SoftDiceLoss, BinaryCrossEntropyLoss2d


experiment = Experiment(api_key="YFDQ2jeiF9jHKIezwWK2MBS2P",project_name="general", workspace="leytzher")


# 1. Define train and test datasets

train_dataset = SeismicData(image_path='./seismic/train/images/', mask_path='./seismic/train/masks/')
test_dataset = SeismicData(image_path='./seismic/test/images/', mask_path='./seismic/test/masks/')

# 2. Initialize data loaders. The SeismicData class does data augmentation on sampling

train_data_load = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=16)
test_data_load = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

# 3. Initialize Neural Network 
device = torch.device("cuda:0")

model = UNet()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Save history to csv file 
header = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
save_file_name = "./history/RMS/historyRMS.csv"
save_dir = './history/RMS'

# save images
model_save_dir = "../history/RMS/saved_models3"
image_save_path = "../history/RMS/result_images3"

# 4.  Start training

print("[INFO] Starting Training...")
epochs = 700

def valid_loss(model, test_data):
    model.eval()
    for batch, (images, masks) in enumerate(test_data):
        images = Variable(images.float().cuda())
        masks = Variable(masks.cuda())
        outputs = model(images)
        loss = torch.mean(BinaryCrossEntropyLoss2d().forward(outputs,masks.float()) + SoftDiceLoss().forward(outputs,masks.float()))
        return loss


for i in range(0,epochs):
    model.train()
    for batch,(images,masks) in enumerate(train_data_load):
        images = Variable(images.float().cuda())
        masks = Variable(masks.cuda())
        # Calculate output
        outputs = model(images)
        loss = torch.mean(BinaryCrossEntropyLoss2d().forward(outputs,masks.float()) + SoftDiceLoss().forward(outputs,masks.float()))
        val_loss = valid_loss(model,test_data_load)
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        # Report every 5 epochs
            
    print(f'[INFO] Epoch {i+1}, Train loss: {loss}; Validation loss: {val_loss}')




