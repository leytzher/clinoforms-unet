import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import glob
import re
import os
import random
from random import randint



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        ### Start downward path:
        # Conv Block 1 - Down 1
        self.conv1_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Block 2 - Down 2
        self.conv2_block = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Block 3 - Down 3
        self.conv3_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 4 - Down 4
        self.conv4_block = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)

         # Conv Block 5 - Down 5 ((Bottom of the network))
        self.conv5_block = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        ## Start upwards path
        
        # Up 1
        self.up_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        
        # Upconvolution Block 1
        self.conv_up_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Up 2
        self.up_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        
        # Upconvolution Block 2
        self.conv_up_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Up 3
        self.up_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        
        # Upconvolution Block 3
        self.conv_up_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Up 4
        self.up_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        
        # Upconvolution Block 4
        self.conv_up_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Final output
        self.conv_final = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, padding=0, stride=1)

        self.final = nn.Sequential(
             nn.Sigmoid(),
         )
        

    def forward(self,x):
        #print('input', x.shape)
        # Conv1 block (Down)
        x = self.conv1_block(x)
        #print('after conv1', x.shape)
        # Save output for future concatenation
        conv1_out = x
        conv1_dim_h = x.shape[2]
        conv1_dim_w = x.shape[3]
        # Max pooling
        x = self.max1(x)
        #print('before conv2', x.shape)
        
        # Conv2 block (Down)
        x = self.conv2_block(x)
        #print('after conv2', x.shape)
        # Save output for future concatenation
        conv2_out = x
        conv2_dim_h = x.shape[2]
        conv2_dim_w = x.shape[3]
        # Max pooling
        x = self.max2(x)
        #print('before conv3', x.shape)

        # Conv3 block (Down)
        x = self.conv3_block(x)
        #print('after conv3', x.shape)
        # Save output for future concatenation
        conv3_out = x
        conv3_dim_h = x.shape[2]
        conv3_dim_w = x.shape[3]
        # Max pooling
        x = self.max3(x)
        #print('before conv4', x.shape)

         # Conv4 block (Down)
        x = self.conv4_block(x)
        #print('after conv5', x.shape)
        # Save output for future concatenation
        conv4_out = x
        conv4_dim_h = x.shape[2]
        conv4_dim_w = x.shape[3]
        # Max pooling
        x = self.max4(x)
        #print('before conv6', x.shape)
        
        # Bottom of the network
        x = self.conv5_block(x)
        #print("At bottom of the network",x.shape)
        
        # Conv1 block (Up)
        x = self.up_1(x)
        #print('up_1', x.shape)
        lower_h = int((conv4_dim_h - x.shape[2])/2)
        upper_h = int((conv4_dim_h - lower_h))
        lower_w = int((conv4_dim_w - x.shape[3])/2)
        upper_w = int((conv4_dim_w - lower_w))
        conv4_out_modified = conv4_out[:,:,lower_h:upper_h,lower_w:upper_w]
        #print("Shape of conv4-out-mod",conv4_out_modified.shape)
        x = torch.cat([x,conv4_out_modified], dim=1)
        #print('after cat_1', x.shape)
        x = self.conv_up_1(x)
        #print('after conv_1', x.shape)

        # Conv2 block (Up)
        x = self.up_2(x)
        #print('up_2', x.shape)
        lower_h = int((conv3_dim_h - x.shape[2])/2)
        upper_h = int((conv3_dim_h - lower_h))
        lower_w = int((conv3_dim_w - x.shape[3])/2)
        upper_w = int((conv3_dim_w - lower_w))
        conv3_out_modified = conv3_out[:,:,lower_h:upper_h,lower_w:upper_w]
        #print("Shape of conv3-out-mod",conv3_out_modified.shape)
        x = torch.cat([x,conv3_out_modified], dim=1)
        #print('after cat_2', x.shape)
        x = self.conv_up_2(x)
        #print('after conv_2', x.shape)
        
        # Conv3 block (Up)
        x = self.up_3(x)
        #print('up_3', x.shape)
        lower_h = int((conv2_dim_h - x.shape[2])/2)
        upper_h = int((conv2_dim_h - lower_h))
        lower_w = int((conv2_dim_w - x.shape[3])/2)
        upper_w = int((conv2_dim_w - lower_w))
        conv2_out_modified = conv2_out[:,:,lower_h:upper_h,lower_w:upper_w]
        x = torch.cat([x,conv2_out_modified], dim=1)
        #print('after cat_3', x.shape)
        x = self.conv_up_3(x)
        #print('after conv_3', x.shape)
        
        # Conv4 block (Up)
        x = self.up_4(x)
        #print('up_4', x.shape)
        lower_h = int((conv1_dim_h - x.shape[2])/2)
        upper_h = int((conv1_dim_h - lower_h))
        lower_w = int((conv1_dim_w - x.shape[3])/2)
        upper_w = int((conv1_dim_w - lower_w))
        conv1_out_modified = conv1_out[:,:,lower_h:upper_h,lower_w:upper_w]
        x = torch.cat([x,conv1_out_modified], dim=1)
        #print('after cat_4', x.shape)
        x = self.conv_up_4(x)
        #print('after conv_4', x.shape)
               
        # Final 
        x = self.conv_final(x)
        #print('after final', x.shape)
        
        x = self.final(x)
        return x
        
