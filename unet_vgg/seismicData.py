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
from torchvision import models, transforms
import torchvision


def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def gaussian_noise(image):
    ''' 
    Add gaussian noise to image
    '''
    row, col = image.shape
    mean = 0
    gauss = np.random.normal(mean,1,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image+gauss
    return noisy


class SeismicData(data.Dataset):
    """ Load Seismic Dataset.
    Args:
        image_path(str): the path where the image is located
        mask_path(str): the path where the mask is located
        option(str): decide which dataset to import
    """
    def __init__(self, image_path, mask_path):
        self.image_path = image_path
        self.mask_path = mask_path
        self.mask_arr = glob.glob(str(mask_path)+"/*")
        self.image_arr = glob.glob(str(image_path)+str("/*"))
        self.data_len = len(self.mask_arr)
        
    def __getitem__(self, index):
        """Get specific data corresponding to the index
        Args:
            index: index of the data
        Returns:
            Tensor: specific data on index which is converted to Tensor
        """
        single_image_name = self.image_arr[index]
        
        imgID = fname = re.findall("Line\d+",single_image_name)[0]
        single_mask_name = os.path.join(self.mask_path, f"{imgID}_mask.png")
        
        # Read image and mask
        img = cv2.imread(single_image_name)
        mask = cv2.imread(single_mask_name)
        
        # convert image and mask to grayscale
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Resize to 256*1216 for U-net
        img = img[:,:1216]
        mask = mask[:,:1216]
        
        # Normalize 
        img = img/255
        mask = mask/255
#         plt.imshow(img)
#         plt.show()
#         plt.imshow(mask)
#         plt.show()
        
        # Data augmentation on sampling
        ## Image flip
        if randint(0,1) == 1:
            # we flip horizontally the image and its mask
            img = cv2.flip(img,1)
            mask = cv2.flip(mask,1)
        
        # addNoise = randint(0,2)
        # if addNoise == 1:  ## Add some gaussian noise
        #     img = gaussian_noise(img)
        # elif addNoise == 2:  # Add salt/pepper noise
        #     img = sp_noise(img,0.05)
                 
        
        
        img_as_tensor = torch.from_numpy(img)
        mask_as_tensor = torch.from_numpy(mask).int()
        
        # Reshape to (ch,h,w)
        img_as_tensor = torch.reshape(img_as_tensor,(3,img_as_tensor.shape[0],img_as_tensor.shape[1]))
        mask_as_tensor = torch.reshape(mask_as_tensor,(1,mask_as_tensor.shape[0],mask_as_tensor.shape[1]))

        
        # Normalize image
        # Note. The model expects 3-channel RGB images of shape (3xHxW) where H and W are at least 224.
        # The images must be loaded into a range of [0,1] and then normalized using
        # mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
        # Use the following code to normalize:
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_as_tensor = normalize(img_as_tensor.float())
#         plt.imshow(img)
#         plt.show()
#         plt.imshow(mask)
#         plt.show()
        return (img_as_tensor, mask_as_tensor)
        
    def __len__(self):
        return self.data_len   
