# Unet using vgg11 pretrained weights in the encoding section 
  
from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision


class UnetVGG(nn.Module):
    def __init__(self):
        super().__init__()
        # Load VGG11
        vgg11 = models.vgg11(pretrained=True)
        encoder = vgg11.features

        # Build encoding path based on VGG11:
        self.convblock_0 = encoder[0:2]   # 256x1216 (64)
        self.pool0 = encoder[2]           # 128x608 (64)
        self.convblock_1 = encoder[3:5]   # 128x608 (128)
        self.pool1 = encoder[5]           # 64x304 (128)
        self.convblock_2 = encoder[6:10]  # 64x304 (256)
        self.pool2 = encoder[10]          # 32x152 (256)
        self.convblock_3 = encoder[11:15] # 32x152 (512)
        self.pool3 = encoder[15]          # 16x76 (512)
        self.convblock_4 = encoder[16:20] # 16x76 (512)
        self.pool4 = encoder[20]          # 8x38 (512)
        # 
        # Bottleneck layer (make 2 convolutions)
        self.convblock_5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # Build decoding path 
        ### 
        # Block 0
        self.up_0 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        # Upconvolution Block 0
        self.conv_up_0 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        ###
        # Block 1
        self.up_1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2)
        # Upconvolution Block 1
        self.conv_up_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        ### 
        # Block 2
        self.up_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        # Upconvolution Block 2
        self.conv_up_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Block 3
        self.up_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        # Upconvolution Block 3
        self.conv_up_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Block 4
        self.up_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        # Upconvolution Block 4
        self.conv_up_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv_final = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0, stride=1)

        self.final = nn.Sequential(
             nn.Sigmoid(),
         )

    
    def forward(self,x):
        # Propagate through the network:
        #print(f"original image {x.shape}")
        x = self.convblock_0(x)
        conv0_out = x
        conv0_dim_h = x.shape[2]
        conv0_dim_w = x.shape[3]
        #print(f"convblock_0 {x.shape}")

        x = self.pool0(x)
        #print(f"pool0 {x.shape}")
        x = self.convblock_1(x)
        conv1_out = x
        conv1_dim_h = x.shape[2]
        conv1_dim_w = x.shape[3]
        #print(f"convblock_1 {x.shape}")

        x = self.pool1(x)
        #print(f"pool1 {x.shape}")

        x = self.convblock_2(x)
        conv2_out = x
        conv2_dim_h = x.shape[2]
        conv2_dim_w = x.shape[3]
        #print(f"convblock_2 {x.shape}")

        x = self.pool2(x)
        #print(f"pool2 {x.shape}")

        x = self.convblock_3(x)
        conv3_out = x
        conv3_dim_h = x.shape[2]
        conv3_dim_w = x.shape[3]
        #print(f"convblock_3 {x.shape}")

        x = self.pool3(x)
        #print(f"pool3 {x.shape}")

        x = self.convblock_4(x)
        conv4_out = x
        conv4_dim_h = x.shape[2]
        conv4_dim_w = x.shape[3]
        #print(f"convblock_4 {x.shape}")


        x = self.pool4(x)
        #print(f"pool4 {x.shape}")

        x = self.convblock_5(x)
        #print(f"convblock_5 {x.shape}")

        # Moving up
        x = self.up_0(x)
        #print("****************")
        #print(f"up_0 {x.shape}")

        lower_h = int((conv4_dim_h - x.shape[2])/2)
        upper_h = int((conv4_dim_h - lower_h))
        lower_w = int((conv4_dim_w - x.shape[3])/2)
        upper_w = int((conv4_dim_w - lower_w))
        conv4_out_modified = conv4_out[:,:,lower_h:upper_h,lower_w:upper_w]
        x = torch.cat([x,conv4_out_modified], dim=1)
        x = self.conv_up_0(x)
        #print(f"conv_up_0 {x.shape}")

        x = self.up_1(x)
        #print(f"up_1 {x.shape}")

        lower_h = int((conv3_dim_h - x.shape[2])/2)
        upper_h = int((conv3_dim_h - lower_h))
        lower_w = int((conv3_dim_w - x.shape[3])/2)
        upper_w = int((conv3_dim_w - lower_w))
        conv3_out_modified = conv3_out[:,:,lower_h:upper_h,lower_w:upper_w]
        x = torch.cat([x,conv3_out_modified], dim=1)
        #print(x.shape)
        x = self.conv_up_1(x)
        #print(f"conv_up_1 {x.shape}")

        x = self.up_2(x)
        #print(f"up_2 {x.shape}")

        lower_h = int((conv2_dim_h - x.shape[2])/2)
        upper_h = int((conv2_dim_h - lower_h))
        lower_w = int((conv2_dim_w - x.shape[3])/2)
        upper_w = int((conv2_dim_w - lower_w))
        conv2_out_modified = conv2_out[:,:,lower_h:upper_h,lower_w:upper_w]
        x = torch.cat([x,conv2_out_modified], dim=1)
        x = self.conv_up_2(x)
        #print(f"conv_up_2 {x.shape}")

        x = self.up_3(x)
        #print(f"up_3 {x.shape}")

        lower_h = int((conv1_dim_h - x.shape[2])/2)
        upper_h = int((conv1_dim_h - lower_h))
        lower_w = int((conv1_dim_w - x.shape[3])/2)
        upper_w = int((conv1_dim_w - lower_w))
        conv1_out_modified = conv1_out[:,:,lower_h:upper_h,lower_w:upper_w]
        x = torch.cat([x,conv1_out_modified], dim=1)
        x = self.conv_up_3(x)
        #print(f"conv_up_3 {x.shape}")

        x = self.up_4(x)
        #print(f"up_3 {x.shape}")

        lower_h = int((conv0_dim_h - x.shape[2])/2)
        upper_h = int((conv0_dim_h - lower_h))
        lower_w = int((conv0_dim_w - x.shape[3])/2)
        upper_w = int((conv0_dim_w - lower_w))
        conv0_out_modified = conv0_out[:,:,lower_h:upper_h,lower_w:upper_w]
        x = torch.cat([x,conv0_out_modified], dim=1)
        x = self.conv_up_4(x)   
        #print(f"conv_up_4 {x.shape}")

        x = self.conv_final(x)
        #print(f"conv_final {x.shape}")

        x = self.final(x)
        #print(f"final {x.shape}")


        return x





# model = UnetVGG()
# im = torch.randn(1,3,256,1216)
# x = model(im)
# print(x)
# print(x.shape)
# print("***************************")
# cnt = 0
# for child in model.children():
#     if cnt < 10:
#         for param in child.parameters():
#             param.requires_grad = False
#     cnt+=1

