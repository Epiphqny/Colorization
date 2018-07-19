import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class ScaleLayer(nn.Module):

   def __init__(self, init_value=1e-3):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]))

   def forward(self, input):
       return input * self.scale

def weights_init(model):
    if type(model) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_normal_(model.weight.data)
        nn.init.constant_(model.bias.data, 0.1)

class Color_model(nn.Module):
    def __init__(self):
        super(Color_model, self).__init__()
        self.conv1 = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 64),
        )
        
        self.conv2 = nn.Sequential(
            # conv2
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 128),
        )
        
        self.conv3 = nn.Sequential(
            # conv3
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 256),
        )

        self.conv4 = nn.Sequential(
            # conv4
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 512),
        )

        self.conv5 = nn.Sequential(
            # conv5
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 512),
        )

        self.conv6 = nn.Sequential(
            # conv6
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 512),
        )

        self.conv7 = nn.Sequential(
            # conv7
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = 512),
        )

        self.conv8 = nn.Sequential(
            # conv8
            nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 4, stride = 2, padding = 1, dilation = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
            nn.ReLU(),
        )
            # conv8_313
        self.classify=nn.Conv2d(in_channels = 2752, out_channels = 313, kernel_size = 1, stride = 1,dilation = 1)
            #eltwise product
            # softmax
            #nn.Softmax(dim=1),
	    # decoding
            #nn.Conv2d(in_channels = 313, out_channels = 2, kernel_size = 1, stride = 1)
        
        self.apply(weights_init)

    def forward(self, gray_image):
        conv1=self.conv1(gray_image)
        conv1_resize=F.max_pool2d(conv1,kernel_size=2,padding=0)
        conv2=self.conv2(conv1)
        conv3=self.conv3(conv2)
        conv3_resize=F.upsample(conv3,scale_factor=2)
        conv4=self.conv4(conv3)
        conv4_resize=F.upsample(conv4,scale_factor=2)
        conv5=self.conv5(conv4)
        conv5_resize=F.upsample(conv5,scale_factor=2)
        conv6=self.conv6(conv5)
        conv6_resize=F.upsample(conv6,scale_factor=2)
        conv7=self.conv7(conv6)
        conv7_resize=F.upsample(conv7,scale_factor=2)
        conv8=self.conv8(conv7)
        combine=torch.cat((conv1_resize,conv2,conv3_resize,conv4_resize,conv5_resize,conv6_resize,conv7_resize,conv8),dim=1)
        features=self.classify(combine)
        features=features/0.38
        return features
