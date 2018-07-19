import torch
from torch.autograd import Variable
from skimage.color import lab2rgb
from model_column import Color_model
from data_loader import ValImageFolder
import numpy as np
from skimage.color import rgb2lab, rgb2gray
import torch.nn as nn 
from PIL import Image
import scipy.misc
from torchvision import datasets, transforms
from training_layers import decode
import torch.nn.functional as F
import os
import json

scale_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
])

def load_image(image_path,transform=None):
    image = Image.open(image_path)
    image = image.resize([256, 256], Image.LANCZOS)
    #if transform is not None:
    #    image = transform(image)
    #image_small=transforms.Scale(56)(image)
    #image_small=np.expand_dims(rgb2lab(image_small)[:,:,0],axis=-1)
    image=rgb2lab(image)[:,:,0]-50.
    image=torch.from_numpy(image).unsqueeze(0)
    
    return image

def main():
    #data_dir = "/mnt/cephfs/lab/wangyuqing/1000"
    #dirs=os.listdir(data_dir)
    color_model = nn.DataParallel(Color_model()).cuda().eval()
    color_model.load_state_dict(torch.load('../model/models_column/model-32-304.ckpt'))
    modules=list(list(color_model.children())[0].children())[:-2]
    #print(color_model)
    
    model=nn.Sequential(*modules)
    
    #fmax=open('../data/conv8_max.txt','w')
    #favg=open('../data/conv8_avg.txt','w')
    #for file in dirs:
        #try:
            #image=load_image(data_dir+'/'+file, scale_transform)
    image=load_image('../data/1.jpg',scale_transform)
    image=image.unsqueeze(0).float().cuda()
    feature=model(image)
            #maxpool=F.max_pool2d(feature,kernel_size=32)
            #maxpool=maxpool.view(512)
            #maxpool=F.normalize(maxpool,dim=0)
            #print(maxpool)
            
            #maxpool=maxpool.data.cpu().numpy().tolist()
            #print(max(maxpool))
            
            #max_item=[file,maxpool]
            #fmax.write('%s\n' % json.dumps(max_item))
    avgpool=F.avg_pool2d(feature,kernel_size=32)
    avgpool=avgpool.view(512)
    avgpool=F.normalize(avgpool,dim=0)
    avgpool=avgpool.data.cpu().numpy()
    np.save('../data/test_conv7.npy',avgpool)
            
            #avg_item=[file,avgpool]
            #favg.write('%s\n' % json.dumps(avg_item))
            
        #except:
        #    pass
    #fmax.close()
    #favg.close()
    
if __name__ == '__main__':
    main()
