import os
from os.path import isdir, exists, abspath, join

import random

import numpy as np
from numpy import newaxis
from PIL import Image, ImageEnhance

import torch 
import torchvision.transforms.functional as TF
import torchvision.transforms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class DataLoader():
    
    def __init__(self, root_dir='data', batch_size=16, test_percent=0.1):
            
        self.batch_size = batch_size
        self.test_percent = test_percent
        self.root_dir = abspath(root_dir)
        self.data_files = join(self.root_dir,'train.png')
        
        #print (self.data_files)

    def __iter__(self):
        
        n_train = self.n_train()
        
        if self.mode == 'train':
            current = 0
            endId = 100
        
        elif self.mode == 'test':
            #current = n_train
            current = 0 
            endId = 1 #len(self.data_files)
            self.batch_size = 1  
        
        while current < endId:
            
            label_list=[]
        
            data_image_list = []
        
            labels_list = []
            
            input_image = Image.open(self.data_files)
                       
            for i in range (self.batch_size):
                                       
                obj = torchvision.transforms.RandomCrop((128,128))

                labels = obj(input_image)
                 
                value = np.random.randint(1,6)

                if self.mode == 'train':
                    
                    labels = self.applyDataAugmentation(labels, value) 

                label_list.append(labels)
    
                data_images = labels
                
                obj1 =transforms.Compose([transforms.ToTensor()])
                
                labels = obj1(labels)
                
                data_images = obj1(data_images)

                data_images = self.mask(data_images)
                
                data_images = data_images.unsqueeze(0)

                labels = labels.unsqueeze(0)
                
                #labels = labels/255.
               
                if i == 0: 
                    data_image_list= data_images
                    labels_list = labels
    
                else:
                    data_image_list = torch.cat((data_image_list, data_images), dim=0)
                    labels_list = torch.cat((labels_list, labels), dim=0)

            data_image_list = np.asarray(data_image_list)

            yield (data_image_list, labels_list)
        

    def setMode(self, mode):
        self.mode = mode
        
    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))
    
    def applyDataAugmentation(self, label_image, value):
        
        self.image = label_image
        self.value = value
                    
        if (self.value==1):
                                    
            self.image = TF.hflip(self.image)
        
        elif (self.value==2):
            self.image = TF.vflip(self.image)

        elif (self.value==3):
                                              
            self.image = TF.adjust_hue(self.image, random.uniform(-0.1,0.1))

        elif (self.value==4):
            
            self.image = self.image.transpose(Image.ROTATE_90)
        
        elif (self.value==5):
            
            self.image = TF.resize(self.image, (128,128)) 
            
        else:
            self.image = label_image

        return self.image
    
    
    def mask(self, data_image):
            
        mask = torch.ones([128,128])

        for i in range(5):
                             
            width = random.randint(0,119)
            
            height = random.randint(0,63)
            
            #print ('Data Image Shape' +str(data_image.shape))
            
            if random.random() > 0.4:
                mask[height:height+64,width:width+8] = 0
                data_image[:,height:height+64,width:width+8] = 0
            else:
                mask[width:width+8,height:height+64] = 0
                data_image[:,width:width+8,height:height+64] = 0

        mask = mask.reshape(1,128,128)
           
        data_image = torch.cat((data_image,mask),dim=0)

        return data_image