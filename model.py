import torch
import torch.nn as nn
import torch.nn.functional as F
#import tensorflow as tf

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        # todo
        self.d1 = downStep(4,64, False)
        self.d2 = downStep(64,128, True)
        self.d3 = downStep(128, 256, True)
        self.d4 = downStep(256,512, True)
        self.d5 = downStep(512,1024, True)
        
        self.u1 = upStep(1024, 512, True)
        self.u2 = upStep(512, 256, True)
        self.u3 = upStep(256, 128, True)
        self.u4 = upStep(128, 64, False)
        self.u5 = nn.Conv2d(64, 3, kernel_size=1)

        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        # todo
        f1 = self.d1(x)
        f2 = self.d2(f1)
        f3 = self.d3(f2)
        f4 = self.d4(f3)
        f5 = self.d5(f4)
        
        x = self.u1(f5,f4)
        x = self.u2((x),f3)
        x = self.u3((x),f2)
        x = self.u4((x),f1)
        x = self.u5((x))
        print (x.shape)
        x = self.sig(x)
        
        return x

class downStep(nn.Module):
    def __init__(self, inC, outC, max_pool=True):
        super(downStep, self).__init__()
        # todo
        self.inC = inC
        self.outC = outC
        self.max_pool = max_pool
                
        self.conv1 = nn.Conv2d(inC, outC, kernel_size=3, stride=1,padding=1)    
        self.batchnorm1 = nn.BatchNorm2d(outC)
        self.conv2 = nn.Conv2d(outC, outC, kernel_size=3, stride=1,padding=1)
        self.batchnorm2 = nn.BatchNorm2d(outC)

        
    def forward(self, x):
        
        if self.max_pool:
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        
        return x

class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()
        # todo
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.inC = inC
        self.outC = outC
        self.withReLU = withReLU
        
        self.conv_up = nn.ConvTranspose2d(inC, outC, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(inC, outC, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(outC)
        self.conv3 = nn.Conv2d(outC, outC, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(outC)
        
    def forward(self, x, x_down):
        # todo

        x = self.conv_up(x)
      
        batch_size, n_channels, x_height , x_width = x.size()
        batch_size, n_channels, x_down_height, x_down_width = x_down.size()

        #height_pad = int((x_down_height - x_height)/2)
        #width_pad = int((x_down_width - x_width)/2)
        
        #height_pad_end = int(x_down_height - (x_down_height - x_height)/2)
        #width_pad_end = int(x_down_width - (x_down_width - x_width)/2)
        
        #x_down = x_down[:,:,height_pad:height_pad_end, width_pad:width_pad_end]
        
        x = torch.cat([x_down, x], dim=1)
        
        if self.withReLU == True:
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        
        else:
            x = self.conv2(x)
            x = self.conv3(x)
        
        return x
