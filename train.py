import sys
import os
from os.path import join
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from torchvision import transforms

import matplotlib.pyplot as plt

from model import UNet
from dataloader import DataLoader

def train_net(net,
              epochs=2,
              data_dir='data/',
              n_classes=2,
              lr=0.001,
              #val_percent=0.1,
              save_cp=True,
              gpu=False):
    
    loader = DataLoader(data_dir)
    test_loader = DataLoader(data_dir)

    N_train = loader.n_train()
    
    optimizer = optim.Adam(net.parameters(),
                            lr=lr, #0.99
                            weight_decay=1e-8)

    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        
        print('Epoch %d/%d' % (epoch + 1, epochs))
        print('Training...')
        net.train()
        loader.setMode('train')

        epoch_loss = 0
        
        counter = 0
                
        for i, (img, label) in enumerate(loader):
            
            shape = img.shape
            
            #print ('GT shape ' + str(label.shape))

            label_copy = label
            
            label = label.float()
            
            # todo: create image tensor: (N,C,H,W) - (batch size=1,channels=1,height,width)
            
            img_tensor = torch.from_numpy(img).float()

            masked_img_tensor = img_tensor[0]

            # todo: load image tensor to gpu
            
            if gpu:
                print ('Inside GPU')
                
                img_tensor = Variable(img_tensor.cuda())
            
                label = Variable(label.cuda())
        
            optimizer.zero_grad()
            
            #todo: get prediction and getLoss()
            
            pred = net.forward(img_tensor)

            loss = getLoss(pred,label)

            epoch_loss += loss.item()
            
            print('Training sample %d / %d - Loss: %.6f' % (i+1, N_train, loss.item()))
            
            print ('Counter', counter)
            
            # optimize weights
            
            loss.backward()
            
            optimizer.step()

            label_copy = label_copy[0]

            label_copy = label_copy.permute(1,2,0)
                                 
            masked_img_tensor = masked_img_tensor.permute(1,2,0)   
            
            masked_img_tensor = masked_img_tensor[:,:,0:3]
            
            pred = pred[0]
            
            pred = pred.permute(1,2,0)
            
            counter = counter + 1 
            
            if (counter==100):
                break
            
        plt.subplot(1, 3, 1)
        plt.imshow(label_copy)
        plt.title(str(epoch) + 'train-gt')
                        
        plt.subplot(1, 3, 2)
        plt.imshow((masked_img_tensor))
        plt.title(str(epoch) + 'train-in')
                        
        plt.subplot(1, 3, 3)
        plt.imshow(pred.cpu().detach().numpy())
        plt.title(str(epoch) + 'train-out')

        plt.savefig(str(epoch)+'-train.png')

        test_loader.setMode('test')

        net.eval()

        print ('Testing')

        with torch.no_grad():
            
            test_counter = 0

            for _, (img_test, label_test) in enumerate(test_loader):

                label_test_copy = label_test     

                label_test = label_test.float()

                img_test_tensor = torch.from_numpy(img_test).float()

                masked_img_test_tensor = img_test_tensor[0]

                if gpu:

                    print ('Inside GPU')

                    img_test_tensor = Variable(img_test_tensor.cuda())

                    label_test = Variable(label_test.cuda())

                predicted = net(img_test_tensor)

                loss = getLoss(predicted,label_test)

                print('test loss' + str(loss))

                label_test_copy = label_test_copy[0]

                label_test_copy = label_test_copy.permute(1,2,0)

                masked_img_test_tensor = masked_img_test_tensor.permute(1,2,0)   

                masked_img_test_tensor = masked_img_test_tensor[:,:,0:3]

                predicted = predicted[0]

                predicted = predicted.permute(1,2,0)
                
                test_counter = test_counter + 1
                
                print ('Test Counter', test_counter)
                
                if (test_counter >=1):
                    break

            plt.subplot(1, 3, 1)
            plt.imshow(label_test_copy)
            plt.title(str(epoch) + 'test-gt')

            plt.subplot(1, 3, 2)
            plt.imshow((masked_img_test_tensor))
            plt.title(str(epoch) + 'test-in')

            plt.subplot(1, 3, 3)
            plt.imshow(predicted.cpu().detach().numpy())
            plt.title(str(epoch) + 'test-out')

            plt.savefig(str(epoch) + '-test.png')

            torch.save(net.state_dict(), join(data_dir, 'checkpoints') + '/CP%d.pth' % (epoch + 1))

#         plt.imsave('home/moorthy/sfuhome/Visual_Computing_Lab2/Assignment2/Inpainting_UNet/saved_images/', label_copy)
#         plt.imsave('/sfuhome/Visual_Computing_Lab2/Assignment2/Inpainting_UNet/saved_images/', masked_img_tensor)
#         plt.imsave('/sfuhome/Visual_Computing_Lab2/Assignment2/Inpainting_UNet/saved_images/', pred.cpu().detach().numpy())

        print ('Images Saved')
        print('Checkpoint %d saved !' % (epoch + 1))
        print('Epoch %d finished! - Loss: %.6f' % (epoch+1, epoch_loss / i))


def getLoss(pred_label, target_label):
    loss = ((pred_label - target_label) ** 2).mean()
    return loss

def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=1, type='int', help='number of epochs')
    parser.add_option('-c', '--n-classes', dest='n_classes', default=3, type='int', help='number of classes')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='data/', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=True, help='use cuda')
    parser.add_option('-l', '--load', dest='load', default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_classes=args.n_classes)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from %s' % (args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    train_net(net=net,
        epochs=args.epochs,
        n_classes=args.n_classes,
        gpu=args.gpu,
        data_dir=args.data_dir)
