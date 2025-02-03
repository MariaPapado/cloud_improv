import os
import tools
import torch
import numpy as np
import rasterio as rio
import warnings
import cv2
import cloud_detection
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import shutil
#import torchnet as tnt
import argparse
import myUNF
import torch.nn.init as init
from sklearn.metrics import confusion_matrix
from unet_model import *


parser = argparse.ArgumentParser()
#parser.add_argument("--data_root", type=str, default="/home/mariapap/DATASETS/second_dataset/SECOND_train_set/")
#parser.add_argument("--train_txt_file", type=str, default="/home/mariapap/DATASETS/second_dataset/SECOND_train_set/list/train.txt")
#parser.add_argument("--val_txt_file", type=str, default="/home/mariapap/DATASETS/second_dataset/SECOND_train_set/list/val.txt")
#parser.add_argument("--batch-size", type=int, default=6)
#parser.add_argument("--val-batch-size", type=int, default=8)
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')

parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')

args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('ok')

net = myUNF.UNetFormer(num_classes=2)
#net = UNet(n_channels=3, n_classes=2, bilinear=True)
net.load_state_dict(torch.load('./builds_saved_models/net_11.pt'))
#model_weights = torch.load('/home/mariapap/CODE/cloud-detection/cloud_weights/net_30_RGB.pt')

#for layer_name, param in model_weights.items():
#    print(f"Layer: {layer_name} | Shape: {param.shape}")
#print('aaaa', model_weights['decoder.segmentation_head.2.0.weight'].shape)


#net.load_state_dict(model_weights)


#for name, param in .items():
#    print(f"Layer: {name} | Shape: {param.shape}")net.load_state_dict(model_weights)

#net.decoder.segmentation_head[2] = torch.nn.Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1), bias=False)
net.to(device)

with open("/home/mariapap/DATASETS/cloud11/lists/train_list.txt", "r") as file:
    train_ids = [line.strip() for line in file.readlines()]

with open("/home/mariapap/DATASETS/cloud11/lists/val_list.txt", "r") as file:
    val_ids = [line.strip() for line in file.readlines()]

data_path = '/home/mariapap/DATASETS/cloud11/PATCHES/'

trainset = cloud_detection.CloudDetection(data_path, train_ids, 'train')
valset = cloud_detection.CloudDetection(data_path, val_ids, 'val')

batch_size = 4
val_batch_size = 2
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                      pin_memory=False, drop_last=False)
valloader = DataLoader(valset, batch_size=val_batch_size, shuffle=False,
                                    pin_memory=True, drop_last=False)


w_tensor=torch.FloatTensor(2)
w_tensor[0]= 0.45
w_tensor[1]= 0.55
#w_tensor[2]= 0.99
#w_tensor[3]= 0.99
w_tensor = w_tensor.to(device)

#criterion = torch.nn.CrossEntropyLoss(w_tensor).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)

base_lr = 0.0001 #0.0005
base_wd = 0.01

optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.1, patience=2, verbose=True)

batch_size = 4
val_batch_size=2

epochs = 50
num_classes = 4

print('loaders')

total_iters = len(trainloader) * epochs
print('totaliters', total_iters)
save_folder = 'saved_models' #where to save the models and training progress
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.mkdir(save_folder)
ff=open('./' + save_folder + '/progress.txt','w')
iter_ = 0

iters = len(trainloader)


for epoch in range(1, epochs+1):

    net.train()
    train_losses = []
#    confusion_matrix = tnt.meter.ConfusionMeter(2, normalized=True)
    cm_total = np.zeros((2,2))
    for i, batch in enumerate(tqdm(trainloader)):
        Ximg, y = batch
        #print(Ximg.shape, y.shape)
       # print('uni', np.unique(y.data.cpu().numpy()))

        imgs, labels = batch
#################################################################################################################################

        #img_save = imgs[0].permute(1,2,0).data.cpu().numpy()
        #lbl_save = labels[0].squeeze().data.cpu().numpy()
        
        #cv2.imwrite('./checks/imgs/img_{}.png'.format(i), img_save[:,:,[2,1,0]]*256)
        #cv2.imwrite('./checks/labels/lbl_{}.png'.format(i), lbl_save*256)

#################################################################################################################################


        imgs, labels = imgs.float().to(device), labels.long().to(device)
        #print(imgs)
        optimizer.zero_grad()
     #   preds, d0, d1, d2, d3, d4, amp41, amp31, amp21, amp11, amp01 = net(imgs)
#        preds, _ = net(imgs)
        preds, _ = net(imgs)

        #print('bbbbbb', imgs.shape, labels.shape, preds.shape)

    
        label_conf, pred_conf = labels.data.cpu().numpy().flatten(), torch.argmax(preds,1).data.cpu().numpy().flatten()
        cm = confusion_matrix(label_conf, pred_conf, labels=[0, 1])
        cm_total += cm
        
        loss = criterion(preds, labels)
        train_losses.append(loss.item())

        loss.backward()
        optimizer.step()
        iter_ += 1
        #lr_scheduler.step(epoch + i / iters)
        #lr_ = base_lr * (1.0 - iter_ / total_iters) ** 0.9
        #for param_group in optimizer.param_groups:
        #    param_group['lr'] = lr_


            
        if iter_ % 20 == 0:
            pred = preds[0]
            pred = torch.softmax(pred, 0)
            pred = np.argmax(pred.data.cpu().numpy(), axis=0)
            gt = labels.data.cpu().numpy()[0]
            print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss_CH: {:.6f}\tAccuracy: {}'.format(
                      epoch, epochs, i, len(trainloader),100.*i/len(trainloader), loss.item(), tools.accuracy(pred, gt)))
        
        



    train_acc=(np.trace(cm_total)/float(np.ndarray.sum(cm_total))) *100
    print('TRAIN_LOSS: ', '%.3f' % np.mean(train_losses), 'TRAIN_ACC: ', '%.3f' % train_acc)
    print(cm_total)
    #prec, rec, f1 = tools.metrics(cm_total)
    #print ('Precision: {}\nRecall: {}\nF1: {}'.format(prec, rec, f1)) 
    
    cm_total = np.zeros((2,2))
    with torch.no_grad():
        net.eval()
        val_losses = []

        for i, batch in enumerate(tqdm(valloader)):
            imgs, labels = batch
            imgs, labels = imgs.float().to(device), labels.long().to(device)
#            preds, d0, d1, d2, d3, d4, amp41, amp31, amp21, amp11, amp01 = net(imgs)            
            preds = net(imgs)
            label_conf, pred_conf = labels.data.cpu().numpy().flatten(), torch.argmax(torch.softmax(preds, 1),1).data.cpu().numpy().flatten()
            cm = confusion_matrix(label_conf, pred_conf, labels=[0, 1])
            cm_total += cm

            loss = criterion(preds, labels)
            val_losses.append(loss.item())


        #scheduler.step(np.mean(val_losses))    
        test_acc=(np.trace(cm_total)/float(np.ndarray.sum(cm_total))) *100
        print('VAL_LOSS: ', '%.3f' % np.mean(val_losses), 'VAL_ACC: ', '%.3f' % test_acc)
        print('VALIDATION CONFUSION MATRIX')    
        print(cm_total)
        #prec, rec, f1 = tools.metrics(cm_total)
        #print ('Precision: {}\nRecall: {}\nF1: {}'.format(prec, rec, f1)) 
    

    #tools.write_results(ff, save_folder, epoch, train_acc, test_acc, np.mean(train_losses), np.mean(val_losses), cm_total, prec, rec, f1, optimizer.param_groups[0]['lr'])

    #save model in every epoch
    torch.save(net.state_dict(), './' + save_folder + '/net_{}.pt'.format(epoch))

