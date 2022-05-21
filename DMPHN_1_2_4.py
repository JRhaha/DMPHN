import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import random
import models
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from datasets import GoProDataset,MyDataset
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="Deep Multi-Patch Hierarchical Network")
parser.add_argument("-e","--epochs",type = int, default = 2400)
parser.add_argument("-se","--start_epoch",type = int, default = 0)
parser.add_argument("-b","--batchsize",type = int, default = 2)
parser.add_argument("-s","--imagesize",type = int, default = 256)
parser.add_argument("-l","--learning_rate", type = float, default = 0.0001)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()

#Hyper Parameters
METHOD = "DMPHN_1_2_4"
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
GPU = args.gpu
BATCH_SIZE = args.batchsize
IMAGE_SIZE = args.imagesize

def save_deblur_images(images, iteration, epoch):

    if os.path.exists('./newcheckpoints/' + METHOD) == False:
        os.system('./newcheckpoints/' + METHOD)  
    filename = './newcheckpoints/' + METHOD + "/epoch" + str(epoch)  + "Iter_" + str(iteration) + "_deblur.png"
    torchvision.utils.save_image(images, filename)

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5*math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    #记录训练次数
    total_train_step=0
    #记录测试次数
    total_test_step=0
    writer=SummaryWriter("./logs")
    print("init data folders")

    encoder_lv1 = models.Encoder()
    encoder_lv2 = models.Encoder()    
    encoder_lv3 = models.Encoder()

    decoder_lv1 = models.Decoder()
    decoder_lv2 = models.Decoder()    
    decoder_lv3 = models.Decoder()
    
    encoder_lv1.apply(weight_init).cuda(GPU)    
    encoder_lv2.apply(weight_init).cuda(GPU)
    encoder_lv3.apply(weight_init).cuda(GPU)

    decoder_lv1.apply(weight_init).cuda(GPU)    
    decoder_lv2.apply(weight_init).cuda(GPU)
    decoder_lv3.apply(weight_init).cuda(GPU)
    
    encoder_lv1_optim = torch.optim.Adam(encoder_lv1.parameters(),lr=LEARNING_RATE)
    encoder_lv1_scheduler = StepLR(encoder_lv1_optim,step_size=1000,gamma=0.1)
    encoder_lv2_optim = torch.optim.Adam(encoder_lv2.parameters(),lr=LEARNING_RATE)
    encoder_lv2_scheduler = StepLR(encoder_lv2_optim,step_size=1000,gamma=0.1)
    encoder_lv3_optim = torch.optim.Adam(encoder_lv3.parameters(),lr=LEARNING_RATE)
    encoder_lv3_scheduler = StepLR(encoder_lv3_optim,step_size=1000,gamma=0.1)

    decoder_lv1_optim = torch.optim.Adam(decoder_lv1.parameters(),lr=LEARNING_RATE)
    decoder_lv1_scheduler = StepLR(decoder_lv1_optim,step_size=1000,gamma=0.1)
    decoder_lv2_optim = torch.optim.Adam(decoder_lv2.parameters(),lr=LEARNING_RATE)
    decoder_lv2_scheduler = StepLR(decoder_lv2_optim,step_size=1000,gamma=0.1)
    decoder_lv3_optim = torch.optim.Adam(decoder_lv3.parameters(),lr=LEARNING_RATE)
    decoder_lv3_scheduler = StepLR(decoder_lv3_optim,step_size=1000,gamma=0.1)

    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv1.pkl")):
        encoder_lv1.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv1.pkl")))
        print("load encoder_lv1 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv2.pkl")):
        encoder_lv2.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv2.pkl")))
        print("load encoder_lv2 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv3.pkl")):
        encoder_lv3.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv3.pkl")))
        print("load encoder_lv3 success")

    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv1.pkl")):
        decoder_lv1.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv1.pkl")))
        print("load encoder_lv1 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv2.pkl")):
        decoder_lv2.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv2.pkl")))
        print("load decoder_lv2 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv3.pkl")):
        decoder_lv3.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv3.pkl")))
        print("load decoder_lv3 success")
    
    if os.path.exists('./checkpoints/' + METHOD) == False:
        os.system('mkdir ./checkpoints/' + METHOD)    
            
    for epoch in range(args.start_epoch, EPOCHS):
        encoder_lv1_scheduler.step(epoch)
        encoder_lv2_scheduler.step(epoch)
        encoder_lv3_scheduler.step(epoch)

        decoder_lv1_scheduler.step(epoch)
        decoder_lv2_scheduler.step(epoch)
        decoder_lv3_scheduler.step(epoch)     
        
        print("Training...")
        
        train_dataset = MyDataset(
            #JJR:更改为自己的数据目录
            blur_image_files = './datas/myData/train_blur_file.txt',
            sharp_image_files = './datas/myData/train_sharp_file.txt',
            root_dir = './datas/myData',
            # blur_image_files = './datas/GoPro/train_blur_file.txt',
            # sharp_image_files = './datas/GoPro/train_sharp_file.txt',
            # root_dir = './datas/GoPro/',
            crop = True,
            crop_size = IMAGE_SIZE,
            transform = transforms.Compose([
                transforms.ToTensor()
                ]))
        train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
        start = 0
        total_train_loss=0
        for iteration, images in enumerate(train_dataloader):            
            mse = nn.MSELoss().cuda(GPU)          
            
            gt = Variable(images['sharp_image'] - 0.5).cuda(GPU)            
            H = gt.size(2)
            W = gt.size(3)

            images_lv1 = Variable(images['blur_image'] - 0.5).cuda(GPU)
            images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
            images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
            images_lv3_1 = images_lv2_1[:,:,:,0:int(W/2)]
            images_lv3_2 = images_lv2_1[:,:,:,int(W/2):W]
            images_lv3_3 = images_lv2_2[:,:,:,0:int(W/2)]
            images_lv3_4 = images_lv2_2[:,:,:,int(W/2):W]

            feature_lv3_1 = encoder_lv3(images_lv3_1)
            feature_lv3_2 = encoder_lv3(images_lv3_2)
            feature_lv3_3 = encoder_lv3(images_lv3_3)
            feature_lv3_4 = encoder_lv3(images_lv3_4)
            feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
            feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
            feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
            residual_lv3_top = decoder_lv3(feature_lv3_top)
            residual_lv3_bot = decoder_lv3(feature_lv3_bot)

            feature_lv2_1 = encoder_lv2(images_lv2_1 + residual_lv3_top)
            feature_lv2_2 = encoder_lv2(images_lv2_2 + residual_lv3_bot)
            feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
            residual_lv2 = decoder_lv2(feature_lv2)

            feature_lv1 = encoder_lv1(images_lv1 + residual_lv2) + feature_lv2
            deblur_image = decoder_lv1(feature_lv1)

            loss_lv1 = mse(deblur_image, gt)

            loss = loss_lv1
            
            encoder_lv1.zero_grad()
            encoder_lv2.zero_grad()
            encoder_lv3.zero_grad()

            decoder_lv1.zero_grad()
            decoder_lv2.zero_grad()
            decoder_lv3.zero_grad()

            loss.backward()

            encoder_lv1_optim.step()
            encoder_lv2_optim.step()
            encoder_lv3_optim.step()

            decoder_lv1_optim.step()
            decoder_lv2_optim.step()
            decoder_lv3_optim.step() 
            total_train_step = total_train_step + 1
            total_train_loss = total_train_loss + loss
            if (iteration+1)%10 == 0:
                stop = time.time()
                print("epoch:", epoch, "iteration:", iteration+1, "loss:%.4f"%loss.item(), 'time:%.4f'%(stop-start))
                start = time.time()
        writer.add_images("Deblur_Epoch:{}".format(epoch),deblur_image.data+0.5, epoch)
        writer.add_images("Sharp_Epoch:{}".format(epoch),images['sharp_image'], epoch)
        writer.add_scalar("train_loss", total_train_loss.item(),epoch)
                

        if (epoch)%200==0:
            if os.path.exists('./checkpointsnew/' + METHOD + '/epoch' + str(epoch)) == False:
                os.system('mkdir ./checkpointsnew/' + METHOD + '/epoch' + str(epoch))
            
            print("Testing...")
            #JJR: 更改为自己的数据目录
            test_dataset = MyDataset(
                blur_image_files = './datas/myData/test_blur_file.txt',
                sharp_image_files = './datas/myData/test_sharp_file.txt',
                root_dir = './datas/myData/',
                crop = True,
                crop_size = IMAGE_SIZE,
                transform = transforms.Compose([
                    transforms.ToTensor()
                ]))
            test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False)
            total_test_loss=0
            test_time = 0.0       		
            for iteration, images in enumerate(test_dataloader):
                with torch.no_grad():                                   
                    images_lv1 = Variable(images['blur_image'] - 0.5).cuda(GPU)
                    start = time.time()
                    H = images_lv1.size(2)
                    W = images_lv1.size(3)                    
                    images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
                    images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
                    images_lv3_1 = images_lv2_1[:,:,:,0:int(W/2)]
                    images_lv3_2 = images_lv2_1[:,:,:,int(W/2):W]
                    images_lv3_3 = images_lv2_2[:,:,:,0:int(W/2)]
                    images_lv3_4 = images_lv2_2[:,:,:,int(W/2):W]
                    
                    feature_lv3_1 = encoder_lv3(images_lv3_1)
                    feature_lv3_2 = encoder_lv3(images_lv3_2)
                    feature_lv3_3 = encoder_lv3(images_lv3_3)
                    feature_lv3_4 = encoder_lv3(images_lv3_4)
                    feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
                    feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
                    feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
                    residual_lv3_top = decoder_lv3(feature_lv3_top)
                    residual_lv3_bot = decoder_lv3(feature_lv3_bot)
                    
                    feature_lv2_1 = encoder_lv2(images_lv2_1 + residual_lv3_top)
                    feature_lv2_2 = encoder_lv2(images_lv2_2 + residual_lv3_bot)
                    feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
                    residual_lv2 = decoder_lv2(feature_lv2)
                    
                    feature_lv1 = encoder_lv1(images_lv1 + residual_lv2) + feature_lv2
                    deblur_image = decoder_lv1(feature_lv1)

                    loss = mse(deblur_image, gt)
                    total_test_loss=loss.item()+total_test_loss


                    stop = time.time()
                    test_time += stop - start
                    print('RunTime:%.4f'%(stop-start), '  Average Runtime:%.4f'%(test_time/(iteration+1)))
                    #JJR:保存的路径
                    save_deblur_images(deblur_image.data + 0.5, iteration, epoch)
            print("整体测试集上的Loss：{}".format(total_test_loss))#item：把tensorboard转化为真实数据    
            writer.add_scalar("test_loss",total_test_loss,epoch)
            
        torch.save(encoder_lv1.state_dict(),str('./newcheckpoints/' + METHOD + "/encoder_lv1.pkl"))
        torch.save(encoder_lv2.state_dict(),str('./newcheckpoints/' + METHOD + "/encoder_lv2.pkl"))
        torch.save(encoder_lv3.state_dict(),str('./newcheckpoints/' + METHOD + "/encoder_lv3.pkl"))
        
        torch.save(decoder_lv1.state_dict(),str('./newcheckpoints/' + METHOD + "/decoder_lv1.pkl"))
        torch.save(decoder_lv2.state_dict(),str('./newcheckpoints/' + METHOD + "/decoder_lv2.pkl"))
        torch.save(decoder_lv3.state_dict(),str('./newcheckpoints/' + METHOD + "/decoder_lv3.pkl"))
        print("模型已保存")
    writer.close()  


if __name__ == '__main__':
    main()

        

        

