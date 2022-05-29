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
from datasets import GoProDataset
import time
from PIL import Image
from attention import se_block, cbam_block, eca_block

attention_blocks = [se_block, cbam_block, eca_block]
parser = argparse.ArgumentParser(description="Deep Multi-Patch Hierarchical Network")
parser.add_argument("-e","--epochs",type = int, default = 2600)
parser.add_argument("-se","--start_epoch",type = int, default = 0)
parser.add_argument("-b","--batchsize",type = int, default = 2)
parser.add_argument("-s","--imagesize",type = int, default = 256)
parser.add_argument("-l","--learning_rate", type = float, default = 0.0001)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()

#Hyper Parameters
METHOD = "DMPHN_1_2_4_8_ssim_attention"
SAMPLE_DIR = "test_samples"
EXPDIR = "DMPHN_1_2_4_8_ssim_attention_test"
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
GPU = args.gpu
BATCH_SIZE = args.batchsize
IMAGE_SIZE = args.imagesize

def save_images(images, name):
    filename = './test_results/' + EXPDIR + "/" + name
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
    print("init data folders")

    encoder_lv1 = models.Encoder().apply(weight_init).cuda(GPU)
    encoder_lv2 = models.Encoder().apply(weight_init).cuda(GPU)
    encoder_lv3 = models.Encoder().apply(weight_init).cuda(GPU)
    encoder_lv4 = models.Encoder().apply(weight_init).cuda(GPU)
    attention_lv1 = attention_blocks[0](128).apply(weight_init).cuda(GPU)
    attention_lv2 = attention_blocks[0](128).apply(weight_init).cuda(GPU)
    attention_lv3 = attention_blocks[0](128).apply(weight_init).cuda(GPU)
    attention_lv4 = attention_blocks[0](128).apply(weight_init).cuda(GPU)
    decoder_lv1 = models.Decoder().apply(weight_init).cuda(GPU)
    decoder_lv2 = models.Decoder().apply(weight_init).cuda(GPU)
    decoder_lv3 = models.Decoder().apply(weight_init).cuda(GPU)
    decoder_lv4 = models.Decoder().apply(weight_init).cuda(GPU)

    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv1.pkl")):
        encoder_lv1.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv1.pkl")))
        print("load encoder_lv1 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv2.pkl")):
        encoder_lv2.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv2.pkl")))
        print("load encoder_lv2 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv3.pkl")):
        encoder_lv3.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv3.pkl")))
        print("load encoder_lv3 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv4.pkl")):
        encoder_lv4.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv4.pkl")))
        print("load encoder_lv4 success")

    if os.path.exists(str('./checkpoints/' + METHOD + "/attention_lv1.pkl")):
        attention_lv1.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/attention_lv1.pkl")))
        print("load attention_lv1 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/attention_lv2.pkl")):
        attention_lv2.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/attention_lv2.pkl")))
        print("load attention_lv2 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/attention_lv3.pkl")):
        attention_lv3.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/attention_lv3.pkl")))
        print("load attention_lv3 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/attention_lv4.pkl")):
        attention_lv4.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/attention_lv4.pkl")))
        print("load attention_lv4 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv1.pkl")):
        decoder_lv1.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv1.pkl")))
        print("load encoder_lv1 success")

    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv2.pkl")):
        decoder_lv2.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv2.pkl")))
        print("load decoder_lv2 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv3.pkl")):
        decoder_lv3.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv3.pkl")))
        print("load decoder_lv3 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv4.pkl")):
        decoder_lv4.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv4.pkl")))
        print("load decoder_lv4 success")
    
    if os.path.exists('./test_results/' + EXPDIR) == False:
        os.system('mkdir ./test_results/' + EXPDIR)     
            
    iteration = 0.0
    test_time = 0.0
    for images_name in os.listdir(SAMPLE_DIR):
    	with torch.no_grad():   
            img=Image.open(SAMPLE_DIR + '/' + images_name).convert('RGB')  
            img_w=img.width
            img_h=img.height
            img_w=int(img_w/8)*8           
            img_h=int(img_h/8)*8  
            resize=transforms.Resize([img_h,img_w]) 
            images_lv1 = resize(img)         
            images_lv1 = transforms.ToTensor()(images_lv1)
            images_lv1 = Variable(images_lv1 - 0.5).unsqueeze(0).cuda(GPU)
            start = time.time()           
            H = images_lv1.size(2)
            W = images_lv1.size(3)

            images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
            images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
            images_lv3_1 = images_lv2_1[:,:,:,0:int(W/2)]
            images_lv3_2 = images_lv2_1[:,:,:,int(W/2):W]
            images_lv3_3 = images_lv2_2[:,:,:,0:int(W/2)]
            images_lv3_4 = images_lv2_2[:,:,:,int(W/2):W]
            images_lv4_1 = images_lv3_1[:,:,0:int(H/4),:]
            images_lv4_2 = images_lv3_1[:,:,int(H/4):int(H/2),:]
            images_lv4_3 = images_lv3_2[:,:,0:int(H/4),:]
            images_lv4_4 = images_lv3_2[:,:,int(H/4):int(H/2),:]
            images_lv4_5 = images_lv3_3[:,:,0:int(H/4),:]
            images_lv4_6 = images_lv3_3[:,:,int(H/4):int(H/2),:]
            images_lv4_7 = images_lv3_4[:,:,0:int(H/4),:]
            images_lv4_8 = images_lv3_4[:,:,int(H/4):int(H/2),:]

            feature_lv4_1 = encoder_lv4(images_lv4_1)
            feature_lv4_2 = encoder_lv4(images_lv4_2)
            feature_lv4_3 = encoder_lv4(images_lv4_3)
            feature_lv4_4 = encoder_lv4(images_lv4_4)
            feature_lv4_5 = encoder_lv4(images_lv4_5)
            feature_lv4_6 = encoder_lv4(images_lv4_6)
            feature_lv4_7 = encoder_lv4(images_lv4_7)
            feature_lv4_8 = encoder_lv4(images_lv4_8)

            feature_lv4_1 = attention_lv4(feature_lv4_1)
            feature_lv4_2 = attention_lv4(feature_lv4_2)
            feature_lv4_3 = attention_lv4(feature_lv4_3)
            feature_lv4_4 = attention_lv4(feature_lv4_4)
            feature_lv4_5 = attention_lv4(feature_lv4_5)
            feature_lv4_6 = attention_lv4(feature_lv4_6)
            feature_lv4_7 = attention_lv4(feature_lv4_7)
            feature_lv4_8 = attention_lv4(feature_lv4_8)

            feature_lv4_top_left = torch.cat((feature_lv4_1, feature_lv4_2), 2)
            feature_lv4_top_right = torch.cat((feature_lv4_3, feature_lv4_4), 2)
            feature_lv4_bot_left = torch.cat((feature_lv4_5, feature_lv4_6), 2)
            feature_lv4_bot_right = torch.cat((feature_lv4_7, feature_lv4_8), 2)
            feature_lv4_top = torch.cat((feature_lv4_top_left, feature_lv4_top_right), 3)
            feature_lv4_bot = torch.cat((feature_lv4_bot_left, feature_lv4_bot_right), 3)
            feature_lv4 = torch.cat((feature_lv4_top, feature_lv4_bot), 2)
            residual_lv4_top_left = decoder_lv4(feature_lv4_top_left)
            residual_lv4_top_right = decoder_lv4(feature_lv4_top_right)
            residual_lv4_bot_left = decoder_lv4(feature_lv4_bot_left)
            residual_lv4_bot_right = decoder_lv4(feature_lv4_bot_right)
            residual_lv4_top_left = torch.clamp(residual_lv4_top_left, -0.5, 0.5)
            residual_lv4_top_right = torch.clamp(residual_lv4_top_right, -0.5, 0.5)
            residual_lv4_bot_left = torch.clamp(residual_lv4_bot_left, -0.5, 0.5)
            residual_lv4_bot_right = torch.clamp(residual_lv4_bot_right, -0.5, 0.5)

            feature_lv3_1 = encoder_lv3(images_lv3_1 + residual_lv4_top_left)
            feature_lv3_2 = encoder_lv3(images_lv3_2 + residual_lv4_top_right)
            feature_lv3_3 = encoder_lv3(images_lv3_3 + residual_lv4_bot_left)
            feature_lv3_4 = encoder_lv3(images_lv3_4 + residual_lv4_bot_right)

            feature_lv3_1 = attention_lv3(feature_lv3_1)
            feature_lv3_2 = attention_lv3(feature_lv3_2)
            feature_lv3_3 = attention_lv3(feature_lv3_3)
            feature_lv3_4 = attention_lv3(feature_lv3_4)

            feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3) + feature_lv4_top
            feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3) + feature_lv4_bot
            feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
            residual_lv3_top = decoder_lv3(feature_lv3_top)
            residual_lv3_bot = decoder_lv3(feature_lv3_bot)
            residual_lv3_top = torch.clamp(residual_lv3_top, min=-0.5 ,max=0.5)
            residual_lv3_bot = torch.clamp(residual_lv3_bot, min=-0.5 ,max=0.5)
            
            feature_lv2_1 = encoder_lv2(images_lv2_1 + residual_lv3_top)
            feature_lv2_2 = encoder_lv2(images_lv2_2 + residual_lv3_bot)
            feature_lv2_1 = attention_lv2(feature_lv2_1)
            feature_lv2_2 = attention_lv2(feature_lv2_1)

            feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
            residual_lv2 = decoder_lv2(feature_lv2)
            residual_lv2 = torch.clamp(residual_lv2, min=-0.5 ,max=0.5)


            feature_lv1 = encoder_lv1(images_lv1 + residual_lv2)
            feature_lv1 = attention_lv1(feature_lv1)
            feature_lv1 = feature_lv1 + feature_lv2
            deblur_image = decoder_lv1(feature_lv1)
            deblur_image = torch.clamp(deblur_image, min=-0.5 ,max=0.5)
            
            stop = time.time()
            test_time += stop-start
            print('RunTime:%.4f'%(stop-start), '  Average Runtime:%.4f'%(test_time/(iteration+1)))
            save_images(deblur_image.data + 0.5, images_name) 
            iteration += 1
            
if __name__ == '__main__':
    main()

        

        

