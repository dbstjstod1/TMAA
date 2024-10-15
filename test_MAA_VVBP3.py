import argparse
import struct
import pdb
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#from model import *
#from model_attention import *
from model_MAA_VVBP import *
from dataset_MAA_VVBP3 import *
from util import *

import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from vgg_perceptual_loss import *

def run():

## 라이브러리 추가하기

    
    ## Parser 생성하기
    #torch.multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Train the UNet",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
    parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
    parser.add_argument("--num_epoch", default=400, type=int, dest="num_epoch")
    
    parser.add_argument("--data_dir", default="./data", type=str, dest="data_dir")
    parser.add_argument("--ckpt_dir", default="./checkpoint3", type=str, dest="ckpt_dir")
    
    parser.add_argument("--log_dir", default="./log3", type=str, dest="log_dir")
    parser.add_argument("--result_dir", default="./result3", type=str, dest="result_dir")
    
    parser.add_argument("--mode", default="train", type=str, dest="mode")
    parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")
    parser.add_argument("--fine_tuning", default="on", type=str, dest="fine_tuning")
    
    args = parser.parse_args()
    
    ## 트레이닝 파라메터 설정하기
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    
    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    
    log_dir = args.log_dir
    result_dir = args.result_dir
    
    mode = args.mode
    train_continue = args.train_continue
    fine_tuning = args.fine_tuning
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)
    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)
    print("mode: %s" % mode)
    
    ## 불러올 image size
    nu = 512
    nv = 512
    
    ## 디렉토리 생성하기
    if not os.path.exists(result_dir):
        os.makedirs(os.path.join(result_dir, 'png'))
        os.makedirs(os.path.join(result_dir, 'numpy'))
        os.makedirs(os.path.join(result_dir, 'raw'))
        os.makedirs(os.path.join(result_dir, 'raw','input'))
        os.makedirs(os.path.join(result_dir, 'raw','output'))
        os.makedirs(os.path.join(result_dir, 'raw','label'))
        os.makedirs(os.path.join(result_dir, 'raw','residual'))
        
    ## 네트워크 학습하기
    if mode == 'train':
        #transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
        transform = transforms.Compose([ToTensor()])
        dataset_train = Dataset(os.path.join(data_dir, 'train'), nu, nv, transform=transform)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=4)
        
        dataset_val = Dataset(os.path.join(data_dir, 'val'), nu, nv, transform=transform)
        loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)
    
        # 그밖에 부수적인 variables 설정하기
        num_data_train = len(dataset_train)
        num_data_val = len(dataset_val)
    
        num_batch_train = np.ceil(num_data_train / batch_size)
        num_batch_val = np.ceil(num_data_val / batch_size)
    else:
        #transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
        transform = transforms.Compose([ToTensor()])
        dataset_test = Dataset(os.path.join(data_dir, 'test'), nu, nv, transform=transform)
        loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
    
        # 그밖에 부수적인 variables 설정하기
        num_data_test = len(dataset_test)
    
        num_batch_test = np.ceil(num_data_test / batch_size)
    
    ## 네트워크 생성하기
    #_net = U_Net_Recon().cuda()
    #_net = Sparse_Net_comparison().cuda()
    _net = MAA_VVBP3().cuda()
    
    ## Multi-GPU Option
    if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
    	print('Multi GPU activate')
    	net = nn.DataParallel(_net).to(device)
    else:
        print('Single GPU activate')
        net = _net.to(device)
        #net = nn.DataParallel(_net).to(device)  
        
    ## 손실함수 정의하기
    #fn_loss = nn.BCEWithLogitsLoss().to(device)
    fn_loss = nn.MSELoss().to(device)
    #fn_loss = VGGPerceptualLoss().to(device)
    ## Optimizer 설정하기
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    #optim = torch.optim.Adam(list(net1.parameters())+list(net2.parameters()), lr=lr)
    #optim = torch.optim.SGD(net.parameters(), lr=lr, momentum= 0.99)
    
    ##Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.86)
    ## 그밖에 부수적인 functions 설정하기
    #fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    #fn_denorm = lambda x, mean, std: (x * std) + mean
    #fn_class = lambda x: 1.0 * (x > 0.5)
    
    ## Tensorboard 를 사용하기 위한 SummaryWriter 설정
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))
    
    ## 네트워크 학습시키기
    st_epoch = 0
    
    # TRAIN MODE
    if mode == 'train':
        
        if train_continue == "on":
            net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
            
        for epoch in range(st_epoch + 1, num_epoch + 1):
            
            net.train()
            loss_arr = []
    
            for batch, data in enumerate(loader_train, 1):
                # forward pass
                input = data['input'].to(device)
                input2 = data['input2'].to(device)
                label = data['label'].to(device)             
                output, label = net(input,input2,label)
    
                # backward pass
                optim.zero_grad()
    
                loss = fn_loss(output, label)
                loss.backward()
    
                optim.step()
    
                # 손실함수 계산
                loss_arr += [loss.item()]
    
                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.9f" %
                      (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))
                
            writer_train.add_scalar('loss', np.mean(loss_arr), epoch)
    
            with torch.no_grad():
                
                net.eval()
                loss_arr = []
    
                for batch, data in enumerate(loader_val, 1):
                    # forward pass
                    input = data['input'].to(device)
                    input2 = data['input2'].to(device)
                    label = data['label'].to(device)             
                    output, label = net(input,input2,label)
    
                    # 손실함수 계산하기
                    loss = fn_loss(output, label)
                    loss_arr += [loss.item()]
    
                    print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.9f" %
                          (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))
    
            writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
            
            scheduler.step()
            
            if epoch % 10 == 0:
                save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)
                
        writer_train.close()
        writer_val.close()
    
    # TEST MODE
    
    else:
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
        
        
        with torch.no_grad():
            net.eval()
            loss_arr = []
    
            for batch, data in enumerate(loader_test, 1):
                # forward pass
                input = data['input'].to(device)
                input2 = data['input2'].to(device)
                label = data['label'].to(device)             
                output, label, residual = net(input,input2,label)
                            
                # 손실함수 계산하기
                loss = fn_loss(output, label)
    
                loss_arr += [loss.item()]
    
                print("TEST: BATCH %04d / %04d | LOSS %.9f" %
                      (batch, num_batch_test, np.mean(loss_arr)))
    
                # Tensorboard 저장하기
                fn_tonumpy = lambda x: x.to('cpu').detach().numpy()
                nu2 = 512
                nv2 = 512
                
                label = fn_tonumpy(label)/1000
                #input = fn_tonumpy(input)
                output = fn_tonumpy(output)/1000
                residual = fn_tonumpy(residual)

                id = batch 

                
                f = open(os.path.join(result_dir, 'raw', 'label', 'label_%04d.raw' % id), "wb")
                label_p = np.reshape(label.squeeze(), nu2*nv2)
                myfmt = 'f' * len(label_p)
                bin = struct.pack(myfmt, *label_p)
                f.write(bin)
                f.close
                
                #f = open(os.path.join(result_dir, 'raw', 'input', 'input_%04d.raw' % id), "wb")
                #input_p1 = np.reshape(input.squeeze(), nu2*nv2)
                #myfmt = 'f' * len(input_p1)
                #bin = struct.pack(myfmt, *input_p1)
                #f.write(bin)
                #f.close
                
                f = open(os.path.join(result_dir, 'raw', 'output', 'output_%04d.raw' % id), "wb")
                output_p = np.reshape(output.squeeze(), nu2*nv2)
                #output_p = output_p*input_p1
                myfmt = 'f' * len(output_p)
                bin = struct.pack(myfmt, *output_p)
                f.write(bin)
                f.close
    
                f = open(os.path.join(result_dir, 'raw', 'residual', 'residual_%04d.raw' % id), "wb")
                residual_p1 = np.reshape(residual.squeeze(), nu2*nv2)
                myfmt = 'f' * len(residual_p1)
                bin = struct.pack(myfmt, *residual_p1)
                f.write(bin)
                f.close
                    
        print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.9f" %
              (batch, num_batch_test, np.mean(loss_arr)))
    
if __name__ == '__main__':
    run()
