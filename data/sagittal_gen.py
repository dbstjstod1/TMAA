import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import struct
import pdb
from torch_radon import Radon, RadonFanbeam
import os

device = torch.device('cuda')
dir_orig = './orig'
dir_input = './input_sagittal'
dir_target = './label_sagittal'

n_angles = 96
n_angles_full = 768
input_size = 512
vox_scale = 1/0.28
fn_tonumpy = lambda x: x.to('cpu').detach().numpy()
for k in range(0,208):
    img = np.fromfile(os.path.join(dir_orig,'%04d.raw' % k), dtype=np.float32).reshape([512, 512])
    bp_stack = torch.zeros((1,input_size*input_size,n_angles)).to(device)
    for i in range(0,n_angles):
        angles = np.linspace(2*np.pi/n_angles*i, 2*np.pi, 1, endpoint=False)
        radon = RadonFanbeam(input_size,angles,730*vox_scale,470*vox_scale,864,0.4*vox_scale,clip_to_circle=False)
       
        with torch.no_grad():
            x = torch.FloatTensor(img).to(device)
            sino = radon.forward(x)
            filtered_sino = radon.filter_sinogram(sino,filter_name='ram-lak')
            bp = radon.backward(filtered_sino)
            bp = torch.reshape(bp,[1,int(input_size)*int(input_size)])
            bp_stack[:,:,i]=(bp/int(n_angles))*10000
            
    #bp_stack = torch.sum(bp_stack,2)
    bp_stack = fn_tonumpy(bp_stack)
    f = open(os.path.join(dir_input,'%04d.raw' % k), "wb")
    bp_stack_p = np.reshape(bp_stack, input_size*input_size*n_angles)
    myfmt = 'f' * len(bp_stack_p)
    bin = struct.pack(myfmt, *bp_stack_p)
    f.write(bin)
    f.close



for k in range(0,208):
    img = np.fromfile(os.path.join(dir_orig,'%04d.raw' % k), dtype=np.float32).reshape([512, 512])
    bp_stack_full = torch.zeros((1,input_size*input_size,n_angles_full)).to(device)
    bp_stack_down = torch.zeros((1,input_size*input_size,n_angles)).to(device)
    for i in range(0,n_angles_full):
        angles = np.linspace(2*np.pi/n_angles_full*i, 2*np.pi, 1, endpoint=False)
        radon = RadonFanbeam(input_size,angles,730*vox_scale,470*vox_scale,864,0.4*vox_scale,clip_to_circle=False)
        
        with torch.no_grad():
            x = torch.FloatTensor(img).to(device)
            sino = radon.forward(x)
            filtered_sino = radon.filter_sinogram(sino,filter_name='ram-lak')
            bp = radon.backward(filtered_sino)
            bp = torch.reshape(bp,[1,int(input_size)*int(input_size)])
            bp_stack_full[:,:,i]=(bp/int(n_angles_full))*10000  

    for j in range(0,n_angles):
        bp_stack_down[:,:,j] = bp_stack_full[:,:,8*j] + bp_stack_full[:,:,8*j+1] + bp_stack_full[:,:,8*j+2] + bp_stack_full[:,:,8*j+3] + bp_stack_full[:,:,8*j+4] + bp_stack_full[:,:,8*j+5] + bp_stack_full[:,:,8*j+6] + bp_stack_full[:,:,8*j+7]
    
    #bp_stack_down = torch.sum(bp_stack_down,2)
    bp_stack_down = fn_tonumpy(bp_stack_down)

    f = open(os.path.join(dir_target,'%04d.raw' % k), "wb")
    bp_stack_full_p = np.reshape(bp_stack_down, input_size*input_size*n_angles)
    myfmt = 'f' * len(bp_stack_full_p)
    bin = struct.pack(myfmt, *bp_stack_full_p)
    f.write(bin)
    f.close
