import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import struct
import pdb
from torch_radon import Radon, RadonFanbeam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_patch(full_input_img, full_target_img, patch_size):
    assert full_input_img.shape == full_target_img.shape

    h=768
    w=864
    new_h, new_w = patch_size, patch_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)
    patch_input_img = full_input_img[:,:,top:top + new_h, left:left + new_w]
    patch_target_img = full_target_img[:,:,top:top + new_h, left:left + new_w]

    return patch_input_img, patch_target_img
    
def Sino_generation(input, n_angles):
    batch_size = input.shape[0]
    channel_size = input.shape[1]
    input_size = input.shape[2]
    vox_scale = 1/0.28
    sino_stack = torch.zeros((batch_size,channel_size,n_angles,864)).to(device)
    for j in range(0,batch_size):
        angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
        radon = RadonFanbeam(input_size,angles,730*vox_scale,470*vox_scale,864,0.4*vox_scale,clip_to_circle=False)
            
        with torch.no_grad():
            x = input[j,:,:,:]
            #x = torch.squeeze(x)
            sino = radon.forward(x)
            sino = torch.unsqueeze(sino,0)
            sino_stack[j,:,:,:]=sino
                
    return sino_stack
   
def Sino_generation_upscale(input, n_angles, extended_view):
    batch_size = input.shape[0]
    channel_size = input.shape[1]
    input_size = input.shape[2]
    upsample = nn.Upsample(size=[extended_view, 864])
    vox_scale = 1/0.28   
    extended_sino = torch.zeros((batch_size,channel_size,extended_view,864)).to(device)
    for j in range(0,batch_size):
        angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
        radon = RadonFanbeam(input_size,angles,730*vox_scale,470*vox_scale,864,0.4*vox_scale,clip_to_circle=False)
            
        with torch.no_grad():
            x = input[j,:,:,:]
            #x = torch.squeeze(x)
            sino = radon.forward(x)
            sino = torch.unsqueeze(sino,0)
            sino_ext = upsample(sino)
            sino_ext = torch.unsqueeze(sino_ext,0)
            extended_sino[j,:,:,:]=sino_ext
                
    return extended_sino
    
def Sino_upscale(input, extended_view):
    batch_size = input.shape[0]
    channel_size = input.shape[1]

    upsample = nn.Upsample(size=[extended_view, 864])
    
    extended_sino = torch.zeros((batch_size,channel_size,extended_view,864)).to(device)
    for j in range(0,batch_size):
    	x = input[j,:,:,:]

    	#pdb.set_trace()
    	x = torch.unsqueeze(x,0)
    	sino_ext = upsample(x)
    	sino_ext = torch.squeeze(sino_ext)
    	extended_sino[j,:,:,:]=sino_ext
                
    return extended_sino
    
def FBP(input, n_angles, target_size):
    batch_size = input.shape[0]
    channel_size = input.shape[1]
    vox_scale = 1/0.28   
    fbp = torch.zeros((batch_size,channel_size,target_size,target_size)).to(device)
    for j in range(0,batch_size):
        angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
        radon = RadonFanbeam(target_size,angles,730*vox_scale,470*vox_scale,864,0.4*vox_scale,clip_to_circle=False)
            
        with torch.no_grad():
            x = input[j,:,:,:]
            x = torch.squeeze(x)
            filtered_sino = radon.filter_sinogram(x,filter_name='ram-lak')
            bp = radon.backward(filtered_sino)
            bp = torch.unsqueeze(bp,0)
            fbp[j,:,:,:]=bp
                
    return fbp

def Random_Crop(input, target, crop_size):
    crop_size = crop_size
    random_idx = np.random.randint(0,int(input.shape[2])/int(crop_size))
    input_cropped = input[:,:,int(crop_size)*random_idx:int(crop_size)*random_idx+int(crop_size),:]
    target_cropped = target[:,:,int(crop_size)*random_idx:int(crop_size)*random_idx+int(crop_size),:]          
    return input_cropped, target_cropped

def Seqauntial_Crop(input, target, crop_size):
    crop_size = crop_size
    idx = 0
    input_cropped = input[:,:,int(crop_size)*idx:int(crop_size)*idx+int(crop_size),:]
    target_cropped = target[:,:,int(crop_size)*idx:int(crop_size)*idx+int(crop_size),:]          
    return input_cropped, target_cropped

def Image_conversion(input, target):
    input_image = torch.sum(input,3)/1000      
    target_image = torch.sum(target,3)/1000   
    #pdb.set_trace()
    return input_image, target_image
