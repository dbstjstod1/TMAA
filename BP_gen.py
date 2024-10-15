import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import struct
import pdb
from torch_radon import Radon, RadonFanbeam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
def VVBP_gen(input, n_angles):
    batch_size = input.shape[0]
    channel_size = n_angles
    input_size = input.shape[2]

    bp_stack = torch.zeros((batch_size,channel_size,input_size,input_size)).to(device)
    for j in range(0,batch_size):
        for i in range(0,n_angles):
        
            angles = np.linspace(2*np.pi/n_angles*i, 2*np.pi, 1, endpoint=False)
            radon = RadonFanbeam(input_size,angles,480,355,1024,1.5,clip_to_circle=False)
            
            with torch.no_grad():
                x = input[j,:,:,:]
                x = torch.squeeze(x)
                sino = radon.forward(x)
                filtered_sino = radon.filter_sinogram(sino)
                bp = radon.backward(filtered_sino)
                bp_stack[j,i,:,:]=bp/n_angles*10000
    #bp_stack, indices = torch.sort(bp_stack,dim=1)     
                    
    return bp_stack

def VVBP_gen_upscale(input, n_angles, upscale_factor):
    batch_size = input.shape[0]
    channel_size = n_angles
    input_size = input.shape[2]

    bp_stack = torch.zeros((batch_size,channel_size,input_size,input_size)).to(device)
    bp_stack_upscaled = torch.zeros((batch_size,channel_size*upscale_factor,input_size,input_size)).to(device)
    upsample = nn.Upsample(size=[int(n_angles)*int(upscale_factor),int(input_size),int(input_size)])
    for j in range(0,batch_size):
        for i in range(0,n_angles):
        
            angles = np.linspace(2*np.pi/n_angles*i, 2*np.pi, 1, endpoint=False)
            radon = RadonFanbeam(input_size,angles,480,355,1024,1.5,clip_to_circle=False)
            
            with torch.no_grad():
                x = input[j,:,:,:]
                x = torch.squeeze(x)
                sino = radon.forward(x)
                filtered_sino = radon.filter_sinogram(sino)
                bp = radon.backward(filtered_sino)
                bp_stack[j,i,:,:]=bp/(n_angles*upscale_factor)*10000
    bp_stack = torch.unsqueeze(bp_stack,1) 
    bp_stack_upscaled = torch.unsqueeze(bp_stack_upscaled,1) 
    bp_stack_upscaled = upsample(bp_stack)
    bp_stack_upscaled = torch.squeeze(bp_stack_upscaled,1)
                    
    return bp_stack_upscaled
    
def VVBP_gen_down(input, n_angles):
    batch_size = input.shape[0]
    channel_size = n_angles
    input_size = input.shape[2]

    bp_stack = torch.zeros((batch_size,channel_size,input_size,input_size)).to(device)
    for j in range(0,batch_size):
        for i in range(0,n_angles):
        
            angles = np.linspace(2*np.pi/n_angles*i, 2*np.pi, 1, endpoint=False)
            radon = RadonFanbeam(input_size,angles,480,355,1024,1.5,clip_to_circle=False)
            
            with torch.no_grad():
                x = input[j,:,:,:]
                x = torch.squeeze(x)
                sino = radon.forward(x)
                filtered_sino = radon.filter_sinogram(sino)
                bp = radon.backward(filtered_sino)
                bp_stack[j,i,:,:]=bp/n_angles
    bp_stack, indices = torch.sort(bp_stack,dim=1)
    bp_stack_odd = bp_stack[:,0::2,:,:]
    bp_stack_even = bp_stack[:,1::2,:,:]
    down_bp_stack = (bp_stack_odd + bp_stack_even) 
               
    return down_bp_stack
    
def Folded_VVBP_gen(input, n_angles):
    batch_size = input.shape[0]
    channel_size = n_angles
    folded_channel_size = channel_size/2
    input_size = input.shape[2]

    bp_stack = torch.zeros((batch_size,channel_size,input_size,input_size)).to(device)
    for j in range(0,batch_size):
        for i in range(0,n_angles):
        
            angles = np.linspace(2*np.pi/n_angles*i, 2*np.pi, 1, endpoint=False)
            radon = RadonFanbeam(input_size,angles,480,355,1024,1.5,clip_to_circle=False)
            
            with torch.no_grad():
                x = input[j,:,:,:]
                x = torch.squeeze(x)
                sino = radon.forward(x)
                filtered_sino = radon.filter_sinogram(sino)
                bp = radon.backward(filtered_sino)            
                bp_stack[j,i,:,:]=bp/n_angles
    bp_stack, indices = torch.sort(bp_stack,dim=1)   
    folded_bp_stack_bottom = bp_stack[:,:int(folded_channel_size),:,:]
    folded_bp_stack_top = torch.flip(bp_stack[:,int(folded_channel_size):,:,:],dims=(1,))
    folded_bp_stack = (folded_bp_stack_bottom + folded_bp_stack_top)
    folded_bp_stack, indices = torch.sort(folded_bp_stack,dim=1)     
           
    return folded_bp_stack
    
def Manifold_VVBP(input, n_angles):
    batch_size = input.shape[0]
    channel_size = n_angles
    input_size = input.shape[2]

    bp_stack = torch.zeros((batch_size,1,input_size*input_size,channel_size)).to(device)
    for j in range(0,batch_size):
        for i in range(0,n_angles):
        
            angles = np.linspace(2*np.pi/n_angles*i, 2*np.pi, 1, endpoint=False)
            radon = RadonFanbeam(input_size,angles,480,355,1024,1.5,clip_to_circle=False)
            
            with torch.no_grad():
                x = input[j,:,:,:]
                x = torch.squeeze(x)
                sino = radon.forward(x)
                filtered_sino = radon.filter_sinogram(sino)
                bp = radon.backward(filtered_sino)
                #pdb.set_trace()
                #bp = torch.transpose(bp,1,2)
                bp = torch.reshape(bp,[1,int(input_size)*int(input_size)])
                bp_stack[j,:,:,i]=bp/n_angles*10000  
                     
    return bp_stack

def Manifold_VVBP_upscale(input, n_angles, upscale_factor):
    batch_size = input.shape[0]
    channel_size = n_angles
    input_size = input.shape[2]

    bp_stack = torch.zeros((batch_size,channel_size,input_size,input_size)).to(device)
    bp_stack_upscaled = torch.zeros((batch_size,channel_size*upscale_factor,input_size,input_size)).to(device)
    upsample = nn.Upsample(size=[int(n_angles)*int(upscale_factor),int(input_size),int(input_size)])
    for j in range(0,batch_size):
        for i in range(0,n_angles):
        
            angles = np.linspace(2*np.pi/n_angles*i, 2*np.pi, 1, endpoint=False)
            radon = RadonFanbeam(input_size,angles,480,355,1024,1.5,clip_to_circle=False)
            
            with torch.no_grad():
                x = input[j,:,:,:]
                x = torch.squeeze(x)
                sino = radon.forward(x)
                filtered_sino = radon.filter_sinogram(sino)
                bp = radon.backward(filtered_sino)
                bp_stack[j,i,:,:]=bp/(n_angles*upscale_factor)*10000
    bp_stack = torch.unsqueeze(bp_stack,1) 
    bp_stack_upscaled = torch.unsqueeze(bp_stack_upscaled,1) 
    bp_stack_upscaled = upsample(bp_stack)
    bp_stack_upscaled = torch.squeeze(bp_stack_upscaled,1)
    bp_stack_upscaled = torch.transpose(bp_stack_upscaled,1,3)
    bp_stack_upscaled = torch.reshape(bp_stack_upscaled,[int(batch_size),1,int(input_size)*int(input_size),int(n_angles)*int(upscale_factor)])
                    
    return bp_stack_upscaled

def Manifold_VVBP_down(input, n_angles):
    batch_size = input.shape[0]
    channel_size = n_angles
    input_size = input.shape[2]

    bp_stack = torch.zeros((batch_size,1,input_size*input_size,channel_size)).to(device)
    for j in range(0,batch_size):
        for i in range(0,n_angles):
        
            angles = np.linspace(2*np.pi/n_angles*i, 2*np.pi, 1, endpoint=False)
            radon = RadonFanbeam(input_size,angles,480,355,1024,1.5,clip_to_circle=False)
            
            with torch.no_grad():
                x = input[j,:,:,:]
                x = torch.squeeze(x)
                sino = radon.forward(x)
                filtered_sino = radon.filter_sinogram(sino)
                bp = radon.backward(filtered_sino)
                bp = torch.reshape(bp,[1,int(input_size)*int(input_size)])
                bp_stack[j,:,:,i]=bp/n_angles  
    bp_stack_odd = bp_stack[:,:,:,0::2]
    bp_stack_even = bp_stack[:,:,:,1::2]
    down_bp_stack = (bp_stack_odd + bp_stack_even)                
    return down_bp_stack

def Manifold_VVBP_down2(input, n_angles):
    batch_size = input.shape[0]
    channel_size = n_angles
    input_size = input.shape[2]

    bp_stack = torch.zeros((batch_size,1,input_size*input_size,channel_size)).to(device)
    for j in range(0,batch_size):
        for i in range(0,n_angles):
        
            angles = np.linspace(2*np.pi/n_angles*i, 2*np.pi, 1, endpoint=False)
            radon = RadonFanbeam(input_size,angles,480,355,1024,1.5,clip_to_circle=False)
            
            with torch.no_grad():
                x = input[j,:,:,:]
                x = torch.squeeze(x)
                sino = radon.forward(x)
                filtered_sino = radon.filter_sinogram(sino)
                bp = radon.backward(filtered_sino)
                #bp = torch.transpose(bp,1,2)
                bp = torch.reshape(bp,[1,int(input_size)*int(input_size)])
                bp_stack[j,:,:,i]=bp/n_angles*10000 
    bp_stack_odd = bp_stack[:,:,:,0::2]
    bp_stack_even = bp_stack[:,:,:,1::2]
    down_bp_stack = (bp_stack_odd + bp_stack_even)
    
    bp_stack_odd2 = down_bp_stack[:,:,:,0::2]
    bp_stack_even2 = down_bp_stack[:,:,:,1::2]
    down_bp_stack2 = (bp_stack_odd2 + bp_stack_even2)                  
    return down_bp_stack2

def Manifold_VVBP_down3(input, n_angles):
    batch_size = input.shape[0]
    channel_size = n_angles
    input_size = input.shape[2]

    bp_stack = torch.zeros((batch_size,1,input_size*input_size,channel_size)).to(device)
    for j in range(0,batch_size):
        for i in range(0,n_angles):
        
            angles = np.linspace(2*np.pi/n_angles*i, 2*np.pi, 1, endpoint=False)
            radon = RadonFanbeam(input_size,angles,480,355,1024,1.5,clip_to_circle=False)
            
            with torch.no_grad():
                x = input[j,:,:,:]
                x = torch.squeeze(x)
                sino = radon.forward(x)
                filtered_sino = radon.filter_sinogram(sino)
                bp = radon.backward(filtered_sino)
                #bp = torch.transpose(bp,1,2)
                bp = torch.reshape(bp,[1,int(input_size)*int(input_size)])
                bp_stack[j,:,:,i]=bp/n_angles*10000  
                
    bp_stack_odd = bp_stack[:,:,:,0::2]
    bp_stack_even = bp_stack[:,:,:,1::2]
    down_bp_stack = (bp_stack_odd + bp_stack_even)
    
    bp_stack_odd2 = down_bp_stack[:,:,:,0::2]
    bp_stack_even2 = down_bp_stack[:,:,:,1::2]
    down_bp_stack2 = (bp_stack_odd2 + bp_stack_even2)  
    
    bp_stack_odd3 = down_bp_stack2[:,:,:,0::2]
    bp_stack_even3 = down_bp_stack2[:,:,:,1::2]
    down_bp_stack3 = (bp_stack_odd3 + bp_stack_even3)  
               
    return down_bp_stack3

def Manifold_VVBP_down2_transpose(input, n_angles):
    batch_size = input.shape[0]
    channel_size = n_angles
    input_size = input.shape[2]

    bp_stack = torch.zeros((batch_size,1,input_size*input_size,channel_size)).to(device)
    for j in range(0,batch_size):
        for i in range(0,n_angles):
        
            angles = np.linspace(2*np.pi/n_angles*i, 2*np.pi, 1, endpoint=False)
            radon = RadonFanbeam(input_size,angles,480,355,1024,1.5,clip_to_circle=False)
            
            with torch.no_grad():
                x = input[j,:,:,:]
                x = torch.squeeze(x)
                sino = radon.forward(x)
                filtered_sino = radon.filter_sinogram(sino)
                bp = radon.backward(filtered_sino)
                bp = torch.transpose(bp,1,2)
                bp = torch.reshape(bp,[1,int(input_size)*int(input_size)])
                bp_stack[j,:,:,i]=bp/n_angles*10000  
    bp_stack_odd = bp_stack[:,:,:,0::2]
    bp_stack_even = bp_stack[:,:,:,1::2]
    down_bp_stack = (bp_stack_odd + bp_stack_even)
    
    bp_stack_odd2 = down_bp_stack[:,:,:,0::2]
    bp_stack_even2 = down_bp_stack[:,:,:,1::2]
    down_bp_stack2 = (bp_stack_odd2 + bp_stack_even2)                  
    return down_bp_stack2

def Manifold_VVBP_down3_transpose(input, n_angles):
    batch_size = input.shape[0]
    channel_size = n_angles
    input_size = input.shape[2]

    bp_stack = torch.zeros((batch_size,1,input_size*input_size,channel_size)).to(device)
    for j in range(0,batch_size):
        for i in range(0,n_angles):
        
            angles = np.linspace(2*np.pi/n_angles*i, 2*np.pi, 1, endpoint=False)
            radon = RadonFanbeam(input_size,angles,480,355,1024,1.5,clip_to_circle=False)
            
            with torch.no_grad():
                x = input[j,:,:,:]
                x = torch.squeeze(x)
                sino = radon.forward(x)
                filtered_sino = radon.filter_sinogram(sino)
                bp = radon.backward(filtered_sino)
                bp = torch.transpose(bp,1,2)
                bp = torch.reshape(bp,[1,int(input_size)*int(input_size)])
                bp_stack[j,:,:,i]=bp/n_angles*10000  
    bp_stack_odd = bp_stack[:,:,:,0::2]
    bp_stack_even = bp_stack[:,:,:,1::2]
    down_bp_stack = (bp_stack_odd + bp_stack_even)
    
    bp_stack_odd2 = down_bp_stack[:,:,:,0::2]
    bp_stack_even2 = down_bp_stack[:,:,:,1::2]
    down_bp_stack2 = (bp_stack_odd2 + bp_stack_even2)

    bp_stack_odd3 = down_bp_stack2[:,:,:,0::2]
    bp_stack_even3 = down_bp_stack2[:,:,:,1::2]
    down_bp_stack3 = (bp_stack_odd3 + bp_stack_even3)   
                  
    return down_bp_stack3

def Manifold_VVBP_axial(input, n_angles):
    batch_size = input.shape[0]
    channel_size = n_angles
    input_size = input.shape[2]

    bp_stack = torch.zeros((batch_size,channel_size,input_size,input_size)).to(device)
    for j in range(0,batch_size):
        for i in range(0,n_angles):
        
            angles = np.linspace(2*np.pi/n_angles*i, 2*np.pi, 1, endpoint=False)
            radon = RadonFanbeam(input_size,angles,480,355,1024,1.5,clip_to_circle=False)
            
            with torch.no_grad():
                x = input[j,:,:,:]
                x = torch.squeeze(x)
                sino = radon.forward(x)
                filtered_sino = radon.filter_sinogram(sino)
                bp = radon.backward(filtered_sino)
                bp_stack[j,i,:,:]=bp/n_angles*10000             
    bp_stack = torch.reshape(bp_stack,[batch_size,1,int(n_angles)*int(input_size),int(input_size)])     
                    
    return bp_stack

def FBP(input, n_angles):
    batch_size = input.shape[0]
    channel_size = input.shape[1]
    input_size = input.shape[2]
    vox_scale = 1/0.28

    fbp = torch.zeros((batch_size,channel_size,input_size,input_size)).to(device)
    for j in range(0,batch_size):
        angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
        radon = RadonFanbeam(input_size,angles,730*vox_scale,470*vox_scale,864,0.4*vox_scale,clip_to_circle=False)
            
        with torch.no_grad():
            x = input[j,:,:,:]
            x = torch.squeeze(x)
            sino = radon.forward(x)
            filtered_sino = radon.filter_sinogram(sino,filter_name='ram-lak')
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

def Random_Crop2(input, input2, target, crop_size):
    crop_size = crop_size
    random_idx = np.random.randint(0,int(input.shape[2])/int(crop_size))
    input_cropped = input[:,:,int(crop_size)*random_idx:int(crop_size)*random_idx+int(crop_size),:]
    input_cropped2 = input2[:,:,int(crop_size)*random_idx:int(crop_size)*random_idx+int(crop_size),:]
    target_cropped = target[:,:,int(crop_size)*random_idx:int(crop_size)*random_idx+int(crop_size),:]          
    return input_cropped, input_cropped2, target_cropped

def Seqauntial_Crop(input, target, crop_size, idx):
    crop_size = crop_size
    input_cropped = input[:,:,int(crop_size)*idx:int(crop_size)*idx+int(crop_size),:]
    target_cropped = target[:,:,int(crop_size)*idx:int(crop_size)*idx+int(crop_size),:]          
    return input_cropped, target_cropped

def Seqauntial_Crop2(input, input2, target, crop_size, idx):
    crop_size = crop_size
    input_cropped = input[:,:,int(crop_size)*idx:int(crop_size)*idx+int(crop_size),:]
    input_cropped2 = input2[:,:,int(crop_size)*idx:int(crop_size)*idx+int(crop_size),:]
    target_cropped = target[:,:,int(crop_size)*idx:int(crop_size)*idx+int(crop_size),:]          
    return input_cropped, input_cropped2, target_cropped

def Image_conversion(input, target):
    input_image = torch.sum(input,3)/10000      
    target_image = torch.sum(target,3)/10000   
    #pdb.set_trace()
    return input_image, target_image


def Image_conversion2(input, target):
    batch_size = input.shape[0]
    input_size = input.shape[3]
    n_angle = int(input.shape[2]/input_size)
    input_image = torch.reshape(input,[batch_size,int(n_angle),int(input_size),int(input_size)])
    input_image = torch.sum(input_image,1)/10000
    target_image = torch.reshape(target,[batch_size,int(n_angle),int(input_size),int(input_size)])      
    target_image = torch.sum(target_image,1)/10000   
    return input_image, target_image

def Image_conversion3(input, input2, target):
    batch_size = input.shape[0]
    input_size = input.shape[3]
    n_angle = int(input.shape[2]/input_size)
    input_image = torch.reshape(input,[batch_size,int(n_angle),int(input_size),int(input_size)])
    input_image = torch.sum(input_image,1)/10000
    input_image2 = torch.reshape(input2,[batch_size,int(n_angle),int(input_size),int(input_size)])
    input_image2 = torch.sum(input_image2,1)/10000
    target_image = torch.reshape(target,[batch_size,int(n_angle),int(input_size),int(input_size)])      
    target_image = torch.sum(target_image,1)/10000   
    return input_image, input_image2, target_image

def reshape_VVBP(input, target):
    batch_size = input.shape[0]
    input_size = input.shape[3]
    n_angle = int(input.shape[2]/input.shape[3])

    input_image = torch.reshape(input,[batch_size,int(n_angle),int(input_size),int(input_size)])
    target_image = torch.reshape(target,[batch_size,int(n_angle),int(input_size),int(input_size)])

    return input_image, target_image

def reshape_transpose(input):
    batch_size = input.shape[0]
    n_angle = input.shape[3]
    input_size = np.sqrt(input.shape[2])
    output = torch.reshape(input,[batch_size,int(input_size),int(input_size),int(n_angle)]) #1 is batch_size, 256 is input_size
    output = torch.transpose(output,1,2)
    output = torch.reshape(output,[batch_size,1,int(input_size)*int(input_size),int(n_angle)])
    return output
