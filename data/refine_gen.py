import matplotlib.pyplot as plt
import numpy as np
import torch
import struct
import os
from torch_radon import Radon, RadonFanbeam

device = torch.device('cuda')
dir_data = './orig'
dir_target = './label_refine'
vox_scale = 1/0.28
for i in range(0,208):
	img = np.fromfile(os.path.join(dir_data,'%04d.raw' % i), dtype=np.float32).reshape([512, 512])
	#img = np.load("phantom.npy")
	image_size = img.shape[0]
	print(image_size)
	n_angles = image_size

	# Instantiate Radon transform. clip_to_circle should be True when using filtered backprojection.
	angles1 = np.linspace(0, 2*np.pi, 768, endpoint=False)
	#angles2 = np.linspace(2*np.pi/400*100, 2*np.pi, 1, endpoint=False)
	#angles3 = np.linspace(2*np.pi/400*200, 2*np.pi, 1, endpoint=False)
	radon1 = RadonFanbeam(image_size, angles1,730*vox_scale,470*vox_scale,864,0.4*vox_scale,clip_to_circle=False)
	#radon2 = RadonFanbeam(image_size, angles2, 393, 207, 512,2)
	#radon3 = RadonFanbeam(image_size, angles3, 393, 207, 512,2)

	with torch.no_grad():
	    x = torch.FloatTensor(img).to(device)

	    sinogram1 = radon1.forward(x)
	    filtered_sinogram = radon1.filter_sinogram(sinogram1,filter_name='ram-lak')
	    fbp1 = radon1.backprojection(filtered_sinogram)


	fn_tonumpy = lambda x: x.to('cpu').detach().numpy()
	fbp1 = fn_tonumpy(fbp1)

	f = open(os.path.join(dir_target,'%04d.raw' % i), "wb")
	fbp_p = np.reshape(fbp1, image_size*image_size)
	myfmt = 'f' * len(fbp_p)
	bin = struct.pack(myfmt, *fbp_p)
	f.write(bin)
	f.close
	
dir_data2 = './orig'     
dir_target2 = './input_128view'   
for i in range(0,208):
	img = np.fromfile(os.path.join(dir_data2,'%04d.raw' % i), dtype=np.float32).reshape([512, 512])
	#img = np.load("phantom.npy")
	image_size = img.shape[0]
	print(image_size)
	n_angles = image_size

	# Instantiate Radon transform. clip_to_circle should be True when using filtered backprojection.
	angles1 = np.linspace(0, 2*np.pi, 128, endpoint=False)
	#angles2 = np.linspace(2*np.pi/400*100, 2*np.pi, 1, endpoint=False)
	#angles3 = np.linspace(2*np.pi/400*200, 2*np.pi, 1, endpoint=False)
	radon1 = RadonFanbeam(image_size, angles1,730*vox_scale,470*vox_scale,864,0.4*vox_scale,clip_to_circle=False)
	#radon2 = RadonFanbeam(image_size, angles2, 393, 207, 512,2)
	#radon3 = RadonFanbeam(image_size, angles3, 393, 207, 512,2)

	with torch.no_grad():
	    x = torch.FloatTensor(img).to(device)

	    sinogram1 = radon1.forward(x)
	    filtered_sinogram = radon1.filter_sinogram(sinogram1,filter_name='ram-lak')
	    fbp1 = radon1.backprojection(filtered_sinogram)


	fn_tonumpy = lambda x: x.to('cpu').detach().numpy()
	fbp1 = fn_tonumpy(fbp1)

	f = open(os.path.join(dir_target2,'%04d.raw' % i), "wb")
	fbp_p = np.reshape(fbp1, image_size*image_size)
	myfmt = 'f' * len(fbp_p)
	bin = struct.pack(myfmt, *fbp_p)
	f.write(bin)
	f.close
