import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import struct
import pdb
from torch_radon import Radon, RadonFanbeam
import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from helper import load_tiff_stack_with_metadata, save_to_tiff_stack

device = torch.device('cuda')


dir_target_resample = './input_resample_sino'
dir_target_resample_768 = './input_resample_sino_768view'

dir_target = './input_96view_sino'
dir_target_label = './label_refine_sino'

n_angles_sparse = 96
n_angles_full = 768
ratio = int(n_angles_full/n_angles_sparse)
input_size = 512
vox_scale = 1/0.28
fn_tonumpy = lambda x: x.to('cpu').detach().numpy()

angles_sparse = np.linspace(0, 2*np.pi, n_angles_sparse, endpoint=False)
angles_full = np.linspace(0, 2*np.pi, n_angles_full, endpoint=False)

dir_data = './input_refine'
dir_data_orig = './orig' 
for i in range(0,208):
	img = np.fromfile(os.path.join(dir_data,'output_%04d.raw' % (i)), dtype=np.float32).reshape([512, 512])
	image_size = img.shape[0]

	# Instantiate Radon transform. clip_to_circle should be True when using filtered backprojection.
	radon1 = RadonFanbeam(image_size, angles_sparse,730*vox_scale,470*vox_scale,864,0.4*vox_scale,clip_to_circle=False)


	with torch.no_grad():
	    x = torch.FloatTensor(img).to(device)

	    sinogram1 = radon1.forward(x)

	fn_tonumpy = lambda x: x.to('cpu').detach().numpy()
	fbp1 = fn_tonumpy(sinogram1)

	f = open(os.path.join(dir_target_resample,'%04d.raw' % i), "wb")
	fbp_p = np.reshape(fbp1, n_angles_sparse*864)
	myfmt = 'f' * len(fbp_p)
	bin = struct.pack(myfmt, *fbp_p)
	f.write(bin)
	f.close
	
for i in range(0,208):
	img = np.fromfile(os.path.join(dir_data,'output_%04d.raw' % (i)), dtype=np.float32).reshape([512, 512])
	image_size = img.shape[0]

	# Instantiate Radon transform. clip_to_circle should be True when using filtered backprojection.
	radon1 = RadonFanbeam(image_size, angles_full,730*vox_scale,470*vox_scale,864,0.4*vox_scale,clip_to_circle=False)


	with torch.no_grad():
	    x = torch.FloatTensor(img).to(device)

	    sinogram1 = radon1.forward(x)

	fn_tonumpy = lambda x: x.to('cpu').detach().numpy()
	fbp1 = fn_tonumpy(sinogram1)

	f = open(os.path.join(dir_target_resample_768,'%04d.raw' % i), "wb")
	fbp_p = np.reshape(fbp1, 768*864)
	myfmt = 'f' * len(fbp_p)
	bin = struct.pack(myfmt, *fbp_p)
	f.write(bin)
	f.close    
	
for i in range(0,208):
	img = np.fromfile(os.path.join(dir_data_orig,'%04d.raw' % (i)), dtype=np.float32).reshape([512, 512])
	image_size = img.shape[0]

	# Instantiate Radon transform. clip_to_circle should be True when using filtered backprojection.
	radon1 = RadonFanbeam(image_size, angles_sparse,730*vox_scale,470*vox_scale,864,0.4*vox_scale,clip_to_circle=False)


	with torch.no_grad():
	    x = torch.FloatTensor(img).to(device)

	    sinogram1 = radon1.forward(x)

	fn_tonumpy = lambda x: x.to('cpu').detach().numpy()
	fbp1 = fn_tonumpy(sinogram1)

	f = open(os.path.join(dir_target,'%04d.raw' % i), "wb")
	fbp_p = np.reshape(fbp1, n_angles_sparse*864)
	myfmt = 'f' * len(fbp_p)
	bin = struct.pack(myfmt, *fbp_p)
	f.write(bin)
	f.close
	
for i in range(0,208):
	img = np.fromfile(os.path.join(dir_data_orig,'%04d.raw' % (i)), dtype=np.float32).reshape([512, 512])
	image_size = img.shape[0]

	# Instantiate Radon transform. clip_to_circle should be True when using filtered backprojection.
	radon1 = RadonFanbeam(image_size, angles_full,730*vox_scale,470*vox_scale,864,0.4*vox_scale,clip_to_circle=False)


	with torch.no_grad():
	    x = torch.FloatTensor(img).to(device)

	    sinogram1 = radon1.forward(x)

	fn_tonumpy = lambda x: x.to('cpu').detach().numpy()
	fbp1 = fn_tonumpy(sinogram1)

	f = open(os.path.join(dir_target_label,'%04d.raw' % i), "wb")
	fbp_p = np.reshape(fbp1, n_angles_full*864)
	myfmt = 'f' * len(fbp_p)
	bin = struct.pack(myfmt, *fbp_p)
	f.write(bin)
	f.close    
