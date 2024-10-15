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


dir_target = './residual_768'

n_angles_sparse = 96
n_angles_full = 768
ratio = int(n_angles_full/n_angles_sparse)
input_size = 512
vox_scale = 1/0.28
fn_tonumpy = lambda x: x.to('cpu').detach().numpy()

angles_sparse = np.linspace(0, 2*np.pi, n_angles_sparse, endpoint=False)
angles_full = np.linspace(0, 2*np.pi, n_angles_full, endpoint=False)

dir_data = '../../result_refine_sino_test/raw/output'
for i in range(0,208):
	sino = np.fromfile(os.path.join(dir_data,'output_%04d.raw' % (i+1)), dtype=np.float32).reshape([768, 864])
	#sino_flip = np.copy(np.flip(sino[:, :], axis=0))
	image_size = 512

	
	# Instantiate Radon transform. clip_to_circle should be True when using filtered backprojection.
	radon1 = RadonFanbeam(image_size, angles_full,730*vox_scale,470*vox_scale,864,0.4*vox_scale,clip_to_circle=False)


	with torch.no_grad():
	    sino_float = torch.FloatTensor(sino).to(device)
	    filtered_sinogram = radon1.filter_sinogram(sino_float,filter_name='ram-lak')
	    fbp1 = radon1.backprojection(filtered_sinogram)


	fn_tonumpy = lambda x: x.to('cpu').detach().numpy()
	fbp1 = fn_tonumpy(fbp1)

	f = open(os.path.join(dir_target,'%04d.raw' % i), "wb")
	fbp_p = np.reshape(fbp1, image_size*image_size)
	myfmt = 'f' * len(fbp_p)
	bin = struct.pack(myfmt, *fbp_p)
	f.write(bin)
	f.close
    

