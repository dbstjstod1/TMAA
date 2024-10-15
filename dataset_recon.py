## import

import os
import numpy as np
import torch

## DATA LOADER
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir_bh, nu, nv, transform=None):
        self.data_dir_bh = data_dir_bh
        self.transform = transform
        self.nu = nu
        self.nv = nv
        #self.nz = nz
        #self.nlam = nlam
        self.data_dir_input_bh = os.path.join(self.data_dir_bh, 'residual_sino')
        self.data_dir_label_bh = os.path.join(self.data_dir_bh, 'residual_sino_label')
        
        lst_input_bh = os.listdir(self.data_dir_input_bh)
        lst_label_bh = os.listdir(self.data_dir_label_bh)

        lst_input_bh.sort()
        lst_label_bh.sort()
        
        self.lst_input_bh = lst_input_bh
        self.lst_label_bh = lst_label_bh

    def __len__(self):
        return len(self.lst_label_bh)

    def __getitem__(self, index):
        
        input_bh = np.fromfile(os.path.join(self.data_dir_input_bh, self.lst_input_bh[index]), dtype=np.float32).reshape([1, self.nv, self.nu])
        label_bh = np.fromfile(os.path.join(self.data_dir_label_bh, self.lst_label_bh[index]), dtype=np.float32).reshape([1, self.nv*8, self.nu])

        data = {'input_bh': input_bh, 'label_bh': label_bh}

        if self.transform:
            data = self.transform(data)

        return data


##
class ToTensor(object):
    def __call__(self, data):
        
        label_bh, input_bh = data['label_bh'], data['input_bh']

        label_bh = label_bh.astype(np.float32)        
        input_bh = input_bh.astype(np.float32)

        data = {'input_bh': torch.from_numpy(input_bh), 'label_bh': torch.from_numpy(label_bh)}

        return data

# class Crop(object):
#     def __call__(self, data):
#         label, input = data['label'], data['input']
        
#         input_cropped = input[:,0:self.nv-32,16:self.nu-16].reshape([self.nlam,self.nv-32, self.nu-32])
#         label_cropped = label[:,0:self.nv-32,16:self.nu-16].reshape([1,self.nv-32, self.nu-32])
        
#         data = {'label' : label_cropped, 'input' : input_cropped}
        
#         return data
        
        
        

'''
class ZeroPadd(object):
    def __call__(self, data):
        label, ini, input = data['label'], data['ini'], data['input']

        npadu = 3
        npadlam = 5
        nv, nlam, nu = input.shape

        input_padded = np.zeros([nv, nlam + npadlam*2, nu + npadu*2], dtype=np.float32)
        input_padded[:,npadlam:npadlam+nlam,npadu:npadu+nu] = input

        ini_padded = np.zeros([1, nlam + npadlam*2, nu + npadu*2], dtype=np.float32)
        ini_padded[:,npadlam:npadlam+nlam,npadu:npadu+nu] = ini

        label_padded = np.zeros([1, nlam + npadlam*2, nu + npadu*2], dtype=np.float32)
        label_padded[:,npadlam:npadlam+nlam,npadu:npadu+nu] = label

        
        data = {'label': label_padded, 'ini': ini_padded,'input': input_padded}

        return data


class ViewSelect(object):
    def __call__(self, data):
        label, input = data['label''label'], data['input']

        nseed = 15
        nbun = 7
        nch, ny, nx = input.shape

        A = np.unique(np.random.randint(0,ny,nseed,dtype=int))
        B = np.random.randint(1,nbun,A.size,dtype=int)
        C = A+B
        D = np.ones(nch,dtype=bool)
        for i in range(A.size):
            D[A[i]:C[i]] = False

        input[D,:,:] = 0.0
        data = {'label': label, 'input': input}

        return data


class ViewAverage(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        nch, ny, nx = input.shape
        nd = 10
        nchd = int(nch / nd)
        input_d = np.zeros([nchd, ny, nx], dtype=np.float32)
        for i in range(nchd):
            input_d[i,:,:] = np.mean(input[i*nd: (i+1) * nd,:,:], axis = 0)

        data = {'label': label, 'input': input_d}

        return data
       


class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        #print(label.shape)
        #print(input.shape)

        if np.random.rand() > 0.5:
            label = np.flip(label, 1)
            input = np.flip(input, 1)

        if np.random.rand() > 0.5:
            label = np.flip(label, 2)
            input = np.flip(input, 2)

        data = {'label': label, 'input': input}

        return data


'''



