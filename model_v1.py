import os
import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

## 네트워크 구축하기

class UnPooling2d(nn.Module):
    def __init__(self, nch=[], pool=2, type='nearest'):
        super().__init__()

        if type == 'nearest':
            self.unpooling = nn.Upsample(scale_factor=pool, mode='nearest')
        elif type == 'bilinear':
            self.unpooling = nn.Upsample(scale_factor=pool, mode='bilinear', align_corners=False)
        elif type == 'conv':
            self.unpooling = nn.ConvTranspose2d(nch, nch, kernel_size=pool, stride=pool)

    def forward(self, x):
        return self.unpooling(x)

class SqueezeExcitation(nn.Module):
    def __init__(self,channels,squeeze_channels=None):
        if squeeze_channels is None:
            squeeze_channels = channels//8
        super(SqueezeExcitation,self).__init__()

        self.channels = channels
        self.fc1 = nn.Conv2d(channels,squeeze_channels,kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(squeeze_channels,channels,kernel_size=1, bias=True)
        self.fc3 = nn.Sigmoid()

    def forward(self,x):
        out = F.avg_pool2d(x,x.size()[2:])
        #out = F.avg_pool2d(x)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = self.fc3(out)
        return x*out
    
class resUNet(nn.Module):
    def __init__(self, nch_in=2, nch_out=1, nch_ker=64):
        super(resUNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias, padding_mode= 'replicate')]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr
        
        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker

        # Contracting path
        self.enc1_1 = CBR2d(1 * self.nch_in, 1 * self.nch_ker)
        self.enc1_2 = CBR2d(1 * self.nch_ker, 1 * self.nch_ker)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(1 * self.nch_ker, 2 * self.nch_ker)
        self.enc2_2 = CBR2d(2 * self.nch_ker, 2 * self.nch_ker)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(2 * self.nch_ker, 4 * self.nch_ker)
        self.enc3_2 = CBR2d(4 * self.nch_ker, 4 * self.nch_ker)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(4 * self.nch_ker, 8 * self.nch_ker)
        self.enc4_2 = CBR2d(8 * self.nch_ker, 8 * self.nch_ker)
        
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.enc5_1 = CBR2d(8 * self.nch_ker, 16 * self.nch_ker)        
        self.dec5_1 = CBR2d(16 * self.nch_ker, 8 * self.nch_ker)
        
        self.unpool4 = nn.ConvTranspose2d(8 * self.nch_ker, 8 * self.nch_ker,
                                          kernel_size=2, stride=2, padding=0, bias= True)

        self.dec4_2 = CBR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker)
        self.dec4_1 = CBR2d(8 * self.nch_ker, 4 * self.nch_ker)

        self.unpool3 = nn.ConvTranspose2d(4 * self.nch_ker, 4 * self.nch_ker,
                                          kernel_size=2, stride=2, padding=0, bias= True)

        self.dec3_2 = CBR2d(2 * 4 * self.nch_ker, 4 * self.nch_ker)
        self.dec3_1 = CBR2d(4 * self.nch_ker, 2 * self.nch_ker)

        self.unpool2 = nn.ConvTranspose2d(2 * self.nch_ker, 2 * self.nch_ker,
                                          kernel_size=2, stride=2, padding=0, bias= True)

        self.dec2_2 = CBR2d(2 * 2 * self.nch_ker, 2 * self.nch_ker)
        self.dec2_1 = CBR2d(2 * self.nch_ker, self.nch_ker)

        self.unpool1 = nn.ConvTranspose2d(self.nch_ker, self.nch_ker,
                                          kernel_size=2, stride=2, padding=0, bias= True)

        self.dec1_2 = CBR2d(2 * self.nch_ker, 1 * self.nch_ker)
        self.dec1_1 = CBR2d(1 * self.nch_ker, 1 * self.nch_ker)

        self.fc = nn.Conv2d(1 * self.nch_ker, 1 * self.nch_out, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, feature):
        x2 = torch.cat((x,feature),dim=1)
        enc1_1 = self.enc1_1(x2)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)
        
        enc5_1 = self.enc5_1(pool4)
        dec5_1 = self.dec5_1(enc5_1)
        
        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        out = self.fc(dec1_1)

        return out+x
    
class SCUNet2(nn.Module):
    def __init__(self, nch_in=1, nch_out=1, nch_ker=64):
        super(SCUNet2, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=1,
                                 bias=bias, padding_mode= 'replicate')]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr
        
        def downconv(in_channels,kernel_size=2, stride=2):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                 kernel_size=kernel_size, stride=stride, padding=0, bias= True)]           
            layers += [nn.BatchNorm2d(num_features=in_channels)]
            layers += [nn.ReLU()]

            downconv = nn.Sequential(*layers)

            return downconv
        
        def upconv(in_channels):
            layers = []
            layers += [UnPooling2d(in_channels, 2,'conv')]
            #layers += [nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
            #                     kernel_size=3, stride=1, padding=1,
            #                     bias=True, padding_mode= 'replicate')]
            #layers += [nn.BatchNorm2d(num_features=in_channels)]
            #layers += [nn.ReLU()]
    
            upconv = nn.Sequential(*layers)
    
            return upconv
        
        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        
        # Contracting path
        self.enc1_1 = CBR2d(1 * self.nch_in, 1 * self.nch_ker)
        self.enc1_2 = CBR2d(1 * self.nch_ker, 1 * self.nch_ker)
        #self.squeeze_enc1 = SqueezeExcitation(1 * self.nch_ker)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        #self.pool1 = downconv(1 * self.nch_ker)
        
        self.enc2_1 = CBR2d(1 * self.nch_ker, 2 * self.nch_ker)
        self.enc2_2 = CBR2d(2 * self.nch_ker, 2 * self.nch_ker)
        #self.squeeze_enc2 = SqueezeExcitation(2 * self.nch_ker)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        #self.pool2 = downconv(2 * self.nch_ker)
        
        self.enc3_1 = CBR2d(2 * self.nch_ker, 4 * self.nch_ker)
        self.enc3_2 = CBR2d(4 * self.nch_ker, 4 * self.nch_ker)
        #self.squeeze_enc3 = SqueezeExcitation(4 * self.nch_ker)
        self.pool3 = nn.AvgPool2d(kernel_size=2)
        #self.pool3 = downconv(4 * self.nch_ker)
        
        # Bottleneck
        self.enc4_1 = CBR2d(4 * self.nch_ker, 8 * self.nch_ker)
        self.dec4_1 = CBR2d(8 * self.nch_ker, 4 * self.nch_ker)
        #self.squeeze_dec4 = SqueezeExcitation(4 * self.nch_ker)
        self.upconv3 = upconv(4 * self.nch_ker)
        
        # Expansive path
        self.dec3_2 = CBR2d(2 * 4 * self.nch_ker, 4 * self.nch_ker)
        self.dec3_1 = CBR2d(4 * self.nch_ker, 2 * self.nch_ker)
        #self.squeeze_dec3 = SqueezeExcitation(2 * self.nch_ker)
        self.upconv2 = upconv(2 * self.nch_ker)
        
        self.dec2_2 = CBR2d(2 * 2 * self.nch_ker, 2 * self.nch_ker)
        self.dec2_1 = CBR2d(2 * self.nch_ker, 1 * self.nch_ker)
        #self.squeeze_dec2 = SqueezeExcitation(1 * self.nch_ker)
        self.upconv1 = upconv(1 * self.nch_ker)
        
        self.dec1_2 = CBR2d(2 * self.nch_ker, 1 * self.nch_ker)
        self.dec1_1 = CBR2d(1 * self.nch_ker, 1 * self.nch_ker)
        #self.squeeze_dec1 = SqueezeExcitation(1 * self.nch_ker)

        self.fc = nn.Conv2d(1 * self.nch_ker, self.nch_out, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self,x):

        # Contracting path
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        #SE_enc1_2 = self.squeeze_enc1(enc1_2)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        #SE_enc2_2 = self.squeeze_enc2(enc2_2)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        #SE_enc3_2 = self.squeeze_enc3(enc3_2)
        pool3 = self.pool3(enc3_2)

        # Bottleneck
        enc4_1 = self.enc4_1(pool3)
        dec4_1 = self.dec4_1(enc4_1)
        #SE_dec4_1 = self.squeeze_dec4(dec4_1)
        unpoolconv3 = self.upconv3(dec4_1)

        # Expaning path
        cat3 = torch.cat((unpoolconv3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        #SE_dec3_1 = self.squeeze_dec3(dec3_1)
        unpoolconv2 = self.upconv2(dec3_1)
        
        cat2 = torch.cat((unpoolconv2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        #SE_dec2_1 = self.squeeze_dec2(dec2_1)
        unpoolconv1 = self.upconv1(dec2_1)
        
        cat1 = torch.cat((unpoolconv1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        #SE_dec1_1 = self.squeeze_dec1(dec1_1)

        out = self.fc(dec1_1)
        
        return x*out
    
class RedSCAN(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64):
        super(RedSCAN, self).__init__()

        
        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker

        # Contracting path
        
        self.enc1_1 = nn.Conv2d(1 * self.nch_in, 1 * self.nch_ker, kernel_size=3, stride=1, padding=1,bias=False, padding_mode='zeros')
        self.enc1_2 = nn.Conv2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, padding=1,bias=False, padding_mode='zeros')
        self.dense1 = DenseNet(1 * self.nch_ker, 1 * self.nch_ker)
        self.dense2 = DenseNet(1 * self.nch_ker, 1 * self.nch_ker)
        self.dense3 = DenseNet(1 * self.nch_ker, 1 * self.nch_ker)
        self.dense4 = DenseNet(1 * self.nch_ker, 1 * self.nch_ker)
        self.dense5 = DenseNet(1 * self.nch_ker, 1 * self.nch_ker)
        self.dec1 = nn.Conv2d(5 * self.nch_ker, 1 * self.nch_ker, kernel_size=1, stride=1, padding=0, bias= False)
        self.dec1_1 = nn.Conv2d(1 * self.nch_ker, 1 * self.nch_out, kernel_size=3, stride=1, padding=1,bias=False, padding_mode='zeros')
        #self.dec1_2 = nn.Conv2d(1 * self.nch_ker, 1 * self.nch_out, kernel_size=3, stride=1, padding=1,bias=False, padding_mode='zeros')        
        
        
    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        dense1 = self.dense1(enc1_2)
        dense2 = self.dense2(dense1)
        dense3 = self.dense3(dense2)
        dense4 = self.dense4(dense3)
        dense5 = self.dense5(dense4)
        cat1 = torch.cat((dense1, dense2), dim=1)
        cat2 = torch.cat((cat1, dense3), dim=1)
        cat3 = torch.cat((cat2, dense4), dim=1)
        cat4 = torch.cat((cat3, dense5), dim=1)
        dec1 = self.dec1(cat4)
        dec1_1 = self.dec1_1(dec1)
        #dec1_2 = self.dec1_2(dec1_1+enc1_1)
        out = dec1_1
        return out
    
class BHCUNet(nn.Module):
    def __init__(self, nch_in=1, nch_out=1, nch_ker=64):
        super(BHCUNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding='same',
                                 bias=bias, padding_mode= 'replicate')]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr
        
        def downconv(in_channels,kernel_size=2, stride=2):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                 kernel_size=kernel_size, stride=stride, padding=0, bias= True)]           
            layers += [nn.BatchNorm2d(num_features=in_channels)]
            layers += [nn.ReLU()]

            downconv = nn.Sequential(*layers)

            return downconv
        
        def upconv(in_channels):
            layers = []
            layers += [UnPooling2d(in_channels, 2,'conv')]
            #layers += [nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
            #                     kernel_size=3, stride=1, padding=1,
            #                     bias=True, padding_mode= 'replicate')]
            #layers += [nn.BatchNorm2d(num_features=in_channels)]
            #layers += [nn.ReLU()]
    
            upconv = nn.Sequential(*layers)
    
            return upconv 
        
        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.nch_feature = 16
        
        # Feature path
        #self.feature_conv = CBR2d(1 * self.nch_in, 1 * self.nch_feature)
        
        
        # Contracting path
        self.enc1_1 = CBR2d(2 * self.nch_in, 1 * self.nch_ker)
        self.enc1_2 = CBR2d(1 * self.nch_ker, 1 * self.nch_ker)
        #self.squeeze_enc1 = SqueezeExcitation(1 * self.nch_ker)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        
        self.enc2_1 = CBR2d(1 * self.nch_ker, 2 * self.nch_ker)
        self.enc2_2 = CBR2d(2 * self.nch_ker, 2 * self.nch_ker)
        #self.squeeze_enc2 = SqueezeExcitation(2 * self.nch_ker)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        
        self.enc3_1 = CBR2d(2 * self.nch_ker, 4 * self.nch_ker)
        self.enc3_2 = CBR2d(4 * self.nch_ker, 4 * self.nch_ker)
        #self.squeeze_enc3 = SqueezeExcitation(4 * self.nch_ker)
        self.pool3 = nn.AvgPool2d(kernel_size=2)
        
        self.enc4_1 = CBR2d(4 * self.nch_ker, 8 * self.nch_ker)
        self.enc4_2 = CBR2d(8 * self.nch_ker, 8 * self.nch_ker)
        #self.squeeze_enc4 = SqueezeExcitation(8 * self.nch_ker)
        self.pool4 = nn.AvgPool2d(kernel_size=2)
        
        # Bottleneck
        self.enc5_1 = CBR2d(8 * self.nch_ker, 16 * self.nch_ker)
        self.dec5_1 = CBR2d(16 * self.nch_ker, 8 * self.nch_ker)
        #self.squeeze_dec5 = SqueezeExcitation(8 * self.nch_ker)
        self.unpool4 = upconv(8 * self.nch_ker)
        
        # Expansive path
        self.dec4_2 = CBR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker)
        self.dec4_1 = CBR2d(8 * self.nch_ker, 4 * self.nch_ker)
        #self.squeeze_dec4 = SqueezeExcitation(4 * self.nch_ker)
        self.unpool3 = upconv(4 * self.nch_ker)
        
        
        self.dec3_2 = CBR2d(2 * 4 * self.nch_ker, 4 * self.nch_ker)
        self.dec3_1 = CBR2d(4 * self.nch_ker, 2 * self.nch_ker)
        #self.squeeze_dec3 = SqueezeExcitation(2 * self.nch_ker)
        self.unpool2 = upconv(2 * self.nch_ker)
     
        self.dec2_2 = CBR2d(2 * 2 * self.nch_ker, 2 * self.nch_ker)
        self.dec2_1 = CBR2d(2 * self.nch_ker, 1 * self.nch_ker)
        #self.squeeze_dec2 = SqueezeExcitation(1 * self.nch_ker)
        self.unpool1 = upconv(1 * self.nch_ker)
        
        self.dec1_2 = CBR2d(2 * self.nch_ker, 1 * self.nch_ker)
        self.dec1_1 = CBR2d(1 * self.nch_ker, 1 * self.nch_ker)
        #self.squeeze_dec1 = SqueezeExcitation(1 * self.nch_ker)

        self.fc = nn.Conv2d(1 * self.nch_ker, self.nch_out, kernel_size=1, stride=1, padding=0, bias=True)
        #self.activation1 = nn.Sigmoid()
        #self.activation2 = nn.ReLU()
        
    def forward(self, x,feature):

        #feat = self.feature_conv(feature)
        x2 = torch.cat((x,feature),1)

        #pdb.set_trace()
        
        # Contracting path
        enc1_1 = self.enc1_1(x2)
        enc1_2 = self.enc1_2(enc1_1)
        #SE_enc1_2 = self.squeeze_enc1(enc1_2)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        #SE_enc2_2 = self.squeeze_enc2(enc2_2)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        #SE_enc3_2 = self.squeeze_enc3(enc3_2)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)     
        #SE_enc4_2 = self.squeeze_enc4(enc4_2)
        pool4 = self.pool4(enc4_2)

        # Bottleneck
        enc5_1 = self.enc5_1(pool4)
        dec5_1 = self.dec5_1(enc5_1)
        #SE_dec5_1 = self.squeeze_dec5(dec5_1)
        unpool4 = self.unpool4(dec5_1)


        # Expaning path
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        #SE_dec4_1 = self.squeeze_dec4(dec4_1)
        unpool3 = self.unpool3(dec4_1)

        
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        #SE_dec3_1 = self.squeeze_dec3(dec3_1)
        unpool2 = self.unpool2(dec3_1)

        
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        #SE_dec2_1 = self.squeeze_dec2(dec2_1)
        unpool1 = self.unpool1(dec2_1)

        
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        #SE_dec1_1 = self.squeeze_dec1(dec1_1)

        out = self.fc(dec1_1)
        #out = self.activation2(out)
        
        return out+x
    
    
class BHCUNet2(nn.Module):
    def __init__(self, nch_in=1, nch_out=1, nch_ker=64):
        super(BHCUNet2, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding='same',
                                 bias=bias, padding_mode= 'replicate')]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr
        
        def downconv(in_channels,kernel_size=2, stride=2):
            layers = []
            #layers += [nn.AvgPool2d(kernel_size=2)]
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                 kernel_size=kernel_size, stride=stride, padding=0, bias= True)]           
            layers += [nn.BatchNorm2d(num_features=in_channels)]
            layers += [nn.ReLU()]

            downconv = nn.Sequential(*layers)

            return downconv
        
        def upconv(in_channels):
            layers = []
            layers += [UnPooling2d()]
            #layers += [nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
            #                     kernel_size=3, stride=1, padding=1,
            #                     bias=True, padding_mode= 'replicate')]
            #layers += [nn.BatchNorm2d(num_features=in_channels)]
            #layers += [nn.ReLU()]
    
            upconv = nn.Sequential(*layers)
    
            return upconv 
        
        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.nch_feature = 16
        
        # Feature path
        #self.feature_conv = CBR2d(1 * self.nch_in, 1 * self.nch_feature)
        
        
        # Contracting path
        self.enc1_1 = CBR2d(2 * self.nch_in, 1 * self.nch_ker)
        self.enc1_2 = CBR2d(1 * self.nch_ker, 1 * self.nch_ker)
        #self.squeeze_enc1 = SqueezeExcitation(1 * self.nch_ker)
        self.pool1 = downconv(1 * self.nch_ker)
        
        self.enc2_1 = CBR2d(1 * self.nch_ker, 2 * self.nch_ker)
        self.enc2_2 = CBR2d(2 * self.nch_ker, 2 * self.nch_ker)
        #self.squeeze_enc2 = SqueezeExcitation(2 * self.nch_ker)
        self.pool2 = downconv(2 * self.nch_ker)
        
        self.enc3_1 = CBR2d(2 * self.nch_ker, 4 * self.nch_ker)
        self.enc3_2 = CBR2d(4 * self.nch_ker, 4 * self.nch_ker)
        #self.squeeze_enc3 = SqueezeExcitation(4 * self.nch_ker)
        self.pool3 = downconv(4 * self.nch_ker)
        
        self.enc4_1 = CBR2d(4 * self.nch_ker, 8 * self.nch_ker)
        self.enc4_2 = CBR2d(8 * self.nch_ker, 8 * self.nch_ker)
        #self.squeeze_enc4 = SqueezeExcitation(8 * self.nch_ker)
        self.pool4 = downconv(8 * self.nch_ker)
        
        # Bottleneck
        self.enc5_1 = CBR2d(8 * self.nch_ker, 16 * self.nch_ker)
        self.dec5_1 = CBR2d(16 * self.nch_ker, 8 * self.nch_ker)
        #self.squeeze_dec5 = SqueezeExcitation(8 * self.nch_ker)
        self.unpool4 = upconv(8 * self.nch_ker)
        
        # Expansive path
        self.dec4_2 = CBR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker)
        self.dec4_1 = CBR2d(8 * self.nch_ker, 4 * self.nch_ker)
        #self.squeeze_dec4 = SqueezeExcitation(4 * self.nch_ker)
        self.unpool3 = upconv(4 * self.nch_ker)
        
        
        self.dec3_2 = CBR2d(2 * 4 * self.nch_ker, 4 * self.nch_ker)
        self.dec3_1 = CBR2d(4 * self.nch_ker, 2 * self.nch_ker)
        #self.squeeze_dec3 = SqueezeExcitation(2 * self.nch_ker)
        self.unpool2 = upconv(2 * self.nch_ker)
     
        self.dec2_2 = CBR2d(2 * 2 * self.nch_ker, 2 * self.nch_ker)
        self.dec2_1 = CBR2d(2 * self.nch_ker, 1 * self.nch_ker)
        #self.squeeze_dec2 = SqueezeExcitation(1 * self.nch_ker)
        self.unpool1 = upconv(1 * self.nch_ker)
        
        self.dec1_2 = CBR2d(2 * self.nch_ker, 1 * self.nch_ker)
        self.dec1_1 = CBR2d(1 * self.nch_ker, 1 * self.nch_ker)
        #self.squeeze_dec1 = SqueezeExcitation(1 * self.nch_ker)

        self.fc = nn.Conv2d(1 * self.nch_ker, self.nch_out, kernel_size=1, stride=1, padding=0, bias=True)
        #self.fc = nn.Conv2d(1 * self.nch_ker, self.nch_out, kernel_size=3, stride=1, padding=1, bias=True, padding_mode= 'replicate')

        
    def forward(self, x,feature):

        #feat = self.feature_conv(feature)
        x2 = torch.cat((x,feature),1)

        #pdb.set_trace()
        
        # Contracting path
        enc1_1 = self.enc1_1(x2)
        enc1_2 = self.enc1_2(enc1_1)
        #SE_enc1_2 = self.squeeze_enc1(enc1_2)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        #SE_enc2_2 = self.squeeze_enc2(enc2_2)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        #SE_enc3_2 = self.squeeze_enc3(enc3_2)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)     
        #SE_enc4_2 = self.squeeze_enc4(enc4_2)
        pool4 = self.pool4(enc4_2)

        # Bottleneck
        enc5_1 = self.enc5_1(pool4)
        dec5_1 = self.dec5_1(enc5_1)
        #SE_dec5_1 = self.squeeze_dec5(dec5_1)
        self.unpool4 = nn.Upsample(size=[enc4_2.shape[2],enc4_2.shape[3]], mode='nearest')
        unpool4 = self.unpool4(dec5_1)


        # Expaning path
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        #SE_dec4_1 = self.squeeze_dec4(dec4_1)
        self.unpool3 = nn.Upsample(size=[enc3_2.shape[2],enc3_2.shape[3]], mode='nearest')
        unpool3 = self.unpool3(dec4_1)

        
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        #SE_dec3_1 = self.squeeze_dec3(dec3_1)
        self.unpool2 = nn.Upsample(size=[enc2_2.shape[2],enc2_2.shape[3]], mode='nearest')
        unpool2 = self.unpool2(dec3_1)

        
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        #SE_dec2_1 = self.squeeze_dec2(dec2_1)
        self.unpool1 = nn.Upsample(size=[enc1_2.shape[2],enc1_2.shape[3]], mode='nearest')
        unpool1 = self.unpool1(dec2_1)

        
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        #SE_dec1_1 = self.squeeze_dec1(dec1_1)

        out = self.fc(dec1_1)
        #out = self.activation2(out)
        
        return out+x
    
class DenseNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, grow_rate=32):
        super(DenseNet, self).__init__()
        
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias, padding_mode= 'zeros')]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr
        
        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.grow_rate = grow_rate
        
        #Dense Block
        self.dense1 = CBR2d(in_channels= 1 * self.nch_in, out_channels= self.grow_rate)
        self.dense2 = CBR2d(in_channels= 1 * self.nch_in + 1 * self.grow_rate, out_channels= self.grow_rate)
        self.dense3 = CBR2d(in_channels= 1 * self.nch_in + 2 * self.grow_rate, out_channels= self.grow_rate)
        self.dense4 = CBR2d(in_channels= 1 * self.nch_in + 3 * self.grow_rate, out_channels= self.grow_rate)
        #Final convolution
        self.fc = nn.Conv2d(1 * self.nch_in + 4 * self.grow_rate, 1 * self.nch_out, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        
        dense1 = self.dense1(x)
        cat1 = torch.cat((x, dense1), dim=1)

        dense2 = self.dense2(cat1)
        cat2 = torch.cat((cat1, dense2), dim=1)
        
        dense3 = self.dense3(cat2)
        cat3 = torch.cat((cat2, dense3), dim=1)
        
        dense4 = self.dense4(cat3)
        cat4 = torch.cat((cat3, dense4), dim=1)

        out = self.fc(cat4)
        #out = SqueezeExcitation(out)

        return out + x
      
class BARNet(nn.Module):
    def __init__(self, feature_map):
        super(BARNet, self).__init__()

        # Scatterblock
        self.scblocks = SCUNet(nch_in= 1, nch_out= 1, nch_ker= 64)
        self.upsample = UnPooling2d(nch= 1, pool=2, type='bilinear')
        
        # Beamhardeningblock
        self.bhcblocks = BHCUNet(nch_in= 3, nch_out= 1, nch_ker= 64)
        
        # Conectionblock
        #self.connecblocks = ConnecNet(nch_in= , nch_out= , nch_ker= )
        
        # Feature Image
        self.featuremap = feature_map
        
    def forward(self, x1, x2):
        SCNet = self.scblocks(x1)
        #SCcorrected = self.connecblocks(SCNet,x2)
        SCRes = x1*SCNet
        SCRes = self.upsample(SCRes)
        cat1 = torch.cat((x2, self.featuremap, SCRes), dim=1)
        BHCNet = self.bhcblocks(cat1)

        return BHCNet
