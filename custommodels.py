import numpy as np
import torch
import torch.nn as nn     
import math

class ProdFunc(nn.Module):
    def __init__(self):
        super(ProdFunc, self).__init__()
    def forward(self, x):
        return torch.prod(x)

class babysfirstAE(nn.Module):
    def __init__(self,num_layers,kernel_size,stride,padding,dilation):
        super(babysfirstAE,self).__init__()

        self.channel_num = 1
        self.lrelu = torch.nn.LeakyReLU(.2)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.a = torch.nn.Conv2d(in_channels=self.channel_num,out_channels=32*2,kernel_size=3,stride=2,padding=2,dilation=1)
        self.b = torch.nn.Conv2d(in_channels=32*2,out_channels=32*4,kernel_size=3,stride=2,padding=2,dilation=1)
        self.c = torch.nn.Conv2d(in_channels=32*4,out_channels=32*8,kernel_size=3,stride=2,padding=2,dilation=1)
        #self.d = torch.nn.Conv2d(in_channels=32*8,out_channels=32*8,kernel_size=5,stride=2,padding=2,dilation=1)

        self.flat = nn.Flatten()
        self.inlin = nn.Linear(256*10**2,200)
        self.outlin = nn.Linear(200,256*10**2)
        self.reshp = Reshape(256*10**2,256)

        self.a_r = torch.nn.ConvTranspose2d(256,32*4,kernel_size=2,stride=3,padding=2,output_padding=1,dilation=1)
        self.b_r = torch.nn.ConvTranspose2d(32*4,32*2,kernel_size=2,stride=3,padding=2,dilation=1)
        #self.c_r = torch.nn.ConvTranspose2d(32*4,32*2,kernel_size=5,stride=2,padding=2,dilation=1)
        self.d_r = torch.nn.ConvTranspose2d(32*2,1,kernel_size=2,stride=3,padding=2,output_padding=1,dilation=1)

    def forward(self,input):
        input = PrintLayer()(input)
        a= self.a(input)
        ar = self.lrelu(a)
        ar =  PrintLayer()(ar)
        b = self.b(ar)
        br = self.lrelu(b)
        br =  PrintLayer()(br)
        c = self.c(br)
        cr = self.lrelu(c)
        cr =  PrintLayer()(cr)
        #d = self.d(cr)
        #dr = self.lrelu(d)
        #dr =  PrintLayer()(dr)
        flat = self.flat(cr) #dr
        flat =  PrintLayer()(flat)
        inlin = self.inlin(flat)
        inlin = self.sigmoid(inlin)
        inlin =  PrintLayer()(inlin)

        outlin = self.outlin(inlin)
        outlin =  PrintLayer()(outlin)

        reshp = self.reshp(outlin)
        reshp =  PrintLayer()(reshp)
        a_r = self.a_r(reshp)
        a_rr = self.relu(a_r)
        a_rr =  PrintLayer()(a_rr)

        b_r = self.b_r(a_r)
        b_rr = self.relu(b_r)
        b_rr =  PrintLayer()(b_rr)

        #c_r = self.c_r(b_rr)
        #c_rr = self.relu(c_r)
        #c_rr =  PrintLayer()(c_rr)

        d_r = self.d_r(b_rr) #c_rr

        outpt = self.sigmoid(d_r)
        outpt =  PrintLayer()(outpt)


        return outpt 



   


class SkipVAE(nn.Module):
    def __init__(self,num_layers,kernel_size,stride,padding,dilation,latent=128):
        super(SkipVAE,self).__init__()
        
        self.encoder = SkipEncoder(num_layers,kernel_size,stride,padding,dilation)
        dim,shape,dimshapes = self.encoder.get_dim()
        self.bottleneck = VAE_Bottleneck(dim,shape,latent)
        self.decoder = SkipDecoder(dim,shape,dimshapes,num_layers,kernel_size,stride,padding,dilation)
    def forward(self,input):
        x,skip = self.encoder(input)
        x,mean,sigma = self.bottleneck(x)
        return self.decoder(x,skip),mean,sigma

