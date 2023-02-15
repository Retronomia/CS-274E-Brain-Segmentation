
import numpy as np
import torch
import torch.nn as nn     
import math

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
        self.debug = False
    def forward(self, x):
        if self.debug:
            print(x.shape)
        return x

class Reshape(nn.Module):
    def __init__(self,lth,channels):
        super(Reshape, self).__init__()
        self.lth = lth
        self.channels = channels

    def forward(self, x):
        return x.view(-1,self.channels,int(np.sqrt(self.lth//self.channels)),int(np.sqrt(self.lth//self.channels)))


class Encoder(nn.Module):
    def __init__(self,num_layers,kernel_size,stride,padding,dilation,input_shape=64,input_size=1):
        super(Encoder,self).__init__()

        self.input_size = input_size #number of channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.padding = padding
        self.dilation = dilation
        self.input_shape= input_shape
        self.num_layers = num_layers

        encoderlist = []
        curr_input_size = self.input_size
        curr_input_shape = self.input_shape
        self.dimshapes = []
        for i in range(self.num_layers):
            self.dimshapes.append(curr_input_shape)
            tempshape = int(np.floor(((curr_input_shape+2*self.padding[i]-self.dilation[i]*(self.kernel_size[i]-1)-1)/(self.stride[i]))+1))
            if tempshape < 1:
                raise ValueError(f"Warning: VAE encoder portion only can go up to {i} layers. Cutting off early.")
                print(f"Warning: VAE encoder portion only can go up to {i} layers. Cutting off early.")
                break 
            output_size = int(min(128, 32 * (2 ** i)))
            #print(f"Layer {i+1} (channels {curr_input_size} shape {curr_input_shape}) -> {output_size}, {tempshape}")
            encoderlist.append(
                nn.Conv2d(curr_input_size,output_size,
                kernel_size=self.kernel_size[i],
                stride=self.stride[i],
                padding=self.padding[i],
                dilation=self.dilation[i])
            )
            #encoderlist.append(PrintLayer())
            encoderlist.append(
                nn.BatchNorm2d(output_size)
            )
            #encoderlist.append(PrintLayer())
            encoderlist.append(
                nn.LeakyReLU()
            )
            encoderlist.append(PrintLayer())
            curr_input_size=output_size
            curr_input_shape = tempshape

        self.encoder = nn.Sequential(*encoderlist)
        self.final_dim = curr_input_size
        self.final_shape = curr_input_shape

    def get_dim(self):
        return self.final_dim,self.final_shape,self.dimshapes

    def forward(self,input):
        return self.encoder(input)

class Decoder(nn.Module):
    def __init__(self,input_size,input_shape,dimshapes,num_layers,kernel_size,stride,padding,dilation,output_shape=64):
        super(Decoder,self).__init__()

        self.input_size = input_size
        self.kernel_size = kernel_size.copy()
        self.kernel_size.reverse()
        self.stride = stride.copy()
        self.stride.reverse()
        self.padding = padding.copy()
        self.padding.reverse()
        self.dilation = dilation.copy()
        self.dilation.reverse()

        self.outputWidth = output_shape
        self.outputChannels = 1
        self.num_layers = num_layers


        decoderlist = []

        curr_input_size = self.input_size
        curr_input_shape = input_shape

        dimshapes.reverse()
        decoderlist.append(nn.BatchNorm2d(input_size))
        decoderlist.append(nn.ReLU())
        for i in range(self.num_layers-1):
            output_padding=0
            tempshape = int(np.floor((curr_input_shape-1)*self.stride[i]-2*self.padding[i]+self.dilation[i]*(self.kernel_size[i]-1)+output_padding+1))
            if tempshape < dimshapes[i]:
                output_padding= dimshapes[i]-tempshape
                #print("padding now",output_padding)
                tempshape = int(np.floor((curr_input_shape-1)*self.stride[i]-2*self.padding[i]+self.dilation[i]*(self.kernel_size[i]-1)+output_padding+1))
            if tempshape < 1:
                raise ValueError(f"Warning: VAE decoder portion only can go up to {i} layers. Cutting off early.")
                print(f"Warning: VAE decoder portion only can go up to {i} layers. Cutting off early.")
                break
            output_size = int(min(128, 32 * (2 ** (num_layers-i-2))))
            #print(f"Layer {i+1} (channels {curr_input_size} shape {curr_input_shape}) -> {output_size}, {tempshape}")
            decoderlist.append(
                nn.ConvTranspose2d(curr_input_size,output_size,
                self.kernel_size[i],
                self.stride[i],
                self.padding[i],
                output_padding=output_padding,
                dilation=self.dilation[i])
            )
            #decoderlist.append(PrintLayer())
            decoderlist.append(
                nn.BatchNorm2d(output_size)
            )
            #decoderlist.append(PrintLayer())
            decoderlist.append(
                nn.LeakyReLU()
            )
            decoderlist.append(PrintLayer())
            curr_input_size = output_size
            curr_input_shape = tempshape

        decoderlist.append(
            nn.Conv2d(curr_input_size,self.outputChannels,kernel_size=1,stride=1,padding='same')
        )
        decoderlist.append(nn.Sigmoid())
        decoderlist.append(PrintLayer())
        self.decoder = nn.Sequential(*decoderlist)
 

    def forward(self,input):
        return self.decoder(input)


class Bottleneck(nn.Module):
    def __init__(self,input_size,input_shape,latent=128):
        super(Bottleneck,self).__init__()

        bottlenecklist = []

        self.stride = 1
        self.kernel_size = 1
        self.padding = "same"

        orig_chan = input_size
        bottle_chan = input_size//8
        linear_size = latent
        #print(orig_chan,bottle_chan,input_shape,linear_size)
  
        bottlenecklist.append(nn.Conv2d(orig_chan,bottle_chan,self.kernel_size,self.stride,padding=self.padding))
        bottlenecklist.append(PrintLayer())
        bottlenecklist.append(nn.Flatten())
        bottlenecklist.append(PrintLayer())
        #print(input_size,input_shape,"input",bottle_chan*input_shape**2,"output",linear_size)
        bottlenecklist.append(
            nn.Linear(bottle_chan*input_shape**2,linear_size)
        )
        bottlenecklist.append(PrintLayer())
        bottlenecklist.append(nn.Dropout(p=0.1))
        bottlenecklist.append(
            nn.Linear(linear_size,bottle_chan*input_shape**2)
        )
        bottlenecklist.append(PrintLayer())
        bottlenecklist.append(nn.Dropout(p=0.1))
        bottlenecklist.append(Reshape(bottle_chan*input_shape**2,bottle_chan))
        bottlenecklist.append(PrintLayer())
        bottlenecklist.append(nn.Conv2d(bottle_chan,input_size,self.kernel_size,self.stride,padding="same"))
        bottlenecklist.append(PrintLayer())
        self.bottleneck = nn.Sequential(*bottlenecklist)

    def forward(self,input):
        return self.bottleneck(input)