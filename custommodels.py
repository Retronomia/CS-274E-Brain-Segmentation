import numpy as np
import torch
import torch.nn as nn     
import math

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
        self.debug = False
    def forward(self, x):
        # Do your print / debug stuff here
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

        
from functools import reduce
from operator import __add__
class Conv2dSamePadding(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()

        self.input_size = 1
        self.kernel_size = 5
        self.stride = 1
        self.latent_dim = 64

        encoderlist = []

        input_shape=(64,64)
        intermediateResolutions = (8,8)
        num_pooling = int(math.log(input_shape[1], 2) - math.log(float(intermediateResolutions[0]), 2))
        curr_input_size = self.input_size
        for i in range(num_pooling):
            output_size = int(min(input_shape[1], 32 * (2 ** i)))
            encoderlist.append(
                nn.Conv2d(curr_input_size,output_size,self.kernel_size,self.stride)
            )
            encoderlist.append(PrintLayer())
            encoderlist.append(
                nn.BatchNorm2d(output_size)
            )
            encoderlist.append(PrintLayer())
            encoderlist.append(
                nn.LeakyReLU()
            )
            encoderlist.append(PrintLayer())
            curr_input_size=output_size

        self.encoder = nn.Sequential(*encoderlist)
        self.final_dim = curr_input_size

    def get_dim(self):
        return self.final_dim

    def forward(self,input):
        return self.encoder(input)

class Bottleneck(nn.Module):
    def __init__(self,input_size):
        super(Bottleneck,self).__init__()

        bottlenecklist = []

        bottlenecklist.append(nn.Conv2d(input_size,input_size//8,1,padding="same"))
        bottlenecklist.append(PrintLayer())
        bottlenecklist.append(nn.Flatten())
        bottlenecklist.append(PrintLayer())
        bottlenecklist.append(
            nn.LazyLinear(128)
        )
        bottlenecklist.append(PrintLayer())
        bottlenecklist.append(nn.Dropout(p=0.1))
        bottlenecklist.append(PrintLayer())
        bottlenecklist.append(
            nn.LazyLinear(21632)
        )
        bottlenecklist.append(PrintLayer())
        bottlenecklist.append(nn.Dropout(p=0.1))
        bottlenecklist.append(PrintLayer())
        bottlenecklist.append(Reshape(21632,input_size//8))
        bottlenecklist.append(PrintLayer())
        bottlenecklist.append(nn.Conv2d(input_size//8,input_size,1,padding="same"))
        bottlenecklist.append(PrintLayer())
        self.bottleneck = nn.Sequential(*bottlenecklist)

    def forward(self,input):
        return self.bottleneck(input)

class Decoder(nn.Module):
    def __init__(self,input_size):
        super(Decoder,self).__init__()

        self.input_size = input_size
        self.kernel_size = 5
        self.stride = 1
        self.latent_dim = 64

        outputChannels = 1
        outputWidth=64
        intermediateResolutions = (8,8)
        decoderlist = []
        num_upsampling = int(math.log(outputWidth, 2) - math.log(float(intermediateResolutions[0]), 2))
        curr_input_size = self.input_size
        for i in range(num_upsampling):

            output_size = int(max(32, outputWidth / (2 ** i)))
            decoderlist.append(
                nn.ConvTranspose2d(curr_input_size,output_size,self.kernel_size,self.stride)
            )
            decoderlist.append(PrintLayer())
            decoderlist.append(
                nn.BatchNorm2d(output_size)
            )
            decoderlist.append(PrintLayer())
            decoderlist.append(
                nn.LeakyReLU()
            )
            decoderlist.append(PrintLayer())
            curr_input_size = output_size
        decoderlist.append(
            nn.Conv2d(curr_input_size,outputChannels,kernel_size=1,stride=1,padding='same')
        )
        decoderlist.append(PrintLayer())
        self.decoder = nn.Sequential(*decoderlist)
 

    def forward(self,input):
        return self.decoder(input)

class AutoEnc(nn.Module):
    def __init__(self):
        super(AutoEnc,self).__init__()
        
        self.encoder = Encoder()
        dim = self.encoder.get_dim()
        self.bottleneck = Bottleneck(dim)
        self.decoder = Decoder(dim)

    def forward(self,input):
        x = self.encoder(input)
        x = self.bottleneck(x)
        return self.decoder(x)