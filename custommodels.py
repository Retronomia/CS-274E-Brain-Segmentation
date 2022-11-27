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



class Encoder(nn.Module):
    def __init__(self,num_layers,kernel_size,stride,padding,dilation):
        super(Encoder,self).__init__()

        self.input_size = 1 #number of channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.latent_dim = 64
        self.padding = padding
        self.dilation = dilation
        self.input_shape= 64
        self.num_layers = num_layers

        encoderlist = []
        curr_input_size = self.input_size
        curr_input_shape = self.input_shape
        self.dimshapes = []
        for i in range(self.num_layers):
            self.dimshapes.append(curr_input_shape)
            tempshape = int(np.floor(((curr_input_shape+2*self.padding-self.dilation*(self.kernel_size-1)-1)/(self.stride))+1))
            if tempshape < 1:
                raise ValueError(f"Warning: VAE encoder portion only can go up to {i} layers. Cutting off early.")
                print(f"Warning: VAE encoder portion only can go up to {i} layers. Cutting off early.")
                break 
            output_size = int(min(self.input_shape, 16 * (2 ** i)))
            #print(f"Layer {i+1} (channels {curr_input_size} shape {curr_input_shape}) -> {output_size}, {tempshape}")
            encoderlist.append(
                nn.Conv2d(curr_input_size,output_size,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation)
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
            curr_input_shape = tempshape

        self.encoder = nn.Sequential(*encoderlist)
        self.final_dim = curr_input_size
        self.final_shape = curr_input_shape

    def get_dim(self):
        return self.final_dim,self.final_shape,self.dimshapes

    def forward(self,input):
        return self.encoder(input)

class Bottleneck(nn.Module):
    def __init__(self,input_size,input_shape):
        super(Bottleneck,self).__init__()

        bottlenecklist = []

        self.stride = 1
        self.kernel_size = 1
        self.padding = "same"

        orig_chan = input_size
        bottle_chan = input_size//8
        
        linear_size = 128
  
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
        bottlenecklist.append(PrintLayer())
        bottlenecklist.append(
            nn.Linear(linear_size,bottle_chan*input_shape**2)
        )
        bottlenecklist.append(PrintLayer())
        bottlenecklist.append(nn.Dropout(p=0.1))
        bottlenecklist.append(PrintLayer())
        bottlenecklist.append(Reshape(bottle_chan*input_shape**2,input_size//8))
        bottlenecklist.append(PrintLayer())
        bottlenecklist.append(nn.Conv2d(input_size//8,input_size,self.kernel_size,self.stride,padding="same"))
        bottlenecklist.append(PrintLayer())
        self.bottleneck = nn.Sequential(*bottlenecklist)

    def forward(self,input):
        return self.bottleneck(input)

class Decoder(nn.Module):
    def __init__(self,input_size,input_shape,dimshapes,num_layers,kernel_size,stride,padding,dilation):
        super(Decoder,self).__init__()

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.outputWidth = 64
        self.outputChannels = 1
        self.num_layers = num_layers


        decoderlist = []

        curr_input_size = self.input_size
        curr_input_shape = input_shape

        dimshapes.reverse()
        for i in range(self.num_layers):
            output_padding=0
            tempshape = int(np.floor((curr_input_shape-1)*self.stride-2*self.padding+self.dilation*(self.kernel_size-1)+output_padding+1))
            if tempshape < dimshapes[i]:
                output_padding= dimshapes[i]-tempshape
                #print("padding now",output_padding)
                tempshape = int(np.floor((curr_input_shape-1)*self.stride-2*self.padding+self.dilation*(self.kernel_size-1)+output_padding+1))
            if tempshape < 1:
                raise ValueError(f"Warning: VAE decoder portion only can go up to {i} layers. Cutting off early.")
                print(f"Warning: VAE decoder portion only can go up to {i} layers. Cutting off early.")
                break
            output_size = int(max(16, self.outputWidth / (2 ** i)))
            #print(f"Layer {i+1} (channels {curr_input_size} shape {curr_input_shape}) -> {output_size}, {tempshape}")
            decoderlist.append(
                nn.ConvTranspose2d(curr_input_size,output_size,
                self.kernel_size,
                self.stride,
                self.padding,
                output_padding=output_padding,
                dilation=self.dilation)
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
            curr_input_shape = tempshape

        decoderlist.append(
            nn.Conv2d(curr_input_size,self.outputChannels,kernel_size=1,stride=1,padding='same')
        )
        decoderlist.append(PrintLayer())
        self.decoder = nn.Sequential(*decoderlist)
 

    def forward(self,input):
        return self.decoder(input)

class AutoEnc(nn.Module):
    def __init__(self,num_layers,kernel_size,stride,padding,dilation):
        super(AutoEnc,self).__init__()
        
        self.encoder = Encoder(num_layers,kernel_size,stride,padding,dilation)
        dim,shape,dimshapes = self.encoder.get_dim()
        self.bottleneck = Bottleneck(dim,shape)
        self.decoder = Decoder(dim,shape,dimshapes,num_layers,kernel_size,stride,padding,dilation)

    def forward(self,input):
        x = self.encoder(input)
        x = self.bottleneck(x)
        return self.decoder(x)

class VAE_Bottleneck(nn.Module):
    def __init__(self,input_size,input_shape):
        super(VAE_Bottleneck,self).__init__()

        self.stride = 1
        self.kernel_size = 1
        self.padding = "same"

        orig_chan = input_size
        bottle_chan = input_size//8
        
        linear_size = 128
  
        self.bottleconv = nn.Conv2d(orig_chan,bottle_chan,self.kernel_size,self.stride,padding=self.padding)
        #_ = PrintLayer(bottleconv)
        self.flattenconv = nn.Flatten()

        self.ls_lin = nn.Linear(bottle_chan*input_shape**2,linear_size)
        self.log_sigma = nn.Dropout(p=0.1)

        self.mu_lin = nn.Linear(bottle_chan*input_shape**2,linear_size)
        self.mu = nn.Dropout(p=0.1)

        #self.z_vae = z_mu + tf.random_normal(tf.shape(z_sigma)) * z_sigma
        
        self.out_lin = nn.Linear(linear_size,bottle_chan*input_shape**2)
        self.reshape_out = Reshape(bottle_chan*input_shape**2,input_size//8)
        
        self.convout = nn.Conv2d(input_size//8,input_size,self.kernel_size,self.stride,padding="same")

    def forward(self,input):
        bottleconv = self.bottleconv(input)
        flattenconv = self.flattenconv(bottleconv)

        ls_lin = self.ls_lin(flattenconv)
        log_sigma = self.log_sigma(ls_lin)
        sigma = torch.exp(log_sigma)

        mu_lin = self.ls_lin(flattenconv)
        mu = self.log_sigma(mu_lin)


        norm = torch.randn_like(mu)
        z_vae = torch.mul(sigma,norm).add(mu)

        out_lin = self.out_lin(z_vae)
        reshape_out = self.reshape_out(out_lin)

        convout = self.convout(reshape_out)

        return convout,mu,sigma

class VAE(nn.Module):
    def __init__(self,num_layers,kernel_size,stride,padding,dilation):
        super(VAE,self).__init__()
        
        self.encoder = Encoder(num_layers,kernel_size,stride,padding,dilation)
        dim,shape,dimshapes = self.encoder.get_dim()
        self.bottleneck = VAE_Bottleneck(dim,shape)
        self.decoder = Decoder(dim,shape,dimshapes,num_layers,kernel_size,stride,padding,dilation)
    def forward(self,input):
        x = self.encoder(input)
        x,mean,sigma = self.bottleneck(x)
        return self.decoder(x),mean,sigma