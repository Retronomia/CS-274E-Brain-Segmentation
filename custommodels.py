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

class ProdFunc(nn.Module):
    def __init__(self):
        super(ProdFunc, self).__init__()
    def forward(self, x):
        # Do your print / debug stuff here
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
            tempshape = int(np.floor(((curr_input_shape+2*self.padding[i]-self.dilation[i]*(self.kernel_size[i]-1)-1)/(self.stride[i]))+1))
            if tempshape < 1:
                raise ValueError(f"Warning: VAE encoder portion only can go up to {i} layers. Cutting off early.")
                print(f"Warning: VAE encoder portion only can go up to {i} layers. Cutting off early.")
                break 
            output_size = int(min(128, 32 * (2 ** i))) #int(min(self.input_shape, 16 * (2 ** i))) #int(min(64, 16 * (2 ** i))) #int(min(128, 32 * (2 ** i)))
            #print(f"Layer {i+1} (channels {curr_input_size} shape {curr_input_shape}) -> {output_size}, {tempshape}")
            encoderlist.append(
                nn.Conv2d(curr_input_size,output_size,
                kernel_size=self.kernel_size[i],
                stride=self.stride[i],
                padding=self.padding[i],
                dilation=self.dilation[i])
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

class Decoder(nn.Module):
    def __init__(self,input_size,input_shape,dimshapes,num_layers,kernel_size,stride,padding,dilation):
        super(Decoder,self).__init__()

        self.input_size = input_size
        self.kernel_size = kernel_size[:]
        self.kernel_size.reverse()
        self.stride = stride[:]
        self.stride.reverse()
        self.padding = padding[:]
        self.padding.reverse()
        self.dilation = dilation[:]
        self.dilation.reverse()

        self.outputWidth = 64
        self.outputChannels = 1
        self.num_layers = num_layers


        decoderlist = []

        curr_input_size = self.input_size
        curr_input_shape = input_shape

        dimshapes.reverse()
        decoderlist.append(nn.BatchNorm2d(input_size))
        decoderlist.append(nn.ReLU())
        for i in range(self.num_layers):
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
            output_size = int(min(128, 32 * (2 ** (num_layers-i-1)))) #int(max(16, self.outputWidth / (2 ** i))) #int(min(64, 16 * (2 ** (num_layers-i-1))))
            #print(f"Layer {i+1} (channels {curr_input_size} shape {curr_input_shape}) -> {output_size}, {tempshape}")
            decoderlist.append(
                nn.ConvTranspose2d(curr_input_size,output_size,
                self.kernel_size[i],
                self.stride[i],
                self.padding[i],
                output_padding=output_padding,
                dilation=self.dilation[i])
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
        decoderlist.append(nn.Sigmoid())
        decoderlist.append(PrintLayer())
        self.decoder = nn.Sequential(*decoderlist)
 

    def forward(self,input):
        return self.decoder(input)


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
            nn.Linear(bottle_chan*input_shape**2,linear_size) #bottle_chan*input_shape**2
        )
        bottlenecklist.append(nn.Sigmoid())
        bottlenecklist.append(PrintLayer())
        #bottlenecklist.append(nn.Dropout(p=0.1))
        bottlenecklist.append(
            nn.Linear(linear_size,bottle_chan*input_shape**2) #bottle_chan*input_shape**2
        )
        bottlenecklist.append(nn.Sigmoid())
        bottlenecklist.append(PrintLayer())
        #bottlenecklist.append(nn.Dropout(p=0.1))
        bottlenecklist.append(Reshape( bottle_chan*input_shape**2, bottle_chan)) #bottle_chan*input_shape**2
        bottlenecklist.append(PrintLayer())
        bottlenecklist.append(nn.Conv2d(bottle_chan,input_size,self.kernel_size,self.stride,padding="same"))
        bottlenecklist.append(PrintLayer())
        self.bottleneck = nn.Sequential(*bottlenecklist)

    def forward(self,input):
        return self.bottleneck(input)


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

class AutoEnc_NoBottleneck(nn.Module):
    def __init__(self,num_layers,kernel_size,stride,padding,dilation):
        super(AutoEnc,self).__init__()
        
        self.encoder = Encoder(num_layers,kernel_size,stride,padding,dilation)
        dim,shape,dimshapes = self.encoder.get_dim()
        #self.bottleneck = Bottleneck(dim,shape)
        self.decoder = Decoder(dim,shape,dimshapes,num_layers,kernel_size,stride,padding,dilation)

    def forward(self,input):
        x = self.encoder(input)
        #x = self.bottleneck(x)
        return self.decoder(x)


class VAE_Bottleneck(nn.Module):
    def __init__(self,input_size,input_shape):
        super(VAE_Bottleneck,self).__init__()

        self.stride = 1
        self.kernel_size = 1
        self.padding = "same"

        orig_chan = input_size
        bottle_chan = input_size//8
        
        linear_size = 64
  
        self.bottleconv = nn.Conv2d(orig_chan,bottle_chan,self.kernel_size,self.stride,padding=self.padding)
        #_ = PrintLayer(bottleconv)
        self.flattenconv = nn.Flatten()

        self.ls_lin = nn.Linear(bottle_chan*input_shape**2,linear_size)
        #self.log_sigma = nn.Dropout(p=0.1)

        self.mu_lin = nn.Linear(bottle_chan*input_shape**2,linear_size)
        #self.mu = nn.Dropout(p=0.1)

        #self.z_vae = z_mu + tf.random_normal(tf.shape(z_sigma)) * z_sigma
        
        self.out_lin = nn.Linear(linear_size,bottle_chan*input_shape**2)
        #self.drop_lin = nn.Dropout(p=0.1)
        self.reshape_out = Reshape(bottle_chan*input_shape**2,bottle_chan)
        
        self.convout = nn.Conv2d(bottle_chan,input_size,self.kernel_size,self.stride,padding="same")

    def forward(self,input):
        bottleconv = self.bottleconv(input)
        flattenconv = self.flattenconv(bottleconv)

        log_sigma = self.ls_lin(flattenconv)
        #log_sigma = self.log_sigma(ls_lin)
        sigma = torch.exp(log_sigma)

        mu = self.ls_lin(flattenconv)
        #mu = self.mu(mu_lin)


        norm = torch.randn_like(mu)
        z_vae = torch.mul(sigma,norm).add(mu)

        out_lin = self.out_lin(z_vae)
        #drop_lin = self.drop_lin(out_lin)

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
        
class ConstrainedAE(nn.Module):
    def __init__(self,num_layers,kernel_size,stride,padding,dilation):
        super(ConstrainedAE,self).__init__()
        
        self.encoder = Encoder(num_layers,kernel_size,stride,padding,dilation)
        dim,shape,dimshapes = self.encoder.get_dim()
        #bottleneck code
        input_size=dim
        input_shape=shape
        self.stride = 1
        self.kernel_size = 1
        self.padding = "same"

        orig_chan = input_size
        bottle_chan = input_size//8
        
        linear_size = 128
  
        self.bottleconv = nn.Conv2d(orig_chan,bottle_chan,self.kernel_size,self.stride,padding=self.padding)

        self.flattenconv = nn.Flatten()

        self.in_lin = nn.Linear(bottle_chan*input_shape**2,linear_size)
        self.lin_drop = nn.Dropout(p=0.1)

        self.out_lin = nn.Linear(linear_size,bottle_chan*input_shape**2)
        self.drop_lin = nn.Dropout(p=0.1)

        self.reshape_out = Reshape(bottle_chan*input_shape**2,input_size//8)
        
        self.convout = nn.Conv2d(input_size//8,input_size,self.kernel_size,self.stride,padding="same")

        #end bottleneck code
        self.decoder = Decoder(dim,shape,dimshapes,num_layers,kernel_size,stride,padding,dilation)

    def forward(self,input):
        x = self.encoder(input)
        #bottleneck
        bottleconv = self.bottleconv(x)
        flattenconv = self.flattenconv(bottleconv)

        in_lin = self.in_lin(flattenconv)
        lin_drop = self.lin_drop(in_lin)


        out_lin = self.out_lin(lin_drop)
        drop_lin = z = self.drop_lin(out_lin)

        reshape_out = self.reshape_out(drop_lin)

        convout = self.convout(reshape_out)

        pred = self.decoder(convout)

        z_rec_pre = self.encoder(pred)
        z_bottleconv = self.bottleconv(z_rec_pre)
        z_flattenconv = self.flattenconv(z_bottleconv)

        z_in_lin = self.in_lin(z_flattenconv)
        z_lin_drop = self.lin_drop(z_in_lin)

        z_out_lin = self.out_lin(z_lin_drop)
        z_rec = self.drop_lin(z_out_lin)
        

        return pred,z,z_rec



class SkipEncoder(nn.Module):
    def __init__(self,num_layers,kernel_size,stride,padding,dilation):
        super(SkipEncoder,self).__init__()

        self.input_size = 1 #number of channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.padding = padding
        self.dilation = dilation
        self.input_shape= 64
        self.num_layers = num_layers

        self.encoderlist = nn.ModuleList()
        curr_input_size = self.input_size
        curr_input_shape = self.input_shape
        self.dimshapes = []
        self.skippos = (self.num_layers//2)-1


        for i in range(self.num_layers):
            self.dimshapes.append(curr_input_shape)
            tempshape = int(np.floor(((curr_input_shape+2*self.padding[i]-self.dilation[i]*(self.kernel_size[i]-1)-1)/(self.stride[i]))+1))
            if tempshape < 1:
                raise ValueError(f"Warning: VAE encoder portion only can go up to {i} layers. Cutting off early.")
                print(f"Warning: VAE encoder portion only can go up to {i} layers. Cutting off early.")
                break 
            output_size = int(min(128, 32 * (2 ** i))) #int(min(self.input_shape, 16 * (2 ** i))) #int(min(64, 16 * (2 ** i))) #int(min(128, 32 * (2 ** i)))
            #print(f"Layer {i+1} (channels {curr_input_size} shape {curr_input_shape}) -> {output_size}, {tempshape}")
            convlayer = nn.Conv2d(curr_input_size,output_size,
                kernel_size=self.kernel_size[i],
                stride=self.stride[i],
                padding=self.padding[i],
                dilation=self.dilation[i])
            batchlayer = nn.BatchNorm2d(output_size)
            activatelayer = nn.LeakyReLU()
            self.encoderlist.append(convlayer)
            self.encoderlist.append(batchlayer)
            self.encoderlist.append(activatelayer)
            curr_input_size=output_size
            curr_input_shape = tempshape

        self.final_dim = curr_input_size
        self.final_shape = curr_input_shape

    def get_dim(self):
        return self.final_dim,self.final_shape,self.dimshapes

    def forward(self,input):
        for i in range(self.num_layers):
            convlayer = self.encoderlist[3*i+0]
            batchlayer = self.encoderlist[3*i+1]
            activatelayer = self.encoderlist[3*i+2]
            if (i)==self.skippos: #skip connection add here
                #print("SkipL:")
                convskip = convlayer(input)
                #print("E:",convskip.shape)
                batchl =batchlayer(convskip)
                input=activatelayer(batchl)
            else: #no skip
                convl = convlayer(input)
                #print("E:",convl.shape)
                batchl = batchlayer(convl)
                input = activatelayer(batchl)
        return input,convskip


class SkipDecoder(nn.Module):
    def __init__(self,input_size,input_shape,dimshapes,num_layers,kernel_size,stride,padding,dilation):
        super(SkipDecoder,self).__init__()

        self.input_size = input_size
        self.kernel_size = kernel_size[:]
        self.kernel_size.reverse()
        self.stride = stride[:]
        self.stride.reverse()
        self.padding = padding[:]
        self.padding.reverse()
        self.dilation = dilation[:]
        self.dilation.reverse()

        self.outputWidth = 64
        self.outputChannels = 1
        self.num_layers = num_layers
        self.skippos = (self.num_layers)-((((self.num_layers))//2))-1

        self.initbatch = nn.BatchNorm2d(input_size)
        self.initactivate = nn.ReLU()
        self.decoderlist = nn.ModuleList()

        curr_input_size = self.input_size
        curr_input_shape = input_shape

        dimshapes.reverse()
        for i in range(self.num_layers):
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
            output_size = int(min(128, 32 * (2 ** (num_layers-i-2)))) #int(max(16, self.outputWidth / (2 ** i))) #int(min(64, 16 * (2 ** (num_layers-i-1))))
            #print("Output size:",output_size)
            #print(f"Layer {i+1} (channels {curr_input_size} shape {curr_input_shape}) -> {output_size}, {tempshape}")
            convlayer = nn.ConvTranspose2d(curr_input_size,output_size,
                self.kernel_size[i],
                self.stride[i],
                self.padding[i],
                output_padding=output_padding,
                dilation=self.dilation[i])
            batchlayer =  nn.BatchNorm2d(output_size)
            activatelayer = nn.LeakyReLU()
            self.decoderlist.append(convlayer)
            self.decoderlist.append(batchlayer)
            self.decoderlist.append(activatelayer)
            curr_input_size = output_size
            curr_input_shape = tempshape

        self.outconv = nn.Conv2d(curr_input_size,self.outputChannels,kernel_size=1,stride=1,padding='same')
        self.outactivate = nn.Sigmoid()


    def forward(self,input,skip):
        #print("Input:",input.shape)
        #print(self.skippos)
        input =  self.initbatch(input)
        input = self.initactivate(input)
        for i in range(self.num_layers):
            convlayer = self.decoderlist[3*i+0]
            batchlayer = self.decoderlist[3*i+1]
            activatelayer = self.decoderlist[3*i+2]
            if (i)==self.skippos: #skip connection add here
                #print("SkipL")
                convskip = convlayer(input)
                #print("D:",convskip.shape)
                #print(convskip.shape,skip.shape)
                convskip = torch.add(convskip,skip)
                #print(convskip.shape)
                batchl =batchlayer(convskip)
                #print('batch')
                input=activatelayer(batchl)
                #print('activate')
            else: #no skip
                convl = convlayer(input)
                #print("D:",convl.shape)
                batchl = batchlayer(convl)
                #print('batch')
                input = activatelayer(batchl)
                #print('active')
        input = self.outconv(input)
        input = self.outactivate(input)
        return input


class SkipAE(nn.Module):
    def __init__(self,num_layers,kernel_size,stride,padding,dilation):
        super(SkipAE,self).__init__()
        self.encoder = SkipEncoder(num_layers,kernel_size,stride,padding,dilation)
        dim,shape,dimshapes = self.encoder.get_dim()
        self.bottleneck = Bottleneck(dim,shape)
        self.decoder = SkipDecoder(dim,shape,dimshapes,num_layers,kernel_size,stride,padding,dilation)

    def forward(self,input):
        x,skip = self.encoder(input)
        x = self.bottleneck(x)
        return self.decoder(x,skip)

