from models.helper_funcs.modelhelper import *


class SkipEncoder(nn.Module):
    def __init__(self, num_layers, kernel_size, stride, padding, dilation, input_shape=64, input_size=1):
        super(SkipEncoder, self).__init__()

        self.input_size = input_size  # number of channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.padding = padding
        self.dilation = dilation
        self.input_shape = input_shape
        self.num_layers = num_layers

        self.encoderlist = nn.ModuleList()
        curr_input_size = self.input_size
        curr_input_shape = self.input_shape
        self.dimshapes = []
        self.skippos = (self.num_layers//2)-1

        for i in range(self.num_layers):
            self.dimshapes.append(curr_input_shape)
            tempshape = int(np.floor(
                ((curr_input_shape+2*self.padding[i]-self.dilation[i]*(self.kernel_size[i]-1)-1)/(self.stride[i]))+1))
            if tempshape < 1:
                raise ValueError(
                    f"Warning: VAE encoder portion only can go up to {i} layers. Cutting off early.")
                print(
                    f"Warning: VAE encoder portion only can go up to {i} layers. Cutting off early.")
                break
            # int(min(self.input_shape, 16 * (2 ** i))) #int(min(64, 16 * (2 ** i))) #int(min(128, 32 * (2 ** i)))
            output_size = int(min(128, 32 * (2 ** i)))
            #print(f"Layer {i+1} (channels {curr_input_size} shape {curr_input_shape}) -> {output_size}, {tempshape}")
            convlayer = nn.Conv2d(curr_input_size, output_size,
                                  kernel_size=self.kernel_size[i],
                                  stride=self.stride[i],
                                  padding=self.padding[i],
                                  dilation=self.dilation[i])
            batchlayer = nn.BatchNorm2d(output_size)
            activatelayer = nn.LeakyReLU()
            self.encoderlist.append(convlayer)
            self.encoderlist.append(batchlayer)
            self.encoderlist.append(activatelayer)
            curr_input_size = output_size
            curr_input_shape = tempshape

        self.final_dim = curr_input_size
        self.final_shape = curr_input_shape

    def get_dim(self):
        return self.final_dim, self.final_shape, self.dimshapes

    def forward(self, input):
        for i in range(self.num_layers):
            convlayer = self.encoderlist[3*i+0]
            batchlayer = self.encoderlist[3*i+1]
            activatelayer = self.encoderlist[3*i+2]
            if (i) == self.skippos:  # skip connection add here
                # print("SkipL:")
                convskip = convlayer(input)
                # print("E:",convskip.shape)
                batchl = batchlayer(convskip)
                input = activatelayer(batchl)
            else:  # no skip
                # print("NoskipL:")
                convl = convlayer(input)
                # print("E:",convl.shape)
                batchl = batchlayer(convl)
                input = activatelayer(batchl)
        return input, convskip


class SkipDecoder(nn.Module):
    def __init__(self, input_size, input_shape, dimshapes, num_layers, kernel_size, stride, padding, dilation, output_shape=64):
        super(SkipDecoder, self).__init__()

        self.input_size = input_size
        self.kernel_size = kernel_size[:]
        self.kernel_size.reverse()
        self.stride = stride[:]
        self.stride.reverse()
        self.padding = padding[:]
        self.padding.reverse()
        self.dilation = dilation[:]
        self.dilation.reverse()

        self.outputWidth = output_shape
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
            output_padding = 0
            tempshape = int(np.floor(
                (curr_input_shape-1)*self.stride[i]-2*self.padding[i]+self.dilation[i]*(self.kernel_size[i]-1)+output_padding+1))
            if tempshape < dimshapes[i]:
                output_padding = dimshapes[i]-tempshape
                #print("padding now",output_padding)
                tempshape = int(np.floor(
                    (curr_input_shape-1)*self.stride[i]-2*self.padding[i]+self.dilation[i]*(self.kernel_size[i]-1)+output_padding+1))
            if tempshape < 1:
                raise ValueError(
                    f"Warning: VAE decoder portion only can go up to {i} layers. Cutting off early.")
                print(
                    f"Warning: VAE decoder portion only can go up to {i} layers. Cutting off early.")
                break
            # int(max(16, self.outputWidth / (2 ** i))) #int(min(64, 16 * (2 ** (num_layers-i-1))))
            output_size = int(min(128, 32 * (2 ** (num_layers-i-2))))
            #print("Output size:",output_size)
            #print(f"Layer {i+1} (channels {curr_input_size} shape {curr_input_shape}) -> {output_size}, {tempshape}")
            convlayer = nn.ConvTranspose2d(curr_input_size, output_size,
                                           self.kernel_size[i],
                                           self.stride[i],
                                           self.padding[i],
                                           output_padding=output_padding,
                                           dilation=self.dilation[i])
            batchlayer = nn.BatchNorm2d(output_size)
            activatelayer = nn.LeakyReLU()
            self.decoderlist.append(convlayer)
            self.decoderlist.append(batchlayer)
            self.decoderlist.append(activatelayer)
            curr_input_size = output_size
            curr_input_shape = tempshape

        self.outconv = nn.Conv2d(
            curr_input_size, self.outputChannels, kernel_size=1, stride=1, padding='same')
        self.outactivate = nn.Sigmoid()

    def forward(self, input, skip):
        # print("Input:",input.shape)
        # print(self.skippos)
        input = self.initbatch(input)
        input = self.initactivate(input)
        for i in range(self.num_layers):
            convlayer = self.decoderlist[3*i+0]
            batchlayer = self.decoderlist[3*i+1]
            activatelayer = self.decoderlist[3*i+2]
            if (i) == self.skippos:  # skip connection add here
                # print("SkipL")
                convskip = convlayer(input)
                # print("D:",convskip.shape)
                # print(convskip.shape,skip.shape)
                convskip = torch.add(convskip, skip)
                # print(convskip.shape)
                batchl = batchlayer(convskip)
                # print('batch')
                input = activatelayer(batchl)
                # print('activate')
            else:  # no skip
                # print("Noskipl")
                convl = convlayer(input)
                # print("D:",convl.shape)
                batchl = batchlayer(convl)
                # print('batch')
                input = activatelayer(batchl)
                # print('active')
        input = self.outconv(input)
        input = self.outactivate(input)
        # print(input.shape)
        return input


class SkipAE(nn.Module):
    def __init__(self, num_layers, kernel_size, stride, padding, dilation, latent=128, image_shape=64):
        super(SkipAE, self).__init__()
        self.encoder = SkipEncoder(
            num_layers, kernel_size, stride, padding, dilation, image_shape, inp_size=1)
        dim, shape, dimshapes = self.encoder.get_dim()
        self.bottleneck = Bottleneck(dim, shape, latent)
        self.decoder = SkipDecoder(
            dim, shape, dimshapes, num_layers, kernel_size, stride, padding, dilation, image_shape)

    def forward(self, input):
        x, skip = self.encoder(input)
        x = self.bottleneck(x)
        return self.decoder(x, skip)
