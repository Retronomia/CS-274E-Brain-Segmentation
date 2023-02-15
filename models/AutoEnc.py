from models.helper_funcs.modelhelper import *
class AutoEnc(nn.Module):
    def __init__(self,num_layers,kernel_size,stride,padding,dilation,latent=128,image_shape=64):
        super(AutoEnc,self).__init__()
        input_size=1
        self.encoder = Encoder(num_layers,kernel_size,stride,padding,dilation,image_shape,input_size)
        dim,shape,dimshapes = self.encoder.get_dim()
        self.bottleneck = Bottleneck(dim,shape,latent)
        self.decoder = Decoder(dim,shape,dimshapes,num_layers,kernel_size,stride,padding,dilation,image_shape)

    def forward(self,input):
        x = self.encoder(input)
        x = self.bottleneck(x)
        return self.decoder(x)