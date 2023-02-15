from models.helper_funcs.modelhelper import *
class AutoEnc_NoBottleneck(nn.Module):
    def __init__(self,num_layers,kernel_size,stride,padding,dilation,latent=None):
        super(AutoEnc_NoBottleneck,self).__init__()
        
        self.encoder = Encoder(num_layers,kernel_size,stride,padding,dilation)
        dim,shape,dimshapes = self.encoder.get_dim()
        #self.bottleneck = Bottleneck(dim,shape)
        self.decoder = Decoder(dim,shape,dimshapes,num_layers,kernel_size,stride,padding,dilation)

    def forward(self,input):
        x = self.encoder(input)
        #x = self.bottleneck(x)
        return self.decoder(x)
