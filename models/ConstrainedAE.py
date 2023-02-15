from models.helper_funcs.modelhelper import *
class ConstrainedAE(nn.Module):
    def __init__(self,num_layers,kernel_size,stride,padding,dilation,latent=128):
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
        
        linear_size = latent
  
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
