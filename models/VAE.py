from models.helper_funcs.modelhelper import *

class VAE_Bottleneck(nn.Module):
    def __init__(self,input_size,input_shape,latent=128):
        super(VAE_Bottleneck,self).__init__()

        self.stride = 1
        self.kernel_size = 1
        self.padding = "same"

        orig_chan = input_size
        bottle_chan = input_size//8
        
        linear_size = latent
  
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
    def __init__(self,num_layers,kernel_size,stride,padding,dilation,latent=128):
        super(VAE,self).__init__()
        
        self.encoder = Encoder(num_layers,kernel_size,stride,padding,dilation)
        dim,shape,dimshapes = self.encoder.get_dim()
        self.bottleneck = VAE_Bottleneck(dim,shape,latent)
        self.decoder = Decoder(dim,shape,dimshapes,num_layers,kernel_size,stride,padding,dilation)
    def forward(self,input):
        x = self.encoder(input)
        x,mean,sigma = self.bottleneck(x)
        return self.decoder(x),mean,sigma