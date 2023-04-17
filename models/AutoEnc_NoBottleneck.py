from models.helper_funcs.modelhelper import *


class AutoEnc_NoBottleneck(nn.Module):
    def __init__(self, num_layers, kernel_size, stride, padding, dilation, latent=None, image_shape=64, inp_size=1):
        super(AutoEnc_NoBottleneck, self).__init__()

        self.encoder = Encoder(num_layers, kernel_size,
                               stride, padding, dilation, image_shape, inp_size)
        dim, shape, dimshapes = self.encoder.get_dim()

        #self.mid = nn.Conv2d(dim,dim,kernel_size=1,stride=1,padding='same')

        #self.bottleneck = Bottleneck(dim,shape)
        self.decoder = Decoder(dim, shape, dimshapes, num_layers,
                               kernel_size, stride, padding, dilation, image_shape)

    def forward(self, input):
        x = self.encoder(input)
        #x = self.mid(x)
        # print(x.shape)
        #x = self.bottleneck(x)
        return self.decoder(x)
