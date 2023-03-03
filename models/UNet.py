from models.helper_funcs.modelhelper import *
import torch.nn.functional as F

class ConvStep(nn.Module):
    def __init__(self,in_channels,mid_channels,out_channels):
        super(ConvStep,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,input):
        return self.conv(input)

class DownStep(nn.Module):
    def __init__(self,in_channels,mid_channels,out_channels):
        super(DownStep,self).__init__()
        self.down = nn.MaxPool2d(kernel_size=2)
        self.conv = ConvStep(in_channels,mid_channels,out_channels)
    def forward(self,input):
        input = self.down(input)
        input = self.conv(input)
        return input
    
class UpStep(nn.Module):
    def __init__(self,in_channels,mid_channels,out_channels):
        super(UpStep,self).__init__()
        self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv = ConvStep(in_channels,mid_channels,out_channels)
    def forward(self,i1,i2):
        #print("i1: ",i1.size())
        #print("i2: ",i2.size())
        i1 = self.up(i1)
        #print("i1: ",i1.size())

        dim1 = i2.size()[2] - i1.size()[2]
        dim2 = i2.size()[3] - i1.size()[3]

        i1 = F.pad(i1,(dim2//2,dim2-(dim2//2),dim1//2,dim1-(dim1//2)))
        #print("i1: ",i1.size())
        output = torch.cat([i2,i1],dim=1)
        #print("cat: ",output.size())
        output = self.conv(output)
        return output

class UNet(nn.Module):
    def __init__(self,num_layers,kernel_size,stride,padding,dilation,latent=128,image_shape=64,inp_size=1):
        super(UNet,self).__init__()

        #input
        #first conv 3x3 relu
        #second conv 3x3 relu -> save a
        self.first_conv = ConvStep(inp_size,64,64)


        #max pool 2x2
        #conv 3x3 relu
        #conv 3x3 relu -> save b
        self.down1 = DownStep(64,128,128)

        #max pool 2x2
        #conv 3x3 relu
        #conv 3x3 relu -> save c
        self.down2 = DownStep(128,256,256)

        #max pool 2x2
        #conv 3x3 relu
        #conv 3x3 relu -> save d
        self.down3 = DownStep(256,512,512)

        #max pool 2x2
        #conv 3x3 relu
        #conv 3x3 relu
        self.down4 = DownStep(512,1024,1024//2)
        
        #up conv 2x2
        #concatenate prev and save d
        #conv 3x3 relu
        #conv 3x3 relu
        self.up1 = UpStep(1024,512,512//2)


        #up conv 2x2
        #concatenate prev and save c
        #conv 3x3 relu
        #conv 3x3 relu
        self.up2 = UpStep(512,256,256//2)


        #up conv 2x2
        #concatenate prev and save b
        #conv 3x3 relu
        #conv 3x3 relu
        self.up3 = UpStep(256,128,128//2)


        #up conv 2x2
        #concatenate prev and save a
        #conv 3x3 relu
        #conv 3x3 relu
        self.up4 = UpStep(128,64,64)


        #conv 1x1
        self.out = nn.Sequential(
            nn.Conv2d(64, inp_size, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self,input):
        a = self.first_conv(input)
        #_ = PrintLayer()(a)
        b = self.down1(a)
        #_ = PrintLayer()(b)
        c = self.down2(b)
        #_ = PrintLayer()(c)
        d = self.down3(c)
        #_ = PrintLayer()(d)

        bridge = self.down4(d)
        #_ = PrintLayer()(bridge)

        up = self.up1(bridge,d)
        #_ = PrintLayer()(up)
        up = self.up2(up,c)
        #_ = PrintLayer()(up)
        up = self.up3(up,b)
        #_ = PrintLayer()(up)
        up = self.up4(up,a)
        #_ = PrintLayer()(up)
        
        fix_dim = self.out(up)
        #_ = PrintLayer()(fix_dim)
        return fix_dim