from data_loader import loadData
train,val,test,loader = loadData("wmh_semi")



# from utils import *
# '''
# ae = load_class("AutoEnc")
# encoderdict = dict()
# encoderdict['num_layers'] = 4
# encoderdict['kernel_size'] = [3,2,4,2] 
# encoderdict['stride'] = [1,2,1,1] 
# encoderdict['padding'] = [1,1,1,2] 
# encoderdict['dilation'] = [1,1,1,1] 
# encoderdict['latent'] = 128 
# ae = ae(**encoderdict)


# x=torch.rand((200,1,64,64))
# ae(x)
# '''
# from utils import *

# ae = load_class("AutoEnc")
# encoderdict = dict()
# encoderdict['num_layers'] = 2
# encoderdict['kernel_size'] = [3,2] 
# encoderdict['stride'] = [1,2] 
# encoderdict['padding'] = [1,1] 
# encoderdict['dilation'] = [1,1] 
# encoderdict['latent'] = 128 
# ae = ae(**encoderdict)


# x=torch.rand((200,1,64,64))
# ae(x)
