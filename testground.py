from data_loader import loadData
from utils import *
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
'''
loss_values = torch.tensor([], dtype=torch.float32, device=device)
loss_values = torch.cat([loss_values, torch.tensor([123],device=device)], dim=0)
loss_values = torch.cat([loss_values, torch.tensor([122133],device=device)], dim=0)

print(loss_values.shape)
print(loss_values)'''

#train,val,test,loader = loadData("wmh_sp")
#train,val,test,loader = loadData("wmh_usp")
#print(train[0:4])
'''def true_alpha(data):
    f = 0
    tot = len(data)
    for img,mask in data:
        if any(e !=0 for e in np.unique(mask)):
            f+=1
            plt.imshow(img[0])
            plt.show()
            break
    print("Result:",f/tot)
test_ds = loader(val)
true_alpha(test_ds)'''
'''train_ds = loader(train)
#train_loader = DataLoader(train_ds, batch_size=batch_size,shuffle=True) 
plt.imshow(train_ds[2][0][0])
plt.show()'''

'''
ae = load_class("AutoEnc")
encoderdict = dict()
encoderdict['num_layers'] = 4
encoderdict['kernel_size'] = [3,2,4,2] 
encoderdict['stride'] = [1,2,1,1] 
encoderdict['padding'] = [1,1,1,2] 
encoderdict['dilation'] = [1,1,1,1] 
encoderdict['latent'] = 128 
ae = ae(**encoderdict)


x=torch.rand((200,1,64,64))
ae(x)
'''
# from utils import *

# img_size = 240

# ae = load_class("AutoEnc")
# encoderdict = dict()
# encoderdict['num_layers'] = 5
# encoderdict['kernel_size'] = [5,5,5,5,5] 
# encoderdict['stride'] = [1,2,1,1,1] 
# encoderdict['padding'] = [1,1,1,2,1] 
# encoderdict['dilation'] = [1,1,1,1,1] 
# encoderdict['latent'] = 128 
# encoderdict['image_shape']=img_size
# ae = ae(**encoderdict)


# x=torch.rand((4,1,img_size,img_size))
# ae(x)


# a = torch.randn((200,1,64,64))
# b =torch.quantile(a,.9,dim=0)
# print(a.shape)
# print(b.shape)
# print(torch.unique((a > b).float()))