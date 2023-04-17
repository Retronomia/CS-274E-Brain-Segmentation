import torch
import torch.nn as nn


def load_loss(loss_name):
    '''Returns loss func from this file'''
    return globals()[loss_name]


def L1_Loss():
    return nn.L1Loss()  # (prediction,true)


def L2_Loss(prediction, true):
    return nn.MSELoss()  # (prediction,true)


def Bernoulli_Loss(prediction, true):
    return nn.BCELoss()  # (prediction,true)


def KL_Loss():
    def kl_loss(pred, real, mu, sigma):
        #rec = torch.sum(nn.BCELoss(reduction='none')(pred,real),axis=[1, 2, 3])
        phi = 1
        l1_loss = nn.L1Loss(reduction='none')(pred, real)
        rec = phi * torch.sum(l1_loss, axis=[1, 2, 3])
        kl = .5 * torch.sum(torch.square(mu)+torch.square(sigma) -
                            torch.log(torch.square(sigma))-1, axis=1)
        # print("rec",torch.mean(rec).item(),"kl",torch.mean(kl).item())
        return torch.mean(rec+kl)
    return kl_loss


def KL_SP_Loss():
    def anomaly_score(pred, real):
        return torch.abs(pred-real)

    def maskloss(pred, real, mask, reduction='mean'):
        def formula(pred, real, mask):
            anom = anomaly_score(pred, real)
            #anom_mod = anom.clone()
            #anom_mod[anom_mod == 0] = 1e-6
            # return (1 - mask)*anom + mask/anom_mod
            return mask * (-1 * anom + 1) + (1-mask) * anom
        res = formula(pred, real, mask)
        if reduction == 'mean':
            #mask = torch.logical_or(mask == 0, mask == 1)
            #res = res[mask]
            return torch.mean(res)
        else:
            return res

    def kl_loss(pred, real, mask, mu, sigma):
        rec = torch.sum(maskloss(pred, real, mask,
                        reduction='none'), axis=[1, 2, 3])
        #phi = 10
        #l1_loss = nn.L1Loss(reduction='none')(pred,real)
        #rec = phi * torch.sum(l1_loss, axis=[1, 2, 3])
        kl = .5 * torch.sum(torch.square(mu)+torch.square(sigma) -
                            torch.log(torch.square(sigma))-1, axis=1)
        # print("rec",torch.mean(rec).item(),"kl",torch.mean(kl).item())
        return torch.mean(rec+kl)
    return kl_loss


def Custom_Loss():
    def anomaly_score(pred, real):
        return torch.abs(pred-real)

    def maskloss(pred, real, mask, reduction='mean'):
        def formula(pred, real, mask):
            anom = anomaly_score(pred, real)
            #anom_mod = anom.clone()
            # anom_mod[anom_mod==0]=1e-6
            #skew_anom = 1
            #skew_norm = 1
            # skew_norm*(1 - mask)*anom + skew_anom*mask/anom_mod #original loss
            # return (skew_norm*(1 - mask)*torch.exp((2*anom))) + (skew_anom*mask/(anom+1e-6))-1 #loss I tried that one time

            # symmetrical loss
            #((skew_norm*-1*(1 - mask))/(anom-1+1e-6)) + (skew_anom*mask/(anom+1e-6))-(skew_norm*(1-mask))-(skew_anom*mask)
            #mask * (-1 * anom + 1) + (1-mask) * anom
            # ((skew_norm*-1*(1 - mask))/(anom-1+1e-6)) + (skew_anom*mask/(anom+1e-6))-(skew_norm*(1-mask))-(skew_anom*mask)
            # mask * (-1 * anom + 1) + (1-mask) * anom
            return mask * (-1 * anom + 1) + (1-mask) * anom
        res = formula(pred, real, mask)
        if reduction == 'mean':
            #mask = torch.logical_or(mask ==0,mask==1)
            # res=res[mask]
            return torch.mean(res)
        else:
            return res

    return maskloss


def CAE_Loss():
    def caeloss(pred, real, z, z_rec, reduction='mean'):
        def formula(pred, real, z, z_rec, reduction='mean'):
            rho = 1
            l2 = torch.mean(torch.nn.MSELoss(reduction='none')
                            (pred, real), axis=[1, 2, 3])
            rec_z = torch.mean(torch.nn.MSELoss(
                reduction='none')(z, z_rec), axis=1)
            return l2+rho*rec_z
        res = formula(pred, real, z, z_rec)
        if reduction == 'mean':
            return torch.mean(res)
        else:
            return res
    return caeloss


def CAE_SP_Loss():
    def anomaly_score(pred, real):
        return torch.abs(pred-real)

    def maskloss(pred, real, mask, reduction='mean'):
        def formula(pred, real, mask):
            anom = anomaly_score(pred, real)
            #anom_mod = anom.clone()
            #anom_mod[anom_mod == 0] = 1e-6
            # return (1 - mask)*anom + mask/anom_mod
            return mask * (-1 * anom + 1) + (1-mask) * anom
        res = formula(pred, real, mask)
        if reduction == 'mean':
            #mask = torch.where(mask == 2)
            #res[mask] = 0
            return torch.mean(res)
        else:
            #mask = torch.where(mask == 2)
            #res[mask] = 0
            return res

    def caeloss(pred, real, mask, z, z_rec, reduction='mean'):
        def formula(pred, real, mask, z, z_rec, reduction='mean'):
            rho = 1

            # torch.mean(torch.nn.MSELoss(reduction='none')(pred,real),axis=[1,2,3])
            l2 = torch.sum(maskloss(pred, real, mask,
                           reduction='none'), axis=[1, 2, 3])

            rec_z = torch.mean(torch.nn.MSELoss(
                reduction='none')(z, z_rec), axis=1)
            return l2+rho*rec_z
        res = formula(pred, real, mask, z, z_rec)
        if reduction == 'mean':
            #mask = torch.logical_or(mask ==0,mask==1)
            # res=res[mask]
            return torch.mean(res)
        else:
            return res
    return caeloss


def VQ_Model_Loss():
    return L1_Loss()


def VQ_Model_SP_Loss():
    return Custom_Loss()
