import os
import PIL
import tempfile
import random
from monai.utils import set_determinism
from monai.apps import download_and_extract
import numpy as np
import torch 
import matplotlib.pyplot as plt
import h5py
from monai.transforms.utils import rescale_array


def resetSeeds():
    set_determinism(seed=0)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

def downloadMedNIST(root_dir):
    resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
    md5 = "0bc7306e7427e00ad1c5526a6677552d"

    compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")
    data_dir = os.path.join(root_dir, "MedNIST")
    if not os.path.exists(data_dir):
        download_and_extract(resource, compressed_file, root_dir, md5)

def loadDSprites():
    resetSeeds()
    filename = "dsprites-dataset-master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5"

    with h5py.File(filename, "r") as f:
        print("Keys: %s" % f.keys())

        mask_imgs = f['imgs'][()]
        latent_imgs = f['latents']

    num_key_frames = len(mask_imgs)

    np.random.shuffle(mask_imgs)
    print(f"Number of frames: {num_key_frames}")
    return mask_imgs

def loadMedNISTData(root_dir):
    downloadMedNIST(root_dir)
    data_dir = os.path.join(root_dir, "MedNIST")

    class_names = sorted(x for x in os.listdir(data_dir)
                        if os.path.isdir(os.path.join(data_dir, x)))
    num_class = len(class_names)

    image_files = dict()
    num_each = dict()
    for class_name in class_names:
        image_files[class_name] = [
            os.path.join(data_dir, class_name, x)
            for x in os.listdir(os.path.join(data_dir, class_name))
        ]
        num_each[class_name] = len(image_files[class_name])

    data_len = sum([v for v in num_each.values()])
    image_width, image_height = PIL.Image.open(image_files[class_names[0]][0]).size

    plt.subplots(2,3, figsize=(6, 6))
    for i, class_name in enumerate(class_names):
        sel_rand = np.random.randint(num_each[class_name])
        im = PIL.Image.open(image_files[class_name][sel_rand])
        arr = np.array(im)
        plt.subplot(2,3, i + 1)
        plt.xlabel(f"{class_name}:{sel_rand}")
        plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()

    print(f"Total image count: {data_len}")
    print(f"Image dimensions: {image_width} x {image_height}")
    print(f"Label names: {class_names}")
    print(f"Label counts: {num_each}")

    return image_files

def splitData(split_tuple: tuple,image_files: dict,mask_imgs: list):
    resetSeeds()
    '''
    Function to allow data to be pulled from multiple classes with specific proportions.
    Input:
    split_dict: dict- dictionary of proportions of data to be pulled, with keys of (train,val,test) where each value is 0<=p<=1 and sums to 1 max
    Output:
    returns train_x,val_x,test_x, which are lists of whatever data was passed into image_files with elements of format (data,int) where int is
    either 0 (do not add sprite) or 1 (do add sprite)
    '''
    class_name = 'HeadCT'

    train_frac,val_frac,test_frac = split_tuple
    if train_frac + val_frac + test_frac >1:
        raise ValueError(f"Proportions of class {class_name} ({train_frac},{val_frac},{test_frac}) sum to greater than 1.")
    
    dat_len = len(image_files[class_name])
    split_indices = np.arange(dat_len)
    np.random.shuffle(split_indices)

    train_indices = split_indices[0:int(dat_len*train_frac)]
    val_indices = split_indices[int(dat_len*train_frac):int(dat_len*train_frac)+int(dat_len*val_frac)]
    test_indices = split_indices[int(dat_len*train_frac)+int(dat_len*val_frac):int(dat_len*train_frac)+int(dat_len*val_frac)+int(dat_len*test_frac)]

    add_train = []
    add_val = []
    add_test = []
    for i in train_indices:
        rawimg = np.array(PIL.Image.open(image_files[class_name][i]))
        rawimg = rescale_array(rawimg,0,1)
        rawimg = np.expand_dims(rawimg, axis=0)
        makenorm = rawimg * 0
        add_train.append((rawimg,makenorm))

    percent_sprite = .3 #30% chance to have a sprite
    for i in val_indices[0:int(len(val_indices)*percent_sprite)]: #add sprite
        tempmask = mask_imgs[i]*1
        tempmask = np.expand_dims(tempmask, axis=0)
        tempimg = np.array(PIL.Image.open(image_files[class_name][i]))
        tempimg = rescale_array(tempimg,0,1)
        tempimg = np.expand_dims(tempimg, axis=0)
        tempimg[tempmask.astype(bool)]= 1   #random.uniform(0,1)
        add_val.append((tempimg,tempmask))
    for i in val_indices[int(len(val_indices)*percent_sprite):]: #no sprite
        rawimg = np.array(PIL.Image.open(image_files[class_name][i]))
        rawimg = rescale_array(rawimg,0,1)
        rawimg = np.expand_dims(rawimg, axis=0)
        makenorm = rawimg * 0
        add_val.append((rawimg,makenorm))

    for i in test_indices[0:int(len(test_indices)*percent_sprite)]: #add sprite
        tempmask = mask_imgs[i]*1
        tempmask = np.expand_dims(tempmask, axis=0)
        tempimg = np.array(PIL.Image.open(image_files[class_name][i]))
        tempimg = rescale_array(tempimg,0,1)
        tempimg = np.expand_dims(tempimg, axis=0)
        tempimg[tempmask.astype(bool)]= 1   #random.uniform(0,1)
        add_test.append((tempimg,tempmask))
    for i in test_indices[int(len(test_indices)*percent_sprite):]: #no sprite
        rawimg = np.array(PIL.Image.open(image_files[class_name][i]))
        rawimg = rescale_array(rawimg,0,1)
        rawimg = np.expand_dims(rawimg, axis=0)
        makenorm = rawimg * 0
        add_test.append((rawimg,makenorm))

    print(f"For {class_name} added {len(add_train)} train, {len(add_val)} val, {len(add_test)} test.")
    print(f"Val has {int(len(val_indices)*percent_sprite)}/{len(val_indices)} sprited images, Test has {int(len(test_indices)*percent_sprite)}/{len(test_indices)} sprited images, ")
    return np.array(add_train),np.array(add_val),np.array(add_test)