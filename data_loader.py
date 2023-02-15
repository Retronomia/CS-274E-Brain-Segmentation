from utils import *
import tempfile
from monai.apps import download_and_extract
import h5py
import matplotlib.pyplot as plt
import PIL
from monai.transforms.utils import rescale_array
import nibabel as nib
from PIL import Image as im
import seaborn as sns
from scipy.ndimage import rotate, affine_transform


def downloadMedNIST(root_dir):
    '''Downloads MedNIST dataset. Code from MedNIST tutorial.'''
    resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
    md5 = "0bc7306e7427e00ad1c5526a6677552d"

    compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")
    data_dir = os.path.join(root_dir, "MedNIST")
    if not os.path.exists(data_dir):
        download_and_extract(resource, compressed_file, root_dir, md5)

def loadDSprites():
    '''Load sprite images from dataset.'''
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
    '''Grab dict of image file paths'''
    downloadMedNIST(root_dir)
    data_dir = os.path.join(root_dir, "MedNIST")

    class_names = sorted(x for x in os.listdir(data_dir)
                        if os.path.isdir(os.path.join(data_dir, x)))

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

    #plt.subplots(2,3, figsize=(6, 6))
    #for i, class_name in enumerate(class_names):
    #    sel_rand = np.random.randint(num_each[class_name])
    #    im = PIL.Image.open(image_files[class_name][sel_rand])
    #    arr = np.array(im)
    #    plt.subplot(2,3, i + 1)
    #    plt.xlabel(f"{class_name}:{sel_rand}")
    #    plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
    #plt.tight_layout()
    #plt.show()

    print(f"Total image count: {data_len}")
    print(f"Image dimensions: {image_width} x {image_height}")
    print(f"Label names: {class_names}")
    print(f"Label counts: {num_each}")

    return image_files

def loadMNIST(split_tuple: tuple,image_files: dict,mask_imgs: list,sprite_chances: tuple,maskmax: bool):
    '''Split HeadCT data into test,train,validate'''
    resetSeeds()
    train_sp,val_sp,test_sp = sprite_chances
    class_name = 'HeadCT'

    train_frac,val_frac,test_frac = split_tuple
    if train_frac + val_frac + test_frac >1:
        raise ValueError(f"Proportions of class {class_name} ({train_frac},{val_frac},{test_frac}) sum to greater than 1.")
    
    filteredfiles = []
    for fileimg in image_files[class_name]:
        tempimg = np.array(PIL.Image.open(fileimg))
        tempimg = rescale_array(tempimg,0,1)
        if np.sum(tempimg)/(64*64) >= 5/(64*64):
            filteredfiles.append(fileimg)
        #else:
        #    print("ERROR!",np.sum(tempimg)/(64*64))


    dat_len = len(filteredfiles)
    split_indices = np.arange(dat_len)
    np.random.shuffle(split_indices)

    train_indices = split_indices[0:int(dat_len*train_frac)]
    val_indices = split_indices[int(dat_len*train_frac):int(dat_len*train_frac)+int(dat_len*val_frac)]
    test_indices = split_indices[int(dat_len*train_frac)+int(dat_len*val_frac):int(dat_len*train_frac)+int(dat_len*val_frac)+int(dat_len*test_frac)]

    def genArray(type_indices,p_sprite,filteredfiles,mask_imgs):
        made_arr = []
        for i in type_indices[0:int(len(type_indices)*p_sprite)]: #add sprite
            tempmask = mask_imgs[i]*1
            #
            #tempmask = resize(tempmask,(128,128),anti_aliasing=False)
            #
            tempmask = np.expand_dims(tempmask,axis=0)
            tempimg = np.array(PIL.Image.open(filteredfiles[i]))

            tempimg = rescale_array(tempimg,0,1)

            if np.sum(tempimg)/(64*64) <= 10/(64*64):
                print("ERROR!",np.sum(tempimg)/(64*64))
                plt.imshow(tempimg, cmap="gray", vmin=0, vmax=1)
                plt.show()

            #
            #tempimg = resize(tempimg,(128,128),anti_aliasing=False)
            #
            tempimg = np.expand_dims(tempimg,axis=0)
            if maskmax:
                tempimg[tempmask.astype(bool)] = 1
            else:
                tempimg[tempmask.astype(bool)] = random.uniform(0,1)
            made_arr.append((tempimg,tempmask))
        for i in type_indices[int(len(type_indices)*p_sprite):]: #no sprite
            tempimg = np.array(PIL.Image.open(filteredfiles[i]))
            tempimg = rescale_array(tempimg,0,1)
            if np.sum(tempimg)/(64*64) <= 10/(64*64):
                print("ERROR!",np.sum(tempimg)/(64*64))
                plt.imshow(tempimg, cmap="gray", vmin=0, vmax=1)
                plt.show()
            #
            #tempimg = resize(tempimg,(128,128),anti_aliasing=False)
            #
            tempimg = np.expand_dims(tempimg, axis=0)
            makenorm = tempimg * 0
            made_arr.append((tempimg,makenorm))

        made_arr = np.array(made_arr)
        return made_arr

    add_train = genArray(train_indices,train_sp,filteredfiles,mask_imgs)
    add_val = genArray(val_indices,val_sp,filteredfiles,mask_imgs)
    add_test = genArray(test_indices,test_sp,filteredfiles,mask_imgs)

    print(f"For {class_name} added {len(add_train)} train, {len(add_val)} val, {len(add_test)} test.")
    print(f"Train has {int(len(train_indices)*train_sp)}/{len(train_indices)} sprited images, Val has {int(len(val_indices)*val_sp)}/{len(val_indices)} sprited images, Test has {int(len(test_indices)*test_sp)}/{len(test_indices)} sprited images.")
    
    return add_train,add_val,add_test


def loadWMH(split_tuple: tuple,init_path: str,supervised: True):
    resetSeeds()
    
    def getPatients(init_path):
        for dir in os.listdir(init_path):
            img_source = dir
            #print(f"Img Source ({img_source}):")
            for patient_num in os.listdir(init_path / img_source):
                if patient_num.isnumeric() == False:
                    scanner_path = patient_num
                    for patient_num in os.listdir(init_path / img_source / scanner_path):
                        temp_path = init_path / img_source / scanner_path / patient_num
                        #print(init_path / img_source / scanner_path / patient_num)
                        yield temp_path
                else:
                    temp_path = init_path / img_source / patient_num
                    #print(init_path / img_source / patient_num)
                    yield temp_path

    #t1_name = "pre/T1.nii.gz"
    flair_name = "orig/FLAIR_stripped_registered.nii.gz"
    mask_name = "orig/anomaly_segmentation.nii.gz"

    print("Loading training data...")
    train_files = []
    for file in getPatients(init_path / "train"):
        #t1_path = file / t1_name
        flair_path = file / flair_name
        mask_path = file / mask_name
        train_files.append((flair_path,mask_path)) #(t1_path,flair_path,mask_path)

    print("Loading test data...")
    test_files = []
    for file in getPatients(init_path / "test"):
        #t1_path = file / t1_name
        flair_path = file / flair_name
        mask_path = file / mask_name
        test_files.append((flair_path,mask_path)) #(t1_path,flair_path,mask_path)

    train_frac,val_frac = split_tuple
    if train_frac + val_frac >1:
        raise ValueError(f"Proportions ({train_frac},{val_frac}) sum to greater than 1.")
    
    def getimg(fileimg,is_sup,apply_trans=False):
        #t1_imgs = []
        flair_imgs = []
        mask_imgs = []

        #t1 = nib.load(fileimg[0]).get_fdata()
        #assert np.min(t1)==0
        #####t1 = t1 + np.abs(np.min(t1))
        #t1 = t1 / np.max(t1)

        try:
            flair = nib.load(fileimg[0]).get_fdata() #nib.load(fileimg[1]).get_fdata()

            assert np.min(flair)==0
            #flair = flair + np.abs(np.min(flair))
            flair = flair / np.max(flair)

            mask =  nib.load(fileimg[1]).get_fdata() #nib.load(fileimg[2]).get_fdata()
            #print(np.unique(mask))
            #print(np.max(mask),np.min(mask))
            #sns.heatmap(mask[:,:,mask.shape[2]//2+0])
            #plt.show()
            #mask = mask + np.abs(np.min(mask))
            #mask = mask / np.max(mask)
            assert np.min(mask)==0

            tempdim = flair.shape[2]//2
        except:
            print(f"{fileimg[0]} broken or {fileimg[1]} broken")
            return flair_imgs,mask_imgs
        
        
        img_dim = 200
        n_resample = 1
        for diff in [-1,0,1]:
            for _n in range(n_resample):
                rotation = random.uniform(-15,15)
                shearing = random.uniform(-1.,.1)
                x_scale = random.uniform(.9,1.1)
                y_scale = random.uniform(.9,1.1)
                transform = [[x_scale, 0, 0],[shearing, y_scale, 0],[0, 0, 1]]
                #print(rotation,shearing,x_scale,y_scale)

                def manip_img(temp_im):
                    #print("New Img")
                    #print(temp_im.shape)

                    #temp_im = rotate(temp_im,rotation,reshape=False)
                    '''height, width =temp_im.shape
                    #print(np.min(temp_im),np.max(temp_im))

                    #print(np.min(temp_im))
                    #sns.heatmap(temp_im)
                    #plt.show()
                    temp_im = affine_transform(temp_im,transform,offset=(0, -height//2, 0),output_shape=(height, width+height//2),prefilter=False)
                    #sns.heatmap(temp_im)
                    #plt.show()
                    #print(np.min(temp_im),np.max(temp_im))
                    #print(temp_im.shape)

                    temp_im = im.fromarray(temp_im)
                    #print(np.min(temp_im))
                    temp_im = temp_im.rotate(rotation,expand=True)

                    #print(np.min(temp_im))
                    #print(temp_im.size)
                    curr_height,curr_width = temp_im.size
                    new_width = img_dim if curr_width >= img_dim else curr_width
                    new_height = img_dim if curr_height >= img_dim else curr_height
                    left = (curr_width - new_width)/2
                    top = (curr_height - new_height)/2
                    right = (curr_width + new_width)/2
                    bottom = (curr_height + new_height)/2

                    #sns.heatmap(temp_im)
                    #plt.show()
                    temp_im = temp_im.crop((left, top, right, bottom))
                    temp_im = np.array(temp_im)

                    #sns.heatmap(temp_im)
                    #plt.show()
                    #print(temp_im.shape)

                    temp_im = np.pad(temp_im, [(0,img_dim-new_height), (0,img_dim-new_width)], mode='constant')

                    #print(temp_im.shape)
                    assert temp_im.shape==(img_dim,img_dim)'''
                    return temp_im

                #print("T1")
                #t1
                #t1_im = t1[:,:,tempdim+diff]
                #t1_im = manip_img(t1_im)
                #t1_im = np.expand_dims(t1_im, axis=0)

                #mean = np.mean(t1)
                #std = np.std(t1)
                #mx = np.max(t1)
                #if mx != 0:
                #    t1_im = t1_im/mx
                #t1_im = (t1_im-mean)/std

                #print(np.min(t1_im))

                #flair
                flair_im = flair[:,:,tempdim+diff]
                flair_im = manip_img(flair_im)
                flair_im = np.expand_dims(flair_im, axis=0)

                #mean = np.mean(flair)
                #std = np.std(flair)
                #mx = np.max(flair)
                #if mx != 0:
                #    flair_im = flair_im/mx
                #flair_im = (flair_im-mean)/std

                #mask
                mask_im = mask[:,:,tempdim+diff]
                mask_im = manip_img(mask_im)
                #mx = np.max(mask_im)
                #if mx != 0:
                #    mask_im = mask_im/mx
                mask_im[mask_im!=1]= 0
                mask_im = np.expand_dims(mask_im, axis=0)

                if is_sup==True or np.max(mask_im) != 0:
                    #t1_imgs.append(t1_im)
                    flair_imgs.append(flair_im)
                    mask_imgs.append(mask_im)

        return flair_imgs,mask_imgs #t1_imgs,flair_imgs,mask_imgs

    def load_files(files,is_sup):
        filtered_files = []
        for fileimg in files:
            flair_imgs,mask_imgs = getimg(fileimg,is_sup) #t1_imgs,flair_imgs,mask_imgs
            for i in range(len(flair_imgs)):
                #merge_imgs = np.stack((t1_imgs[i],flair_imgs[i]), axis=0)
                #merge_imgs = np.moveaxis(merge_imgs,0,-1)
                filtered_files.append((flair_imgs[i],mask_imgs[i])) #(t1_imgs[i],flair_imgs[i],mask_imgs[i])
        return filtered_files
    
    dat_len_train = len(train_files)
    split_indices = np.arange(dat_len_train)
    np.random.shuffle(split_indices)
    train_indices = split_indices[0:int(dat_len_train*train_frac)]
    val_indices = split_indices[int(dat_len_train*train_frac):int(dat_len_train*train_frac)+int(dat_len_train*val_frac)]

    val_files = [train_files[i] for i in val_indices]
    train_files = [train_files[i] for i in train_indices]

    print(f"Found {len(test_files)} test patients, {len(val_files)} val patients, {len(train_files)} train patients.")

    filtered_train_files = load_files(train_files,supervised)
    filtered_val_files = load_files(val_files,True)
    filtered_test_files = load_files(test_files,True)
  
    print(f"Added {len(filtered_train_files)} train, {len(filtered_val_files)} val, {len(filtered_test_files)} test.")

    return filtered_train_files,filtered_val_files,filtered_test_files


class MedNISTDataset(torch.utils.data.Dataset):
    '''Used to load data in DataLoader'''
    def __init__(self, image_files):
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.image_files[index][0],self.image_files[index][1]

class WMHDataset(torch.utils.data.Dataset):
    '''Used to load data in DataLoader'''
    def __init__(self, image_files):
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        #input image, image mask
        #return np.concatenate((self.image_files[index][0],self.image_files[index][1]),axis=0),self.image_files[index][2]
        return self.image_files[index][0],self.image_files[index][1] #self.image_files[index][1],self.image_files[index][2]


def loadData(exp_name):
    folder= Path("data_folder")
    if exp_name=="mnist_usp":
        #directory = os.environ.get("MONAI_DATA_DIRECTORY")
        #root_dir = tempfile.mkdtemp() if directory is None else directory
        root_dir = Path("./monaidata")
        ensure_folder_exists(root_dir)

        train_file_name = "mnist_usp_train.npy"
        val_file_name = "mnist_usp_val.npy"
        test_file_name = "mnist_usp_test.npy"
        dat_split = (.6,.2,.2)
        sprite_chances= (0,.1,.1)
        maskmax=True
        loader = MedNISTDataset
    elif exp_name=="mnist_sp":
        directory = os.environ.get("MONAI_DATA_DIRECTORY")
        root_dir = tempfile.mkdtemp() if directory is None else directory
        print(f"Created temp directory {root_dir}.")

        train_file_name = "mnist_sp_train.npy"
        val_file_name = "mnist_sp_val.npy"
        test_file_name = "mnist_sp_test.npy"
        dat_split = (.6,.2,.2)
        sprite_chances= (.1,.1,.1)
        maskmax=True
        loader = MedNISTDataset
    elif exp_name=="mnist_semi":
        directory = os.environ.get("MONAI_DATA_DIRECTORY")
        root_dir = tempfile.mkdtemp() if directory is None else directory
        print(f"Created temp directory {root_dir}.")
        train_file_name = ["mnist_sp_train.npy","mnist_usp_train.npy"]
        val_file_name = ["mnist_sp_val.npy","mnist_usp_val.npy"]
        test_file_name = ["mnist_sp_test.npy","mnist_usp_test.npy"]
        dat_split = (.6,.2,.2)
        sprite_chances= [(.1,.1,.1),(0,.1,.1)]
        maskmax=True
        loader = MedNISTDataset
    elif exp_name=="wmh_usp":
        root_dir= Path(r"E:\SCHOOL\FINISHED\CS274E\CS-274E-Brain-Segmentation\WMH-dataset")
        train_file_name = "wmh_usp_train.npy"
        val_file_name ="wmh_usp_val.npy"
        test_file_name ="wmh_usp_test.npy"
        dat_split = (.8,.2)
        is_supervised=False
        maskmax=True
        loader = WMHDataset
    elif exp_name=="wmh_sp":
        root_dir= Path(r"E:\SCHOOL\FINISHED\CS274E\CS-274E-Brain-Segmentation\WMH-dataset")
        train_file_name =  "wmh_sp_train.npy"
        val_file_name =  "wmh_sp_val.npy"
        test_file_name = "wmh_sp_test.npy"
        dat_split = (.8,.2)
        is_supervised=True
        loader = WMHDataset
    elif exp_name=="wmh_semi":
        root_dir= Path(r"E:\SCHOOL\FINISHED\CS274E\CS-274E-Brain-Segmentation\WMH-dataset")
        train_file_name =  ["wmh_sp_train.npy","wmh_usp_train.npy"]
        val_file_name =  ["wmh_sp_val.npy","wmh_usp_val.npy"]
        test_file_name = ["wmh_sp_test.npy","wmh_usp_test.npy"]
        dat_split = (.8,.2)
        is_supervised= [True,False]
        loader = WMHDataset
    else:
        raise ValueError(f"exp_name {exp_name} invalid.")
    
    if type(train_file_name) is str:
        if all_files_exist([folder/train_file_name,folder/val_file_name,folder/test_file_name]):
            with open(folder / train_file_name, 'rb') as f:
                train_x = np.load(f)
                with open(folder / val_file_name, 'rb') as f:
                    val_x = np.load(f)
                with open(folder / test_file_name, 'rb') as f:
                    test_x = np.load(f)
                print(f"Loaded files from {folder}.")
        else:
            ensure_folder_exists(folder)
            if "mnist" in exp_name:
                image_files = loadMedNISTData(root_dir)
                mask_imgs = loadDSprites()
                train_x,val_x,test_x  = loadMNIST(dat_split,image_files,mask_imgs,sprite_chances,maskmax)
            elif "wmh" in exp_name:
                train_x,val_x,test_x  = loadWMH(dat_split,root_dir,is_supervised)
            np.save(os.path.join(folder,train_file_name),train_x)
            np.save(os.path.join(folder,val_file_name),val_x)
            np.save(os.path.join(folder,test_file_name),test_x)
        print(f"Loaded {len(train_x)} train, {len(val_x)} val, {len(test_x)} test.")
        return train_x,val_x,test_x,loader
    else:
        full_train_x = []
        full_val_x = []
        full_test_x = []
        for i in range(len(train_file_name)):
            if all_files_exist([folder/train_file_name[i],folder/val_file_name[i],folder/test_file_name[i]]):
                with open(folder / train_file_name[i], 'rb') as f:
                    train_x = np.load(f)
                    with open(folder / val_file_name[i], 'rb') as f:
                        val_x = np.load(f)
                    with open(folder / test_file_name[i], 'rb') as f:
                        test_x = np.load(f)
                    print(f"Loaded files from {folder}.")
            else:
                ensure_folder_exists(folder)
                if "mnist" in exp_name:
                    image_files = loadMedNISTData(root_dir)
                    mask_imgs = loadDSprites()
                    train_x,val_x,test_x  = loadMNIST(dat_split,image_files,mask_imgs,sprite_chances[i],maskmax)
                elif "wmh" in exp_name:
                    train_x,val_x,test_x  = loadWMH(dat_split,root_dir,is_supervised[i])
                np.save(os.path.join(folder,train_file_name[i]),train_x)
                np.save(os.path.join(folder,val_file_name[i]),val_x)
                np.save(os.path.join(folder,test_file_name[i]),test_x)
            print(f"Loaded {len(train_x)} train, {len(val_x)} val, {len(test_x)} test.")
            full_train_x.append(train_x)
            full_val_x.append(val_x)
            full_test_x.append(test_x)
        print(f"Loaded {len(full_train_x)} train, {len(full_val_x)} val, {len(full_test_x)} test total.")
        full_train_x=tuple(full_train_x)
        full_val_x=tuple(full_val_x)
        full_test_x=tuple(full_test_x)
        return full_train_x,full_val_x,full_test_x,loader