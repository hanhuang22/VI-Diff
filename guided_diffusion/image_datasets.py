import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from scipy import ndimage
import os
import matplotlib.pyplot as plt
import random

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        # resize image rather than crop
        pil_image = pil_image.resize((self.resolution, self.resolution*2))
        arr = np.array(pil_image)

        # if self.random_crop:
        #     arr = random_crop_arr(pil_image, self.resolution)
        # else:
        #     arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def load_SYSU_data(
    data_dir,
    batch_size,
    image_size,
    deterministic=False,
    random_crop=True,
    contour=True,
    dataset='sysu',
    hist_match=False,
    return_path=False,
    class_cond=False,
    balance_sample=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    if dataset.lower() == 'sysu':
        dataset_class = SYSUFolder
        modality = 'all'
    elif dataset.lower() in  ['sysu_rgb', 'sysu_ir']:
        dataset_class = SYSUModFolder
        if dataset.lower().find('rgb'):
            modality = 'rgb'
        elif dataset.lower().find('ir'):
            modality = 'ir'
        else:
            modality = None
    else:
        raise NotImplementedError

    dataset = dataset_class(
        data_dir,
        image_size,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        contour=contour,
        hist_match=hist_match,
        return_path=return_path,
        modality=modality,
        class_cond=class_cond,
        balance_sample=balance_sample,
    )
    if deterministic or balance_sample:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

def load_RegDB_data(
    data_dir,
    batch_size,
    image_size,
    deterministic=False,
    random_crop=True,
    contour=True,
    dataset='regdb',
    hist_match=False,
    return_path=False,
    class_cond=False,
    balance_sample=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    if dataset.lower() == 'regdb':
        dataset_class = RegDBFolder
        modality = 'all'
    elif dataset.lower() in  ['rgb', 'ir']:
        dataset_class = RegDBModFolder
        if dataset.lower().find('rgb'):
            modality = 'rgb'
        elif dataset.lower().find('ir'):
            modality = 'ir'
        else:
            modality = None
    else:
        raise NotImplementedError

    dataset = dataset_class(
        data_dir,
        image_size,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        contour=contour,
        hist_match=hist_match,
        return_path=return_path,
        modality=modality,
        class_cond=class_cond,
        balance_sample=balance_sample,
    )
    if deterministic or balance_sample:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


class SYSUBaseDataset(object):
    def __init__(
            self,
            image_size,
            random_crop,
            contour,
            hist_match,
    ):
        self.contour = contour
        if self.contour:
            img_norm = T.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
        else:
            img_norm = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        if random_crop:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize(((image_size+16)*2, image_size+8), antialias=True),
                T.RandomCrop((image_size*2, image_size)),
                img_norm,
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize((image_size*2, image_size), antialias=True),
                img_norm,
            ])
        self.H = self.high_pass_filter(image_size*2, image_size, 5)
        self.hist_match = hist_match
        if hist_match:
            self.cdf_all = np.load('hist_matching/cdf_all.npy')
    
    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def highpass(self, img_path):
        data = np.mean(Image.open(img_path), axis=2)
        lowpass = ndimage.gaussian_filter(data, 3)
        gauss_highpass = data - lowpass
        gauss_highpass = self.normalization(gauss_highpass) * 255

        img = gauss_highpass 
        img = Image.fromarray(img.astype('uint8'))
        return img

    def high_pass_filter(self, h, w, d0):
        H=np.empty(shape=[h,w],dtype=float)
        mid_x=int(w/2)
        mid_y=int(h/2)
        for y in range(0,h):
            for x in range(0,w):
                d=np.sqrt((x-mid_x)**2+(y-mid_y)**2)
                if d<=d0:
                    H[y,x]=0
                else:
                    H[y,x]=1
        return H

    def fft_highpass(self, data):
        """
        data: numpy array from pil image
        """
        data_fft2 = np.mean(data, axis=2)
        data_fft2 = np.fft.fft2(data_fft2.astype(float))
        data_fft2 = np.fft.fftshift(data_fft2)
        data_fft2 = data_fft2 * self.H
        img_highpass = np.fft.ifft2(np.fft.ifftshift(data_fft2)).real
        img_arr_min = img_highpass.min()
        img_arr_max = img_highpass.max()
        img_highpass = (img_highpass - img_arr_min) / (img_arr_max - img_arr_min) * 255
        img_highpass = img_highpass.astype(np.uint8)
        img_highpass = np.expand_dims(img_highpass, axis=2)
        return img_highpass

    def fft_highpass_new(self, image, cutoff_freq=2):
        gray_image = image.convert('L')
        # Convert the grayscale image to a NumPy array
        img_array = np.array(gray_image)
        # Perform the 2D FFT on the image array
        fft_image = np.fft.fft2(img_array)
        # Shift the zero-frequency component to the center of the spectrum
        shifted_fft = np.fft.fftshift(fft_image)
        # Create a high-pass filter mask
        rows, cols = img_array.shape
        mask = np.ones((rows, cols))
        center_row, center_col = rows // 2, cols // 2
        mask[center_row - cutoff_freq: center_row + cutoff_freq, center_col - cutoff_freq: center_col + cutoff_freq] = 0
        # Apply the mask to the shifted FFT image
        filtered_fft = shifted_fft * mask
        # Shift the zero-frequency component back to the corners
        shifted_filtered_fft = np.fft.ifftshift(filtered_fft)
        # Perform the inverse 2D FFT to obtain the filtered image
        filtered_image = np.fft.ifft2(shifted_filtered_fft)
        # Take the absolute value to get the intensity values
        filtered_image = np.abs(filtered_image)
        filtered_image = np.expand_dims(filtered_image, axis=2)
        return filtered_image

    def pre_process(self, img_path):
        # img_data =  np.array(Image.open(img_path).resize([128,256]))
        # img_highpass = self.fft_highpass(img_data)
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(img_highpass, cmap='gray')
        img = Image.open(img_path).resize([128,256])
        img_data = np.array(img)
        img_highpass = self.fft_highpass_new(img)
        if self.hist_match:
            img_highpass = self.hist_matching(img_highpass)
        # plt.subplot(122)
        # plt.imshow(img_highpass, cmap='gray')
        # plt.savefig('histmatch.png')
        temp_data = np.concatenate([img_data, img_highpass], axis=2).astype(np.uint8)
        temp_data = self.transform(temp_data)
        img = temp_data[:3]
        img_highpass = temp_data[3:]
        return img, img_highpass

    def hist(self, img_arr):
        img_arr = np.reshape(img_arr, (-1, ))
        hist_arr = np.zeros(256)
        for i in range(len(img_arr)):
            hist_arr[img_arr[i]] += 1
        return hist_arr

    def cdf(self, hist_arr):
        cdf_arr = np.cumsum(hist_arr)
        cdf_arr /= cdf_arr[-1]
        return cdf_arr

    def hist_matching(self, img_arr, cdf_t=None):
        if cdf_t == None:
            cdf_t = self.cdf_all
        cdf_s = self.cdf(self.hist(img_arr))
        b = np.interp(cdf_s, cdf_t, np.arange(256)) # find closest matches to b_t
        pix_repl = {i:b[i] for i in range(256)} # dictionary to replace the pixels
        mp = np.arange(0,256)
        for (k, v) in pix_repl.items():
            mp[k] = v
        s = img_arr.shape
        img_arr = np.reshape(mp[img_arr.ravel()], img_arr.shape)
        img_arr = np.reshape(img_arr, s)
        return img_arr

class SYSUFolder(SYSUBaseDataset, ImageFolder):
    def __init__(
            self, 
            data_dir, 
            image_size,
            shard=0,
            num_shards=1,
            return_path=False,
            random_crop=True,
            contour=True,
            hist_match=False,
            class_cond=False,
            balance_sample=False,
            **kwargs,
    ):
        ImageFolder.__init__(self, data_dir)
        SYSUBaseDataset.__init__(self, image_size, random_crop, contour, hist_match)
        self.local_imgs = self.imgs[shard:][::num_shards] # mpi multi-gpus
        self.class_cond = class_cond
        self.return_path = return_path

        self.balance_sample = balance_sample
        self.rgb_imgs = []
        self.ir_imgs = []
    
    def split_class(self):
        if len(self.rgb_imgs) == 0 or len(self.ir_imgs) == 0:
            self.rgb_imgs.clear()
            self.ir_imgs.clear()
            for i in range(len(self.local_imgs)):
                if self.local_imgs[i][1] == 0:
                    self.rgb_imgs.append(self.local_imgs[i][0])
                else:
                    self.ir_imgs.append(self.local_imgs[i][0])

    def __len__(self):
        return len(self.local_imgs)

    def __getitem__(self, index):
        if not self.balance_sample:
            path = self.local_imgs[index][0] 
            label = self.local_imgs[index][1]
        else:
            self.split_class()
            if index % 2 == 0:
                path = random.choice(self.rgb_imgs)
                self.rgb_imgs.remove(path)
            else:
                path = random.choice(self.ir_imgs)
                self.ir_imgs.remove(path)
            label = index % 2

        out_dict = {}
        if self.class_cond:
            out_dict["y"] = np.array(label, dtype=np.int64)

        if self.contour:
            img, img_highpass = self.pre_process(path)
            out_dict["contour"] = img_highpass
        else:
            img = self.loader(path)
            img = self.transform(img)
        
        if self.return_path:
            out_dict["path"] = path

        return img, out_dict


class SYSUModFolder(SYSUBaseDataset, Dataset):
    def __init__(
            self, 
            data_dir, 
            image_size,
            shard=0,
            num_shards=1,
            return_path=False,
            random_crop=False,
            contour=False,
            hist_match=False,
            modality=None,
            class_cond=False,
            **kwargs,
    ):
        SYSUBaseDataset.__init__(self, image_size, random_crop, contour, hist_match)

        self.ids = set()
        self.imgs = []
        for img_name in sorted(os.listdir(data_dir)):
            img_id = img_name.split('_')[1]
            self.imgs.append([os.path.join(data_dir, img_name), img_id])
            self.ids.add(img_id)
        self.ids = list(sorted(self.ids))
        self.local_imgs = self.imgs[shard:][::num_shards] # mpi multi-gpus
        self.modality = modality
        self.return_path = return_path
        self.class_cond = class_cond

    def __len__(self):
        return len(self.local_imgs)

    def __getitem__(self, index):
        img_path = self.local_imgs[index][0] 
        img_id = self.local_imgs[index][1]

        out_dict = {}
        out_dict['id'] = np.array(self.ids.index(img_id), dtype=np.int64)

        if self.class_cond:
            if self.modality == 'rgb':
                out_dict['y'] = np.array(0, dtype=np.int64)
            elif self.modality == 'ir':
                out_dict['y'] = np.array(1, dtype=np.int64)

        if self.contour:
            img, img_highpass = self.pre_process(img_path)
            out_dict["contour"] = img_highpass
        else:
            img = Image.open(img_path).resize([128,256])
            img = self.transform(img)
        
        if self.return_path:
            out_dict["path"] = img_path

        return img, out_dict



class FakeImageFolder(SYSUBaseDataset, Dataset):
    def __init__(
            self, 
            data_dir, 
            image_size,
            shard=0,
            num_shards=1,
            return_path=False,
            random_crop=True,
            contour=True,
    ):
        SYSUBaseDataset.__init__(self, image_size, random_crop, contour)

        self.imgs = []
        for img_name in sorted(os.listdir(data_dir)):
            if img_name.endswith('fake.jpg') and img_name.startswith(('cam1', 'cam2', 'cam4', 'cam5')):
                self.imgs.append([os.path.join(data_dir, img_name), 1])

        self.local_imgs = self.imgs[shard:][::num_shards] # mpi multi-gpus
        
        self.return_path = return_path
        

    def __len__(self):
        return len(self.local_imgs)

    def __getitem__(self, index):
        path = self.local_imgs[index][0] 
        label = self.local_imgs[index][1]

        out_dict = {}
        out_dict["y"] = np.array(label, dtype=np.int64)

        img, img_highpass = self.pre_process(path)
        out_dict["contour"] = img_highpass
        
        if self.return_path:
            out_dict["path"] = path

        return img, out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


class RegDBFolder(SYSUBaseDataset, ImageFolder):
    def __init__(
            self, 
            data_dir, 
            image_size,
            shard=0,
            num_shards=1,
            return_path=False,
            random_crop=True,
            contour=True,
            hist_match=False,
            class_cond=False,
            balance_sample=False,
            **kwargs,
    ):
        ImageFolder.__init__(self, data_dir)
        SYSUBaseDataset.__init__(self, image_size, random_crop, contour, hist_match)
        self.local_imgs = self.imgs[shard:][::num_shards] # mpi multi-gpus
        self.class_cond = class_cond
        self.return_path = return_path

        self.balance_sample = balance_sample
        self.rgb_imgs = []
        self.ir_imgs = []
    
    def split_class(self):
        if len(self.rgb_imgs) == 0 or len(self.ir_imgs) == 0:
            self.rgb_imgs.clear()
            self.ir_imgs.clear()
            for i in range(len(self.local_imgs)):
                if self.local_imgs[i][1] == 0:
                    self.rgb_imgs.append(self.local_imgs[i][0])
                else:
                    self.ir_imgs.append(self.local_imgs[i][0])

    def __len__(self):
        return len(self.local_imgs)

    def pre_process(self, img_path:str):
        img = Image.open(img_path)
        img_data = np.array(img)
        if img_path.find('_v_') != -1:
            edge_root = '/path/to/HED/edge/trainA'
        else:
            edge_root = '/path/to/HED/edge/trainB'
        img_edge = np.array(Image.open(os.path.join(edge_root , os.path.basename(img_path).replace('.bmp', '.png'))))
        img_edge = np.expand_dims(img_edge, axis=2)
        temp_data = np.concatenate([img_data, img_edge], axis=2).astype(np.uint8)
        temp_data = self.transform(temp_data)
        img = temp_data[:3]
        img_edge = temp_data[3:]
        return img, img_edge

    def __getitem__(self, index):
        if not self.balance_sample:
            path = self.local_imgs[index][0] 
            label = self.local_imgs[index][1]
        else:
            self.split_class()
            if index % 2 == 0:
                path = random.choice(self.rgb_imgs)
                self.rgb_imgs.remove(path)
            else:
                path = random.choice(self.ir_imgs)
                self.ir_imgs.remove(path)
            label = index % 2

        out_dict = {}
        if self.class_cond:
            out_dict["y"] = np.array(label, dtype=np.int64)

        if self.contour:
            img, img_edge = self.pre_process(path)
            out_dict["contour"] = img_edge
        else:
            img = self.loader(path)
            img = self.transform(img)
        
        if self.return_path:
            out_dict["path"] = path

        return img, out_dict


class RegDBModFolder(SYSUBaseDataset, Dataset):
    def __init__(
            self, 
            data_dir, 
            image_size,
            shard=0,
            num_shards=1,
            return_path=False,
            random_crop=False,
            contour=False,
            hist_match=False,
            modality=None,
            class_cond=False,
            **kwargs,
    ):
        SYSUBaseDataset.__init__(self, image_size, random_crop, contour, hist_match)

        self.ids = set()
        self.imgs = []
        for img_name in sorted(os.listdir(data_dir)):
            img_id = img_name.split('_')[1]
            self.imgs.append([os.path.join(data_dir, img_name), img_id])
            self.ids.add(img_id)
        self.ids = list(sorted(self.ids))
        self.local_imgs = self.imgs[shard:][::num_shards] # mpi multi-gpus
        self.modality = modality
        self.return_path = return_path
        self.class_cond = class_cond

    def __len__(self):
        return len(self.local_imgs)

    def pre_process(self, img_path:str):
        img = Image.open(img_path)
        img_data = np.array(img)
        if img_path.find('_v_') != -1:
            edge_root = '/path/to/HED/edge/trainA'
        else:
            edge_root = '/path/to/HED/edge/trainB'
        img_edge = np.array(Image.open(os.path.join(edge_root , os.path.basename(img_path).replace('.bmp', '.png'))))
        img_edge = np.expand_dims(img_edge, axis=2)
        temp_data = np.concatenate([img_data, img_edge], axis=2).astype(np.uint8)
        temp_data = self.transform(temp_data)
        img = temp_data[:3]
        img_edge = temp_data[3:]
        return img, img_edge

    def __getitem__(self, index):
        img_path = self.local_imgs[index][0] 
        img_id = self.local_imgs[index][1]

        out_dict = {}
        out_dict['id'] = np.array(self.ids.index(img_id), dtype=np.int64)

        if self.class_cond:
            if self.modality == 'rgb':
                out_dict['y'] = np.array(0, dtype=np.int64)
            elif self.modality == 'ir':
                out_dict['y'] = np.array(1, dtype=np.int64)

        if self.contour:
            img, img_highpass = self.pre_process(img_path)
            out_dict["contour"] = img_highpass
        else:
            img = Image.open(img_path).resize([128,256])
            img = self.transform(img)
        
        if self.return_path:
            out_dict["path"] = img_path

        return img, out_dict