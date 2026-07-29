from torch.utils.data import Dataset, Subset
import numpy as np
import os
from scipy import interpolate
from einops import rearrange
import json
import csv
import torch
from pathlib import Path
import torchvision.transforms as transforms
from scipy.interpolate import interp1d
from typing import Callable, Optional, Tuple, Union
from natsort import natsorted
from glob import glob
import pickle

from transformers import AutoProcessor
def identity(x):
    return x

def split_eeg_imagenet_by_class(data, train_ratio=0.6, seed=2022):
    class_to_indices = {}
    for index, item in enumerate(data):
        class_key = item.get("label", item["image"].split("_")[0])
        class_to_indices.setdefault(class_key, []).append(index)

    rng = np.random.RandomState(seed)
    train_indices = []
    test_indices = []
    for class_key in sorted(class_to_indices):
        indices = np.array(class_to_indices[class_key], dtype=np.int64)
        indices = rng.permutation(indices)
        train_size = int(len(indices) * train_ratio)
        if len(indices) > 1:
            train_size = min(max(train_size, 1), len(indices) - 1)
        train_indices.extend(indices[:train_size].tolist())
        test_indices.extend(indices[train_size:].tolist())

    return np.array(train_indices, dtype=np.int64), np.array(test_indices, dtype=np.int64)

def split_eeg_imagenet_random(data, train_ratio=0.6, seed=2022):
    indices = np.arange(len(data), dtype=np.int64)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(indices)
    train_size = int(len(indices) * train_ratio)
    if len(indices) > 1:
        train_size = min(max(train_size, 1), len(indices) - 1)
    return indices[:train_size], indices[train_size:]

def _get_eeg_imagenet_window_shape(data):
    sample = data[0]["eeg_data"]
    n_channels = sample.shape[-2]
    start = 40
    end = min(440, sample.shape[-1])
    if start >= end:
        start = 0
        end = sample.shape[-1]
    return start, end, n_channels, end - start

class EEGImageNetPreprocessMixin:
    def _init_eeg_imagenet_preprocess(self, args):
        self.eeg_preprocess = getattr(args, 'eeg_preprocess', 'channel_minmax') or 'channel_minmax'
        self.eeg_scale = getattr(args, 'eeg_scale', 1.0)
        self.eeg_zscore_eps = getattr(args, 'eeg_zscore_eps', 1e-8)
        self.eeg_zscore_mean = None
        self.eeg_zscore_std = None

    def fit_eeg_zscore(self, indices=None):
        if self.eeg_preprocess != 'channel_zscore':
            return
        if indices is None:
            indices = range(len(self.data))
        indices = list(indices)
        if len(indices) == 0:
            raise RuntimeError('Cannot fit EEG-ImageNet z-score stats on an empty index set.')

        channel_sum = torch.zeros(self.data_chan, dtype=torch.float64)
        channel_sumsq = torch.zeros(self.data_chan, dtype=torch.float64)
        count = 0
        for index in indices:
            eeg_data = self.data[int(index)]["eeg_data"].to(torch.float64)
            feat = eeg_data[:, self.eeg_start:self.eeg_end]
            channel_sum += feat.sum(dim=1)
            channel_sumsq += (feat * feat).sum(dim=1)
            count += feat.shape[-1]

        mean = channel_sum / count
        var = (channel_sumsq / count - mean * mean).clamp_min(0.0)
        std = torch.sqrt(var).clamp_min(self.eeg_zscore_eps)
        self.eeg_zscore_mean = mean.float().view(-1, 1)
        self.eeg_zscore_std = std.float().view(-1, 1)
        print('EEG-ImageNet channel z-score fitted on', len(indices), 'samples')

    def set_eeg_zscore_stats(self, mean, std):
        if self.eeg_preprocess != 'channel_zscore':
            return
        if mean is None or std is None:
            raise RuntimeError('EEG-ImageNet z-score stats are not fitted.')
        self.eeg_zscore_mean = mean.clone()
        self.eeg_zscore_std = std.clone()

    def preprocess_eeg_feat(self, feat):
        if self.eeg_preprocess == 'channel_minmax':
            eeg_chan_min, _ = torch.min(feat, dim=-1, keepdims=True)
            eeg_chan_max, _ = torch.max(feat, dim=-1, keepdims=True)
            return (feat - eeg_chan_min) / (eeg_chan_max - eeg_chan_min + 1e-8)
        if self.eeg_preprocess == 'scale':
            return feat * self.eeg_scale
        if self.eeg_preprocess == 'channel_zscore':
            if self.eeg_zscore_mean is None or self.eeg_zscore_std is None:
                raise RuntimeError('Call fit_eeg_zscore or set_eeg_zscore_stats before using channel_zscore.')
            mean = self.eeg_zscore_mean.to(device=feat.device, dtype=feat.dtype)
            std = self.eeg_zscore_std.to(device=feat.device, dtype=feat.dtype)
            return (feat - mean) / (std + self.eeg_zscore_eps)
        if self.eeg_preprocess == 'none':
            return feat
        raise ValueError(f'Unknown EEG-ImageNet preprocess mode: {self.eeg_preprocess}')

def pad_to_patch_size(x, patch_size):
    assert x.ndim == 2
    return np.pad(x, ((0,0),(0, patch_size-x.shape[1]%patch_size)), 'wrap')

def pad_to_length(x, length):
    assert x.ndim == 3
    assert x.shape[-1] <= length
    if x.shape[-1] == length:
        return x

    return np.pad(x, ((0,0),(0,0), (0, length - x.shape[-1])), 'wrap')

def normalize(x, mean=None, std=None):
    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    return (x - mean) / (std * 1.0)

def process_voxel_ts(v, p, t=8):
    '''
    v: voxel timeseries of a subject. (1200, num_voxels)
    p: patch size
    t: time step of the averaging window for v. Kamitani used 8 ~ 12s
    return: voxels_reduced. reduced for the alignment of the patch size (num_samples, num_voxels_reduced)

    '''
    # average the time axis first
    num_frames_per_window = t // 0.75 # ~0.75s per frame in HCP
    v_split = np.array_split(v, len(v) // num_frames_per_window, axis=0)
    v_split = np.concatenate([np.mean(f,axis=0).reshape(1,-1) for f in v_split],axis=0)
    # pad the num_voxels
    # v_split = np.concatenate([v_split, np.zeros((v_split.shape[0], p - v_split.shape[1] % p))], axis=-1)
    v_split = pad_to_patch_size(v_split, p)
    v_split = normalize(v_split)
    return v_split

def augmentation(data, aug_times=2, interpolation_ratio=0.5):
    '''
    data: num_samples, num_voxels_padded
    return: data_aug: num_samples*aug_times, num_voxels_padded
    '''
    num_to_generate = int((aug_times-1)*len(data)) 
    if num_to_generate == 0:
        return data
    pairs_idx = np.random.choice(len(data), size=(num_to_generate, 2), replace=True)
    data_aug = []
    for i in pairs_idx:
        z = interpolate_voxels(data[i[0]], data[i[1]], interpolation_ratio)
        data_aug.append(np.expand_dims(z,axis=0))
    data_aug = np.concatenate(data_aug, axis=0)

    return np.concatenate([data, data_aug], axis=0)

def interpolate_voxels(x, y, ratio=0.5):
    ''''
    x, y: one dimension voxels array
    ratio: ratio for interpolation
    return: z same shape as x and y

    '''
    values = np.stack((x,y))
    points = (np.r_[0, 1], np.arange(len(x)))
    xi = np.c_[np.full((len(x)), ratio), np.arange(len(x)).reshape(-1,1)]
    z = interpolate.interpn(points, values, xi)
    return z

def img_norm(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = (img / 255.0) * 2.0 - 1.0 # to -1 ~ 1
    return img

def channel_first(img):
        if img.shape[-1] == 3:
            return rearrange(img, 'h w c -> c h w')
        return img



#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

def is_npy_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'{ext}' == 'npy'# type: ignore

class eeg_pretrain_dataset(Dataset):
    def __init__(self, path='../dreamdiffusion/datasets/mne_data/', roi='VC', patch_size=16, transform=identity, aug_times=2, 
                num_sub_limit=None, include_kam=False, include_hcp=True):
        super(eeg_pretrain_dataset, self).__init__()
        data = []
        images = []
        self.input_paths = [str(f) for f in sorted(Path(path).rglob('*')) if is_npy_ext(f) and os.path.isfile(f)]

        assert len(self.input_paths) != 0, 'No data found'
        self.data_len  = 512
        self.data_chan = 128

    def __len__(self):
        return len(self.input_paths)
    
    def __getitem__(self, index):
        data_path = self.input_paths[index]

        data = np.load(data_path)

        if data.shape[-1] > self.data_len:
            idx = np.random.randint(0, int(data.shape[-1] - self.data_len)+1)

            data = data[:, idx: idx+self.data_len]
        else:
            x = np.linspace(0, 1, data.shape[-1])
            x2 = np.linspace(0, 1, self.data_len)
            f = interp1d(x, data)
            data = f(x2)
        ret = np.zeros((self.data_chan, self.data_len))
        if (self.data_chan > data.shape[-2]):
            for i in range((self.data_chan//data.shape[-2])):

                ret[i * data.shape[-2]: (i+1) * data.shape[-2], :] = data
            if self.data_chan % data.shape[-2] != 0:

                ret[ -(self.data_chan%data.shape[-2]):, :] = data[: (self.data_chan%data.shape[-2]), :]
        elif(self.data_chan < data.shape[-2]):
            idx2 = np.random.randint(0, int(data.shape[-2] - self.data_chan)+1)
            ret = data[idx2: idx2+self.data_chan, :]
        # print(ret.shape)
        elif(self.data_chan == data.shape[-2]):
            ret = data
        ret = ret/10 # reduce an order
        # torch.tensor()
        ret = torch.from_numpy(ret).float()
        return {'eeg': ret } #,


class MyEEGDataset(Dataset, EEGImageNetPreprocessMixin):
    def __init__(self, args):
        self.dataset_dir = args.dataset_dir
        loaded = torch.load(os.path.join(args.dataset_dir, args.data_file))
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self._init_eeg_imagenet_preprocess(args)
        if args.subject != -1:
            chosen_data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                           loaded['dataset'][i]['subject'] == args.subject]
        else:
            chosen_data = loaded['dataset']
        if args.granularity == 'coarse':
            self.data = [i for i in chosen_data if i['granularity'] == 'coarse']
        elif args.granularity == 'all':
            self.data = chosen_data
        else:
            fine_num = int(args.granularity[-1])
            fine_category_range = np.arange(8 * fine_num, 8 * fine_num + 8)
            self.data = [i for i in chosen_data if
                         i['granularity'] == 'fine' and self.labels.index(i['label']) in fine_category_range]
        self.eeg_start, self.eeg_end, self.data_chan, self.data_len = _get_eeg_imagenet_window_shape(self.data)

    def __getitem__(self, index):
        eeg_data = self.data[index]["eeg_data"].float()
        feat = self.preprocess_eeg_feat(eeg_data[:, self.eeg_start:self.eeg_end])
        return {'eeg': feat}

    def __len__(self):
        return len(self.data)

    def normalize_eeg(self, eeg_data):
        eeg_chan_min, _ = torch.min(eeg_data, dim=-1, keepdims=True)
        eeg_chan_max, _ = torch.max(eeg_data, dim=-1, keepdims=True)
        return (eeg_data - eeg_chan_min) / (eeg_chan_max - eeg_chan_min + 1e-8)


class MyEEGImageNetDataset(Dataset, EEGImageNetPreprocessMixin):
    def __init__(self, args, transform_img=identity):
        self.dataset_dir = args.dataset_dir
        self._init_eeg_imagenet_preprocess(args)
        self.transform_img = transform_img
        loaded = torch.load(os.path.join(args.dataset_dir, args.data_file))
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        if isinstance(args.subject, list):
            chosen_data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                           loaded['dataset'][i]['subject'] in args.subject]
        elif args.subject != -1:
            chosen_data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                           loaded['dataset'][i]['subject'] == args.subject]
        else:
            chosen_data = loaded['dataset']
        if args.granularity == 'coarse':
            self.data = [i for i in chosen_data if i['granularity'] == 'coarse']
        elif args.granularity == 'all':
            self.data = chosen_data
        else:
            fine_num = int(args.granularity[-1])
            fine_category_range = np.arange(8 * fine_num, 8 * fine_num + 8)
            self.data = [i for i in chosen_data if
                         i['granularity'] == 'fine' and self.labels.index(i['label']) in fine_category_range]
        self.eeg_start, self.eeg_end, self.data_chan, self.data_len = _get_eeg_imagenet_window_shape(self.data)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def __getitem__(self, index):
        path = self.data[index]["image"]
        img = Image.open(os.path.join(self.dataset_dir, "imageNet_images", path.split('_')[0], path))
        if img.mode == 'L':
            img = img.convert('RGB')
        eeg_data = self.data[index]["eeg_data"].float()
        feat = self.preprocess_eeg_feat(eeg_data[:, self.eeg_start:self.eeg_end])
        image = np.array(img) / 255.0
        img_raw = self.processor(images=img, return_tensors="pt")
        img_raw['pixel_values'] = img_raw['pixel_values'].squeeze(0)
        label = self.data[index]["label"]
        return {'eeg': feat, 'label': label, 'image': self.transform_img(image), 'image_raw': img_raw}

    def __len__(self):
        return len(self.data)

    def normalize_eeg(self, eeg_data):
        eeg_chan_min, _ = torch.min(eeg_data, dim=-1, keepdims=True)
        eeg_chan_max, _ = torch.max(eeg_data, dim=-1, keepdims=True)
        return (eeg_data - eeg_chan_min) / (eeg_chan_max - eeg_chan_min + 1e-8)



def get_img_label(class_index:dict, img_filename:list, naive_label_set=None):
    img_label = []
    wind = []
    desc = []
    for _, v in class_index.items():
        n_list = []
        for n in v[:-1]:
            n_list.append(int(n[1:]))
        wind.append(n_list)
        desc.append(v[-1])

    naive_label = {} if naive_label_set is None else naive_label_set
    for _, file in enumerate(img_filename):
        name = int(file[0].split('.')[0])
        naive_label[name] = []
        nl = list(naive_label.keys()).index(name)
        for c, (w, d) in enumerate(zip(wind, desc)):
            if name in w:
                img_label.append((c, d, nl))
                break
    return img_label, naive_label

class base_dataset(Dataset):
    def __init__(self, x, y=None, transform=identity):
        super(base_dataset, self).__init__()
        self.x = x
        self.y = y
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        if self.y is None:
            return self.transform(self.x[index])
        else:
            return self.transform(self.x[index]), self.transform(self.y[index])
    
def remove_repeats(fmri, img_lb):
    assert len(fmri) == len(img_lb), 'len error'
    fmri_dict = {}
    for f, lb in zip(fmri, img_lb):
        if lb in fmri_dict.keys():
            fmri_dict[lb].append(f)
        else:
            fmri_dict[lb] = [f]
    lbs = []
    fmris = []
    for k, v in fmri_dict.items():
        lbs.append(k)
        fmris.append(np.mean(np.stack(v), axis=0))
    return np.stack(fmris), lbs


def list_get_all_index(list, value):
    return [i for i, v in enumerate(list) if v == value]

EEG_EXTENSIONS = [
    '.mat'
]


def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in EEG_EXTENSIONS)


def make_dataset(dir):

    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir, topdown=False)):#
        for fname in fnames:
            if is_mat_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

from PIL import Image
import numpy as np
 


class EEGDataset(Dataset):
    
    # Constructor
    def __init__(self, eeg_signals_path, imagenet_path, image_transform=identity, subject = 4):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        # if opt.subject!=0:
        #     self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==opt.subject]
        # else:
        # print(loaded)
        if subject!=0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==subject]
        else:
            self.data = loaded['dataset']        
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.imagenet = imagenet_path
        self.image_transform = image_transform
        self.num_voxels = 440
        self.data_len = 512
        if len(self.data) == 0:
            raise RuntimeError(f'No EEG samples found for subject={subject}.')
        self.data_chan = int(self.data[0]["eeg"].size(0))
        # Compute size
        self.size = len(self.data)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):

        eeg = self.data[i]["eeg"].float().t()

        eeg = eeg[20:460,:]

        eeg = np.array(eeg.transpose(0,1))
        x = np.linspace(0, 1, eeg.shape[-1])
        x2 = np.linspace(0, 1, self.data_len)
        f = interp1d(x, eeg)
        eeg = f(x2)
        eeg = torch.from_numpy(eeg).float()

        label = torch.tensor(self.data[i]["label"]).long()

        # Get label
        image_name = self.images[self.data[i]["image"]]
        if not self.imagenet:
            raise ValueError('imagenet_path must be set for EEGCVPR/EEG dataset image loading.')
        image_path = os.path.join(self.imagenet, image_name.split('_')[0], image_name+'.JPEG')
        if not os.path.exists(image_path):
            raise FileNotFoundError(f'ImageNet image not found: {image_path}')
        image_raw = Image.open(image_path).convert('RGB')
        
        
        image = np.array(image_raw) / 255.0
        image_raw = self.processor(images=image_raw, return_tensors="pt")
        image_raw['pixel_values'] = image_raw['pixel_values'].squeeze(0)


        return {'eeg': eeg, 'label': label, 'image': self.image_transform(image), 'image_raw': image_raw}
        # Return
        # return eeg, label

class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train", subject=4):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)

        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if i <= len(self.dataset.data) and 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size

        self.size = len(self.split_idx)
        self.num_voxels = 440
        self.data_len = 512
        self.data_chan = self.dataset.data_chan

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        return self.dataset[self.split_idx[i]]


def create_EEG_dataset(eeg_signals_path='../dreamdiffusion/datasets/eeg_5_95_std.pth', 
            splits_path = '../dreamdiffusion/datasets/block_splits_by_image_single.pth',
            imagenet_path = '/home/mahui/Dataset/IMAGENET/train',
            image_transform=identity, subject = 0):

    if isinstance(image_transform, list):
        dataset_train = EEGDataset(eeg_signals_path, imagenet_path, image_transform[0], subject )
        dataset_test = EEGDataset(eeg_signals_path, imagenet_path, image_transform[1], subject)
    else:
        dataset_train = EEGDataset(eeg_signals_path, imagenet_path, image_transform, subject)
        dataset_test = EEGDataset(eeg_signals_path, imagenet_path, image_transform, subject)
    # split_train = Splitter(dataset_train, split_path = splits_path, split_num = 0, split_name = 'train', subject= subject)
    # split_test = Splitter(dataset_test, split_path = splits_path, split_num = 0, split_name = 'test', subject = subject)
    train_size = int(0.669 * len(dataset_train))
    test_size = int(0.164 * len(dataset_train))
    val_size = len(dataset_train) - train_size - test_size
    print('随机划分数据集')
    dataset_train, val, false_test_dataset = torch.utils.data.random_split(dataset_train,
                                                                           [train_size, val_size, test_size])
    false_train_dataset, val, dataset_test = torch.utils.data.random_split(dataset_test,
                                                                           [train_size, val_size, test_size])
    return (dataset_train, dataset_test)


def create_myEEGImageNetDataset(args, dataset_name, split_ratio=0.6, image_transform=identity):
    if dataset_name == 'EEG-ImageNet':
        if isinstance(image_transform, list):
            train_dataset = MyEEGImageNetDataset(args, transform_img=image_transform[0])
            test_dataset = MyEEGImageNetDataset(args, transform_img=image_transform[1])
        else:
            train_dataset = MyEEGImageNetDataset(args, transform_img=image_transform)
            test_dataset = MyEEGImageNetDataset(args, transform_img=image_transform)
        train_index, test_index = split_eeg_imagenet_random(
            train_dataset.data,
            train_ratio=split_ratio,
            seed=getattr(args, 'seed', 2022))
        print('EEG-ImageNet整体随机划分：训练比例 %.3f，测试比例 %.3f' % (split_ratio, 1.0 - split_ratio))
        train_dataset.fit_eeg_zscore(train_index)
        test_dataset.set_eeg_zscore_stats(train_dataset.eeg_zscore_mean, train_dataset.eeg_zscore_std)
        train_dataset = Subset(train_dataset, train_index)
        test_dataset = Subset(test_dataset, test_index)
        train_dataset.data_len = train_dataset.dataset.data_len
        train_dataset.data_chan = train_dataset.dataset.data_chan
        test_dataset.data_len = test_dataset.dataset.data_len
        test_dataset.data_chan = test_dataset.dataset.data_chan
        print('训练集大小:', len(train_dataset))
        print('测试集大小:', len(test_dataset))
        return train_dataset, test_dataset
    return create_EEG_dataset(
        eeg_signals_path=args.eeg_signals_path,
        splits_path=args.splits_path,
        imagenet_path=getattr(args, 'imagenet_path', None),
        image_transform=image_transform,
        subject=args.subject,
    )


class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img



def normalize2(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img



def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')


if __name__ == '__main__':
    import scipy.io as scio
    import copy
    import shutil
