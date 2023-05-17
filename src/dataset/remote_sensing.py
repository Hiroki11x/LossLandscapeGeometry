import torch
import torch.optim.lr_scheduler
import torch.nn.init
import os
import numpy as np
from skimage import io
import random

# ISPRS color palette
# Let's define the standard ISPRS color palette
palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


# Dataset class
class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files, label_files, train=True, seed=12345,
                 cache=False, augmentation=True, window_size=(256,256)):
        super(ISPRS_dataset, self).__init__()
        
        self.augmentation = augmentation
        self.cache = cache
        self.train = train
        
        # List of files
        self.data_files = [data_files.format(id) for id in ids]
        self.label_files = [label_files.format(id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))
        
        # Initialize cache dicts
        self.data_cache_ = {}
        self.label_cache_ = {}
        self.window_size=window_size
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.RandomState(seed)
            
    
    def __len__(self):
        # Default train epoch size is 10 000 samples
        if self.train:
            return 10000
        else: #For test and valid
            return 1000 #10 for debug et 1000 for true train
    
    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        rnd_flip = random.random()
        rnd_mirror = random.random()
        
        if flip and rnd_flip < 0.5:
            will_flip = True
        if mirror and rnd_mirror < 0.5:
            will_mirror = True
        
        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
            
        return tuple(results)
    
    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)
        #print('random_idx: ', random_idx)
        
        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            data = 1/255 * np.asarray(io.imread(self.data_files[random_idx]).transpose((2,0,1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data
            
        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else: 
            # Labels are converted from RGB to their numeric values
            label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, self.window_size)
        #print('x1 and y1: ', x1, y1)
        data_p = data[:, x1:x2,y1:y2]
        label_p = label[x1:x2,y1:y2]
        
        # Data augmentation
        data_p, label_p = self.data_augmentation(data_p, label_p)

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p))


def sample_n_batches(exp_dict):
    
    dataset = ISPRS_dataset(exp_dict['train_ids'], 
                            data_files=exp_dict['DATA_FOLDER'], 
                            label_files=exp_dict['LABEL_FOLDER'], 
                            train=True,
                            seed=exp_dict['seed'],
                            cache=exp_dict['cache'], 
                            window_size=exp_dict['window_size'])
    
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=exp_dict['batch_size'],
                                               shuffle=True,
                                               num_workers=exp_dict['num_workers'],
                                               drop_last=True)
    
    batch_list = []
    for i, batch in enumerate(train_loader):
        if i >= 10:
            break
        batch_list.append(batch)
    return batch_list