from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from itertools import permutations

class Band2BandDataset(Dataset):
    """
    Custom pytorch dataset for Band2Band trainslation. Requires data to already
    be structured in the form of:
        <head_directory>/<common_file_stem>/<band_number.tif>
    """

    def __init__(self, data_directory, transform=None):
        """
        Initializer for Band2BandDataset

        ...

        Inputs
        ------
        data_directory : str
            location of source data
        transform : torch.utils.data.transforms
            transforms to perform on images (default = None)
        """

        super(Band2BandDataset, self).__init__()

        data_dirs = [os.path.join(data_directory, x) for x in os.listdir(data_directory)]
        pairs = []
        for d in data_dirs:
            fns = [os.path.join(d,x) for x in os.listdir(d)]
            pairs.extend(permutations(fns,2))
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        classes = []
        imgs = []
        for fn in pair:
            img = Image.open(fn)
            if self.transform:
                img = self.transform(img)

            img = img.float()
            #this is hard coded for now but should be a changeable parameter
            img /= 5000
            img = torch.clip(img,0,1)

            imgs.append(img)
            classes.append(int(fn.split('/')[-1].split('.')[0])-1)

        imgs = torch.stack(imgs)
        classes = torch.tensor(classes)

        return imgs, classes



