from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from itertools import permutations
import pandas as pd

class Band2BandDataset(Dataset):
    """
    Custom pytorch dataset for Band2Band trainslation. Requires data to already
    be structured in the form of:
        <head_directory>/<common_file_stem>/<tile_number>/<band_number.tif>
    """

    def __init__(self, data_directory_head, csv_path=None, transform=None):
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
        df = pd.read_csv(csv_path, header=None)
        self.first_in_pair = df[0].tolist()
        self.second_in_pair = df[1].tolist()
        self.data_directory_head = data_directory_head

        self.transform = transform

    def __len__(self):
        return len(self.first_in_pair)

    def __getitem__(self, idx):

        first = os.path.join(self.data_directory_head, self.first_in_pair[idx])
        second = os.path.join(self.data_directory_head, self.second_in_pair[idx])

        classes = []
        imgs = []
        for fn in [first,second]:
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



