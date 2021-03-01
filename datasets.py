from torch.utils.data import Dataset
import os
from PIL import Image
import torch

class Band2BandDataset(Dataset):

    def __init__(self, data_directory, transform=None):
        super(Band2BandDataset, self).__init__()

        self.data_dir = data_directory
        self.transform = transform

    def __len__(self):
        return len(self.data_dir)

    def __get_item__(self, idx):
        classes = []
        imgs = []
        for fn in os.listdir(self.data_dir[idx]):
            img = Image.open(
                    os.path.join(
                                self.data_dir,
                                self.data_dir[idx],
                                fn
                            )
                        )
            if self.transform:
                img = self.transform(img)

            imgs.append(img)
            classes.append(fn.split('_')[-1].split('.')[0])

        imgs = torch.stack(imgs)
        classes = torch.tensor(classes)

        return imgs, classes



