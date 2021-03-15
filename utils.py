import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import yaml
from models import Encoder, Decoder
from Band2BandDataset import Band2BandDataset

def get_config(config_file):
    """
    Load config from yaml format

    Inputs
    ------
    config_file : str

    Outputs
    -------
    cfg : dict
    """

    with open(config_file, 'r') as f:
        cfg = yaml.full_load(f)

    return cfg

def get_encoder(cfg):
    """
    Loads encoder module (defined in models.py file) using parameters specified
    in config

    ...

    Inputs
    ------
    cfg : dict

    Outputs
    -------
    E : Encoder
    """

    E = Encoder(**cfg['model']['encoder'])

    return E

def get_decoder(cfg):
    """
    Loads decoder module (defined in models.py file) using parameters specified
    in config

    ...

    Inputs
    ------
    cfg : dict

    Outputs
    -------
    D : Decoder
    """
    D = Decoder(**cfg['model']['decoder'])

    return D

def get_device_information():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        world_size = torch.cuda.device_count()
    else:
        device = torch.device('cpu')
        world_size = 0

    return device, world_size

def get_train_loader(cfg, world_size, distributed):
    """
    Prepares data loaders from config. Dataloaders are designed to be used with
    torch.nn.parallel.DistributedDataParallel

    Inputs
    ------
    cfg : dict
    world_size : int
        number of processes
    distributed : bool
        whether application is running on multiple nodes/gpus or not

    Outputs
    -------
    train_dl : torch.utils.data.DataLoader
    train_dataset : torch.utils.data.Dataset
    """

    trans = transforms.Compose([
                        transforms.ToTensor()
                        ])
    train_dataset = Band2BandDataset(cfg['data']['dir_head'],
                                    cfg['data']['train_csv'],
                                    trans)

    #setup samplers
    if distributed:
        train_sampler = DistributedSampler(train_dataset)

    #setup dataloaders
    if distributed:
        train_dl = DataLoader(
                        train_dataset,
                        sampler = train_sampler,
                        batch_size=cfg['data']['batch_size'],
                        )
    else:
        train_dl = DataLoader(
                        train_dataset,
                        shuffle = True,
                        batch_size=cfg['data']['batch_size']
                        )

    return train_dataset, train_dl

def get_val_loader(cfg, world_size, distributed):
    """
    Prepares data loaders from config. Dataloaders are designed to be used with
    torch.nn.parallel.DistributedDataParallel

    Inputs
    ------
    cfg : dict
    world_size : int
        number of processes
    distributed : bool
        whether application is running on multiple nodes/gpus or not

    Outputs
    -------
    val_dl : torch.utils.data.DataLoader
    val_dataset : torch.utils.data.Dataset
    """

    trans = transforms.Compose([
                        transforms.ToTensor()
                        ])
    val_dataset = Band2BandDataset(cfg['data']['dir_head'],
                                cfg['data']['val_csv'],
                                trans)

    #setup samplers
    if distributed:
        val_sampler = DistributedSampler(val_dataset)

    #setup dataloaders
    if distributed:
        val_dl = DataLoader(
                        val_dataset,
                        sampler = val_sampler,
                        batch_size = 1
                        )
    else:
        val_dl = DataLoader(
                        val_dataset,
                        shuffle = True,
                        batch_size = 1
                        )

    return val_dataset, val_dl

def get_reconstruction_gen_loss(cfg):
    """
    Uses config to decide which type of loss to use for reconstruction of a
    generated class (meaning we are trying to generate a different class from
    the input image using the decoder)

    Inputs
    ------
    cfg : dict

    Outputs
    -------
    torch loss module
    """

    if cfg['loss']['reconstruction_gen']['type'] == 'L1':
        loss = nn.L1Loss()
    elif cfg['loss']['reconstruction_gen']['type'] == 'L2':
        loss = nn.MSELoss()
    elif cfg['loss']['reconstruction_gen']['type'] == 'Perceptual':
        raise NotImplementedError
    else:
        raise NotImplementedError

    return loss

