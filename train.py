import yaml
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from models import Encoder, Decoder
from datasets import Band2BandDataset

def get_config(config_file):
    with open(config_file, 'r') as f:
        cfg = yaml.full_load(f)

    return cfg

def get_encoder(cfg):
    E = Encoder(**cfg['model']['encoder'])

    return E

def get_decoder(cfg):
    D = Decoder(**cfg['model']['decoder'])

    return D

def get_data_loaders(cfg):
    trans = transforms.Compose([
                        transforms.Resize(cfg['data']['image_size']),
                        transforms.ToTensor()
                        ])
    train_dataset = Band2BandDataset(cfg['data']['train_dir'],trans)
    val_dataset = Band2BandDataset(cfg['data']['val_dir'],trans)

    train_dl = DataLoader(
                    train_dataset,
                    batch_size=1,
                    shuffle=cfg['data']['shuffle']
                    )
    val_dl = DataLoader(
                    val_dataset,
                    batch_size=1,
                    )
    return train_dl, val_dl

def get_optimizer(cfg, E, D):
    if cfg['optimizer']['algorithm'].lower() == 'adam':
        optimizer = torch.optim.Adam(
                list(E.parameters())+list(D.parameters()),
                lr=cfg['optimizer']['learning_rate']
                )
    elif cfg['optimizer']['algorithm'].lower() == 'sgd':
        optimizer = torch.optim.SGD(
                list(E.parameters())+list(D.parameters()),
                lr=cfg['optimizer']['learning_rate'],
                momentum=cfg['optimizer']['momentum']
                )
    else:
        raise NotImplementedError

    return optimizer

def get_reconstruction_loss(cfg):
    if cfg['loss']['reconstruction'] == 'L1':
        loss = nn.L1Loss()
    elif cfg['loss']['reconstruction'] == 'L2':
        loss = nn.MSELoss()
    elif cfg['loss']['reconstruction'] == 'Perceptual':
        raise NotImplementedError
    else:
        raise NotImplementedError

    return loss

def get_embedding_match_loss(cfg):
    if cfg['loss']['matching'] == 'L1':
        loss = nn.L1Loss()
    elif cfg['loss']['matching'] == 'L2':
        loss = nn.MSELoss()
    else:
        raise NotImplementedError

    return loss

def train(config_file):
    cfg = get_config(config_file)
    E = get_encoder(cfg)
    D = get_decoder(cfg)
    train_data_loader, valid_data_loader = get_data_loaders(cfg)
    optimizer = get_optimizer(cfg, E, D)
    reconstruction_loss = get_reconstruction_loss(cfg)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True, type=str)

    args = parser.parse_args()

    train(**vars(args))
