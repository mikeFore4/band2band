import yaml
import os
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torchvision import transforms
from models import Encoder, Decoder
from Band2BandDataset import Band2BandDataset
from tqdm import tqdm

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

def get_data_loaders(cfg):
    """
    Prepares data loaders from config. Dataloaders are designed to be used with
    torch.nn.parallel.DistributedDataParallel

    Inputs
    ------
    cfg : dict

    Outputs
    -------
    train_dl : torch.utils.data.DataLoader
    val_dl : torch.utils.data.DataLoader
    """

    trans = transforms.Compose([
                        transforms.ToTensor()
                        ])
    train_dataset = Band2BandDataset(cfg['data']['train_dir'],trans)
    val_dataset = Band2BandDataset(cfg['data']['val_dir'],trans)

    #setup samplers
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    #setup dataloaders
    train_dl = DataLoader(
                    train_dataset,
                    sampler = train_sampler,
                    batch_size=cfg['data']['batch_size'],
                    )
    val_dl = DataLoader(
                    val_dataset,
                    sampler = val_sampler,
                    batch_size=1,
                    )
    return train_dl, val_dl

def get_optimizer(cfg, E, D):
    """
    Creates torch optimizers for Encoder and Decoder models

    Inputs
    ------
    cfg : dict
    E : Encoder
    D : Decoder

    Outputs
    -------
    torch optimizer
    """

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

def get_reconstruction_self_loss(cfg):
    """
    Uses config to decide which type of loss to use for self reconstruction
    (meaning we are trying to reconstruct the original using the decoder)

    Inputs
    ------
    cfg : dict

    Outputs
    -------
    torch loss module
    """

    if cfg['loss']['reconstruction_self']['type'] == 'L1':
        loss = nn.L1Loss()
    elif cfg['loss']['reconstruction_self']['type'] == 'L2':
        loss = nn.MSELoss()
    elif cfg['loss']['reconstruction_self']['type'] == 'Perceptual':
        raise NotImplementedError
    else:
        raise NotImplementedError

    return loss

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

def get_feature_match_loss(cfg):
    """
    Uses config to decide which type of loss to use for feature matching of
    intermediate features output from the encoder for multiple bands of the
    same image

    Inputs
    ------
    cfg : dict

    Outputs
    -------
    torch loss module
    """

    if cfg['loss']['matching']['type'] == 'L1':
        loss = nn.L1Loss()
    elif cfg['loss']['matching']['type'] == 'L2':
        loss = nn.MSELoss()
    else:
        raise NotImplementedError

    return loss

def setup(cfg, rank):
    """
    Set up DistributedDataParallel code

    Inputs
    ------
    cfg : dict
    rank : int
        rank number of master process
    """

    #set necessary environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(cfg['backend'], rank=rank)

def train(config_file, local_rank):
    """
    Performs training of Encoder and Decoder modules

    Inputs
    ------
    config_file : str
        path to location of yaml config file
    local_rank : int
        rank number of master process for distributed training
    """

    cfg = get_config(config_file)

    #setup processes for distributed training
    setup(cfg, local_rank)

    #initialize encoder
    E = get_encoder(cfg)
    E = E.to(cfg['device'])
    E = nn.parallel.DistributedDataParallel(E)

    #initialize decoder
    D = get_decoder(cfg)
    D = D.to(cfg['device'])
    D = nn.parallel.DistributedDataParallel(D)

    train_data_loader, valid_data_loader = get_data_loaders(cfg)
    optimizer = get_optimizer(cfg, E, D)
    reconstruction_self_loss = get_reconstruction_self_loss(cfg)
    reconstruction_gen_loss = get_reconstruction_gen_loss(cfg)
    match_loss = get_feature_match_loss(cfg)
    logger = SummaryWriter(cfg['log_directory'])
    total_self_loss = 0
    total_gen_loss = 0
    total_match_loss = 0
    if not os.path.exists(cfg['checkpoint_dir']):
        os.mkdir(cfg['checkpoint_dir'])

    num_iter = 0
    while num_iter < cfg['iterations']:
        for imgs,labels in tqdm(train_data_loader):
            #delete when adding DDP
            imgs = imgs.to(cfg['device'])
            labels = labels.to(cfg['device'])

            num_iter += 1
            if num_iter > cfg['iterations']:
                break
            optimizer.zero_grad()
            labels = labels.view(-1,1)
            imgs = imgs.view([-1,imgs.shape[2],imgs.shape[3],imgs.shape[4]])
            #extract embeddings from Encoder
            embs = E(imgs, labels)

            #feature matching loss
            loss_m = match_loss(
                    embs.view([-1,2,embs.shape[1],embs.shape[2],embs.shape[3]])[:,0,:,:],
                    embs.view([-1,2,embs.shape[1],embs.shape[2],embs.shape[3]])[:,1,:,:]
                    )
            total_match_loss += loss_m.item()

            #reconstruct same band
            output = D(embs, labels)
            #reconstruction loss
            loss_rec_same =  reconstruction_self_loss(output,imgs)
            total_self_loss = loss_rec_same.item()

            #generate new band
            labels_swapped = labels.view(-1,2,1)
            labels_swapped = torch.cat([
                                labels_swapped[:,1,:].view(-1,1,1),
                                labels_swapped[:,0,:].view(-1,1,1)
                                ],
                            1)
            output = D(embs,labels)
            imgs = imgs.view(-1,2,imgs.shape[1],imgs.shape[2],imgs.shape[3])
            imgs = torch.cat([
                        imgs[:,1,:,:,:].view(-1,1,imgs.shape[2],imgs.shape[3],imgs.shape[4]),
                        imgs[:,0,:,:,:].view(-1,1,imgs.shape[2],imgs.shape[3],imgs.shape[4])
                        ],
                    1)
            imgs = imgs.view([-1,imgs.shape[2],imgs.shape[3],imgs.shape[4]])
            loss_rec_swap = reconstruction_gen_loss(output,imgs)
            total_gen_loss += loss_rec_swap.item()

            #backpropagate
            total_loss = cfg['loss']['reconstruction_self']['weight']*loss_rec_same + \
                    cfg['loss']['reconstruction_gen']['weight']*loss_rec_swap + \
                    cfg['loss']['matching']['weight']*loss_m
            total_loss.backward()

            #take optimization step
            optimizer.step()

            if num_iter % cfg['log_every'] == 0:
                logger.add_scalar('Loss/train_self_reconstruction',
                        total_self_loss, num_iter)
                logger.add_scalar('Los/train_generative_reconstruction',
                        total_gen_loss, num_iter)
                logger.add_scalar('Loss/train_feature_matching',
                        total_match_loss, num_iter)

            if (num_iter+1) % cfg['checkpoint_every'] == 0:
                torch.save(E.state_dict(),
                        os.path.join(cfg['checkpoint_dir'],f'Encoder_{num_iter}.pth'))
                torch.save(D.state_dict(),
                        os.path.join(cfg['checkpoint_dir'],f'Decoder_{num_iter}.pth'))

    torch.save(E.state_dict(),
            os.path.join(cfg['checkpoint_dir'],f'Encoder_final.pth'))
    torch.save(D.state_dict(),
            os.path.join(cfg['checkpoint_dir'],f'Decoder_final.pth'))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True, type=str)
    parser.add_argument('--local_rank',default=0,type=int)

    args = parser.parse_args()

    train(**vars(args))
