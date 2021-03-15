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
from utils import *

def validate(cfg, device, world_size, local_rank, distributed):
    """
    Performs training of Encoder and Decoder modules

    Inputs
    ------
    config_file : str
        path to location of yaml config file
    local_rank : int
        rank number of master process for distributed training
    """

    #initialize encoder
    print('Initializing Encoder')
    E = get_encoder(cfg)
    E.load_state_dict(torch.load(cfg['checkpoint']['encoder']))
    E = E.to(device)
    if distributed:
        E = nn.parallel.DistributedDataParallel(E, device_ids=[local_rank],
                output_device=local_rank)

    #initialize decoder
    print('Initializing Decoder')
    D = get_decoder(cfg)
    D.load_state_dict(torch.load(cfg['checkpoint']['decoder']))
    D = D.to(device)
    if distributed:
        D = nn.parallel.DistributedDataParallel(D, device_ids=[local_rank],
                output_device=local_rank)

    print('Building dataloaders')
    val_dataset, val_data_loader = get_val_loader(cfg, world_size, distributed)
    reconstruction_gen_loss = get_reconstruction_gen_loss(cfg)
    if local_rank == 0:
        logger = SummaryWriter(cfg['training']['log_directory'])
    val_dl = iter(val_data_loader)
    total_gen_loss = 0
    dataset_len = len(val_dataset)
    with torch.no_grad():
        for val_iter in range(cfg['validation']['iterations']):
            if val_iter >= dataset_len:
                break
            imgs,labels = val_dl.next()

            imgs = imgs.to(device)
            labels = labels.to(device)

            labels = labels.view(-1,1)
            imgs = imgs.view([-1,imgs.shape[2],imgs.shape[3],imgs.shape[4]])
            #extract embeddings from Encoder
            embs = E(imgs, labels)

            #generate new band
            labels_swapped = labels.view(-1,2,1)
            labels_swapped = torch.cat([
                                labels_swapped[:,1,:].view(-1,1,1),
                                labels_swapped[:,0,:].view(-1,1,1)
                                ],
                            1)
            labels_swapped = labels_swapped.view(-1,1)
            imgs_swapped = imgs.view(-1,2,imgs.shape[1],imgs.shape[2],imgs.shape[3])
            imgs_swapped = torch.cat([
                        imgs_swapped[:,1,:,:,:].view(-1,1,imgs_swapped.shape[2],imgs_swapped.shape[3],imgs_swapped.shape[4]),
                        imgs_swapped[:,0,:,:,:].view(-1,1,imgs_swapped.shape[2],imgs_swapped.shape[3],imgs_swapped.shape[4])
                        ],
                    1)
            imgs_swapped = imgs_swapped.view([-1,imgs_swapped.shape[2],imgs_swapped.shape[3],imgs_swapped.shape[4]])
            output_swapped = D(embs, labels_swapped)
            loss_rec_swap = reconstruction_gen_loss(output_swapped,imgs_swapped)
            total_gen_loss += loss_rec_swap.item()/cfg['validation']['iterations']

    logger.add_scalar('Validation_loss/generative_reconstruction',
            total_gen_loss, val_iter)

def run_validation(config_file, local_rank):
    cfg = get_config(config_file)

    device, world_size = get_device_information()
    distributed = world_size > 1

    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(cfg['backend'], rank=local_rank,
                world_size=world_size)
    else:
        world_size = 1
        local_rank = 0

    validate(cfg, device, world_size, local_rank, distributed)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True, type=str)
    parser.add_argument('--local_rank',default=0,type=int)

    args = parser.parse_args()

    run_validation(**vars(args))
