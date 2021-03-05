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

def write_config(cfg):

    with open(os.path.join(cfg['model_dir'],'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, indent=4)

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
    train_dataset = Band2BandDataset(cfg['data']['train_dir'],trans)

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
    val_dataset = Band2BandDataset(cfg['data']['val_dir'],trans)

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

def get_device_information():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        world_size = torch.cuda.device_count()
    else:
        device = torch.device('cpu')
        world_size = 0

    return device, world_size

def validate(cfg, E, D, val_data_loader, reconstruction_gen_loss, dataset_len,
        device):
        val_dl = iter(val_data_loader)
        total_gen_loss = 0
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

        return total_gen_loss

def train(cfg, device, world_size, local_rank, distributed):
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
    E = E.to(device)
    if distributed:
        E = nn.parallel.DistributedDataParallel(E, device_ids=[local_rank],
                output_device=local_rank)

    #initialize decoder
    print('Initializing Decoder')
    D = get_decoder(cfg)
    D = D.to(device)
    if distributed:
        D = nn.parallel.DistributedDataParallel(D, device_ids=[local_rank],
                output_device=local_rank)

    print('Building dataloaders')
    train_dataset, train_data_loader = get_train_loader(cfg, world_size, distributed)
    if cfg['validation']['do_validation']:
        val_dataset, val_data_loader = get_val_loader(cfg, world_size,
                distributed)
    optimizer = get_optimizer(cfg, E, D)
    reconstruction_self_loss = get_reconstruction_self_loss(cfg)
    reconstruction_gen_loss = get_reconstruction_gen_loss(cfg)
    match_loss = get_feature_match_loss(cfg)
    if local_rank == 0:
        logger = SummaryWriter(cfg['training']['log_directory'])
    total_self_loss = 0
    total_gen_loss = 0
    total_match_loss = 0
    if not os.path.exists(cfg['training']['checkpoint_dir']):
        os.mkdir(cfg['training']['checkpoint_dir'])

    print('Beginning training')
    num_iter = 0
    while num_iter < cfg['training']['iterations']:
        for imgs,labels in tqdm(train_data_loader):
            #delete when adding DDP
            imgs = imgs.to(device)
            labels = labels.to(device)

            num_iter += 1
            if num_iter > cfg['training']['iterations']:
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
            total_self_loss += loss_rec_same.item()

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
            total_gen_loss += loss_rec_swap.item()

            #backpropagate
            total_loss = cfg['loss']['reconstruction_self']['weight']*loss_rec_same + \
                    cfg['loss']['reconstruction_gen']['weight']*loss_rec_swap + \
                    cfg['loss']['matching']['weight']*loss_m
            total_loss.backward()
            #loss_rec_swap.backward()

            #take optimization step
            optimizer.step()

            if num_iter % cfg['training']['log_every'] == 0:
                if local_rank == 0:
                    total_self_loss /= cfg['training']['log_every']
                    total_gen_loss /= cfg['training']['log_every']
                    total_match_loss /= cfg['training']['log_every']
                    logger.add_scalar('Train_loss/self_reconstruction',
                            total_self_loss, num_iter)
                    logger.add_scalar('Train_loss/generative_reconstruction',
                            total_gen_loss, num_iter)
                    logger.add_scalar('Train_loss/feature_matching',
                            total_match_loss, num_iter)
                    total_self_loss = 0
                    total_gen_loss = 0
                    total_match_loss = 0

            if (num_iter+1) % cfg['training']['checkpoint_every'] == 0:
                torch.save(E.state_dict(),
                        os.path.join(cfg['training']['checkpoint_dir'],f'Encoder_{num_iter}.pth'))
                torch.save(D.state_dict(),
                        os.path.join(cfg['training']['checkpoint_dir'],f'Decoder_{num_iter}.pth'))

            if cfg['validation']['do_validation']:
                if (num_iter+1) % cfg['validation']['val_every'] == 0:
                    val_loss = validate(cfg, E, D, val_data_loader,
                            reconstruction_gen_loss, len(val_dataset),
                            device)
                    logger.add_scalar('Validation_loss/generative_reconstruction',
                            val_loss, num_iter)

    torch.save(E.state_dict(),
            os.path.join(cfg['training']['checkpoint_dir'],f'Encoder_final.pth'))
    torch.save(D.state_dict(),
            os.path.join(cfg['training']['checkpoint_dir'],f'Decoder_final.pth'))

def run_training(config_file, local_rank):
    cfg = get_config(config_file)
    if not os.path.exists(cfg['model_dir']):
        os.mkdir(cfg['model_dir'])
    write_config(cfg)

    device, world_size = get_device_information()
    distributed = world_size > 1

    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(cfg['backend'], rank=local_rank,
                world_size=world_size)
    else:
        world_size = 1
        local_rank = 0

    train(cfg, device, world_size, local_rank, distributed)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True, type=str)
    parser.add_argument('--local_rank',default=0,type=int)

    args = parser.parse_args()

    run_training(**vars(args))