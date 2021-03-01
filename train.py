import yaml
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torchvision import transforms
from models import Encoder, Decoder
from datasets import Band2BandDataset
from tqdm import tqdm

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
                        transforms.CenterCrop(cfg['data']['image_size']),
                        transforms.ToTensor()
                        ])
    train_dataset = Band2BandDataset(cfg['data']['train_dir'],trans)
    val_dataset = Band2BandDataset(cfg['data']['val_dir'],trans)

    train_dl = DataLoader(
                    train_dataset,
                    batch_size=cfg['data']['batch_size'],
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

def get_reconstruction_self_loss(cfg):
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
    if cfg['loss']['matching']['type'] == 'L1':
        loss = nn.L1Loss()
    elif cfg['loss']['matching']['type'] == 'L2':
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
    reconstruction_self_loss = get_reconstruction_self_loss(cfg)
    reconstruction_gen_loss = get_reconstruction_gen_loss(cfg)
    match_loss = get_feature_match_loss(cfg)
    logger = SummaryWriter(cfg['log_directory'])
    total_self_loss = 0
    total_gen_loss = 0
    total_match_loss = 0
    if not os.path.exists(cfg['checkpoint_dir']):
        os.mkdir(cfg['checkpoint_dir'])

    #delete when adding DDP
    E = E.to('cuda')
    D = D.to('cuda')

    num_iter = 0
    while num_iter < cfg['iterations']:
        for imgs,labels in tqdm(train_data_loader):
            #delete when adding DDP
            imgs = imgs.to('cuda')
            labels = labels.to('cuda')

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

    args = parser.parse_args()

    train(**vars(args))
