import pyyaml
from models import Encoder, Decoder

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
    raise NotImplementedError

def get_optimizer(cfg, E, D):
    raise NotImplementedError

def get_reconstruction_loss(cfg):
    if cfg['loss']['reconstruction'] == 'L1':
        raise NotImplementedError
    elif cfg['loss']['reconstruction'] == 'L2':
        raise NotImplementedError
    elif cfg['loss']['reconstruction'] == 'Perceptual':
        raise NotImplementedError
    else:
        raise NotImplementedError

def get_embedding_match_loss(cfg):
    if cfg['loss']['matching'] == 'L1':
        raise NotImplementedError
    elif cfg['loss']['matching'] == 'L2':
        raise NotImplementedError
    else:
        raise NotImplementedError

def train(config_file):
    cfg = get_config(config_file)
    E = get_encoder(cfg)
    D = get_decoder(cfg)
    train_data_loader, valid_data_loader = get_data_loaders(cfg)
    optimizer = get_optimizer(cfg, E, D)
    reconstruction_loss = get_reconstruction_loss(cfg)

    for i in range(cfg['num_epochs']):
        #do stuff

