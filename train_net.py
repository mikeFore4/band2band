import os
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
#from tqdm import tqdm
import utils
from azureml_env_adapter import set_environment_variables
import mlflow

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

def train(cfg, world_size, local_rank, distributed):
    """
    Performs training of Encoder and Decoder modules

    Inputs
    ------
    config_file : str
        path to location of yaml config file
    local_rank : int
        rank number of master process for distributed training
    """

    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(local_rank)
    #initialize encoder
    print('Initializing Encoder')
    E = utils.get_encoder(cfg)
    E = E.cuda()
    if distributed:
        E = nn.parallel.DistributedDataParallel(E, device_ids=[local_rank],
                output_device=local_rank)

    #initialize decoder
    print('Initializing Decoder')
    D = utils.get_decoder(cfg)
    D = D.cuda()
    if distributed:
        D = nn.parallel.DistributedDataParallel(D, device_ids=[local_rank],
                output_device=local_rank)

    print('Building dataloaders')
    train_dataset, train_data_loader = utils.get_train_loader(cfg, world_size, distributed)
    if cfg['validation']['do_validation']:
        val_dataset, val_data_loader = utils.get_val_loader(cfg, world_size,
                distributed)
    optimizer = utils.get_optimizer(cfg, E, D)
    reconstruction_self_loss = utils.get_loss(cfg['loss']['reconstruction_self']['type'])
    reconstruction_gen_loss = utils.get_loss(cfg['loss']['reconstruction_gen']['type'])
    match_loss = utils.get_loss(cfg['loss']['matching']['type'])
    val_loss = utils.get_loss(cfg['loss']['validate'])
    if local_rank == 0:
        logger = utils.get_logger(cfg)
    total_self_loss = 0
    total_gen_loss = 0
    total_match_loss = 0
    os.makedirs(cfg['training']['checkpoint_dir'], exist_ok = True)

    print('Beginning training')
    t_dl = iter(train_data_loader)
    train_ds_len = len(train_dataset)
    #for num_iter in tqdm(range(cfg['training']['iterations'])):
    for num_iter in range(cfg['training']['iterations']):
        if num_iter % train_ds_len == 0:
            t_dl = iter(train_data_loader)
        imgs, labels = t_dl.next()
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

        if num_iter % cfg['training']['logging']['log_every'] == 0:
            if local_rank == 0:
                total_self_loss /= cfg['training']['logging']['log_every']
                total_gen_loss /= cfg['training']['logging']['log_every']
                total_match_loss /= cfg['training']['logging']['log_every']
                utils.log_metric(logger, cfg, 'Train_loss/self_reconstruction',
                        total_self_loss, num_iter)
                utils.log_metric(logger, cfg, 'Train_loss/generative_reconstruction',
                        total_gen_loss, num_iter)
                utils.log_metric(logger, cfg, 'Train_loss/feature_matching',
                        total_match_loss, num_iter)
                total_self_loss = 0
                total_gen_loss = 0
                total_match_loss = 0
        if cfg['validation']['do_validation']:
            if (num_iter+1) % cfg['validation']['val_every'] == 0:
                val_loss = validate(cfg, E, D, val_data_loader,
                        val_loss, len(val_dataset),
                        device)
                if local_rank == 0:
                    utils.log_metric(logger, cfg, 'Validation_loss/generative_reconstruction',
                            val_loss, num_iter)
                    if (num_iter+1) == cfg['validation']['val_every']:
                        best_val = val_loss
                    if val_loss < best_val:
                        best_val = val_loss
                        torch.save(E.state_dict(),
                                os.path.join(cfg['training']['checkpoint_dir'],f'Encoder_best.pth'))
                        torch.save(D.state_dict(),
                                os.path.join(cfg['training']['checkpoint_dir'],f'Decoder_best.pth'))


    torch.save(E.state_dict(),
            os.path.join(cfg['training']['checkpoint_dir'],f'Encoder_final.pth'))
    torch.save(D.state_dict(),
            os.path.join(cfg['training']['checkpoint_dir'],f'Decoder_final.pth'))

def run_training(cfg, local_rank):
    os.makedirs(cfg['model_dir'], exist_ok = True)
    utils.write_config(cfg)

    if 'WORLD_SIZE' in os.environ.keys():
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        _, world_size = utils.get_device_information()

    distributed = world_size > 1

    if distributed:
        #torch.cuda.set_device(local_rank)
        dist.init_process_group(cfg['backend'], init_method='env://')
    else:
        world_size = 1
        local_rank = 0

    if cfg['training']['logging']['logger'] == 'aml':
        with mlflow.start_run():
            train(cfg, world_size, local_rank, distributed)
    else:
        train(cfg, world_size, local_rank, distributed)

def process_args(args):
    cfg = utils.get_config(args.config_file)

    if args.data_path is not None:
        cfg['data']['dir_head'] = args.data_path
    if args.batch_size is not None:
        cfg['data']['batch_size'] = args.batch_size
    if args.matching_weight is not None:
        cfg['loss']['matching']['weight'] = args.matching_weight
    if args.rec_gen_weight is not None:
        cfg['loss']['reconstruction_gen']['weight'] = args.rec_gen_weight
    if args.rec_self_weight is not None:
        cfg['loss']['reconstruction_self']['weight'] = args.rec_self_weight
    if args.const_blocks is not None:
        cfg['model']['const_blocks_per'] = args.const_blocks
    if args.up_down_blocks is not None:
        cfg['model']['up_down_blocks'] = args.up_down_blocks
    if args.const_mult is not None:
        cfg['model']['const_channel_multiplier'] = args.const_mult
    if args.up_down_multiplier is not None:
        cfg['model']['up_down_channel_multiplier'] = args.up_down_multiplier
    if args.pooling_factor is not None:
        cfg['model']['scale_pooling_factor'] = args.pooling_factor
    if args.optimizer is not None:
        cfg['training']['optimizer']['algorithm'] = args.optimizer
    if args.learning_rate is not None:
        cfg['training']['optimizer']['learning_rate'] = args.learning_rate
    if args.momentum is not None:
        cfg['training']['optimizer']['momentum'] = args.momentum

    cfg['model']['encoder']['down_blocks'] = cfg['model']['up_down_blocks']
    cfg['model']['decoder']['up_blocks'] = cfg['model']['up_down_blocks']
    cfg['model']['encoder']['const_channel_multiplier'] = cfg['model']['const_channel_multiplier']
    cfg['model']['decoder']['const_channel_divisor'] = cfg['model']['const_channel_multiplier']
    cfg['model']['encoder']['down_channel_multiplier'] = cfg['model']['up_down_channel_multiplier']
    cfg['model']['decoder']['up_channel_divisor'] = cfg['model']['up_down_channel_multiplier']
    cfg['model']['encoder']['const_blocks'] = cfg['model']['const_blocks_per']
    cfg['model']['decoder']['const_blocks'] = cfg['model']['const_blocks_per']
    cfg['model']['encoder']['pooling_factor'] = cfg['model']['scale_pooling_factor']
    cfg['model']['decoder']['scale_factor'] = cfg['model']['scale_pooling_factor']

    # compute input channels for decoder
    out_channels = cfg['model']['encoder']['first_out_channels']*\
            (cfg['model']['encoder']['down_channel_multiplier']**\
            (cfg['model']['encoder']['down_blocks']-1))
    out_channels = out_channels *\
            (cfg['model']['encoder']['const_channel_multiplier']**\
            (cfg['model']['encoder']['const_blocks']))

    cfg['model']['decoder']['input_channels'] = out_channels

    return cfg

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', default='config.yaml', type=str)
    parser.add_argument('--local_rank',default=0,type=int)
    parser.add_argument('--data_path',type=str)
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--matching_weight',type=float)
    parser.add_argument('--rec_gen_weight',type=float)
    parser.add_argument('--rec_self_weight',type=float)
    parser.add_argument('--const_blocks',type=int)
    parser.add_argument('--up_down_blocks',type=int)
    parser.add_argument('--pooling_factor',type=int)
    parser.add_argument('--up_down_multiplier',type=int)
    parser.add_argument('--const_mult',type=int)
    parser.add_argument('--optimizer',type=str)
    parser.add_argument('--learning_rate',type=float)
    parser.add_argument('--momentum',type=float)

    args = parser.parse_args()
    cfg = process_args(args)

    run_training(cfg, args.local_rank)
