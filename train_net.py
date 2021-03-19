import os
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm
import utils
from azureml_env_adapter import set_environment_variables

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
    E = utils.get_encoder(cfg)
    E = E.to(device)
    if distributed:
        E = nn.parallel.DistributedDataParallel(E, device_ids=[local_rank],
                output_device=local_rank)

    #initialize decoder
    print('Initializing Decoder')
    D = utils.get_decoder(cfg)
    D = D.to(device)
    if distributed:
        D = nn.parallel.DistributedDataParallel(D, device_ids=[local_rank],
                output_device=local_rank)

    print('Building dataloaders')
    train_dataset, train_data_loader = utils.get_train_loader(cfg, world_size, distributed)
    if cfg['validation']['do_validation']:
        val_dataset, val_data_loader = utils.get_val_loader(cfg, world_size,
                distributed)
    optimizer = utils.get_optimizer(cfg, E, D)
    reconstruction_self_loss = utils.get_reconstruction_self_loss(cfg)
    reconstruction_gen_loss = utils.get_reconstruction_gen_loss(cfg)
    match_loss = utils.get_feature_match_loss(cfg)
    if local_rank == 0:
        logger = utils.get_logger(cfg)
    total_self_loss = 0
    total_gen_loss = 0
    total_match_loss = 0
    os.makedirs(cfg['training']['checkpoint_dir'], exist_ok = True)

    print('Beginning training')
    t_dl = iter(train_data_loader)
    train_ds_len = len(train_dataset)
    for num_iter in tqdm(range(cfg['training']['iterations'])):
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
                        reconstruction_gen_loss, len(val_dataset),
                        device)
                if local_rank == 0:
                    utils.log_metric(logger, cfg, 'Validation_loss/generative_reconstruction',
                            val_loss, num_iter)
                    if (num_iter+1) == cfg['validation']['val_every']:
                        best_val = val_loss
                    if val_loss < best_val:
                        best_val = val_loss
                        torch.save(E.state_dict(),
                                os.path.join(cfg['training']['checkpoint_dir'],f'Encoder_{num_iter}.pth'))
                        torch.save(D.state_dict(),
                                os.path.join(cfg['training']['checkpoint_dir'],f'Decoder_{num_iter}.pth'))


    torch.save(E.state_dict(),
            os.path.join(cfg['training']['checkpoint_dir'],f'Encoder_final.pth'))
    torch.save(D.state_dict(),
            os.path.join(cfg['training']['checkpoint_dir'],f'Decoder_final.pth'))

def run_training(cfg, local_rank):
    os.makedirs(cfg['model_dir'], exist_ok = True)
    utils.write_config(cfg)
    set_environment_variables()

    device, world_size = utils.get_device_information()
    distributed = world_size > 1

    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(cfg['backend'], rank=local_rank,
                world_size=world_size, init_method='env://')
    else:
        world_size = 1
        local_rank = 0

    train(cfg, device, world_size, local_rank, distributed)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', default='config.yaml', type=str)
    parser.add_argument('--local_rank',default=0,type=int)
    parser.add_argument('--data-path',type=str)

    args = parser.parse_args()

    cfg = utils.get_config(args.config_file)
    if args.data_path is not None:
        cfg['data']['dir_head'] = args.data_path

    run_training(cfg, args.local_rank)
