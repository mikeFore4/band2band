import torch
import torch.distributed as dist
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from utils import get_encoder, get_decoder, get_config, get_device_information
from tqdm import tqdm
import argparse
import os

def get_label_tensor(input_label, output_label):
    tens = torch.tensor([input_label, output_label])

    return tens

def load_image(input_path):
    trans = transforms.Compose([
                        transforms.ToTensor()
                        ])
    img = Image.open(input_path)
    img = trans(img)

    img = img.float()
    #this is hard coded for now but should be a changeable parameter
    img /= 5000
    img = torch.clip(img,0,1)
    img = img.unsqueeze(0)

    return img

def write_image(img, path):
    img = img.squeeze(0)
    img *= 5000
    save_image(img, path)

def infer(cfg, device, world_size, local_rank, distributed, input_dir,
        output_dir, input_class, output_class):
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

    with torch.no_grad():
        for fn in tqdm(os.listdir(input_dir)):

            input_path = os.path.join(input_dir, fn)
            output_path = os.path.join(output_dir, fn)
            img = load_image(input_path)
            input_label = torch.tensor([input_class]).view(-1,1)
            output_label = torch.tensor([output_class]).view(-1,1)

            embs = E(img, input_label)
            output_swapped = D(embs, output_label)

            write_image(output_swapped, output_path)

def run_inference(config_file, input_dir, output_dir, input_class, output_class, local_rank):
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

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    infer(cfg, device, world_size, local_rank, distributed, input_dir,
            output_dir, input_class, output_class)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',type=str)
    parser.add_argument('--output-dir',type=str)
    parser.add_argument('--input-class',type=int)
    parser.add_argument('--output-class',type=int)
    parser.add_argument('--config-file',default='config.yaml',type=str)
    parser.add_argument('--local_rank', default=0, type=int)

    args = parser.parse_args()

    run_inference(**vars(args))

