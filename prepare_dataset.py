import os
import math
from PIL import Image
from tqdm import tqdm
import argparse

def process_data(input_dir, output_dir, tile_size=256):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    splits = os.listdir(input_dir)
    for spl in splits:
        print(f'Processing {spl} data...')
        input_split = os.path.join(input_dir, spl)
        output_split = os.path.join(output_dir, spl)
        process_data_split(input_split, output_split, tile_size)

def process_data_split(input_dir, output_dir, tile_size=256):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    sub_dirs = os.listdir(input_dir)
    for sd in tqdm(sub_dirs):
        input_sd = os.path.join(input_dir, sd)
        output_sd = os.path.join(output_dir, sd)
        if not os.path.exists(output_sd):
            os.mkdir(output_sd)
        process_single_image_set(input_sd, output_sd, tile_size)

def process_single_image_set(input_dir, output_dir, tile_size):
    fns = os.listdir(input_dir)
    for f in fns:
        tile_image(input_dir, output_dir, f, tile_size)

def tile_image(input_dir, output_dir, img_filename, tile_size):
    img = Image.open(os.path.join(input_dir,img_filename))
    n_tile_width = list(range(0,math.floor(img.size[0]/tile_size)))
    n_tile_height = list(range(0,math.floor(img.size[1]/tile_size)))
    tile_combinations = [(a,b) for a in n_tile_width for b in n_tile_height]

    tile_counter = 0
    for tile_touple in tile_combinations:
        x_start_point = tile_touple[0]*tile_size
        y_start_point = tile_touple[1]*tile_size

        crop_box = (x_start_point, y_start_point, x_start_point+tile_size, y_start_point+tile_size)
        tile_crop = img.crop(crop_box)

        num_zeros_0 = 5-len(str(tile_touple[0]))
        num_zeros_1 = 5-len(str(tile_touple[1]))
        tile_dir = os.path.join(output_dir,
                f"tile_{'0'*num_zeros_0 + str(tile_touple[0])}_{'0'*num_zeros_1 + str(tile_touple[1])}")
        if not os.path.exists(tile_dir):
            os.mkdir(tile_dir)

        tile_crop.save(os.path.join(tile_dir, img_filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--tile_size',default=256,type=int)

    args = parser.parse_args()

    process_data(**vars(args))
