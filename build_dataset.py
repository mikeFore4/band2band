import os
from itertools import permutations
import argparse
import csv
from tqdm import tqdm

def create_pair_csv(data_dir, output_path):
    #extract all directories for common image filename stems
    image_groups = [os.path.join(data_dir,x) for x in os.listdir(data_dir)]

    #extract every tile directory
    tile_dirs = []
    for ig in image_groups:
        tile_dirs.extend([os.path.join(ig,x) for x in os.listdir(ig)])

    #get all pairwise combinations of tiles of the same image
    pairs = []
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        for d in tqdm(tile_dirs):
            fns = [os.path.join(d,x) for x in os.listdir(d)]
            perms = permutations(fns,2)
            for p in perms:
                writer.writerow(p)

def process_dir(data_dir):
    subsets = os.listdir(data_dir)

    for sub in subsets:
        create_pair_csv(os.path.join(data_dir, sub),
                os.path.join(data_dir,f'{sub}_pairs.csv'))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=str)

    args = parser.parse_args()

    process_dir(**vars(args))
