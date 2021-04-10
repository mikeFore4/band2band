import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from glob import glob

def read_image(fn):
    img = Image.open(fn)
    img = np.asarray(img)

    return img

def get_stats(img):
    stats = {
            'min': img.min(),
            'max': img.max(),
            'mean': img.mean(),
            '.25': np.quantile(img, .25),
            '.5': np.quantile(img, .5),
            '.75': np.quantile(img, .75),
            'std': np.sqrt(img.var())
            }

    return stats

def process_files(data_dir, output_csv):
    gl_path = os.path.join(data_dir, '**')
    files = glob(gl_path, recursive=True)
    files = [f for f in files if f.split('.')[-1] =='tif']
    all_stats = []
    for full_path in tqdm(files):
        fn = full_path.split('/')[-1]
        img = read_image(full_path)
        stats = get_stats(img)
        stats['band'] = int(fn.split('.')[0])
        all_stats.append(stats)

    df = pd.DataFrame.from_dict(all_stats)
    df.to_csv(output_csv, index=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--output-csv',type=str)

    args = parser.parse_args()

    process_files(**vars(args))
