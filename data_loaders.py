#%%
import numpy as np
import rasterio
#%%
def return_band_array(img_name, band_norm_dict):
    value_array = None

    for band in band_norm_dict.keys():
        tif_name = ''.join([img_name,band,'.tif'])
        file_path = '/home/gecorbet1015/attached_drive/pix2pix/HLS_Raw/'+tif_name
        img = rasterio.open(file_path).read(1)
        #img = img.flatten()
        img = img/band_norm_dict[band]['max']
        np.clip(img,0,1,img)
        #img = np.int_(img*255)

        if value_array is None:
            value_array=[img]
        else:
            value_array = np.append(value_array,[img],axis=0)

    rgb = np.dstack((value_array[0],value_array[1],value_array[2]))
    rgb_y = value_array[3]

    return([rgb,rgb_y])
#%%

