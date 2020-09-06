import os
import numpy as np
import matplotlib.pyplot as plt
import glob

import AMSR2L1RPre

def get_uint8_image(image, vmin, vmax, pmin, pmax):
    ''' Scale image from float (or any) input array to uint8
    Parameters
    ----------
        image : 2D matrix
        vmin : float - minimum value
        vmax : float - maximum value
    Returns
    -------
        2D matrix
    '''
    if vmin is None:
        vmin = np.nanpercentile(image, pmin)
    if vmax is None:
        vmax = np.nanpercentile(image, pmax)
    # redistribute into range [0,255]
    uint8Image = 1 + 254 * (image - vmin) / (vmax - vmin)
    uint8Image[uint8Image < 1] = 1
    uint8Image[uint8Image > 255] = 255
    uint8Image[~np.isfinite(image)] = 0

    return uint8Image.astype('uint8')


if __name__ == '__main__':

    # file_name = 'E:/SeaIceDrift/data/AMSR2/L1R/GW1AM2_201805150109_193D_L1SGRTBR_2220220.h5'
    # out_path = 'E:/SeaIceDrift/data/AMSR2/L1R/GW1AM2_201805150109_193D_L1SGRTBR_2220220_Fram_L_Mask.tiff'
    # amsr2_pre = AMSR2L1RPre.AMSR2L1RPre(file_name)
    # amsr2_pre.reproj_roi(out_path, 500.0, -25.0, 25.0, 76.0, 84.0, res='L')
    # # img_value = get_uint8_image(img_value, None, None, 10, 99)
    # # out_path = 'fram.png'
    # # plt.imsave(out_path, img_value, cmap='gray', origin='lower')
    # # plt.imsave(out_path, img_value, cmap='gray')
    # plt.imsave('mask_.png', amsr2_pre.mask)


    file_dir = 'E:/SeaIceDrift/data/AMSR2/L1R'
    file_list = glob.glob('%s/GW1AM2*.h5' % file_dir)
    for file_path in file_list:
        amsr2_pre = AMSR2L1RPre.AMSR2L1RPre(file_path)
        out_path = '%s_Fram_H.tiff' % os.path.splitext(file_path)[0]
        amsr2_pre.reproj_roi(out_path, 1000.0, -25.0, 25.0, 76.0, 84.0, res='H')
        print(os.path.basename(out_path))

    pass