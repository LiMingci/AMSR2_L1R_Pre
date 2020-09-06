import os
import numpy as np
import matplotlib.pyplot as plt
import glob

import AMSR2L1RPre


if __name__ == '__main__':

    file_name = './data/GW1AM2_201312290732_022D_L1SGRTBR_2220220.h5'
    out_path = './data/GW1AM2_201312290732_022D_L1SGRTBR_2220220_Fram_L.tiff'
    amsr2_pre = AMSR2L1RPre.AMSR2L1RPre(file_name)
    amsr2_pre.reproj_roi(out_path, 500.0, -25.0, 25.0, 76.0, 84.0, res='L')

    pass