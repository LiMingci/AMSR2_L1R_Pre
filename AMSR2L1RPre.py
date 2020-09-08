import numpy as np
from scipy import interpolate
from scipy import ndimage

import h5py
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import cv2
from nansat import NSR
from nansat.vrt import VRT

# ['Area Mean Height',
# 'Attitude Data',
# 'Brightness Temperature (original,89GHz-A,H)',
# 'Brightness Temperature (original,89GHz-A,V)',
# 'Brightness Temperature (original,89GHz-B,H)',
# 'Brightness Temperature (original,89GHz-B,V)',
# 'Brightness Temperature (res06,10.7GHz,H)',
# 'Brightness Temperature (res06,10.7GHz,V)',
# 'Brightness Temperature (res06,18.7GHz,H)',
# 'Brightness Temperature (res06,18.7GHz,V)',
# 'Brightness Temperature (res06,23.8GHz,H)',
# 'Brightness Temperature (res06,23.8GHz,V)',
# 'Brightness Temperature (res06,36.5GHz,H)',
# 'Brightness Temperature (res06,36.5GHz,V)',
# 'Brightness Temperature (res06,6.9GHz,H)',
# 'Brightness Temperature (res06,6.9GHz,V)',
# 'Brightness Temperature (res06,7.3GHz,H)',
# 'Brightness Temperature (res06,7.3GHz,V)',
# 'Brightness Temperature (res06,89.0GHz,H)',
# 'Brightness Temperature (res06,89.0GHz,V)',
# 'Brightness Temperature (res10,10.7GHz,H)',
# 'Brightness Temperature (res10,10.7GHz,V)',
# 'Brightness Temperature (res10,18.7GHz,H)',
# 'Brightness Temperature (res10,18.7GHz,V)',
# 'Brightness Temperature (res10,23.8GHz,H)',
# 'Brightness Temperature (res10,23.8GHz,V)',
# 'Brightness Temperature (res10,36.5GHz,H)',
# 'Brightness Temperature (res10,36.5GHz,V)',
# 'Brightness Temperature (res10,89.0GHz,H)',
# 'Brightness Temperature (res10,89.0GHz,V)',
# 'Brightness Temperature (res23,18.7GHz,H)',
# 'Brightness Temperature (res23,18.7GHz,V)',
# 'Brightness Temperature (res23,23.8GHz,H)',
# 'Brightness Temperature (res23,23.8GHz,V)',
# 'Brightness Temperature (res23,36.5GHz,H)',
# 'Brightness Temperature (res23,36.5GHz,V)',
# 'Brightness Temperature (res23,89.0GHz,H)',
# 'Brightness Temperature (res23,89.0GHz,V)',
# 'Brightness Temperature (res36,36.5GHz,H)',
# 'Brightness Temperature (res36,36.5GHz,V)',
# 'Brightness Temperature (res36,89.0GHz,H)',
# 'Brightness Temperature (res36,89.0GHz,V)',
# 'Earth Azimuth',
# 'Earth Incidence',
# 'Land_Ocean Flag 6 to 36', 'Land_Ocean Flag 89',
# 'Latitude of Observation Point for 89A',
# 'Latitude of Observation Point for 89B',
# 'Longitude of Observation Point for 89A',
# 'Longitude of Observation Point for 89B',
# 'Navigation Data', 'Pixel Data Quality 6 to 36',
# 'Pixel Data Quality 89',
# 'Position in Orbit',
# 'Scan Data Quality',
# 'Scan Time',
# 'Sun Azimuth',
# 'Sun Elevation']


def get_uint8_image(image, vmin, vmax, pmin, pmax):
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


def unsharp_masking_sharp(img):
    blur_img = cv2.GaussianBlur(img, (0, 0), 9)
    usm = cv2.addWeighted(img, 1.0, blur_img, -0.5, 0)
    return usm


def laplcian_sharp(img):
    kernel = np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]], np.float32)
    lps = cv2.filter2D(img, -1, kernel=kernel)
    return lps


class AMSR2L1RPre(object):
    def __init__(self, filename, mask_res=0.1):
        self.filename = filename
        with h5py.File(filename, 'r') as f:
            self.hr_lat = f['Latitude of Observation Point for 89A'][:]
            self.hr_lon = f['Longitude of Observation Point for 89A'][:]
            cols = self.hr_lat.shape[1]
            self.lr_lat = self.hr_lat[..., 0:cols+1:2]
            self.lr_lon = self.hr_lon[..., 0:cols+1:2]
            self.hr_shape = self.hr_lat.shape
            self.lr_shape = self.lr_lat.shape
            self.bt_89h = f['Brightness Temperature (original,89GHz-A,H)'][:]
            self.bt_89v = f['Brightness Temperature (original,89GHz-A,V)'][:]
            self.bt_36h = f['Brightness Temperature (res36,36.5GHz,H)'][:]
            self.bt_36v = f['Brightness Temperature (res36,36.5GHz,V)'][:]

        self.mask_res = mask_res
        self.min_lon, self.min_lat, self.mask = self.get_imgage_mask(mask_res)


    def proj_to_wgs84_nsidc_sea_ice_stere_n(self, x, y, inverse=False):
        '''
        :param x: 1D array -->lon
        :param y: 1D array -->lat
        :param iinverse:
        :return:
        '''
        # WGS84
        srs_src = NSR(4326)
        # WGS 84 / NSIDC Sea Ice Polar Stereographic
        srs_dst = NSR(3413)
        src_points = (x, y)
        if inverse:
            dst_point = VRT.transform_coordinates(srs_dst, src_points, srs_src)
        else:
            dst_point = VRT.transform_coordinates(srs_src, src_points, srs_dst)
        return dst_point


    def get_imgage_mask(self, mask_res=0.1):
        # 1. 按照经纬度计算输入影像的包围盒
        min_lon = self.hr_lon.min()
        max_lon = self.hr_lon.max()
        min_lat = self.hr_lat.min()
        max_lat = self.hr_lat.max()

        # 2. 按照指定经纬度的分辨率创建mask
        mask_width = int(np.ceil((max_lon - min_lon) / mask_res))
        mask_height = int(np.ceil((max_lat - min_lat) / mask_res))
        mask = np.full((mask_height, mask_width), False, dtype=np.bool)

        # 3. 填充mask
        img_lon = self.hr_lon.copy()
        img_lat = self.hr_lat.copy()
        img_lon -= min_lon
        img_lat -= min_lat
        img_lon /= mask_res
        img_lat /= mask_res
        img_lon_idx = img_lon.astype(np.int)
        img_lat_idx = img_lat.astype(np.int)
        mask[img_lat_idx.flatten(), img_lon_idx.flatten()] = True
        mask = ndimage.binary_dilation(mask, np.ones((9, 9)))
        mask = ndimage.binary_erosion(mask, np.ones((11, 11)))
        return min_lon, min_lat, mask


    def reproj_roi(self, out_path, gsd, lonw, lone, lats, latn, res='H'):
        '''

        :param out_path:
        :param gsd:
        :param lonw:
        :param lone:
        :param lats:
        :param latn:
        :return:
        '''
        corner_lon = [lonw, lone, lone, lonw]
        corner_lat = [lats, lats, latn, latn]
        corner_xy = self.proj_to_wgs84_nsidc_sea_ice_stere_n(corner_lon, corner_lat)
        min_x = corner_xy[0].min()
        max_x = corner_xy[0].max()
        min_y = corner_xy[1].min()
        max_y = corner_xy[1].max()
        img_width = int(np.ceil((max_x - min_x) / gsd))
        img_height = int(np.ceil((max_y - min_y) / gsd))
        img_pixel_crd_x = np.linspace(min_x, max_x, img_width)
        img_pixel_crd_y = np.linspace(max_y, min_y, img_height)
        grid_crd_x, grid_crd_y = np.meshgrid(img_pixel_crd_x, img_pixel_crd_y)
        grid_lonlat = self.proj_to_wgs84_nsidc_sea_ice_stere_n(grid_crd_x.flatten(), grid_crd_y.flatten(), True)
        grid_lon = grid_lonlat[0]
        grid_lat = grid_lonlat[1]
        grid_lon_idx = (grid_lon - self.min_lon) / self.mask_res
        grid_lat_idx = (grid_lat - self.min_lat) / self.mask_res
        grid_lon_idx = grid_lon_idx.astype(np.int)
        grid_lat_idx = grid_lat_idx.astype(np.int)
        gpi = self.mask[grid_lat_idx, grid_lon_idx]
        gpi = gpi.reshape(grid_crd_x.shape)

        if res == 'H':
            hr_xy = self.proj_to_wgs84_nsidc_sea_ice_stere_n(self.hr_lon.flatten(), self.hr_lat.flatten())
            hr_x = hr_xy[0].reshape(self.hr_shape)
            hr_y = hr_xy[1].reshape(self.hr_shape)
            hr_pts = np.asarray(list(zip(hr_x.flatten(), hr_y.flatten())))
            hr_value = self.bt_89h.flatten()
            dst_pts = np.asarray(list(zip(grid_crd_x.flatten(), grid_crd_y.flatten())))
            dst_value = interpolate.griddata(hr_pts, hr_value, dst_pts, method='linear', rescale=True)
            dst_value = dst_value.reshape(grid_crd_x.shape)

        else:
            lr_xy = self.proj_to_wgs84_nsidc_sea_ice_stere_n(self.lr_lon.flatten(), self.lr_lat.flatten())
            lr_x = lr_xy[0].reshape(self.lr_shape)
            lr_y = lr_xy[1].reshape(self.lr_shape)
            lr_pts = np.asarray(list(zip(lr_x.flatten(), lr_y.flatten())))
            lr_value = self.bt_36h.flatten()
            dst_pts = np.asarray(list(zip(grid_crd_x.flatten(), grid_crd_y.flatten())))
            dst_value = interpolate.griddata(lr_pts, lr_value, dst_pts, method='cubic', rescale=True)
            dst_value = dst_value.reshape(grid_crd_x.shape)

        dst_value[~gpi] = np.nan
        file_format = 'GTiff'
        driver = gdal.GetDriverByName(file_format)
        dst_ds = driver.Create(out_path, xsize=img_width, ysize=img_height, bands=1, eType=gdal.GDT_Byte)
        srs = NSR(3413)
        dst_ds.SetProjection(srs.ExportToWkt())
        dst_ds.SetGeoTransform([min_x, gsd, 0, max_y, 0, -gsd])
        img_uint8 = get_uint8_image(dst_value, None, None, 1, 99)
        # img_uint8 = unsharp_masking_sharp(img_uint8)
        # img_uint8 = laplcian_sharp(img_uint8)
        dst_ds.WriteRaster(0, 0, img_width, img_height, img_uint8.tostring())
        dst_ds = None
        # return dst_value

