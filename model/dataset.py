import copy
#相关配置
import glob
from torch.utils.data import DataLoader
import tifffile
# from osgeo import gdal
import torch

from torch.utils.data import Dataset
import os
import numpy as np
from torchvision import transforms
import cv2

# def readTif(fileName):
#     dataset = gdal.Open(fileName)
#     if dataset == None:
#         print(fileName+"文件无法打开")
#         return
#     im_width = dataset.RasterXSize #栅格矩阵的列数
#     im_height = dataset.RasterYSize #栅格矩阵的行数
#     im_bands = dataset.RasterCount #波段数
#     im_data = dataset.ReadAsArray(0,0,im_width,im_height)#获取数据
#     im_geotrans = dataset.GetGeoTransform()#获取仿射矩阵信息
#     im_proj = dataset.GetProjection()#获取投影信息
#
#     # im_blueBand = im_data[0, 0:im_height, 0:im_width]  # 获取蓝波段
#     # im_greenBand = im_data[1, 0:im_height, 0:im_width]  # 获取绿波段
#     # im_redBand = im_data[2, 0:im_height, 0:im_width]  # 获取红波段
#     # im_nirBand = im_data[3, 0:im_height, 0:im_width]  # 获取近红外波段
#
#     # return im_data, im_proj,im_geotrans,im_bands
#     return im_data
# def writeTiff(im_data,im_width,im_height,im_bands,im_geotrans,im_proj,path):
#     if 'int8' in im_data.dtype.name:
#         datatype = gdal.GDT_Byte
#     elif 'int16' in im_data.dtype.name:
#         datatype = gdal.GDT_UInt16
#     else:
#         datatype = gdal.GDT_Float32
#
#     if len(im_data.shape) == 3:
#         im_bands, im_height, im_width = im_data.shape
#     elif len(im_data.shape) == 2:
#         im_data = np.array([im_data])
#     else:
#         im_bands, (im_height, im_width) = 1,im_data.shape
#         #创建文件
#     driver = gdal.GetDriverByName("GTiff")
#     dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
#     if(dataset!= None):
#         dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
#         dataset.SetProjection(im_proj) #写入投影
#     for i in range(im_bands):
#         dataset.GetRasterBand(i+1).WriteArray(im_data[i])
#     del dataset

def get_files_with_extension(folder_path,extension):
    file_list=[]
    for filename in os.listdir(folder_path):
        if filename.endswith(extension) :
            file_list.append(filename)
    return file_list

class Mydata_train(Dataset):
    def __init__(self,root_path,s1_name,s2c_name,s2_name):
        with open(s1_name,'r',encoding='utf-8') as f:
            name=f.readlines()
            all_s1_name = list(map(lambda x: x.strip(),name))[:200]
        with open(s2c_name,'r',encoding='utf-8') as f:
            name=f.readlines()
            all_s2c_name = list(map(lambda x: x.strip(),name))[:200]
        with open(s2_name,'r',encoding='utf-8') as f:
            name=f.readlines()
            all_s2_name = list(map(lambda x: x.strip(),name))[:200]
        self.all_s1_name=all_s1_name
        self.all_s2_name=all_s2_name
        self.all_s2c_name=all_s2c_name
        self.root_path=root_path
        self.data_len=len(all_s1_name)
    def __len__(self):
        """
        Returns:
            length (int): length of the data
        """

        return self.data_len

    def __getitem__(self, index):  #index为索引

        # print('开始执行')

        ##root_path=''  ##'../../dataset/m1554803/ROIs1868_summer_'
        s2_path=self.root_path+'s2/'+self.all_s2_name[index]
        s2c_path=self.root_path+'s2_cloudy/'+self.all_s2c_name[index]
        s1_path=self.root_path+'s1/'+self.all_s1_name[index]

        cloud=tifffile.imread(s2c_path)

        cloud=np.transpose(cloud,(2,0,1))/10000

        label=tifffile.imread(s2_path)
        label=np.transpose(label,(2,0,1))/10000

        vv_vh=tifffile.imread(s1_path)
        vv_vh=np.transpose(vv_vh,(2,0,1))/-32.5
        # print([cloud.shape,vv_vh.shape])
        cloud_sar=np.concatenate((cloud,vv_vh),axis=0)


        cloud_sar=torch.from_numpy(cloud_sar)
        label=torch.from_numpy(label)


        return (cloud_sar,label)
class Mydata_train1(Dataset):
    def __init__(self,root_path,s1_name,s2c_name='',s2_name=''):

        with open(s1_name,'r',encoding='utf-8') as f:
            name=f.readlines()
            all_s1_name = list(map(lambda x: x.strip(),name))
        with open(s2c_name,'r',encoding='utf-8') as f:
            name=f.readlines()
            all_s2c_name = list(map(lambda x: x.strip(),name))
        with open(s2_name,'r',encoding='utf-8') as f:
            name=f.readlines()
            all_s2_name = list(map(lambda x: x.strip(),name))
        self.all_s1_name=all_s1_name
        self.all_s2_name=all_s2_name
        self.all_s2c_name=all_s2c_name
        self.root_path=root_path
        self.data_len=len(all_s1_name)
    def __len__(self):
        """
        Returns:
            length (int): length of the data
        """

        return self.data_len

    def __getitem__(self, index):  #index为索引

        # print('开始执行')

        ##root_path=''  ##'../../dataset/m1554803/ROIs1868_summer_'
        s2_path=self.root_path+'s2/'+self.all_s2_name[index]
        s2c_path=self.root_path+'s2_cloudy/'+self.all_s2c_name[index]
        s1_path=self.root_path+'s1/'+self.all_s1_name[index]
        # mask_path=self.root_path+'cmask/'+self.all_s2c_name[index]
        # # print(self.all_s2_name[index])
        # mask=tifffile.imread(mask_path)
        # mask=np.expand_dims(mask,axis=0)

        cloud=tifffile.imread(s2c_path)

        cloud=(np.transpose(cloud,(2,0,1)).clip(0,10000))/10000

        label=tifffile.imread(s2_path)
        label=(np.transpose(label,(2,0,1)).clip(0,10000))/10000

        sar=tifffile.imread(s1_path)
        sar=(np.transpose(sar,(2,0,1)))
        sar[0]=sar[0].clip(-25.0,0)/-25.0
        sar[1]=sar[1].clip(-32.5,0)/-32.5


        cloud=torch.from_numpy(cloud)
        sar=torch.from_numpy(sar)
        label=torch.from_numpy(label)
        # mask=torch.from_numpy(mask)

        return (cloud,sar,label)
