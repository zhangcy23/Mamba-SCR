import glob
import shutil

# from itk.itkSegmentationLevelSetImageFilterPython import segmentation_level_set_image_filter
# from osgeo import gdal
import torch
import numpy as np
import tifffile
import cv2
import torch

from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader
import math
import copy
from skimage.metrics import peak_signal_noise_ratio as cal_panr
from skimage.metrics import structural_similarity as cal_ssim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
# 启用异常处理
# gdal.UseExceptions()



def get_files_with_extension(folder_path,extension):
    file_list=[]
    for filename in os.listdir(folder_path):
        if filename.endswith(extension) :
            file_list.append(filename)
    return file_list

class Mydata_train1(Dataset):
    def __init__(self, root_path, s1_name, s2c_name, s2_name):

        with open(s2_name, 'r', encoding='utf-8') as f:
            name = f.readlines()
            all_s2_name = [x.strip() for x in name ]
        with open(s1_name, 'r', encoding='utf-8') as f:
            name = f.readlines()
            all_s1_name = [x.strip() for x in name]

        with open(s2c_name, 'r', encoding='utf-8') as f:
            name = f.readlines()
            all_s2c_name = [x.strip() for x in name ]

        self.all_s1_name = all_s1_name
        self.all_s2_name = all_s2_name
        self.all_s2c_name = all_s2c_name
        self.root_path = root_path
        self.data_len = len(all_s1_name)

    def __len__(self):
        """
        Returns:
            length (int): length of the data
        """

        return self.data_len

    def __getitem__(self, index):  # index为索引

        # print('开始执行')
        ##root_path=''  ##'../../dataset/m1554803/ROIs1868_summer_'
        s2_path = self.root_path + 's2/' + self.all_s2_name[index]
        s2c_path = self.root_path + 's2_cloudy/' + self.all_s2c_name[index]
        s1_path = self.root_path + 's1/' + self.all_s1_name[index]
        # mask_path = self.root_path + 'cmask/' + self.all_s2c_name[index]
        # mask = tifffile.imread(mask_path)
        # mask = np.expand_dims(mask, axis=0)

        cloud = tifffile.imread(s2c_path)

        cloud = (np.transpose(cloud, (2, 0, 1)).clip(0, 10000)) / 10000

        label = tifffile.imread(s2_path)
        label = (np.transpose(label, (2, 0, 1)).clip(0, 10000)) / 10000

        # sar = tifffile.imread(s1_path)
        # sar = (np.transpose(sar, (2, 0, 1)).clip(-32.5, 0)) / -32.5
        sar=tifffile.imread(s1_path)
        sar=(np.transpose(sar,(2,0,1)))
        sar[0]=sar[0].clip(-25.0,0)/-25.0
        sar[1]=sar[1].clip(-32.5,0)/-32.5
        cloud = torch.from_numpy(cloud)
        sar = torch.from_numpy(sar)
        label = torch.from_numpy(label)
        # mask = torch.from_numpy(mask)
        return (cloud, sar, label, self.all_s2_name[index])
    #实现图片类型的转换，定义相关的存储路径

def percent_show(img,pmin,pmax):
    if len(img.shape) ==3:
        n_img=np.zeros(img.shape)
        for i in range(img.shape[0]):
            lower_bound = np.percentile(img[i], pmin)
            upper_bound = np.percentile(img[i], pmax)
            truncated_image = np.clip(img[i], lower_bound, upper_bound)
            n_img[i] = ((truncated_image - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
    if len(img.shape) ==4:
        n_img=np.zeros(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                lower_bound = np.percentile(img[i][j], pmin)
                upper_bound = np.percentile(img[i][j], pmax)
                truncated_image = np.clip(img[i][j], lower_bound, upper_bound)
                n_img[i][j] = ((truncated_image - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
    return n_img
def img_show(img):
    #用于显示归一化后的图片
    gray_new=np.zeros(img.shape)
    for i in range(3):
        # truncated_down = np.percentile(img[i], 0.000000001)
        truncated_down=0
        truncated_up = np.percentile(img[i], 99.999)
        gray_new[i] = (255 / (truncated_up - truncated_down)) * img[i]
    gray_new=np.where(gray_new>255,255,gray_new)
    gray_new=np.where(gray_new<0,0,gray_new)
    return gray_new.astype(np.uint8)
def test_pre1(power_path):
    path1=power_path
    test_data = Mydata_train1('../../dataset/m1554803/','../../dataset/m1554803/data_txt/test.txt')
    test_data_load = DataLoader(dataset=test_data, batch_size=2,shuffle=True, drop_last=False)
    with torch.no_grad():
        mm = 0
        amin = 0.5
        amax = 99.5
        model = torch.load(power_path)

        # test_save_path = path1[:29] + 'ceshi/test/' + path1[35:-3] + '/'
        # if os.path.exists(test_save_path):
        #     shutil.rmtree(test_save_path)
        test_rgb_save_path = path1[:29] + 'ceshi/test/' + path1[35:-3] + '_rgb/'
        if os.path.exists(test_rgb_save_path):
            shutil.rmtree(test_rgb_save_path)
        os.makedirs(test_rgb_save_path)
        # print(test_save_path)
        print(test_rgb_save_path)

        for batch, (img, sar, label, name) in enumerate(test_data_load):
            mm += 1
            img = img.type(torch.FloatTensor).cuda()
            sar = sar.type(torch.FloatTensor).cuda()
            # mask = mask.type(torch.FloatTensor).cuda()
            label = label.type(torch.FloatTensor).cuda()
            outputs = model(img, sar)


            i = int(batch * img.shape[0] / len(test_data) * 100)
            print("\r", end="")
            print("test progress: {}%: ".format(i), "▋" * (batch // 1000), end="")
            sys.stdout.flush()


            outputs=np.array(outputs.cpu())
            outputs=np.where(outputs>1,1,outputs)
            outputs=np.where(outputs<0,0,outputs).astype(np.float64)
            label=label.cpu().numpy().astype(np.float64)

            ##保存示例图片 (输入图像，VV图像,预测图像，标签)


            img = img.cpu().numpy() ##[batch,13,256,256]
            label=label
            outputs = outputs
            sar=sar.cpu().numpy()
            #百分比截断显示

            img=percent_show(img,amin,amax)
            label=percent_show(label,amin,amax)
            outputs = percent_show(outputs,amin,amax)
            sar=sar

            for n in range(img.shape[0]):
                plt.figure(figsize=(60, 15))
                gs = gridspec.GridSpec(1, 4)

                rgb_img=np.stack((img[n][3],img[n][2],img[n][1]),axis=0).astype(np.uint8)
                vv=(sar[n][1])
                rgb_pre=np.stack((outputs[n][3],outputs[n][2],outputs[n][1]),axis=0).astype(np.uint8)
                rgb_label=np.stack((label[n][3],label[n][2],label[n][1]),axis=0).astype(np.uint8)

                # cloud_tu=np.stack((tihuan_images[n][15],tihuan_images[n][14],tihuan_images[n][13]),axis=0)
                plt.rc("font")
                # plt.subplot(gs[0:2,:],title='pre')
                plt.subplot(gs[0,0])
                plt.imshow(np.transpose(rgb_img,axes=(1,2,0)))
                plt.title('input_cloud',x=0.5,y=0.02,c='red',size=50)
                plt.axis('off')
                plt.subplot(gs[0,1])
                plt.imshow(vv)
                plt.title('vv',x=0.5,y=0.02,c='red',size=50)
                plt.axis('off')

                plt.subplot(gs[0,2])
                plt.imshow(np.transpose(rgb_pre,axes=(1,2,0)))
                plt.title('pre',x=0.5,y=0.02,c='red',size=50)
                plt.axis('off')
                plt.subplot(gs[0, 3])
                plt.imshow(np.transpose(rgb_label,axes=(1,2,0)))
                plt.title('target',x=0.5,y=0.02,c='red',size=50)
                plt.axis('off')
                plt.subplots_adjust(hspace=0.4)
                plt.tight_layout()
                index = str(name[n]).find('/')
                if index==-1:
                    print( str(name[n]))
                    continue
                else:
                    fdir = str(name[n]).split('/')[0]+'/'
                    save_dir=test_rgb_save_path+fdir
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    # plt.savefig(save_path+str(n)+'_'+str(psnr)[:5]+'_'+str(ssim)[:5]+'.png', dpi=200)
                    plt.savefig(test_rgb_save_path+str(name[n])[:-4]+'.png', dpi=200)
                plt.close()
                # mm+=1
            # print(mm)


model_path = './history4/2025-01-22-12-37-48/model/*.pt'
all_model_name=glob.glob(model_path)
test_pre1(all_model_name[-1])









