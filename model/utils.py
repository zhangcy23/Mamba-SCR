#图片预处理
#图片预处理
import copy
import random
import matplotlib.gridspec as gridspec
import torch
import numpy as np
import os
from torch import nn
from skimage.metrics import mean_squared_error as cal_mse
from skimage.metrics import peak_signal_noise_ratio as cal_panr
from skimage.metrics import structural_similarity as cal_ssim
import matplotlib.pyplot as plt
import sys
import errno
from pytorch_wavelets import DWTForward, DWTInverse
class Loss_SAM(nn.Module):
    def __init__(self):
        super(Loss_SAM, self).__init__()
        self.eps = 2.2204e-10

    def forward(self, im1, im2):
        assert im1.shape == im2.shape

        # 检查输入
        if torch.isnan(im1).any() or torch.isinf(im1).any():
            raise ValueError("im1 contains NaN or Inf")
        if torch.isnan(im2).any() or torch.isinf(im2).any():
            raise ValueError("im2 contains NaN or Inf")

        im1 = torch.permute(im1, (1, 0, 2, 3))
        im2 = torch.permute(im2, (1, 0, 2, 3))

        C, B, H, W = im1.shape
        im1 = torch.reshape(im1, (C, B * H * W))
        im2 = torch.reshape(im2, (C, B * H * W))

        core = im1 * im2
        mole = torch.sum(core, dim=0)

        norm_threshold = 1e-10
        im1_norm = torch.sqrt(torch.sum(im1 ** 2, dim=0))
        im2_norm = torch.sqrt(torch.sum(im2 ** 2, dim=0))

        # 避免极小的范数
        im1_norm = torch.where(im1_norm < norm_threshold, torch.tensor(norm_threshold, device=im1.device), im1_norm)
        im2_norm = torch.where(im2_norm < norm_threshold, torch.tensor(norm_threshold, device=im2.device), im2_norm)

        deno = im1_norm * im2_norm

        cos_theta = (mole + self.eps) / deno
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

        # 计算光谱角
        sam = torch.rad2deg(torch.acos(cos_theta))

        return torch.mean(sam)

# 示例用法
# loss_fn = Loss_SAM()
# loss = loss_fn(im1, im2)  # im1 和 im2 是输入的图像张量

def cal_sam(im1, im2):
    eps = 2.2204e-16

    im1=np.transpose(im1,(1,0,2,3))
    im2=np.transpose(im2,(1,0,2,3))
    C,B,H, W = im1.shape
    im1 = np.reshape(im1, (C,B*H * W))
    im2 = np.reshape(im2, (C,B*H * W))
    core = np.multiply(im1, im2)
    mole = np.sum(core, axis=0)
    im1_norm = np.sqrt(np.sum(np.square(im1), axis=0))
    im2_norm = np.sqrt(np.sum(np.square(im2), axis=0))
    deno = np.multiply(im1_norm, im2_norm)
    sam = np.rad2deg(np.arccos(((mole + eps) / (deno + eps)).clip(-0.9999, 0.9999)))
    return np.mean(sam)
def percent_show(img,pmin,pmax):
    if len(img.shape) ==3:
        n_img=np.zeros(img.shape)
        for i in range(img.shape[0]):
            lower_bound = np.percentile(img[i], pmin)
            upper_bound = np.percentile(img[i], pmax)
            truncated_image = np.clip(img, lower_bound, upper_bound)
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


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')




#验证集精度评价


def get_loss_train1(epoch,model, data_train,criterion1, criterion2,save_recall_fig,num):
    """
        Calculate loss over train set
    """
    model.eval()
    ceshi_image=0
    with torch.no_grad():
        mm=0
        l1_loss_sum=0
        all_mse_sum=0
        all_psnr_sum=0
        all_ssim_sum=0
        all_cc_sum=0
        all_sam_sum=0
        l2_loss_sum=0
        save_path = save_recall_fig + str(epoch) + '/'
        if not os.path.exists(save_path) and epoch%5==0:
            os.makedirs(save_path)
        for batch, (img,sar,label) in enumerate(data_train):
            mm+=1

            img = img.type(torch.FloatTensor).cuda()
            sar=sar.type(torch.FloatTensor).cuda()
            label = label.type(torch.FloatTensor).cuda()
            outputs = model(img,sar)

            l1loss= criterion1(outputs, label)
            l1_loss_sum += l1loss.item()

            outputs=np.array(outputs.cpu())*255
            outputs=np.where(outputs>255,255,outputs)
            outputs=np.where(outputs<0,0,outputs).astype(np.float64)
            label=label*255
            label=label.cpu().numpy().astype(np.float64)
            # print([outputs1.max(),mubiao.max()])
            mse_sum=0
            psnr_sum=0
            ssim_sum=0
            cc_sum=0

            i = int(batch * img.shape[0] / num * 100)
            print("\r", end="")
            print("test progress: {}%: ".format(i), "▋" * (batch // 1000), end="")
            sys.stdout.flush()

            ##保存示例图片 (输入图像，VV图像,预测图像，标签)
            if epoch%5==0:
                if ceshi_image<2:
                    ceshi_image=ceshi_image+1

                    img1 = copy.deepcopy((img*255).cpu().numpy()).astype(np.uint8)
                    label1=copy.deepcopy(label).astype(np.uint8)
                    outputs1 = copy.deepcopy(outputs).astype(np.uint8)
                    sar1=copy.deepcopy((sar*255).cpu().numpy()).astype(np.uint8)
                    for n in range(img.shape[0]):
                        plt.figure(figsize=(60, 15))
                        gs = gridspec.GridSpec(1, 4)

                        rgb_img=np.stack((img1[n][3],img1[n][2],img1[n][1]),axis=0)
                        vv=sar1[n][0]
                        rgb_pre=np.stack((outputs1[n][3],outputs1[n][2],outputs1[n][1]),axis=0)
                        rgb_label=np.stack((label1[n][3],label1[n][2],label1[n][1]),axis=0)

                        # cloud_tu=np.stack((tihuan_images[n][15],tihuan_images[n][14],tihuan_images[n][13]),axis=0)
                        plt.rc("font", family='Microsoft YaHei')
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
                        # plt.savefig(save_path+str(n)+'_'+str(psnr)[:5]+'_'+str(ssim)[:5]+'.png', dpi=200)
                        plt.savefig(save_path+str(ceshi_image)+str(n)+'.png', dpi=200)
                        plt.close()
            ##计算评价指标
            for i in range(outputs.shape[0]):
                mse=cal_mse(outputs[i],label[i])
                psnr=cal_panr(outputs[i],label[i],data_range=255)
                ssim=cal_ssim(outputs[i],label[i],data_range=255)
                cc=np.corrcoef(outputs[i].flatten(),label[i].flatten())[0][1]
                mse_sum+=mse
                psnr_sum+=psnr
                ssim_sum+=ssim
                cc_sum+=cc
            sam=cal_sam(outputs,label)
            all_sam_sum=all_sam_sum+sam
            all_ssim_sum=all_ssim_sum+ssim_sum/label.shape[0]
            all_mse_sum=all_mse_sum+mse_sum/label.shape[0]
            all_psnr_sum=all_psnr_sum+psnr_sum/label.shape[0]
            all_cc_sum=all_cc_sum+cc_sum/label.shape[0]
        avg_ssim=all_ssim_sum/mm
        avg_sam = all_sam_sum/ mm
        avg_psnr=all_psnr_sum/mm
        avg_mse=all_mse_sum/mm
        avg_cc=all_cc_sum/mm

        avg_l1loss=l1_loss_sum/mm
        avg_l2loss = l2_loss_sum / mm
        print([epoch,avg_mse,avg_psnr,avg_ssim,avg_cc,avg_sam,avg_l1loss,batch],flush=True)
    return avg_mse,avg_psnr,avg_ssim,avg_cc,avg_sam,avg_l1loss



