#训练函数
import shutil
import time


# from Unet import UNet
# from mamba_dwt_new import mamba_scr
# from mamba_0111 import mamba_scr
from mamba_0113 import mamba_scr
from torch import nn
from dataset import *
import xlwt
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from utils import train_model,get_loss_train,Loss_SAM,get_loss_train1

# from mamba_cr import mamba_cr
# from 测试集预测 import test_pre

# from prefetch_generator import BackgroundGenerator





if __name__ == '__main__':
    #设置Tensorboard
    #创建一个tensorboard对象
    besti=0 #最优的训练次数
    modelname = 'mamba-scr'
    # modelname = 'conv232'
    # modelname = 'conv264'
    nowtime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    writer_val = SummaryWriter("history4/" + str(nowtime) + '/' + modelname + '/val')
    save_model='history4/'+str(nowtime)+'/model/'
    if os.path.exists(save_model):
        shutil.rmtree(save_model)
    os.makedirs(save_model)
    path1='../../dataset/m1554803/data_txt/train_summmer.txt'
    # train_data = Mydata_train1('../../dataset/m1554803/ROIs1868_summer_','../../dataset/m1554803/data_txt/summer/train_s1_summer.txt',
    #                           '../../dataset/m1554803/data_txt/summer/train_cloud_summer.txt','../../dataset/m1554803/data_txt/summer/train_target_summer.txt')
    # val_data = Mydata_train1('../../dataset/m1554803/ROIs1868_summer_','../../dataset/m1554803/data_txt/summer/test_s1_summer.txt',
    #                           '../../dataset/m1554803/data_txt/summer/test_cloud_summer.txt','../../dataset/m1554803/data_txt/summer/test_target_summer.txt')
    train_data = Mydata_train1('../../dataset/m1554803/','../../dataset/m1554803/train.txt')
    val_data = Mydata_train1('../../dataset/m1554803/','../../dataset/m1554803/val.txt')
    print([len(train_data),len(val_data)],flush=True)
    train_data_load = DataLoader(dataset=train_data, batch_size=2, shuffle=True, drop_last=True)
    val_data_load = DataLoader(dataset=val_data, batch_size=2,shuffle=True,drop_last=True)
    #model=UNet(13,12)
    # model=UNet(14,12)
    model=UNet(15,13)
    model=torch.load(model_path).cuda()
    model=mamba_scr(dim=64,depth=[2,2,2,2])
    # model=model.cuda()

    total_num=sum(p.numel() for p in model.parameters())
    print(total_num)
    # model=model.cuda()

    epoch = 10000

    # criterion1=nn.MSELoss().cuda()
    criterion1=nn.L1Loss().cuda()
    criterion2 = Loss_SAM().cuda()

    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001, betas=(0.5, 0.999))

    save_train_recall_fig = 'history4/' + str(nowtime) + '/trainrecallfig/'
    save_test_recall_fig = 'history4/' + str(nowtime) + '/testrecallfig/'

    if os.path.exists(save_test_recall_fig):
        shutil.rmtree(save_test_recall_fig)
    os.makedirs(save_test_recall_fig)

    eva_log_path='history4/'+str(nowtime)+'/evl.xls'
    evl=xlwt.Workbook(encoding='utf-8',style_compression=0)
    # sheet_train = evl.add_sheet('train', cell_overwrite_ok=True)
    sheet_test = evl.add_sheet('test', cell_overwrite_ok=True)
    col = ('epoch', 'avg_mse', 'avg_psnr', 'avg_ssim','cc','sam', 'trian_time', 'test_time', 'all_time')
    for d in range(9):
        sheet_test.write(0,d,col[d])
    print("Initializing Training!",flush=True)
    best_psnr=0
    best_ssim=0
    for i in range(77,epoch):
        start_time = time.time()
        l1_loss_train,sam_loss_train=train_model(model,train_data_load,criterion1,criterion2,optimizer,len(train_data))
        time_now_train=time.time()
        time_train = time_now_train - start_time
        print('     train_time {:.0f}m {:.0f}s'.format(
            time_train // 60, time_train % 60))
        time_train=time_train // 60+0.01*time_train % 60

        mse,psnr,ssim,cc,sam,l1_loss = get_loss_train(i, model, val_data_load,criterion1,criterion2, save_test_recall_fig, len(val_data))
        time_now_test=time.time()
        time_val = time_now_test - time_now_train
        print('     test_time {:.0f}m {:.0f}s'.format(
            time_val // 60, time_val % 60),flush=True)
        time_val= time_val // 60+0.01*time_val % 60
        evl.save(eva_log_path)
        writer_val.add_scalar('mse',mse,i+1)
        writer_val.add_scalar('psnr',psnr, i + 1)
        writer_val.add_scalar('ssim',ssim, i + 1)
        writer_val.add_scalar('l1_loss', l1_loss, i + 1)
        writer_val.add_scalar('cc', cc, i + 1)
        writer_val.add_scalar('sam', sam, i + 1)


        time_elapsed = time.time() - start_time
        print('this epoch complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60),flush=True)
        time_all=time_elapsed//60+0.01*time_elapsed % 60
        # torch.save(model,
        #            save_model + str(i + 1) + '_' + str(miout)[0:5] + '.pt')
        evl_test = [i + 1, mse, psnr, ssim,cc,sam,time_train,time_val,time_all,l1_loss_train,sam_loss_train]
        for m in range(11):
            sheet_test.write(i + 1, m, evl_test[m])

        torch.save(model,save_model+str(i) + '_' + str(psnr)[0:5]+'_'+str(ssim)[0:5] + '.pt')
        if best_psnr<psnr or best_ssim<ssim:
            besti=i
            best_psnr=max(best_psnr,psnr)
            best_ssim=max(best_ssim,ssim)
            # if psnr>25 and ssim>0.8:
            #     torch.save(model,save_model+str(i) + '_' + str(psnr)[0:5]+'_'+str(ssim)[0:5] + '.pt')
            print([(i,psnr,ssim),(besti,best_psnr,best_ssim)],flush=True)
        else:
            if i-besti>=1000:
                print('已经1000次迭代没有优化网络了，所以终止训练')
                print('最佳验证epoch{}'.format(besti))
                path1 = save_model + str(besti) + '_*'
                path2=glob.glob(path1)[0]

                # test_pre(path2)
                #yuce_pinjie(path2)
                break
            else:
                print([(i-besti),(i,psnr,ssim),(besti,best_psnr,best_ssim)],flush=True)
                continue
    writer_val.close()