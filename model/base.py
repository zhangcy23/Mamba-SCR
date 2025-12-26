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
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as selective_scan_fn_v1
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref as selective_scan_ref_v1
import errno
from mamba_ssm import Mamba

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.forward_core = self.forward_corev0
        # self.forward_core = self.forward_corev0_seq
        # self.forward_core = self.forward_corev1

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)

        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward_corev0_seq(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)

        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = []
        for i in range(4):
            yi = self.selective_scan(
                xs[:, i], dts[:, i],
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.view(B, K, -1, L)  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        Ds = self.Ds.view(-1)  # (k * d)
        dt_projs_bias = self.dt_projs_bias.view(-1)  # (k * d)

        # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y = self.forward_core(x)
        # print([y.shape,'ss2d'])
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
class VSSBlock_c1(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            h=0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.model1 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=hidden_dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.model2 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=hidden_dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        # self.ln_1 = norm_layer(hidden_dim)
        # self.self_attention = SS2D_c(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        # self.drop_path = DropPath(drop_path)
        # self.gamma1 = nn.Parameter(torch.ones(hidden_dim))
        self.ln = nn.LayerNorm(normalized_shape=hidden_dim)
    def forward(self, input: torch.Tensor):
        x=input
        b, c, w, h = x.shape
        x = self.ln(x.reshape(b, -1, c))  #(B,L,C)
        x_flipped = torch.flip(x, dims=[-1])
        # x=torch.permute(input,(0,2,3,1))  #B,C,H,W

        x = x + self.model1(x)+self.model2(x_flipped)
        x=x.reshape(b, c, w, h)
        # print([x.shape,'xxxx'])  #[torch.Size([1, 24, 24, 1024]), 'xxxx']
        return x

class e_mamba_c(nn.Module):  # 使用VSSM，进行融合后的主干特征提取
    def __init__(self, dim_c, n):   #此处的dim变为H*W,序列长度变为原来的通道数
        super().__init__()
        vssm_block_c = []


        for i in range(n):
            vssm_block_c.append(VSSBlock_c1(dim_c, hnorm_layer=nn.LayerNorm))

        self.vssm_c = nn.Sequential(*vssm_block_c)


    def forward(self, x ):

        x = self.vssm_c(x) # torch.Size([1, 24, 24, 1024])

        return x
def train_model_one_epoch(model,data_train,criterion1,criterion2,optimizer,num):
    model.train()   #criterion1:diceloss,criterion2:focalloss,criterion3  celoss
    l1_loss_sum=0
    l2_loss_sum=0
    loss_dwt_sum=0
    mm=0
    aa = DWTForward(J=1, mode='zero', wave='db1').cuda(device=0)
    for batch,(img,sar,label) in enumerate(data_train):
        mm+=1
        optimizer.zero_grad()
        img=img.type(torch.FloatTensor).cuda()
        sar=sar.type(torch.FloatTensor).cuda()
        label=label.type(torch.FloatTensor).cuda()
        # mask=mask.type(torch.FloatTensor).cuda()
        outputs=model(img,sar)
        # yl, yh = aa(outputs)
        # yh_out = torch.reshape(yh[0], (yh[0].shape[0], yh[0].shape[1] * yh[0].shape[2], yh[0].shape[3], yh[0].shape[4]))
        #
        # yl1, yh1 = aa(label)
        # yh_out1 = torch.reshape(yh1[0],
        #                         (yh1[0].shape[0], yh1[0].shape[1] * yh1[0].shape[2], yh1[0].shape[3], yh1[0].shape[4]))
        #
        # # l1loss= criterion1(outputs, label)
        # loss_dwt = criterion1(yh_out, yh_out1) + criterion1(yl, yl1)


        loss1 = criterion1(outputs, label)
        loss2 = criterion2( label,outputs)/500
        # loss=loss1+loss2+loss_dwt*10
        loss=loss1+loss2
        l2_loss_sum+=loss2.item()
        l1_loss_sum+=loss1.item()
        # loss_dwt_sum+=loss_dwt.item()

        loss.backward()
        optimizer.step()
        i = int(batch * img.shape[0] / num * 100)
        print("\r", end="")
        print("train progress: {}%: ".format(i), "▋" * (batch // 1000), end="")
        sys.stdout.flush()
    # print('\n')
    print((l1_loss_sum/batch,l2_loss_sum/batch,loss_dwt_sum/batch, mm,batch),flush=True)  #(0.01073792177097251, 0.008707242843229324, 0.0040305583097506315, 25, 24)
    return l1_loss_sum/batch,l2_loss_sum/batch
def get_loss_train_one_epoch(epoch,model, data_train,criterion1, criterion2,save_recall_fig,num):
    """
        Calculate loss over train set
    """
    model.eval()
    ceshi_image=0
    with torch.no_grad():
        mm=0
        amin = 1
        amax = 80
        l1_loss_sum=0
        all_mse_sum=0
        all_psnr_sum=0
        all_ssim_sum=0
        all_cc_sum=0
        all_sam_sum=0
        l2_loss_sum=0
        save_path = save_recall_fig + str(epoch) + '/'
        aa = DWTForward(J=1, mode='zero', wave='db1').cuda(device=0)
        if not os.path.exists(save_path) and epoch%1==0:
            os.makedirs(save_path)

        for batch, (img,sar,label) in enumerate(data_train):
            mm+=1

            img = img.type(torch.FloatTensor).cuda()
            sar=sar.type(torch.FloatTensor).cuda()
            # mask=mask.type(torch.FloatTensor).cuda()
            label = label.type(torch.FloatTensor).cuda()
            outputs = model(img,sar)

            yl, yh = aa(outputs)
            yh_out = torch.reshape(yh[0],(yh[0].shape[0],yh[0].shape[1]*yh[0].shape[2],yh[0].shape[3],yh[0].shape[4]))


            yl1, yh1 = aa(label)
            yh_out1 = torch.reshape(yh1[0],(yh1[0].shape[0],yh1[0].shape[1]*yh1[0].shape[2],yh1[0].shape[3],yh1[0].shape[4]))


            # l1loss= criterion1(outputs, label)
            l1loss = criterion1(yh_out, yh_out1)+10*criterion1(yl,yl1)
            l1_loss_sum += l1loss.item()

            outputs=np.array(outputs.cpu())
            outputs=np.where(outputs>1,1,outputs)
            outputs=np.where(outputs<0,0,outputs).astype(np.float64)
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
            if epoch%1==0:
                if ceshi_image<20:
                    ceshi_image=ceshi_image+1

                    img1 = copy.deepcopy((img).cpu().numpy())  ##[batch,13,256,256]
                    label1=copy.deepcopy(label)
                    outputs1 = copy.deepcopy(outputs)
                    sar1=copy.deepcopy((sar).cpu().numpy())
                    #百分比截断显示
                    # img1=percent_show(img1,0.01,99.9)
                    # label1=percent_show(label1,0.01,99.9)
                    # outputs1 = percent_show(outputs1,0.01,99.9)
                    # sar1=percent_show(sar1,0.01,99.9)

                    img1=percent_show(img1,amin,amax)
                    label1=percent_show(label1,amin,amax+19)
                    outputs1 = percent_show(outputs1,amin,amax+19)
                    sar1=percent_show(sar1,amin,amax+19)

                    for n in range(img.shape[0]):
                        plt.figure(figsize=(60, 15))
                        gs = gridspec.GridSpec(1, 4)

                        rgb_img=np.stack((img1[n][3],img1[n][2],img1[n][1]),axis=0).astype(np.uint8)
                        vv=(sar1[n][0]).astype(np.uint8)
                        rgb_pre=np.stack((outputs1[n][3],outputs1[n][2],outputs1[n][1]),axis=0).astype(np.uint8)
                        rgb_label=np.stack((label1[n][3],label1[n][2],label1[n][1]),axis=0).astype(np.uint8)

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
                        # plt.savefig(save_path+str(n)+'_'+str(psnr)[:5]+'_'+str(ssim)[:5]+'.png', dpi=200)
                        plt.savefig(save_path+str(ceshi_image)+str(n)+'.png', dpi=200)
                        plt.close()
            ##计算评价指标
            for i in range(outputs.shape[0]):
                mse=cal_mse(outputs[i],label[i])
                psnr=cal_panr(outputs[i],label[i],data_range=1)
                ssim=cal_ssim(outputs[i],label[i],data_range=1)
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