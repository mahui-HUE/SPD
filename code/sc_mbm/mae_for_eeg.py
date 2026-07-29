import os
import sys
sys.path.append('../DreamDiffusion_20250917_review/code/')

from torch.nn import ChannelShuffle
from wandb.sdk.lib.timed_input import timed_input




# print(sys.path)
import sc_mbm.utils as ut
import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block
import torch.nn.functional as F
import numpy as np
import math


def _load_index_tensor(index_spec, index_base=0, expected_len=None, max_channels=None):
    if index_spec is None:
        return None
    if torch.is_tensor(index_spec):
        index_tensor = index_spec.detach().clone().long()
    elif isinstance(index_spec, np.ndarray):
        index_tensor = torch.from_numpy(index_spec).long()
    elif isinstance(index_spec, (list, tuple)):
        index_tensor = torch.tensor(index_spec, dtype=torch.long)
    elif isinstance(index_spec, str):
        index_spec = index_spec.strip()
        if os.path.exists(index_spec):
            if index_spec.lower().endswith(('.pt', '.pth')):
                loaded = torch.load(index_spec, map_location='cpu')
                if isinstance(loaded, dict):
                    for key in ('indices', 'index', 'electrode_indices', 'fixed_electrode_indices'):
                        if key in loaded:
                            loaded = loaded[key]
                            break
                index_tensor = torch.as_tensor(loaded, dtype=torch.long)
            elif index_spec.lower().endswith('.npy'):
                index_tensor = torch.from_numpy(np.load(index_spec)).long()
            else:
                with open(index_spec, 'r') as f:
                    text = f.read()
                index_tensor = torch.tensor([int(x) for x in text.replace(',', ' ').split()], dtype=torch.long)
        else:
            index_tensor = torch.tensor([int(x) for x in index_spec.replace(',', ' ').split()], dtype=torch.long)
    else:
        raise TypeError('fixed_electrode_indices must be a tensor, list, numpy array, path, or comma-separated string.')

    index_tensor = index_tensor.flatten() - int(index_base)
    if expected_len is not None and index_tensor.numel() != expected_len:
        raise ValueError('fixed_electrode_indices length must equal kept channels: '
                         f'{index_tensor.numel()} != {expected_len}')
    if index_tensor.numel() != torch.unique(index_tensor).numel():
        raise ValueError('fixed_electrode_indices contains duplicate indices.')
    if max_channels is not None and (
        torch.any(index_tensor < 0) or torch.any(index_tensor >= max_channels)
    ):
        raise ValueError(f'fixed_electrode_indices must be in [0, {max_channels - 1}] after index_base conversion.')
    return index_tensor
class PatchEmbed1D(nn.Module):
    """ 1 Dimensional version of data (fmri voxels) to Patch Embedding
    """
    def __init__(self, time_len=224, patch_size=1, in_chans=128, embed_dim=256,group=1):
        super().__init__()
        """增加group项是为了能够在映射时暂时不让通道发生信息交互，默认1表示不使用group。"""
        num_patches = time_len // patch_size
        self.patch_shape = patch_size
        self.time_len = time_len
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,groups=group)

    def forward(self, x, **kwargs):
        B, C, V = x.shape # batch, channel, voxels
        # assert V == self.num_voxels, \
        #     f"Input fmri length ({V}) doesn't match model ({self.num_voxels})."
        x = self.proj(x).transpose(1, 2).contiguous() # put embed_dim at the last dimension
        return x

class MAEforEEG(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, time_len=512, patch_size=4, embed_dim=1024, in_chans=62,
                 depth=24, num_heads=16, decoder_embed_dim=512, 
                 decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, focus_range=None, focus_rate=None, img_recon_weight=1.0, 
                 use_nature_img_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed1D(time_len, patch_size, in_chans, embed_dim)

        num_patches = int(time_len / patch_size)

        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans * patch_size, bias=True) # encoder to decoder
        # --------------------------------------------------------------------------

        # nature image decoder specifics
        if use_nature_img_loss:
            self.nature_img_decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.nature_img_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            self.nature_img_decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

            self.nature_img_decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(2)])

            self.nature_img_decoder_norm = norm_layer(decoder_embed_dim)
            self.nature_img_decoder_pred = nn.Sequential(
                nn.Conv1d(num_patches, 512, kernel_size=1, stride=1, bias=True),
                nn.Linear(decoder_embed_dim, 28*28, bias=True)
            )
            # --------------------------------------------------------------------------

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.focus_range = focus_range
        self.focus_rate = focus_rate
        self.img_recon_weight = img_recon_weight
        self.use_nature_img_loss = use_nature_img_loss
   
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = ut.get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        if self.use_nature_img_loss:
            nature_img_decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.nature_img_decoder_pos_embed.shape[-1], self.num_patches, cls_token=True)
            self.nature_img_decoder_pos_embed.data.copy_(torch.from_numpy(nature_img_decoder_pos_embed).float().unsqueeze(0))
            torch.nn.init.normal_(self.nature_img_mask_token, std=.02)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def patchify(self, imgs):
        """
        imgs: (N, 1, num_voxels)
        imgs: [N, chan, T]
        x: (N, L, patch_size)
        x: [N, chan * 4, T/4]
        """
        p = self.patch_embed.patch_size
        assert imgs.ndim == 3 and imgs.shape[1] % p == 0

        # h = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1] // p, -1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, 1, num_voxels)
        """
        p = self.patch_embed.patch_size
        h = x.shape[1]
        
        imgs = x.reshape(shape=(x.shape[0], -1, x.shape[2] // p))
        return imgs.transpose(1,2)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        if self.focus_range is not None:
            len_mask = L - len_keep
            weights = [1-self.focus_rate] * L
            weights[self.focus_range[0] // self.patch_size : self.focus_range[1] // self.patch_size
                        ] = [self.focus_rate] * (self.focus_range[1] // self.patch_size - self.focus_range[0] // self.patch_size)
            weights = torch.tensor(weights).repeat(N, 1).to(x.device)
            ids_mask = torch.multinomial(weights, len_mask, replacement=False)
            
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        if self.focus_range is not None:
            for i in range(N):
                noise[i, ids_mask[i,:]] = 1.1  # set mask portion to 1.1 

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # print('encoder embed')
        # print(x.shape)
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore = None):
        # embed tokens
        x = self.decoder_embed(x)
        # print('decoder embed')
        # print(x.shape)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # x = x_
        # add pos embed
        x = x + self.decoder_pos_embed
        # x = x + self.decoder_pos_embed[:, 1:, :]

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # print(x.shape)
        # predictor projection
        x = self.decoder_pred(x)
        # print(x.shape)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_nature_img_decoder(self, x, ids_restore):
        # embed tokens
        x = self.nature_img_decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.nature_img_mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.nature_img_decoder_pos_embed

        # apply Transformer blocks
        for blk in self.nature_img_decoder_blocks:
            x = blk(x)
        x = self.nature_img_decoder_norm(x)
        # remove cls token
        x = x[:, 1:, :]
        # predictor projection
        # x = x.mean(dim=1, keepdim=True)
        x = self.nature_img_decoder_pred(x)
        x = x.view(x.shape[0], 512, 28, 28)

        return x # n, 512, 28, 28
        
    def forward_nature_img_loss(self, inputs, reconstructions):
        loss = ((torch.tanh(inputs) - torch.tanh(reconstructions))**2).mean()
        if torch.isnan(reconstructions).sum():
            print('nan in reconstructions')
        if torch.isnan(inputs).sum():
            print('nan in inputs')
    
        return loss   

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, num_voxels]
        imgs: [N, chan, T]
        pred: [N, L, p]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        imgs = imgs.transpose(1,2)
        target = self.patchify(imgs)
        # target = imgs.transpose(1,2)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # loss = loss.mean()
        loss = (loss * mask).sum() / mask.sum()  if mask.sum() != 0 else (loss * mask).sum() # mean loss on removed patches
        return loss

    def forward(self, imgs, img_features=None, valid_idx=None, mask_ratio=0.75):
        # latent = self.forward_encoder(imgs, mask_ratio)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
            # print(x)
        # print(latent.shape)
        # # print(mask)
        # print(mask.shape)
        # # print(ids_restore)
        # print(ids_restore.shape)
        
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p]
        # pred = self.forward_decoder(latent)  # [N, L, p]
        # pred = pred
        # print(pred.shape)
        # mask=None
        loss = self.forward_loss(imgs, pred, mask)
        # print(self.unpatchify(pred.transpose(1,2)).shape)

        if self.use_nature_img_loss and img_features is not None:
            # valid_idx = torch.nonzero(nature_image.sum(dim=(1,2,3)) != 0).squeeze(1)
            if len(valid_idx) != 0:
                nature_image_recon = self.forward_nature_img_decoder(latent[valid_idx], ids_restore[valid_idx])
                loss_nature_image_recon = self.forward_nature_img_loss(img_features, nature_image_recon)
                if torch.isnan(loss_nature_image_recon).sum():
                    print(loss_nature_image_recon)
                    print("loss_nature_image_recon is nan")
                    
                loss = loss + self.img_recon_weight*loss_nature_image_recon

        return loss, pred, mask

class TimeBranch(nn.Module):
    """时间分支：处理时间维度自注意力"""
    def __init__(self, num_blocks, embed_dim, num_heads, mlp_ratio, norm_layer=nn.LayerNorm):
        super().__init__()
        # 堆叠num_blocks层ViT Block
        self.blocks = nn.ModuleList([
            # ViTBlock(dim=dim, num_heads=num_heads, mlp_dim=mlp_dim)
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        # x输入维度：(B, T, dim)，其中T=time_steps
        for block in self.blocks:
            x = block(x)
        return x  # 输出维度：(B, T, dim)

class ChannelBranch(nn.Module):
    """通道分支：处理通道维度自注意力"""
    def __init__(self, num_blocks, embed_dim, num_heads, mlp_ratio, norm_layer=nn.LayerNorm):
        super().__init__()
        # 堆叠num_blocks层ViT Block（与时间分支结构完全一致）
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        # x输入维度：(B, C, dim)，其中C=channels
        for block in self.blocks:
            x = block(x)
        return x  # 输出维度：(B, channels, T)


class CrossFusionLayer(nn.Module):
    """交叉融合层：拼接+1×1卷积压缩维度"""

    def __init__(self, input_dim, masked_Channels):
        super().__init__()
        self.input_dim = input_dim
        self.masked_Channels = masked_Channels#mask之后的电极通道数
        # self.conv1x1 = nn.Conv1d(
        #     in_channels=input_dim+masked_Channels,  # 拼接后维度
        #     out_channels=input_dim+masked_Channels,  #不压缩维度，只融合
        #     kernel_size=1
        # )
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(input_dim+masked_Channels,(input_dim+masked_Channels)//4,kernel_size=1),
            nn.Conv1d((input_dim+masked_Channels)//4,input_dim+masked_Channels,kernel_size=1)
        )
        # self.pool2C = nn.AdaptiveAvgPool1d(masked_Channels)#自适应池化用于将T维度压缩成C
        # self.pool2T= nn.AdaptiveAvgPool1d(T)#将C恢复成T
        # self.proj_chan2t = self.proj_Chan2T(current_channels=channels,T=T,embed_dim=input_dim)
        # self.proj_t2chan = self.proj_T2Chan(current_channels=T,original_channels=channels,embed_dim=input_dim)
        self.norm = nn.LayerNorm(input_dim+masked_Channels)
        self.activation = nn.GELU()
    def forward(self, time_feat:torch.Tensor, channel_feat:torch.Tensor):
        # time_feat: (B, T, input_dim)  时间分支输出
        # channel_feat: (B, C, input_dim)  通道分支输出
        # T: 时间步数（用于重塑通道分支维度）

        # 1.与通道分支序列长度对齐
        B, C, T = channel_feat.shape#使用原始维度的转置
        N, T, embed_dim_T = time_feat.shape

        assert B == N
        assert C == self.masked_Channels#mask之后的电极通道数
        # time_feat = self.pool2C(time_feat.transpose(1,2)).transpose(1,2)#映射时间分支到通道维度(B,C,input_dim)
        # channel_feat = self.proj_chan2t(channel_feat.transpose(1,2)).transpose(1,2)#映射通道分支维度到时间分支维度
        # new_B, new_C, new_embed_dim_C = channel_feat.shape
        # assert new_C == T//2 and new_embed_dim_C == embed_dim_T * 2
        # 2. 特征拼接
        fused = torch.cat([time_feat, channel_feat.transpose(1,2)], dim=-1)  # (B, T, embed_dim_T+C)

        # 3. 1×1卷积压缩维度
        fused = fused.transpose(1, 2)  # (B,embed_dim_T+C, T)
        fused = self.conv1x1(fused)  # (B,embed_dim_T+C, T)
        fused = fused.transpose(1, 2)  # (B, T, embed_dim_T+C)

        # 4. 激活与归一化
        fused = self.activation(fused)
        fused = self.norm(fused)
        next_channel_feat = fused[:,:,embed_dim_T:].transpose(1,2)#(B,T,C)--(B,C,T)
        # next_time_feat = self.proj_t2chan(fused.transpose(1,2)).transpose(1,2)#将聚合特征映射为原始通道分支维度
        next_time_feat = fused[:,:,:embed_dim_T]##(B,T,embed_dim_T)
        return next_time_feat, next_channel_feat


    # def proj_Chan2T(self, current_channels, T, embed_dim):
    #     target_channels = T//2
    #     padding = 3#设置的超参
    #     kernel_size = current_channels - (target_channels-1-2*padding)#使用卷积特征图大小计算公式使得最终的channels是时间的一半，方便维度变换
    #     proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, padding=padding,stride=1)
    #     return proj
    # def proj_T2Chan(self, current_channels,original_channels, embed_dim):
    #     target_channels = original_channels
    #     padding = 3
    #     kernel_size = target_channels+2*padding+1-current_channels#使用反卷积特征图大小计算公式使得最终的channels恢复为原始电极通道数
    #     proj = nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=kernel_size, padding=padding,stride=1)
    #     return proj


class Stage(nn.Module):
    """模型阶段：由时间分支、通道分支和融合层组成"""

    def __init__(self, time_blocks, channel_blocks, embed_dim, num_heads, mlp_ratio,T,masked_Channels):
        super().__init__()
        self.time_branch = TimeBranch(
            num_blocks=time_blocks,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )
        self.channel_branch = ChannelBranch(
            num_blocks=channel_blocks,
            embed_dim=T,#使用原始（B,T,C)
            num_heads=4,#T一般为100，作为嵌入层维度，设置注意力头为4
            mlp_ratio=mlp_ratio
        )
        self.fusion = CrossFusionLayer(
            input_dim=embed_dim,
            masked_Channels=masked_Channels
        )

    def forward(self, time_input, channel_input):
        # 1. 双分支并行处理
        channel_out = self.channel_branch(channel_input)  # (B, channels, T)
        time_out = self.time_branch(time_input)  # (B, T, dim)

        # 2. 融合（需传入时间步数T用于维度对齐）
        next_time_input, next_channel_input = self.fusion(time_out, channel_out)  # (B, T, dim)
        return next_time_input, next_channel_input


class EEGTwoBranchModel(nn.Module):
    """EEG双分支模型：时间+通道并行处理，共24层ViT Block"""

    def __init__(self,
                 T=100,  # 时间步数
                 channels=62,  # 电极通道数
                 mask_ratio=0.125,# mask之后的电极通道数
                 embed_dim=512,#嵌入层维度
                 num_heads=8,
                 mlp_ratio=4,
                 num_stages=4,
                 norm_layer=nn.LayerNorm,
                 xyz_brain_pth='/home/mahui/Dataset/EEG-ImageNet-Dataset/XYZ_Brain.pth'):  # 4个阶段

        super().__init__()
        self.dim = embed_dim
        self.T = T #num_patch
        self.channels = channels#原始电极通道数
        self.mask_ratio = mask_ratio
        self.masked_Channels = int(channels*(1-mask_ratio))#mask之后剩余的电极通道数
        self.xyz_brain = xyz_brain_pth#3D坐标和脑区编码
        # 将时间点维度按照patch_size打包成token,并且通过分组卷积来实现通道间信息分离。也就是说此操作仅仅是为了将时间步打包成token，对电极通道无影响。

        # 位置编码（假设已实现时间和通道位置编码）
        time_pos_encoding = nn.Parameter(torch.randn(1, T, embed_dim),requires_grad=False)  # 时间位置编码
        time_pos_encoding = self.T_position_encoding(time_pos_encoding)
        self.register_buffer('time_pos_encoding', time_pos_encoding)
        channel_pos_encoding = self.brain_prior_encoding(channels,T)  # 脑先验位置编码

        self.register_buffer('channel_pos_encoding', channel_pos_encoding)#注册为缓冲区，方便在不同设备上移动
        self.proj_chan2embed = nn.Conv1d(
            in_channels=self.masked_Channels,
            out_channels=embed_dim,
            kernel_size=1
        )
        # self.proj_T2embed = nn.Conv1d(
        #     in_channels=self.T,
        #     out_channels=embed_dim,
        #     kernel_size=1
        # )
        self.conv1x1 = nn.Conv1d(
            in_channels=embed_dim+self.masked_Channels,  # 拼接后维度增加
            out_channels=self.masked_Channels,  # 压缩为mask后的电极通道数
            kernel_size=1
        )
        # 4个阶段，每个阶段3时间+3通道Block（共4×6=24层）
        self.stages = nn.ModuleList([
            Stage(
                time_blocks=1,
                channel_blocks=1,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                T=T,
                masked_Channels=self.masked_Channels
            ) for _ in range(num_stages)
        ])

        # 最终输出层
        self.norm = norm_layer(self.masked_Channels)

    def forward(self, x):
        # 输入x维度：(B, T, channels) → 原始EEG数据（batch, 时间, 通道）
        original_input = x
        channel_pos_encoding = self.channel_pos_encoding.repeat(original_input.shape[0],1,1)#(B,T,channels)
        mask, ids_restore = None, None
        if self.mask_ratio != 0:
            original_input, mask,ids_keep, ids_restore = self.random_mask_channels(original_input)
            channel_pos_encoding = torch.gather(channel_pos_encoding,dim=-1, index=ids_keep)#取出新的通道编码

        time_input = self.proj_chan2embed(original_input.transpose(1,2)).transpose(1,2)# 时间分支输入：(B, T, embed_dim)
        time_input = time_input + self.time_pos_encoding  # 加时间位置编码
        channel_input = original_input.transpose(1,2) # 通道分支输入：(B, channels, T)
        channel_input = channel_input + channel_pos_encoding.transpose(1,2)  # 加通道位置编码

        # 2. 多阶段处理（双分支并行+融合）
        for stage in self.stages:
            time_input, channel_input = stage(time_input, channel_input)
        fused_feat = torch.cat([time_input, channel_input.transpose(1,2)], dim=-1)#(B,T,embed_dim+masked_channels)
        fused_feat = self.conv1x1(fused_feat.transpose(1, 2)).transpose(1, 2)#最终的特征输出（B,T,masked_channels)
        return self.norm(fused_feat), mask, ids_restore

    def brain_prior_encoding(self,channels,embed_dim):
        xyz_brain = torch.load(self.xyz_brain,map_location='cpu').numpy()
        xyz_brain = self.absPoisionEncoding(coordinates=xyz_brain,d_model=embed_dim,max_len_channel=channels)
        return torch.from_numpy(xyz_brain).unsqueeze(0).float()#增加batch维度(1,embed_dim,channels)

    # def absPoisionEncoding(self,coordinates, d_model, max_len_channel=62):
    #     assert d_model % 4 == 0  # d_model是嵌入层维度,x,y,z,brain_region均分所有嵌入层维度
    #
    #     d_model = d_model // 4
    #     if d_model % 2 !=0:#务必确保每个位置编码的维度为偶数
    #         d_model = d_model + 1
    #         cut = 1#如果是奇数，那么需要给d_model加一位，并在最后去掉一位。
    #     else:
    #         cut = 0
    #     if isinstance(coordinates, np.ndarray):
    #         div_term = np.exp(np.arange(0, d_model, 2) *
    #                           -(math.log(100.0) / d_model))
    #         # div_term = np.expand_dims(div_term, axis=0)
    #         position_x = coordinates[:, 0:1]
    #         position_y = coordinates[:, 1:2]
    #         position_z = coordinates[:, 2:3]
    #         brain_region = coordinates[:, 3:]
    #         pe_x = np.zeros((max_len_channel, d_model))
    #         pe_y = np.zeros((max_len_channel, d_model))
    #         pe_z = np.zeros((max_len_channel, d_model))
    #         pe_brain_region = np.zeros((max_len_channel, d_model))
    #         pe_x[:, 0::2] = np.sin(position_x * div_term)
    #         pe_x[:, 1::2] = np.cos(position_x * div_term)
    #         pe_y[:, 0::2] = np.sin(position_y * div_term)
    #         pe_y[:, 1::2] = np.cos(position_y * div_term)
    #         pe_z[:, 0::2] = np.sin(position_z * div_term)
    #         pe_z[:, 1::2] = np.cos(position_z * div_term)
    #         pe_brain_region[:, 0::2] = np.sin(brain_region * div_term)
    #         pe_brain_region[:, 1::2] = np.cos(brain_region * div_term)
    #
    #         pe_x = pe_x[:, :(d_model - cut)].transpose(1,0)#注意numpy的transpose和torch的transpose用法不同
    #         pe_y = pe_y[:, :(d_model - cut)].transpose(1,0)
    #         pe_z = pe_z[:, :(d_model - cut)].transpose(1,0)
    #         pe_brain_region = pe_brain_region[:, :(d_model - cut)].transpose(1,0)
    #
    #         return np.concatenate((pe_x, pe_y, pe_z, pe_brain_region), axis=0)  # (维度数=x维度数+y维度数+z维度数+脑区独热编码维度数,电极数)
    #     else:
    #         raise NotImplementedError
    def absPoisionEncoding(self, coordinates, d_model, max_len_channel=62):
        # 新增：先归一化3D坐标
        coordinates = self.normalize_3d_coords(coordinates)
        # 1. 核心校验：保证总维度能被4均分，单维度编码长度为偶数
        assert d_model % 4 == 0, "d_model必须是4的倍数"
        d_per_dim = d_model // 4  # 单维度（X/Y/Z/脑区）的编码长度
        cut = 0
        if d_per_dim % 2 != 0:  # 保证单维度编码长度为偶数（sin/cos各占一半）
            d_per_dim += 1
            cut = 1  # 最后截断多余的1列

        # 2. 初始化交错编码矩阵：[电极数, 总维度]
        pe_4d_interleaved = np.zeros((max_len_channel, d_model))

        # 3. 仅处理numpy数组，避免类型错误
        if not isinstance(coordinates, np.ndarray):
            raise TypeError("coordinates必须是numpy.ndarray类型，形状为[电极数,4]（x/y/z/脑区）")

        # 4. 计算单维度的衰减因子（核心修正：基于d_per_dim计算）
        # div_term长度 = d_per_dim//2（步长2），适配sin/cos的偶数维度
        div_term_3D = np.exp(np.arange(0, d_per_dim, 2) * -(math.log(200.0) / d_per_dim))#3D空间归一化坐标和脑区离散坐标使用不同的scale 200 1000
        div_term_3D = np.expand_dims(div_term_3D, axis=0)  # 扩展维度：[1, d_per_dim//2]，方便广播
        div_term_region = np.exp(np.arange(0, d_per_dim, 2) * -(math.log(1000.0) / d_per_dim))
        div_term_region = np.expand_dims(div_term_region, axis=0)  # 扩展维度：[1, d_per_dim//2]，方便广播

        # 5. 拆分4D坐标（保证形状为[max_len_channel,1]）
        position_x = coordinates[:, 0:1]  # [max_len_channel, 1]
        position_y = coordinates[:, 1:2]  # [max_len_channel, 1]
        position_z = coordinates[:, 2:3]  # [max_len_channel, 1]
        brain_region = coordinates[:, 3:4]  # 修正：取[:,3:4]保证维度为[max_len_channel,1]，避免脑区维度不一致

        # 6. 生成单维度完整编码（修正：维度为[max_len_channel, d_per_dim]，保留高维信息）
        pe_x = np.zeros((max_len_channel, d_per_dim))
        pe_y = np.zeros((max_len_channel, d_per_dim))
        pe_z = np.zeros((max_len_channel, d_per_dim))
        pe_brain_region = np.zeros((max_len_channel, d_per_dim))

        # 正弦（偶数位）+ 余弦（奇数位）编码
        pe_x[:, 0::2] = np.sin(position_x * div_term_3D)  # [max_len_channel, d_per_dim//2]
        pe_x[:, 1::2] = np.cos(position_x * div_term_3D)
        pe_y[:, 0::2] = np.sin(position_y * div_term_3D)
        pe_y[:, 1::2] = np.cos(position_y * div_term_3D)
        pe_z[:, 0::2] = np.sin(position_z * div_term_3D)
        pe_z[:, 1::2] = np.cos(position_z * div_term_3D)
        pe_brain_region[:, 0::2] = np.sin(brain_region * div_term_region)
        pe_brain_region[:, 1::2] = np.cos(brain_region * div_term_region)

        # 7. 截断奇数维度的多余列（恢复偶数维度）
        if cut > 0:
            pe_x = pe_x[:, :-1]
            pe_y = pe_y[:, :-1]
            pe_z = pe_z[:, :-1]
            pe_brain_region = pe_brain_region[:, :-1]
            d_per_dim -= 1  # 更新单维度长度

        # 8. 交错拼接核心逻辑（修正：逐维度填充，保留高维信息）
        # 每4维为一组：第4i维=X的第i维，4i+1=Y的第i维，4i+2=Z的第i维，4i+3=脑区的第i维
        for i in range(d_per_dim):
            pe_4d_interleaved[:, 4 * i] = pe_x[:, i]  # 第4i维 → X的第i维
            pe_4d_interleaved[:, 4 * i + 1] = pe_y[:, i]  # 第4i+1维 → Y的第i维
            pe_4d_interleaved[:, 4 * i + 2] = pe_z[:, i]  # 第4i+2维 → Z的第i维
            pe_4d_interleaved[:, 4 * i + 3] = pe_brain_region[:, i]  # 第4i+3维 → 脑区的第i维

        # 9. 转置返回：(总维度数, 电极数)，适配后续融合逻辑
        return pe_4d_interleaved.transpose(1, 0)

    def normalize_3d_coords(self,coordinates):
        """
        对3D坐标（前3列）做min-max归一化，映射到[-1,1]；脑区列（第4列）保持不变
        input: coordinates - [num_electrodes, 4]（x/y/z/脑区）
        output: normalized_coords - [num_electrodes, 4]
        """
        # 提取3D坐标
        coords_3d = coordinates[:, :3]
        # 按维度计算min/max
        min_vals = coords_3d.min(axis=0, keepdims=True)  # [1,3]
        max_vals = coords_3d.max(axis=0, keepdims=True)  # [1,3]
        # 避免除0（所有电极坐标相同的极端情况）
        ranges = np.maximum(max_vals - min_vals, 1e-8)
        # min-max归一化到[-1,1]
        coords_3d_norm = 2 * (coords_3d - min_vals) / ranges - 1
        # 拼接脑区列
        normalized_coords = np.concatenate([coords_3d_norm, coordinates[:, 3:4]], axis=1)
        return normalized_coords

    def T_position_encoding(self,time_pos_encoding):
        pos_embed = ut.get_1d_sincos_pos_embed(time_pos_encoding.shape[-1], self.T, cls_token=False)
        return torch.from_numpy(pos_embed).float().unsqueeze(0)

    def random_mask_channels(self, x):
        """
        将大脑划分为4个部分，然后在每个部分里去除1-2个通道。用更简洁的方法实现，使用整个大脑的均匀分布来挑选。
        """
        N, T, Channels = x.shape  # batch, length, dim。Channels是整个大脑的电极通道数量,T是每个电极的时间token数
        assert Channels == self.channels
        Channel_keep = int(Channels * (1 - self.mask_ratio))
        noise = torch.rand(N, Channels, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # print('ids_shuffle:', ids_shuffle)
        # print('ids_shuffle.shape',ids_shuffle.shape)

        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # print('ids_restore', ids_restore)
        # print('ids_restore.shape',ids_restore.shape)
        # keep the first subset
        ids_keep = ids_shuffle[:, :Channel_keep].unsqueeze(1).repeat(1, T, 1)
        x_masked = torch.gather(x, dim=-1, index=ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, Channels], device=x.device)#初始化屏蔽矩阵
        mask[:, :Channel_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)#此时mask才真正代表屏蔽矩阵
        return x_masked, mask, ids_keep,ids_restore# x_masked(N,T,Channel_keep)

class EEGHybridBranchModel(nn.Module):
    """EEG双分支模型：时间+通道并行处理，共24层ViT Block"""

    def __init__(self,
                 time_len=400,  # 时间步数
                 patch_size=4,
                 channels=62,  # 电极通道数
                 mask_ratio=0.125,# mask之后的电极通道数
                 embed_dim=512,#嵌入层维度
                 num_heads=8,
                 mlp_ratio=4,
                 depth=4,
                 position_flag=True,
                 norm_layer=nn.LayerNorm,
                 xyz_brain_pth=None,
                 fixed_electrode_indices=None,
                 fixed_electrode_index_base=0):  # 4个阶段

        super().__init__()
        self.dim = embed_dim
        self.num_channel_branch = depth//8
        self.T = time_len//patch_size #num_patch
        self.time_len = time_len
        self.channels = channels#原始电极通道数
        self.mask_ratio = mask_ratio
        self.masked_Channels = int(channels*(1-mask_ratio))#mask之后剩余的电极通道数
        self.xyz_brain = xyz_brain_pth#3D坐标和脑区编码
        fixed_indices = _load_index_tensor(
            fixed_electrode_indices,
            index_base=fixed_electrode_index_base,
            expected_len=self.masked_Channels if mask_ratio != 0 else None,
            max_channels=channels
        )
        if fixed_indices is not None:
            self.register_buffer('fixed_electrode_indices', fixed_indices)
            print('启用固定电极索引:', fixed_indices.tolist())
        else:
            self.fixed_electrode_indices = None
        # 将时间点维度按照patch_size打包成token,并且通过分组卷积来实现通道间信息分离。也就是说此操作仅仅是为了将时间步打包成token，对电极通道无影响。

        # 位置编码（假设已实现时间和通道位置编码）
        time_pos_encoding = nn.Parameter(torch.randn(1, self.T, embed_dim),requires_grad=False)  # 时间位置编码
        time_pos_encoding = self.T_position_encoding(time_pos_encoding)
        self.register_buffer('time_pos_encoding', time_pos_encoding)
        self.position_flag = position_flag  # 是否启用位置编码,这个参数主要是为了第一阶段开启,第二阶段关闭.
        self.channel_pos_dim = 4
        if self.position_flag:
            channel_pos_encoding = self.brain_prior_encoding(channels,time_len)  # 脑先验位置编码
            self.channel_pos_dim = channel_pos_encoding.shape[-1]
            self.register_buffer('channel_pos_encoding', channel_pos_encoding)#注册为缓冲区，方便在不同设备上移动

        self.proj_chan2time = nn.Conv1d(
            in_channels=self.masked_Channels,
            out_channels=time_len,
            kernel_size=1
        )
        self.proj_time2embed = nn.Conv1d(
            in_channels=time_len,
            out_channels=embed_dim,
            kernel_size=1
        )
        # 4个阶段，每个阶段3时间+3通道Block（共4×6=24层）
        self.time_branch = TimeBranch(
            num_blocks= depth-self.num_channel_branch,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )
        self.channel_branch = ChannelBranch(
            num_blocks= self.num_channel_branch,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )
        # self.proj = nn.Linear(self.masked_Channels,self.T)
        self.patch_embed = PatchEmbed1D(time_len=time_len, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim)
        # 最终输出层
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        # 输入x维度：(B, channels，time_len ) → 原始EEG数据（batch, 时间, 通道）
        original_input = x#(B, channels，time_len)
        if self.position_flag:
            channel_pos_encoding = self.channel_pos_encoding.repeat(original_input.shape[0],1,1)#(B,time_len,channels)
        mask, ids_restore = None, None
        if self.mask_ratio != 0:
            original_input, mask,ids_keep, ids_restore = self.random_mask_channels(original_input.transpose(1,2))
            original_input = original_input.transpose(1,2)#(B, channels，time_len)
            if self.position_flag:
                channel_pos_encoding = torch.gather(channel_pos_encoding,dim=-1, index=ids_keep)#取出新的通道编码
        channel_input = original_input.transpose(1,2)  # 通道分支输入：#(B，time_len, channels)
        if self.position_flag:
            channel_input = channel_input + channel_pos_encoding# 加通道位置编码
        else:
            channel_input = channel_input + 0  # 消融实验,不加入先验编码
        channel_feat = self.proj_time2embed(channel_input).transpose(1,2)#(B , channels,embed_dim)
        channel_feat = self.channel_branch(channel_feat) + channel_feat#(B, channels, embed_dim)
        time_input = self.proj_chan2time(channel_feat)#(B,time_len,embed_dim)
        time_feat = self.patch_embed(time_input.transpose(1,2))#(B,T,embed_dim)
        time_feat = time_feat + self.time_pos_encoding  # 加时间位置编码
        time_feat = self.time_branch(time_feat)

        return self.norm(time_feat), mask, ids_restore

    def brain_prior_encoding(self,channels,embed_dim):
        xyz_brain = torch.load(self.xyz_brain,map_location='cpu').numpy()
        xyz_brain = self.absPoisionEncoding(coordinates=xyz_brain,d_model=embed_dim,max_len_channel=channels)
        return torch.from_numpy(xyz_brain).unsqueeze(0).float()#增加batch维度(1,embed_dim,channels)

    def absPoisionEncoding(self,coordinates, d_model, max_len_channel=62):
        assert d_model % 4 == 0  # d_model是嵌入层维度,x,y,z,brain_region均分所有嵌入层维度
        coordinates = self.normalize_3d_coords(coordinates)
        d_model = d_model // 4
        if d_model % 2 !=0:#务必确保每个位置编码的维度为偶数
            d_model = d_model + 1
            cut = 1#如果是奇数，那么需要给d_model加一位，并在最后去掉一位。
        else:
            cut = 0
        if isinstance(coordinates, np.ndarray):
            div_term = np.exp(np.arange(0, d_model, 2) *
                              -(math.log(10000.0) / d_model))
            # div_term = np.expand_dims(div_term, axis=0)
            position_x = coordinates[:, 0:1]
            position_y = coordinates[:, 1:2]
            position_z = coordinates[:, 2:3]
            brain_region = coordinates[:, 3:]
            pe_x = np.zeros((max_len_channel, d_model))
            pe_y = np.zeros((max_len_channel, d_model))
            pe_z = np.zeros((max_len_channel, d_model))
            pe_brain_region = np.zeros((max_len_channel, d_model))
            pe_x[:, 0::2] = np.sin(position_x * div_term)
            pe_x[:, 1::2] = np.cos(position_x * div_term)
            pe_y[:, 0::2] = np.sin(position_y * div_term)
            pe_y[:, 1::2] = np.cos(position_y * div_term)
            pe_z[:, 0::2] = np.sin(position_z * div_term)
            pe_z[:, 1::2] = np.cos(position_z * div_term)
            pe_brain_region[:, 0::2] = np.sin(brain_region * div_term)
            pe_brain_region[:, 1::2] = np.cos(brain_region * div_term)

            pe_x = pe_x[:, :(d_model - cut)].transpose(1,0)#注意numpy的transpose和torch的transpose用法不同
            pe_y = pe_y[:, :(d_model - cut)].transpose(1,0)
            pe_z = pe_z[:, :(d_model - cut)].transpose(1,0)
            pe_brain_region = pe_brain_region[:, :(d_model - cut)].transpose(1,0)

            return np.concatenate((pe_x, pe_y, pe_z, pe_brain_region), axis=0)  # (维度数=x维度数+y维度数+z维度数+脑区独热编码维度数,电极数)
        else:
            raise NotImplementedError
    def normalize_3d_coords(self, coordinates):
        """
        对3D坐标（前3列）做min-max归一化，映射到[-1,1]；脑区列（第4列）保持不变
        input: coordinates - [num_electrodes, 4]（x/y/z/脑区）
        output: normalized_coords - [num_electrodes, 4]
        """
        # 提取3D坐标
        coords_3d = coordinates[:, :3]
        # 按维度计算min/max
        min_vals = coords_3d.min(axis=0, keepdims=True)  # [1,3]
        max_vals = coords_3d.max(axis=0, keepdims=True)  # [1,3]
        # 避免除0（所有电极坐标相同的极端情况）
        ranges = np.maximum(max_vals - min_vals, 1e-8)
        # min-max归一化到[-1,1]
        coords_3d_norm = 2 * (coords_3d - min_vals) / ranges - 1
        # 拼接脑区列
        normalized_coords = np.concatenate([coords_3d_norm, coordinates[:, 3:4]], axis=1)
        return normalized_coords

    def T_position_encoding(self,time_pos_encoding):
        pos_embed = ut.get_1d_sincos_pos_embed(time_pos_encoding.shape[-1], self.T, cls_token=False)
        return torch.from_numpy(pos_embed).float().unsqueeze(0)

    def random_mask_channels(self, x):
        """
        将大脑划分为4个部分，然后在每个部分里去除1-2个通道。用更简洁的方法实现，使用整个大脑的均匀分布来挑选。
        """
        N, T, Channels = x.shape  # batch, length, dim。Channels是整个大脑的电极通道数量,T是每个电极的时间token数
        assert Channels == self.channels
        Channel_keep = int(Channels * (1 - self.mask_ratio))
        noise = torch.rand(N, Channels, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # print('ids_shuffle:', ids_shuffle)
        # print('ids_shuffle.shape',ids_shuffle.shape)

        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # print('ids_restore', ids_restore)
        # print('ids_restore.shape',ids_restore.shape)
        # keep the first subset
        ids_keep = ids_shuffle[:, :Channel_keep].unsqueeze(1).repeat(1, T, 1)
        x_masked = torch.gather(x, dim=-1, index=ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, Channels], device=x.device)#初始化屏蔽矩阵
        mask[:, :Channel_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)#此时mask才真正代表屏蔽矩阵
        return x_masked, mask, ids_keep,ids_restore# x_masked(N,T,Channel_keep)

class EEGHybridBranchModel_V2(nn.Module):
    """EEG双分支模型：时间+通道并行处理，共24层ViT Block"""

    def __init__(self,
                 time_len=400,  # 时间步数
                 patch_size=4,
                 channels=62,  # 电极通道数
                 mask_ratio=0.125,# mask之后的电极通道数
                 embed_dim=512,#嵌入层维度
                 num_heads=8,
                 mlp_ratio=4,
                 depth=4,
                 position_flag=True,
                 norm_layer=nn.LayerNorm,
                 xyz_brain_pth=None,
                 fixed_electrode_indices=None,
                 fixed_electrode_index_base=0):  # 4个阶段

        super().__init__()
        self.dim = embed_dim
        self.num_channel_branch = depth//8
        self.T = time_len//patch_size #num_patch
        self.channels = channels#原始电极通道数
        self.mask_ratio = mask_ratio
        self.masked_Channels = int(channels*(1-mask_ratio))#mask之后剩余的电极通道数
        self.xyz_brain = xyz_brain_pth#3D坐标和脑区编码
        fixed_indices = _load_index_tensor(
            fixed_electrode_indices,
            index_base=fixed_electrode_index_base,
            expected_len=self.masked_Channels if mask_ratio != 0 else None,
            max_channels=channels
        )
        if fixed_indices is not None:
            self.register_buffer('fixed_electrode_indices', fixed_indices)
            print('启用固定电极索引:', fixed_indices.tolist())
        else:
            self.fixed_electrode_indices = None
        # 将时间点维度按照patch_size打包成token,并且通过分组卷积来实现通道间信息分离。也就是说此操作仅仅是为了将时间步打包成token，对电极通道无影响。

        # 位置编码（假设已实现时间和通道位置编码）
        time_pos_encoding = nn.Parameter(torch.randn(1, self.T, embed_dim),requires_grad=False)  # 时间位置编码
        time_pos_encoding = self.T_position_encoding(time_pos_encoding)
        self.register_buffer('time_pos_encoding', time_pos_encoding)
        self.position_flag = position_flag  # 是否启用位置编码,这个参数主要是为了第一阶段开启,第二阶段关闭.
        self.channel_pos_dim = 4
        if self.position_flag:
            channel_pos_encoding = self.brain_prior_encoding(channels,time_len)  # 脑先验位置编码
            self.channel_pos_dim = channel_pos_encoding.shape[-1]
            self.register_buffer('channel_pos_encoding', channel_pos_encoding)#注册为缓冲区，方便在不同设备上移动

        # self.proj_chan2time = nn.Conv1d(
        #     in_channels=self.masked_Channels,
        #     out_channels=time_len,
        #     kernel_size=1
        # )
        self.proj_embed2time = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=time_len,
            kernel_size=1
        )
        self.proj_time2embed = nn.Conv1d(
            in_channels=time_len,
            out_channels=embed_dim,
            kernel_size=1
        )
        self.proj_chan2embed = nn.Conv1d(
            in_channels=self.masked_Channels,
            out_channels=embed_dim,
            kernel_size=1
        )
        # 4个阶段，每个阶段3时间+3通道Block（共4×6=24层）
        self.time_branch = TimeBranch(
            num_blocks= depth-self.num_channel_branch,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )
        self.channel_branch = ChannelBranch(
            num_blocks= self.num_channel_branch,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )
        # self.proj = nn.Linear(self.masked_Channels,self.T)
        self.patch_embed = PatchEmbed1D(time_len=time_len, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim)
        # 最终输出层
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        # 输入x维度：(B, channels，time_len ) → 原始EEG数据（batch, 时间, 通道）
        original_input = x#(B, channels，time_len)
        if self.position_flag:
            channel_pos_encoding = self.channel_pos_encoding.repeat(original_input.shape[0],1,1)#(B,time_len,channels)
        mask, ids_restore = None, None
        if self.mask_ratio != 0:
            original_input, mask,ids_keep, ids_restore = self.random_mask_channels(original_input.transpose(1,2))
            original_input = original_input.transpose(1,2)#(B, channels，time_len)
            if self.position_flag:
                channel_pos_encoding = torch.gather(channel_pos_encoding,dim=-1, index=ids_keep)#取出新的通道编码
        channel_input = original_input.transpose(1,2)  # 通道分支输入：#(B，time_len, channels)
        if self.position_flag:
            channel_input = channel_input + channel_pos_encoding# 加通道位置编码
        else:
            channel_input = channel_input + 0  # 消融实验,不加入先验编码
        channel_feat = self.proj_time2embed(channel_input).transpose(1,2)#(B , channels,embed_dim)
        channel_feat = self.channel_branch(channel_feat) + channel_feat#(B, channels, embed_dim)
        # time_input = self.proj_chan2time(channel_feat)#(B,time_len,embed_dim)
        time_input = self.proj_embed2time(channel_feat.transpose(1,2))#(B,time_len,channels)
        time_input = self.proj_chan2embed(time_input.transpose(1,2))
        time_feat = self.patch_embed(time_input)#(B,T,embed_dim)
        time_feat = time_feat + self.time_pos_encoding  # 加时间位置编码
        time_feat = self.time_branch(time_feat)

        return self.norm(time_feat), mask, ids_restore

    def brain_prior_encoding(self,channels,embed_dim):
        xyz_brain = torch.load(self.xyz_brain,map_location='cpu').numpy()
        xyz_brain = self.absPoisionEncoding(coordinates=xyz_brain,d_model=embed_dim,max_len_channel=channels)
        return torch.from_numpy(xyz_brain).unsqueeze(0).float()#增加batch维度(1,embed_dim,channels)

    # def absPoisionEncoding(self, coordinates, d_model, max_len_channel=62):
    #     # 新增：先归一化3D坐标
    #     #coordinates = self.normalize_3d_coords(coordinates)
    #     # 1. 核心校验：保证总维度能被4均分，单维度编码长度为偶数
    #     assert d_model % 4 == 0, "d_model必须是4的倍数"
    #     d_per_dim = d_model // 4  # 单维度（X/Y/Z/脑区）的编码长度
    #     cut = 0
    #     if d_per_dim % 2 != 0:  # 保证单维度编码长度为偶数（sin/cos各占一半）
    #         d_per_dim += 1
    #         cut = 1  # 最后截断多余的1列
    #
    #     # 2. 初始化交错编码矩阵：[电极数, 总维度]
    #     pe_4d_interleaved = np.zeros((max_len_channel, d_model))
    #
    #     # 3. 仅处理numpy数组，避免类型错误
    #     if not isinstance(coordinates, np.ndarray):
    #         raise TypeError("coordinates必须是numpy.ndarray类型，形状为[电极数,4]（x/y/z/脑区）")
    #
    #     # 4. 计算单维度的衰减因子（核心修正：基于d_per_dim计算）
    #     # div_term长度 = d_per_dim//2（步长2），适配sin/cos的偶数维度
    #     div_term_3D = np.exp(
    #         np.arange(0, d_per_dim, 2) * -(math.log(10000.0) / d_per_dim))  # 3D空间归一化坐标和脑区离散坐标使用不同的scale 200 1000
    #     div_term_3D = np.expand_dims(div_term_3D, axis=0)  # 扩展维度：[1, d_per_dim//2]，方便广播
    #     div_term_region = np.exp(np.arange(0, d_per_dim, 2) * -(math.log(10000.0) / d_per_dim))
    #     div_term_region = np.expand_dims(div_term_region, axis=0)  # 扩展维度：[1, d_per_dim//2]，方便广播
    #
    #     # 5. 拆分4D坐标（保证形状为[max_len_channel,1]）
    #     position_x = coordinates[:, 0:1]  # [max_len_channel, 1]
    #     position_y = coordinates[:, 1:2]  # [max_len_channel, 1]
    #     position_z = coordinates[:, 2:3]  # [max_len_channel, 1]
    #     brain_region = coordinates[:, 3:4]  # 修正：取[:,3:4]保证维度为[max_len_channel,1]，避免脑区维度不一致
    #
    #     # 6. 生成单维度完整编码（修正：维度为[max_len_channel, d_per_dim]，保留高维信息）
    #     pe_x = np.zeros((max_len_channel, d_per_dim))
    #     pe_y = np.zeros((max_len_channel, d_per_dim))
    #     pe_z = np.zeros((max_len_channel, d_per_dim))
    #     pe_brain_region = np.zeros((max_len_channel, d_per_dim))
    #
    #     # 正弦（偶数位）+ 余弦（奇数位）编码
    #     pe_x[:, 0::2] = np.sin(position_x * div_term_3D)  # [max_len_channel, d_per_dim//2]
    #     pe_x[:, 1::2] = np.cos(position_x * div_term_3D)
    #     pe_y[:, 0::2] = np.sin(position_y * div_term_3D)
    #     pe_y[:, 1::2] = np.cos(position_y * div_term_3D)
    #     pe_z[:, 0::2] = np.sin(position_z * div_term_3D)
    #     pe_z[:, 1::2] = np.cos(position_z * div_term_3D)
    #     pe_brain_region[:, 0::2] = np.sin(brain_region * div_term_region)
    #     pe_brain_region[:, 1::2] = np.cos(brain_region * div_term_region)
    #
    #     # 7. 截断奇数维度的多余列（恢复偶数维度）
    #     if cut > 0:
    #         pe_x = pe_x[:, :-1]
    #         pe_y = pe_y[:, :-1]
    #         pe_z = pe_z[:, :-1]
    #         pe_brain_region = pe_brain_region[:, :-1]
    #         d_per_dim -= 1  # 更新单维度长度
    #
    #     # 8. 交错拼接核心逻辑（修正：逐维度填充，保留高维信息）
    #     # 每4维为一组：第4i维=X的第i维，4i+1=Y的第i维，4i+2=Z的第i维，4i+3=脑区的第i维
    #     for i in range(d_per_dim):
    #         pe_4d_interleaved[:, 4 * i] = pe_x[:, i]  # 第4i维 → X的第i维
    #         pe_4d_interleaved[:, 4 * i + 1] = pe_y[:, i]  # 第4i+1维 → Y的第i维
    #         pe_4d_interleaved[:, 4 * i + 2] = pe_z[:, i]  # 第4i+2维 → Z的第i维
    #         pe_4d_interleaved[:, 4 * i + 3] = pe_brain_region[:, i]  # 第4i+3维 → 脑区的第i维
    #
    #     # 9. 转置返回：(总维度数, 电极数)，适配后续融合逻辑
    #     return pe_4d_interleaved.transpose(1, 0)
    def absPoisionEncoding(self,coordinates, d_model, max_len_channel=62):
        assert d_model % 4 == 0  # d_model是嵌入层维度,x,y,z,brain_region均分所有嵌入层维度
        # 新增：先归一化3D坐标
        # coordinates = self.normalize_3d_coords(coordinates)
        # 1. 核心校验：保证总维度能被4均分，单维度编码长度为偶数
        assert d_model % 4 == 0, "d_model必须是4的倍数"
        d_per_dim = d_model // 4  # 单维度（X/Y/Z/脑区）的编码长度
        cut = 0
        if d_per_dim % 2 != 0:  # 保证单维度编码长度为偶数（sin/cos各占一半）
            d_per_dim += 1
            cut = 1  # 最后截断多余的1列



        # 3. 仅处理numpy数组，避免类型错误
        if not isinstance(coordinates, np.ndarray):
            raise TypeError("coordinates必须是numpy.ndarray类型，形状为[电极数,4]（x/y/z/脑区）")

        # 4. 计算单维度的衰减因子（核心修正：基于d_per_dim计算）
        # div_term长度 = d_per_dim//2（步长2），适配sin/cos的偶数维度
        div_term_3D = np.exp(
            np.arange(0, d_per_dim, 2) * -(math.log(10000.0) / d_per_dim))  # 3D空间归一化坐标和脑区离散坐标使用不同的scale 200 1000
        div_term_3D = np.expand_dims(div_term_3D, axis=0)  # 扩展维度：[1, d_per_dim//2]，方便广播
        div_term_region = np.exp(np.arange(0, d_per_dim, 2) * -(math.log(10000.0) / d_per_dim))
        div_term_region = np.expand_dims(div_term_region, axis=0)  # 扩展维度：[1, d_per_dim//2]，方便广播
        position_x = coordinates[:, 0:1]
        position_y = coordinates[:, 1:2]
        position_z = coordinates[:, 2:3]
        brain_region = coordinates[:, 3:]
        pe_x = np.zeros((max_len_channel, d_per_dim))
        pe_y = np.zeros((max_len_channel, d_per_dim))
        pe_z = np.zeros((max_len_channel, d_per_dim))
        pe_brain_region = np.zeros((max_len_channel, d_per_dim))
        pe_x[:, 0::2] = np.sin(position_x * div_term_3D)
        pe_x[:, 1::2] = np.cos(position_x * div_term_3D)
        pe_y[:, 0::2] = np.sin(position_y * div_term_3D)
        pe_y[:, 1::2] = np.cos(position_y * div_term_3D)
        pe_z[:, 0::2] = np.sin(position_z * div_term_3D)
        pe_z[:, 1::2] = np.cos(position_z * div_term_3D)
        pe_brain_region[:, 0::2] = np.sin(brain_region * div_term_region)
        pe_brain_region[:, 1::2] = np.cos(brain_region * div_term_region)

        pe_x = pe_x[:, :(d_model - cut)].transpose(1,0)#注意numpy的transpose和torch的transpose用法不同
        pe_y = pe_y[:, :(d_model - cut)].transpose(1,0)
        pe_z = pe_z[:, :(d_model - cut)].transpose(1,0)
        pe_brain_region = pe_brain_region[:, :(d_model - cut)].transpose(1,0)

        return np.concatenate((pe_x, pe_y, pe_z, pe_brain_region), axis=0)  # (维度数=x维度数+y维度数+z维度数+脑区独热编码维度数,电极数)


    def normalize_3d_coords(self, coordinates):
        """
        对3D坐标（前3列）做min-max归一化，映射到[-1,1]；脑区列（第4列）保持不变
        input: coordinates - [num_electrodes, 4]（x/y/z/脑区）
        output: normalized_coords - [num_electrodes, 4]
        """
        # 提取3D坐标
        coords_3d = coordinates[:, :3]
        # 按维度计算min/max
        min_vals = coords_3d.min(axis=0, keepdims=True)  # [1,3]
        max_vals = coords_3d.max(axis=0, keepdims=True)  # [1,3]
        # 避免除0（所有电极坐标相同的极端情况）
        ranges = np.maximum(max_vals - min_vals, 1e-8)
        # min-max归一化到[-1,1]
        coords_3d_norm = 2 * (coords_3d - min_vals) / ranges - 1
        # 拼接脑区列
        normalized_coords = np.concatenate([coords_3d_norm, coordinates[:, 3:4]], axis=1)
        return normalized_coords
    def T_position_encoding(self,time_pos_encoding):
        pos_embed = ut.get_1d_sincos_pos_embed(time_pos_encoding.shape[-1], self.T, cls_token=False)
        return torch.from_numpy(pos_embed).float().unsqueeze(0)

    def random_mask_channels(self, x):
        """
        将大脑划分为4个部分，然后在每个部分里去除1-2个通道。用更简洁的方法实现，使用整个大脑的均匀分布来挑选。
        """
        N, T, Channels = x.shape  # batch, length, dim。Channels是整个大脑的电极通道数量,T是每个电极的时间token数
        assert Channels == self.channels
        Channel_keep = int(Channels * (1 - self.mask_ratio))
        noise = torch.rand(N, Channels, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # print('ids_shuffle:', ids_shuffle)
        # print('ids_shuffle.shape',ids_shuffle.shape)

        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # print('ids_restore', ids_restore)
        # print('ids_restore.shape',ids_restore.shape)
        # keep the first subset
        ids_keep = ids_shuffle[:, :Channel_keep].unsqueeze(1).repeat(1, T, 1)
        x_masked = torch.gather(x, dim=-1, index=ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, Channels], device=x.device)#初始化屏蔽矩阵
        mask[:, :Channel_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)#此时mask才真正代表屏蔽矩阵
        return x_masked, mask, ids_keep,ids_restore# x_masked(N,T,Channel_keep)

class EEGHybridBranchModel_V3(nn.Module):
    """EEG双分支模型：时间+通道并行处理，共24层ViT Block"""

    def __init__(self,
                 time_len=400,  # 时间步数
                 patch_size=4,
                 channels=62,  # 电极通道数
                 mask_ratio=0.125,# mask之后的电极通道数
                 embed_dim=512,#嵌入层维度
                 num_heads=8,
                 mlp_ratio=4,
                 depth=4,
                 position_flag=True,
                 norm_layer=nn.LayerNorm,
                 xyz_brain_pth=None):  # 4个阶段

        super().__init__()
        self.dim = embed_dim
        self.num_channel_branch = depth
        self.T = time_len//patch_size #num_patch
        self.channels = channels#原始电极通道数
        self.mask_ratio = mask_ratio
        self.masked_Channels = int(channels*(1-mask_ratio))#mask之后剩余的电极通道数
        self.xyz_brain = xyz_brain_pth#3D坐标和脑区编码
        # 将时间点维度按照patch_size打包成token,并且通过分组卷积来实现通道间信息分离。也就是说此操作仅仅是为了将时间步打包成token，对电极通道无影响。

        # # 位置编码（假设已实现时间和通道位置编码）
        # time_pos_encoding = nn.Parameter(torch.randn(1, self.T, embed_dim),requires_grad=False)  # 时间位置编码
        # time_pos_encoding = self.T_position_encoding(time_pos_encoding)
        # self.register_buffer('time_pos_encoding', time_pos_encoding)
        self.position_flag = position_flag  # 是否启用位置编码,这个参数主要是为了第一阶段开启,第二阶段关闭.
        if self.position_flag:
            channel_pos_encoding = self.brain_prior_encoding(channels,time_len)  # 脑先验位置编码
            self.register_buffer('channel_pos_encoding', channel_pos_encoding)#注册为缓冲区，方便在不同设备上移动

        self.proj_time2embed = nn.Conv1d(
            in_channels=time_len,
            out_channels=embed_dim,
            kernel_size=1
        )
        self.channel_branch = ChannelBranch(
            num_blocks= self.num_channel_branch,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )
        self.pos_proj = nn.Sequential(
            nn.Linear(4, time_len//4),
            nn.Linear(time_len//4, time_len)
        )
        # self.proj = nn.Linear(self.masked_Channels,self.T)
        # self.patch_embed = PatchEmbed1D(time_len=time_len, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim)
        # 最终输出层
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        # 输入x维度：(B, channels，time_len ) → 原始EEG数据（batch, 时间, 通道）
        original_input = x#(B, channels，time_len)
        if self.position_flag:
            channel_pos_encoding = self.channel_pos_encoding.repeat(original_input.shape[0],1,1)#(B,channels,4)
            channel_pos_encoding =self.pos_proj(channel_pos_encoding).transpose(1,2)#(B,channels,time_len)--(B,time_len,channels)
        mask, ids_restore = None, None
        if self.mask_ratio != 0:
            original_input, mask,ids_keep, ids_restore = self.random_mask_channels(original_input.transpose(1,2))
            original_input = original_input.transpose(1,2)#(B, channels，time_len)
            if self.position_flag:
                # print('channels_pos.shape',channel_pos_encoding.shape)
                # print('ids_keep.shape',ids_keep.shape)
                channel_pos_encoding = torch.gather(channel_pos_encoding,dim=-1, index=ids_keep)#取出新的通道编码
        channel_input = original_input.transpose(1,2)  # 通道分支输入：#(B，time_len, channels)
        if self.position_flag:
            channel_input = channel_input + channel_pos_encoding# 加可学习通道位置编码
        else:
            channel_input = channel_input + 0  # 消融实验,不加入先验编码
        channel_feat = self.proj_time2embed(channel_input).transpose(1,2)#(B , channels,embed_dim)
        channel_feat = self.channel_branch(channel_feat)#(B, channels, embed_dim)
        # time_input = self.proj_chan2time(channel_feat)#(B,time_len,embed_dim)
        # time_feat = self.patch_embed(time_input.transpose(1,2))#(B,T,embed_dim)
        # time_feat = time_feat + self.time_pos_encoding  # 加时间位置编码
        # time_feat = self.time_branch(time_feat)

        return self.norm(channel_feat), mask, ids_restore

    def brain_prior_encoding(self,channels,embed_dim):
        xyz_brain = torch.load(self.xyz_brain,map_location='cpu').float()
        return self.normalize_4d_coords(xyz_brain).unsqueeze(0)#(1,channels,4)
    def normalize_4d_coords(self, coordinates):
        """

        """
        # 提取3D坐标
        coords_3d = coordinates[:, :3]
        # 计算均值和方差
        mean = torch.mean(coords_3d)
        std = torch.std(coords_3d)
        normalized_3d_coords = (coords_3d - mean) / std
        # 拼接脑区列,5个脑区（0-4），除以4来归一化
        normalized_coords = torch.cat([normalized_3d_coords, coordinates[:, 3:4]/4.], dim=1)
        return normalized_coords


    def T_position_encoding(self,time_pos_encoding):
        pos_embed = ut.get_1d_sincos_pos_embed(time_pos_encoding.shape[-1], self.T, cls_token=False)
        return torch.from_numpy(pos_embed).float().unsqueeze(0)

    def random_mask_channels(self, x):
        """
        将大脑划分为4个部分，然后在每个部分里去除1-2个通道。用更简洁的方法实现，使用整个大脑的均匀分布来挑选。
        """
        N, T, Channels = x.shape  # batch, length, dim。Channels是整个大脑的电极通道数量,T是每个电极的时间token数
        assert Channels == self.channels
        Channel_keep = int(Channels * (1 - self.mask_ratio))
        noise = torch.rand(N, Channels, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # print('ids_shuffle:', ids_shuffle)
        # print('ids_shuffle.shape',ids_shuffle.shape)

        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # print('ids_restore', ids_restore)
        # print('ids_restore.shape',ids_restore.shape)
        # keep the first subset
        ids_keep = ids_shuffle[:, :Channel_keep].unsqueeze(1).repeat(1, T, 1)
        x_masked = torch.gather(x, dim=-1, index=ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, Channels], device=x.device)#初始化屏蔽矩阵
        mask[:, :Channel_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)#此时mask才真正代表屏蔽矩阵
        return x_masked, mask, ids_keep,ids_restore# x_masked(N,T,Channel_keep)

class EEGHybridBranchModel_V4(nn.Module):
    """EEG时空混合分支模型：时间+通道并行处理，共24层ViT Block"""

    def __init__(self,
                 time_len=400,  # 时间步数
                 patch_size=4,
                 channels=62,  # 电极通道数
                 mask_ratio=0.125,# mask之后的电极通道数
                 embed_dim=512,#嵌入层维度
                 num_heads=8,
                 mlp_ratio=4,
                 depth=4,
                 position_flag=True,
                 norm_layer=nn.LayerNorm,
                 xyz_brain_pth=None,
                 fixed_electrode_indices=None,
                 fixed_electrode_index_base=0,
                 position_ablation='none',
                 position_ablation_seed=0,
                 position_ablation_ref_index=0,
                 position_ablation_virtual_coord=None,
                 spatial_depth=None,
                 vit_order='spatial_temporal'):  # 4个阶段

        super().__init__()
        self.dim = embed_dim
        if spatial_depth is None:
            spatial_depth = depth // 8
        self.spatial_depth = int(spatial_depth)
        self.temporal_depth = depth - self.spatial_depth
        if self.spatial_depth < 0 or self.temporal_depth < 0:
            raise ValueError(f'spatial_depth must be in [0, depth], got spatial_depth={self.spatial_depth}, depth={depth}')
        if vit_order not in ('spatial_temporal', 'temporal_spatial'):
            raise ValueError("vit_order must be 'spatial_temporal' or 'temporal_spatial'")
        self.vit_order = vit_order
        self.num_channel_branch = self.spatial_depth
        self.T = time_len//patch_size #num_patch
        self.channels = channels#原始电极通道数
        self.mask_ratio = mask_ratio
        self.masked_Channels = int(channels*(1-mask_ratio))#mask之后剩余的电极通道数
        self.xyz_brain = xyz_brain_pth#3D坐标和脑区编码
        self.position_ablation = position_ablation or 'none'
        self.position_ablation_seed = position_ablation_seed
        self.position_ablation_ref_index = position_ablation_ref_index
        self.position_ablation_virtual_coord = position_ablation_virtual_coord
        fixed_indices = _load_index_tensor(
            fixed_electrode_indices,
            index_base=fixed_electrode_index_base,
            expected_len=self.masked_Channels if mask_ratio != 0 else None,
            max_channels=channels
        )
        if fixed_indices is not None:
            self.register_buffer('fixed_electrode_indices', fixed_indices)
            print('启用固定电极索引:', fixed_indices.tolist())
        else:
            self.fixed_electrode_indices = None
        # 将时间点维度按照patch_size打包成token,并且通过分组卷积来实现通道间信息分离。也就是说此操作仅仅是为了将时间步打包成token，对电极通道无影响。

        # 位置编码（假设已实现时间和通道位置编码）
        time_pos_encoding = nn.Parameter(torch.randn(1, self.T, embed_dim),requires_grad=False)  # 时间位置编码
        time_pos_encoding = self.T_position_encoding(time_pos_encoding)
        self.register_buffer('time_pos_encoding', time_pos_encoding)
        self.position_flag = position_flag  # 是否启用位置编码,这个参数主要是为了第一阶段开启,第二阶段关闭.
        self.channel_pos_dim = 4
        if self.position_flag:
            channel_pos_encoding = self.brain_prior_encoding()  # 脑先验位置编码
            self.channel_pos_dim = channel_pos_encoding.shape[-1]
            self.register_buffer('channel_pos_encoding', channel_pos_encoding)#注册为缓冲区，方便在不同设备上移动
            if self.position_ablation != 'none':
                print('位置编码消融模式:', self.position_ablation)

        self.proj_chan2time = nn.Conv1d(
            in_channels=self.masked_Channels,
            out_channels=time_len,
            kernel_size=1
        )

        if self.vit_order == 'spatial_temporal':
            self.proj_time2embed = nn.Conv1d(
                in_channels=time_len,
                out_channels=embed_dim,
                kernel_size=1
            )
        else:
            self.proj_channels2embed = nn.Conv1d(
                in_channels=self.masked_Channels,
                out_channels=embed_dim,
                kernel_size=1
            )
            self.proj_time2channel = nn.Conv1d(
                in_channels=self.T,
                out_channels=self.masked_Channels,
                kernel_size=1
            )
        # 4个阶段，每个阶段3时间+3通道Block（共4×6=24层）
        self.time_branch = TimeBranch(
            num_blocks=self.temporal_depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )
        self.channel_branch = ChannelBranch(
            num_blocks=self.spatial_depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )
        self.pos_proj = nn.Sequential(
            nn.Linear(self.channel_pos_dim, time_len // 4),
            nn.Linear(time_len // 4, time_len)
        )
        # self.proj = nn.Linear(self.masked_Channels,self.T)
        self.patch_embed = PatchEmbed1D(time_len=time_len, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim)
        # 最终输出层
        self.norm = norm_layer(embed_dim)
        print('空间ViT层数:', self.spatial_depth)
        print('时序ViT层数:', self.temporal_depth)
        print('空间/时序顺序:', self.vit_order)

    def forward(self, x):
        # 输入x维度：(B, channels，time_len ) → 原始EEG数据（batch, 时间, 通道）
        original_input = x#(B, channels，time_len)
        if self.position_flag:
            channel_pos_encoding = self.channel_pos_encoding.repeat(original_input.shape[0], 1, 1)#(B,channels,4)
            channel_pos_encoding = self.pos_proj(channel_pos_encoding).transpose(1,2)  # (B,channels,time_len)--(B,time_len,channels)
        mask, ids_restore = None, None
        if self.mask_ratio != 0:
            original_input, mask,ids_keep, ids_restore = self.random_mask_channels(original_input.transpose(1,2))
            original_input = original_input.transpose(1,2)#(B, channels，time_len)
            if self.position_flag:
                channel_pos_encoding = torch.gather(channel_pos_encoding,dim=-1, index=ids_keep)#取出新的通道编码
        channel_input = original_input.transpose(1,2)  # 通道分支输入：#(B，time_len, channels)
        if self.position_flag:
            channel_input = channel_input + channel_pos_encoding# 加通道位置编码
        else:
            channel_input = channel_input + 0  # 消融实验,不加入先验编码
        if self.vit_order == 'spatial_temporal':
            channel_feat = self.proj_time2embed(channel_input).transpose(1,2)#(B , channels,embed_dim)
            channel_feat = self.channel_branch(channel_feat) + channel_feat#(B, channels, embed_dim)
            time_input = self.proj_chan2time(channel_feat)#(B,time_len,embed_dim)
            time_feat = self.patch_embed(time_input.transpose(1,2))#(B,T,embed_dim)
            time_feat = time_feat + self.time_pos_encoding  # 加时间位置编码
            time_feat = self.time_branch(time_feat)
        else:
            time_input = self.proj_channels2embed(channel_input.transpose(1, 2))#(B,embed_dim,time_len)
            time_feat = self.patch_embed(time_input)#(B,T,embed_dim)
            time_feat = time_feat + self.time_pos_encoding
            time_feat = self.time_branch(time_feat)
            channel_feat = self.proj_time2channel(time_feat)#(B,channels,embed_dim)
            channel_feat = self.channel_branch(channel_feat) + channel_feat#(B,channels,embed_dim)
            time_input = self.proj_chan2time(channel_feat)#(B,time_len,embed_dim)
            time_feat = self.patch_embed(time_input.transpose(1,2))#(B,T,embed_dim)
            time_feat = time_feat + self.time_pos_encoding

        return self.norm(time_feat), mask, ids_restore

    def brain_prior_encoding(self):
        xyz_brain = torch.load(self.xyz_brain,map_location='cpu').float()
        return self.normalize_xyz_onehot_region(xyz_brain).unsqueeze(0)#(1,channels,3+num_regions)

    def refresh_channel_pos_encoding(self):
        if not self.position_flag:
            return
        channel_pos_encoding = self.brain_prior_encoding()
        channel_pos_encoding = channel_pos_encoding.to(
            device=self.channel_pos_encoding.device,
            dtype=self.channel_pos_encoding.dtype
        )
        if channel_pos_encoding.shape != self.channel_pos_encoding.shape:
            raise ValueError(
                f'refreshed channel_pos_encoding shape {tuple(channel_pos_encoding.shape)} '
                f'does not match existing shape {tuple(self.channel_pos_encoding.shape)}'
            )
        self.channel_pos_encoding.copy_(channel_pos_encoding)
        if self.position_ablation != 'none':
            print('已在加载权重后重新应用位置编码消融:', self.position_ablation)
            if self.position_ablation == 'same_coord':
                print('same_coord参考电极索引:', self.position_ablation_ref_index)
            if self.position_ablation == 'virtual_coord':
                print('virtual_coord虚拟电极坐标:', self.position_ablation_virtual_coord)

    def normalize_xyz_onehot_region(self, coordinates):
        """

        """
        if coordinates.shape[0] != self.channels:
            raise ValueError(f'xyz_brain channels {coordinates.shape[0]} != model channels {self.channels}')
        if coordinates.shape[1] < 4:
            raise ValueError('xyz_brain must have at least 4 columns: x, y, z, region_id.')
        # 提取3D坐标
        coords_3d = coordinates[:, :3]
        # 计算均值和方差
        mean = torch.mean(coords_3d)
        std = torch.std(coords_3d)
        normalized_3d_coords = (coords_3d - mean) / std
        region_ids = coordinates[:, 3].long()
        region_id_offset = 1 if torch.min(region_ids) >= 1 else 0
        if region_id_offset == 1:
            region_ids = region_ids - 1
        if torch.any(region_ids < 0):
            raise ValueError('region ids must be non-negative, or one-based positive integers.')
        num_regions = int(torch.max(region_ids).item()) + 1
        one_hot_region = F.one_hot(region_ids, num_classes=num_regions).float()
        normalized_3d_coords, one_hot_region = self.apply_position_ablation(
            normalized_3d_coords,
            one_hot_region,
            coords_mean=mean,
            coords_std=std,
            num_regions=num_regions,
            region_id_offset=region_id_offset
        )
        return torch.cat([normalized_3d_coords, one_hot_region], dim=1)

    def apply_position_ablation(self, coords_3d, one_hot_region, coords_mean=None, coords_std=None,
                                num_regions=None, region_id_offset=0):
        mode = self.position_ablation
        if mode in (None, 'none'):
            return coords_3d, one_hot_region
        valid_modes = {'shuffle_xyz', 'shuffle_region', 'shuffle_both', 'same_coord', 'virtual_coord'}
        if mode not in valid_modes:
            raise ValueError(f'Unsupported position_ablation: {mode}. Valid modes: {sorted(valid_modes)}')

        if mode == 'same_coord':
            ref_index = int(self.position_ablation_ref_index)
            if ref_index < 0 or ref_index >= coords_3d.shape[0]:
                raise ValueError(
                    f'position_ablation_ref_index must be in [0, {coords_3d.shape[0] - 1}], got {ref_index}'
                )
            coords_3d = coords_3d[ref_index:ref_index + 1].repeat(coords_3d.shape[0], 1)
            one_hot_region = one_hot_region[ref_index:ref_index + 1].repeat(one_hot_region.shape[0], 1)
            return coords_3d, one_hot_region

        if mode == 'virtual_coord':
            virtual_coord = self.parse_virtual_coord(
                self.position_ablation_virtual_coord,
                device=coords_3d.device,
                dtype=coords_3d.dtype
            )
            if coords_mean is None or coords_std is None or num_regions is None:
                raise ValueError('virtual_coord requires original coordinate mean/std and num_regions.')
            virtual_xyz = (virtual_coord[:3] - coords_mean.to(device=coords_3d.device, dtype=coords_3d.dtype)) / coords_std.to(device=coords_3d.device, dtype=coords_3d.dtype)
            virtual_region_id = int(round(float(virtual_coord[3].item()))) - int(region_id_offset)
            if virtual_region_id < 0 or virtual_region_id >= int(num_regions):
                raise ValueError(
                    f'virtual region_id must map to existing region indices [0, {int(num_regions) - 1}] '
                    f'after one-based conversion, got {virtual_region_id}.'
                )
            virtual_region = F.one_hot(
                torch.tensor(virtual_region_id, device=one_hot_region.device),
                num_classes=int(num_regions)
            ).float().to(dtype=one_hot_region.dtype)
            coords_3d = virtual_xyz.unsqueeze(0).repeat(coords_3d.shape[0], 1)
            one_hot_region = virtual_region.unsqueeze(0).repeat(one_hot_region.shape[0], 1)
            return coords_3d, one_hot_region

        generator = None
        if self.position_ablation_seed is not None:
            generator = torch.Generator(device=coords_3d.device)
            generator.manual_seed(int(self.position_ablation_seed))

        if mode in {'shuffle_xyz', 'shuffle_both'}:
            xyz_perm = torch.randperm(coords_3d.shape[0], generator=generator, device=coords_3d.device)
            coords_3d = coords_3d[xyz_perm]
        if mode in {'shuffle_region', 'shuffle_both'}:
            region_perm = torch.randperm(one_hot_region.shape[0], generator=generator, device=one_hot_region.device)
            one_hot_region = one_hot_region[region_perm]
        return coords_3d, one_hot_region

    @staticmethod
    def parse_virtual_coord(virtual_coord, device=None, dtype=torch.float32):
        if virtual_coord is None:
            raise ValueError('virtual_coord requires position_ablation_virtual_coord="x,y,z,region_id".')
        if isinstance(virtual_coord, str):
            parts = [p.strip() for p in virtual_coord.replace('，', ',').split(',') if p.strip()]
            if len(parts) != 4:
                raise ValueError('position_ablation_virtual_coord must have four comma-separated values: x,y,z,region_id.')
            values = [float(p) for p in parts]
            return torch.tensor(values, device=device, dtype=dtype)
        if isinstance(virtual_coord, torch.Tensor):
            values = virtual_coord.detach().flatten().to(device=device, dtype=dtype)
        else:
            values = torch.tensor(virtual_coord, device=device, dtype=dtype).flatten()
        if values.numel() != 4:
            raise ValueError('position_ablation_virtual_coord must contain exactly four values: x,y,z,region_id.')
        return values


    def T_position_encoding(self,time_pos_encoding):
        pos_embed = ut.get_1d_sincos_pos_embed(time_pos_encoding.shape[-1], self.T, cls_token=False)
        return torch.from_numpy(pos_embed).float().unsqueeze(0)

    def random_mask_channels(self, x):
        """

        """
        N, T, Channels = x.shape  # batch, length, dim。Channels是整个大脑的电极通道数量,T是每个电极的时间token数
        # print('Channels:', Channels)
        # print(self.channels)
        assert Channels == self.channels
        Channel_keep = int(Channels * (1 - self.mask_ratio))
        if self.fixed_electrode_indices is not None:
            keep_indices = self.fixed_electrode_indices.to(x.device)
            mask_indices = torch.tensor(
                [i for i in range(Channels) if i not in set(keep_indices.detach().cpu().tolist())],
                device=x.device,
                dtype=torch.long
            )
            ids_shuffle = torch.cat([keep_indices, mask_indices], dim=0).unsqueeze(0).repeat(N, 1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = keep_indices.unsqueeze(0).unsqueeze(1).repeat(N, T, 1)
            x_masked = torch.gather(x, dim=-1, index=ids_keep)
            mask = torch.ones([N, Channels], device=x.device)
            mask.scatter_(1, keep_indices.unsqueeze(0).repeat(N, 1), 0)
            return x_masked, mask, ids_keep, ids_restore
        noise = torch.rand(N, Channels, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # print('ids_shuffle:', ids_shuffle)
        # print('ids_shuffle.shape',ids_shuffle.shape)

        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # print('ids_restore', ids_restore)
        # print('ids_restore.shape',ids_restore.shape)
        # keep the first subset
        ids_keep = ids_shuffle[:, :Channel_keep].unsqueeze(1).repeat(1, T, 1)
        x_masked = torch.gather(x, dim=-1, index=ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, Channels], device=x.device)#初始化屏蔽矩阵
        mask[:, :Channel_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)#此时mask才真正代表屏蔽矩阵
        return x_masked, mask, ids_keep,ids_restore# x_masked(N,T,Channel_keep)

class MChan_AEforEEG(nn.Module):
    """ Masked Channels Autoencoder with VisionTransformer backbone.
    """

    def __init__(self, time_len=400, patch_size=4, embed_dim=512, in_chans=62,mask_ratio=0.25,
                 depth=4, num_heads=16, decoder_embed_dim=512,
                 decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4.,position_flag=True, norm_layer=nn.LayerNorm, xyz_brain=None,
                 fixed_electrode_indices=None, fixed_electrode_index_base=0):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.time_len = time_len
        self.in_chans = in_chans
        self.num_patches = int(time_len / patch_size)
        self.masked_channels = int(in_chans*(1-mask_ratio))#剩下的电极通道数
        self.droped_channels = in_chans - int(in_chans*(1-mask_ratio))#有多少电极通道被去除

        # 将时间点维度按照patch_size打包成token,并且通过分组卷积来实现通道间信息分离。也就是说此操作仅仅是为了将时间步打包成token，对电极通道无影响。
        # self.patch_embed = PatchEmbed1D(time_len=time_len, patch_size=patch_size, in_chans=in_chans, embed_dim=in_chans,
        #                                 group=in_chans)
        # self.encoder = EEGTwoBranchModel(T=self.num_patches, channels=in_chans,mask_ratio=mask_ratio,embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
        #                                  num_stages=depth, norm_layer=norm_layer, xyz_brain_pth=xyz_brain)
        # --------------------------------------------------------------------------
        if position_flag:
            print('启用位置编码')
        else:
            print('关闭位置编码')
        self.encoder = EEGHybridBranchModel(time_len=time_len,patch_size=patch_size,channels=in_chans,mask_ratio=mask_ratio,embed_dim=embed_dim,num_heads=num_heads,
                                            mlp_ratio=mlp_ratio,depth=depth,position_flag=position_flag,norm_layer=norm_layer,xyz_brain_pth=xyz_brain)#编码器第一二阶段位置编码情况可能不同
        self.decoder = EEGHybridBranchModel(time_len=time_len,patch_size=patch_size,channels=in_chans,mask_ratio=0,embed_dim=decoder_embed_dim,
                                            num_heads=decoder_num_heads,mlp_ratio=mlp_ratio,depth=decoder_depth,position_flag=position_flag,norm_layer=norm_layer,xyz_brain_pth=xyz_brain)#解码器只参与第一阶段位置编码
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim),
        #                                       requires_grad=False)  # fixed sin-cos embedding
        # self.decoder_embed = nn.Linear(self.num_patches, decoder_embed_dim, bias=True)
        self.decoder_embed = nn.Linear(embed_dim, self.masked_channels*patch_size, bias=True)

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.droped_channels))#(1,1,C)后面还会有初始化。很关键的一点是T维度要是1，否则模型坍塌，所有输出结果都极为相似。这是为了保证，通道之间不同，通道内部相同。

        # self.decoder_blocks = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])
        # self.decoder_blocks = nn.ModuleList([
        #     ChannelBranch(
        #         num_blocks=3,
        #         embed_dim=decoder_embed_dim,
        #         num_heads=decoder_num_heads,
        #         mlp_ratio=mlp_ratio
        #     )
        # for _ in range(decoder_depth)])
        # channel_pos_encoding = self.encoder.brain_prior_encoding(in_chans, self.num_patches)  # 脑先验位置编码
        # self.register_buffer('channel_pos_encoding', channel_pos_encoding)  # 注册为缓冲区，方便在不同设备上移动

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans*patch_size, bias=True)  # encoder to decoder
        # --------------------------------------------------------------------------

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        print('自编码器输入电极通道数为:',self.in_chans)
        print('自编码器保留电极通道数为:',self.masked_channels)
        print('自编码器通道掩码率为:', self.mask_ratio)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        # decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches,
        #                                                cls_token=False)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        #torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def patchify(self, imgs):
        """
        imgs: [N, chan, T]
        x: (N, L, patch_size)
        x: [N, chan * 4, T/4]
        """
        p = self.patch_size

        # h = imgs.shape[2] // p
        assert imgs.shape[1] % p == 0
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1] // p, -1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, 1, num_voxels)
        """
        p = self.patch_size
        h = x.shape[1]

        imgs = x.reshape(shape=(x.shape[0], -1, x.shape[2] // p))
        return imgs.transpose(1, 2)

    def forward_encoder(self, x):
        #x(N,channels,T)
        # embed patches
        # x = self.patch_embed(x)
        # print('encoder embed')
        # print(x.shape)
        # masking: length -> length * mask_ratio
        #print('编码器部分输入维度',x.shape)
        if self.mask_ratio != 0:
            x , mask, ids_restore= self.encoder(x)
            return x, mask, ids_restore
        # apply Transformer blocks
        x,_,_ = self.encoder(x)
        #print('编码器部分输出维度', x.shape)
        return x #(N,T,embed_dim)

    def forward_decoder(self, x, ids_restore=None):
        #x(B,T,embed_dim)
        #print('编码器部分输入维度', x.shape)
        N,T,embed_dim = x.shape

        x = self.decoder_embed(x).view(N,T*self.patch_size,self.masked_channels)#(B,T,masked_channels*patch_size)--(B,T*4,masked_channels)time_len=T*4
        mask_tokens = self.mask_token.repeat(x.shape[0], x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=-1)  # (B,time_len,masked_channels+droped_channels)

        x_ = torch.gather(x_, dim=-1, index=ids_restore.unsqueeze(1).repeat(1, x.shape[1], 1))  # unshuffle
        x,_,_ = self.decoder(x_.transpose(1,2))
        # x_ = x_ + self.channel_pos_encoding
        # # embed tokens
        # # x = self.decoder_embed(x_)#x(B,T,embed_channels)
        # x = self.decoder_embed(x_.transpose(1, 2))#(B,channels,embed_dim)
        # print('decoder embed')
        # print(x.shape)
        # append mask tokens to sequence


        # x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token

        # x = x_
        # add pos embed
        # x = x + self.decoder_pos_embed
        # x = x + self.channel_pos_encoding.transpose(1,2)
        # x = x + self.decoder_pos_embed[:, 1:, :]

        # apply Transformer blocks
        # for blk in self.decoder_blocks:
        #     x = blk(x)
        x = self.decoder_norm(x)
        # print(x.shape)
        # predictor projection
        x = self.decoder_pred(x)
        # print("x.shape",x.shape)

        return x.view(N,self.time_len, self.in_chans).transpose(1,2)#(N,channels,time_point)

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, chanels, timepoint]
        pred: [N,channels,timepoint]
        mask: [N, channels], 0 is keep, 1 is remove,
        """
        # imgs = imgs.transpose(1, 2)#(N,timepoint,channels)
        # target = self.patchify(imgs)#(N,timepoint//4,4*channels) T=timepoint//4
        N,C,T = pred.shape
        # pred = pred.view(N,T*4,C//4)#(N,timepoint,channels)
        # target = imgs.transpose(1,2)
        loss = (pred - imgs) ** 2#(N,channels,timepoint)

        loss = loss.mean(dim=-1)  # [N, channels], mean loss per channel
        # loss = loss.mean()
        loss = (loss * mask).sum() / mask.sum() if mask.sum() != 0 else (
                    loss * mask).sum()  # mean loss on removed patches
        return loss

    def forward(self, x):
        # latent = self.forward_encoder(imgs, mask_ratio)
        # x(N,channels,T)
        if self.mask_ratio != 0:
            latent, mask, ids_restore = self.forward_encoder(x)#latent(N,T,channels)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p]
            loss = self.forward_loss(x, pred, mask)
            return loss, pred, mask
        else:
            latent = self.forward_encoder(x)
            return latent

class MChan_AEforEEG_V2(nn.Module):
    """ Masked Channels Autoencoder with VisionTransformer backbone.
    """

    def __init__(self, time_len=400, patch_size=4, embed_dim=512, in_chans=62,mask_ratio=0.25,
                 depth=4, num_heads=16, decoder_embed_dim=512,
                 decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4.,position_flag=True, norm_layer=nn.LayerNorm, xyz_brain=None,
                 fixed_electrode_indices=None, fixed_electrode_index_base=0):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.time_len = time_len
        self.in_chans = in_chans
        self.num_patches = int(time_len / patch_size)
        self.masked_channels = int(in_chans*(1-mask_ratio))#剩下的电极通道数
        self.droped_channels = in_chans - int(in_chans*(1-mask_ratio))#有多少电极通道被去除

        # 将时间点维度按照patch_size打包成token,并且通过分组卷积来实现通道间信息分离。也就是说此操作仅仅是为了将时间步打包成token，对电极通道无影响。
        # self.patch_embed = PatchEmbed1D(time_len=time_len, patch_size=patch_size, in_chans=in_chans, embed_dim=in_chans,
        #                                 group=in_chans)
        # self.encoder = EEGTwoBranchModel(T=self.num_patches, channels=in_chans,mask_ratio=mask_ratio,embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
        #                                  num_stages=depth, norm_layer=norm_layer, xyz_brain_pth=xyz_brain)
        # --------------------------------------------------------------------------
        if position_flag:
            print('启用位置编码')
        else:
            print('关闭位置编码')
        self.encoder = EEGHybridBranchModel_V2(time_len=time_len,patch_size=patch_size,channels=in_chans,mask_ratio=mask_ratio,embed_dim=embed_dim,num_heads=num_heads,
                                            mlp_ratio=mlp_ratio,depth=depth,position_flag=position_flag,norm_layer=norm_layer,xyz_brain_pth=xyz_brain)#编码器第一二阶段位置编码情况可能不同
        self.decoder = EEGHybridBranchModel_V2(time_len=time_len,patch_size=patch_size,channels=in_chans,mask_ratio=0,embed_dim=decoder_embed_dim,
                                            num_heads=decoder_num_heads,mlp_ratio=mlp_ratio,depth=decoder_depth,position_flag=position_flag,norm_layer=norm_layer,xyz_brain_pth=xyz_brain)#解码器只参与第一阶段位置编码
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim),
        #                                       requires_grad=False)  # fixed sin-cos embedding
        # self.decoder_embed = nn.Linear(self.num_patches, decoder_embed_dim, bias=True)
        self.decoder_embed = nn.Linear(embed_dim, self.masked_channels*patch_size, bias=True)

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.droped_channels))#(1,1,C)后面还会有初始化。很关键的一点是T维度要是1，否则模型坍塌，所有输出结果都极为相似。这是为了保证，通道之间不同，通道内部相同。

        # self.decoder_blocks = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])
        # self.decoder_blocks = nn.ModuleList([
        #     ChannelBranch(
        #         num_blocks=3,
        #         embed_dim=decoder_embed_dim,
        #         num_heads=decoder_num_heads,
        #         mlp_ratio=mlp_ratio
        #     )
        # for _ in range(decoder_depth)])
        # channel_pos_encoding = self.encoder.brain_prior_encoding(in_chans, self.num_patches)  # 脑先验位置编码
        # self.register_buffer('channel_pos_encoding', channel_pos_encoding)  # 注册为缓冲区，方便在不同设备上移动

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans*patch_size, bias=True)  # encoder to decoder
        # --------------------------------------------------------------------------

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        print('自编码器输入电极通道数为:',self.in_chans)
        print('自编码器通道掩码率为:', self.mask_ratio)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        # decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches,
        #                                                cls_token=False)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        #torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def patchify(self, imgs):
        """
        imgs: [N, chan, T]
        x: (N, L, patch_size)
        x: [N, chan * 4, T/4]
        """
        p = self.patch_size

        # h = imgs.shape[2] // p
        assert imgs.shape[1] % p == 0
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1] // p, -1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, 1, num_voxels)
        """
        p = self.patch_size
        h = x.shape[1]

        imgs = x.reshape(shape=(x.shape[0], -1, x.shape[2] // p))
        return imgs.transpose(1, 2)

    def forward_encoder(self, x):
        #x(N,channels,T)
        # embed patches
        # x = self.patch_embed(x)
        # print('encoder embed')
        # print(x.shape)
        # masking: length -> length * mask_ratio
        if self.mask_ratio != 0:
            x , mask, ids_restore= self.encoder(x)
            return x, mask, ids_restore
        # apply Transformer blocks
        x,_,_ = self.encoder(x)
        return x #(N,T,embed_dim+C)

    def forward_decoder(self, x, ids_restore=None):
        #x(B,T,embed_dim)
        N,T,embed_dim = x.shape

        x = self.decoder_embed(x).view(N,T*self.patch_size,self.masked_channels)#(B,T,masked_channels*patch_size)--(B,T*4,masked_channels)time_len=T*4
        mask_tokens = self.mask_token.repeat(x.shape[0], x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=-1)  # (B,time_len,masked_channels+droped_channels)

        x_ = torch.gather(x_, dim=-1, index=ids_restore.unsqueeze(1).repeat(1, x.shape[1], 1))  # unshuffle
        x,_,_ = self.decoder(x_.transpose(1,2))
        # x_ = x_ + self.channel_pos_encoding
        # # embed tokens
        # # x = self.decoder_embed(x_)#x(B,T,embed_channels)
        # x = self.decoder_embed(x_.transpose(1, 2))#(B,channels,embed_dim)
        # print('decoder embed')
        # print(x.shape)
        # append mask tokens to sequence


        # x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token

        # x = x_
        # add pos embed
        # x = x + self.decoder_pos_embed
        # x = x + self.channel_pos_encoding.transpose(1,2)
        # x = x + self.decoder_pos_embed[:, 1:, :]

        # apply Transformer blocks
        # for blk in self.decoder_blocks:
        #     x = blk(x)
        x = self.decoder_norm(x)
        # print(x.shape)
        # predictor projection
        x = self.decoder_pred(x)
        # print("x.shape",x.shape)

        return x.view(N,self.time_len, self.in_chans).transpose(1,2)#(N,channels,time_point)

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, chanels, timepoint]
        pred: [N,channels,timepoint]
        mask: [N, channels], 0 is keep, 1 is remove,
        """
        # imgs = imgs.transpose(1, 2)#(N,timepoint,channels)
        # target = self.patchify(imgs)#(N,timepoint//4,4*channels) T=timepoint//4
        N,C,T = pred.shape
        # pred = pred.view(N,T*4,C//4)#(N,timepoint,channels)
        # target = imgs.transpose(1,2)
        loss = (pred - imgs) ** 2#(N,channels,timepoint)

        loss = loss.mean(dim=-1)  # [N, channels], mean loss per channel
        # loss = loss.mean()
        loss = (loss * mask).sum() / mask.sum() if mask.sum() != 0 else (
                    loss * mask).sum()  # mean loss on removed patches
        return loss

    def forward(self, x):
        # latent = self.forward_encoder(imgs, mask_ratio)
        # x(N,channels,T)
        if self.mask_ratio != 0:
            latent, mask, ids_restore = self.forward_encoder(x)#latent(N,T,channels)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p]
            loss = self.forward_loss(x, pred, mask)
            return loss, pred, mask
        else:
            latent = self.forward_encoder(x)
            return latent

class MChan_AEforEEG_V3(nn.Module):
    """ Masked Channels Autoencoder with VisionTransformer backbone.
    """

    def __init__(self, time_len=400, patch_size=4, embed_dim=512, in_chans=62,mask_ratio=0.25,
                 depth=4, num_heads=16, decoder_embed_dim=512,
                 decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4.,position_flag=True, norm_layer=nn.LayerNorm, xyz_brain=None):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.time_len = time_len
        self.in_chans = in_chans
        self.num_patches = int(time_len / patch_size)
        self.masked_channels = int(in_chans*(1-mask_ratio))#剩下的电极通道数
        self.droped_channels = in_chans - int(in_chans*(1-mask_ratio))#有多少电极通道被去除

        # 将时间点维度按照patch_size打包成token,并且通过分组卷积来实现通道间信息分离。也就是说此操作仅仅是为了将时间步打包成token，对电极通道无影响。
        # self.patch_embed = PatchEmbed1D(time_len=time_len, patch_size=patch_size, in_chans=in_chans, embed_dim=in_chans,
        #                                 group=in_chans)
        # self.encoder = EEGTwoBranchModel(T=self.num_patches, channels=in_chans,mask_ratio=mask_ratio,embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
        #                                  num_stages=depth, norm_layer=norm_layer, xyz_brain_pth=xyz_brain)
        # --------------------------------------------------------------------------
        if position_flag:
            print('启用位置编码')
        else:
            print('关闭位置编码')
        self.encoder = EEGHybridBranchModel_V3(time_len=time_len,patch_size=patch_size,channels=in_chans,mask_ratio=mask_ratio,embed_dim=embed_dim,num_heads=num_heads,
                                            mlp_ratio=mlp_ratio,depth=depth,position_flag=position_flag,norm_layer=norm_layer,xyz_brain_pth=xyz_brain)#编码器第一二阶段位置编码情况可能不同
        self.decoder = EEGHybridBranchModel_V3(time_len=time_len,patch_size=patch_size,channels=in_chans,mask_ratio=0,embed_dim=decoder_embed_dim,
                                            num_heads=decoder_num_heads,mlp_ratio=mlp_ratio,depth=decoder_depth,position_flag=position_flag,norm_layer=norm_layer,xyz_brain_pth=xyz_brain)#解码器只参与第一阶段位置编码
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim),
        #                                       requires_grad=False)  # fixed sin-cos embedding
        # self.decoder_embed = nn.Linear(self.num_patches, decoder_embed_dim, bias=True)
        self.decoder_embed = nn.Linear(embed_dim, time_len, bias=True)

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1,1,time_len))#(1,1,C)后面还会有初始化。注意与原论文对比，思路一致。很关键的一点是T维度要是1，否则模型坍塌，所有输出结果都极为相似。这是为了保证，通道之间不同，通道内部相同。

        # self.decoder_blocks = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])
        # self.decoder_blocks = nn.ModuleList([
        #     ChannelBranch(
        #         num_blocks=3,
        #         embed_dim=decoder_embed_dim,
        #         num_heads=decoder_num_heads,
        #         mlp_ratio=mlp_ratio
        #     )
        # for _ in range(decoder_depth)])
        # channel_pos_encoding = self.encoder.brain_prior_encoding(in_chans, self.num_patches)  # 脑先验位置编码
        # self.register_buffer('channel_pos_encoding', channel_pos_encoding)  # 注册为缓冲区，方便在不同设备上移动

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, time_len, bias=True)  # encoder to decoder
        # --------------------------------------------------------------------------

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        print('自编码器输入电极通道数为:',self.in_chans)
        print('自编码器通道掩码率为:', self.mask_ratio)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        # decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches,
        #                                                cls_token=False)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        #torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def patchify(self, imgs):
        """
        imgs: [N, chan, T]
        x: (N, L, patch_size)
        x: [N, chan * 4, T/4]
        """
        p = self.patch_size

        # h = imgs.shape[2] // p
        assert imgs.shape[1] % p == 0
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1] // p, -1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, 1, num_voxels)
        """
        p = self.patch_size
        h = x.shape[1]

        imgs = x.reshape(shape=(x.shape[0], -1, x.shape[2] // p))
        return imgs.transpose(1, 2)

    def forward_encoder(self, x):
        #x(N,channels,T)
        # embed patches
        # x = self.patch_embed(x)
        # print('encoder embed')
        # print(x.shape)
        # masking: length -> length * mask_ratio
        if self.mask_ratio != 0:
            x , mask, ids_restore= self.encoder(x)
            return x, mask, ids_restore
        # apply Transformer blocks
        x,_,_ = self.encoder(x)
        return x #(N,channels,embed_dim)

    def forward_decoder(self, x, ids_restore=None):
        #x(B,channels,embed_dim)
        N,C,embed_dim = x.shape

        x = self.decoder_embed(x).transpose(1,2)#(B,channels,time_len)--(B,time_len,channels)
        mask_tokens = self.mask_token.repeat(x.shape[0],self.droped_channels, 1).transpose(1,2)#(1,1,time_len)--(N,drop_channels,time_len)--(N,time_len,drop_channels)
        x_ = torch.cat([x, mask_tokens], dim=-1)  # (B,time_len,masked_channels+droped_channels)

        x_ = torch.gather(x_, dim=-1, index=ids_restore.unsqueeze(1).repeat(1, x.shape[1], 1))  # unshuffle
        x,_,_ = self.decoder(x_.transpose(1,2))
        # x_ = x_ + self.channel_pos_encoding
        # # embed tokens
        # # x = self.decoder_embed(x_)#x(B,T,embed_channels)
        # x = self.decoder_embed(x_.transpose(1, 2))#(B,channels,embed_dim)
        # print('decoder embed')
        # print(x.shape)
        # append mask tokens to sequence


        # x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token

        # x = x_
        # add pos embed
        # x = x + self.decoder_pos_embed
        # x = x + self.channel_pos_encoding.transpose(1,2)
        # x = x + self.decoder_pos_embed[:, 1:, :]

        # apply Transformer blocks
        # for blk in self.decoder_blocks:
        #     x = blk(x)
        x = self.decoder_norm(x)
        # print(x.shape)
        # predictor projection
        x = self.decoder_pred(x)
        # print("x.shape",x.shape)

        return x#(N,channels,time_len)

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, chanels, timepoint]
        pred: [N,channels,timepoint]
        mask: [N, channels], 0 is keep, 1 is remove,
        """
        # imgs = imgs.transpose(1, 2)#(N,timepoint,channels)
        # target = self.patchify(imgs)#(N,timepoint//4,4*channels) T=timepoint//4
        N,C,T = pred.shape
        # pred = pred.view(N,T*4,C//4)#(N,timepoint,channels)
        # target = imgs.transpose(1,2)
        loss = (pred - imgs) ** 2#(N,channels,timepoint)

        loss = loss.mean(dim=-1)  # [N, channels], mean loss per channel
        # loss = loss.mean()
        loss = (loss * mask).sum() / mask.sum() if mask.sum() != 0 else (
                    loss * mask).sum()  # mean loss on removed patches
        return loss

    def forward(self, x):
        # latent = self.forward_encoder(imgs, mask_ratio)
        # x(N,channels,time_len)
        if self.mask_ratio != 0:
            latent, mask, ids_restore = self.forward_encoder(x)#latent(N,channels,embed_dim)
            pred = self.forward_decoder(latent, ids_restore)  # (N,channels,time_len)
            loss = self.forward_loss(x, pred, mask)
            return loss, pred, mask
        else:
            latent = self.forward_encoder(x)
            return latent

class MChan_AEforEEG_V4(nn.Module):
    """ Masked Channels Autoencoder with VisionTransformer backbone.
    """

    def __init__(self, time_len=400, patch_size=4, embed_dim=512, in_chans=62,mask_ratio=0.25,
                 depth=4, num_heads=16, decoder_embed_dim=512,
                 decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4.,position_flag=True, norm_layer=nn.LayerNorm, xyz_brain=None,
                 fixed_electrode_indices=None, fixed_electrode_index_base=0,
                 position_ablation='none', position_ablation_seed=0,
                 position_ablation_ref_index=0,
                 position_ablation_virtual_coord=None,
                 spatial_depth=None, vit_order='spatial_temporal'):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.time_len = time_len
        self.in_chans = in_chans
        self.num_patches = int(time_len / patch_size)
        self.masked_channels = int(in_chans*(1-mask_ratio))#剩下的电极通道数
        self.droped_channels = in_chans - int(in_chans*(1-mask_ratio))#有多少电极通道被去除

        # 将时间点维度按照patch_size打包成token,并且通过分组卷积来实现通道间信息分离。也就是说此操作仅仅是为了将时间步打包成token，对电极通道无影响。
        # self.patch_embed = PatchEmbed1D(time_len=time_len, patch_size=patch_size, in_chans=in_chans, embed_dim=in_chans,
        #                                 group=in_chans)
        # self.encoder = EEGTwoBranchModel(T=self.num_patches, channels=in_chans,mask_ratio=mask_ratio,embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
        #                                  num_stages=depth, norm_layer=norm_layer, xyz_brain_pth=xyz_brain)
        # --------------------------------------------------------------------------
        if position_flag:
            print('启用位置编码')
        else:
            print('关闭位置编码')
        self.encoder = EEGHybridBranchModel_V4(time_len=time_len,patch_size=patch_size,channels=in_chans,mask_ratio=mask_ratio,embed_dim=embed_dim,num_heads=num_heads,
                                            mlp_ratio=mlp_ratio,depth=depth,position_flag=position_flag,norm_layer=norm_layer,xyz_brain_pth=xyz_brain,
                                            fixed_electrode_indices=fixed_electrode_indices,
                                            fixed_electrode_index_base=fixed_electrode_index_base,
                                            position_ablation=position_ablation,
                                            position_ablation_seed=position_ablation_seed,
                                            position_ablation_ref_index=position_ablation_ref_index,
                                            position_ablation_virtual_coord=position_ablation_virtual_coord,
                                            spatial_depth=spatial_depth,
                                            vit_order=vit_order)#编码器第一二阶段位置编码情况可能不同
        self.decoder = EEGHybridBranchModel_V4(time_len=time_len,patch_size=patch_size,channels=in_chans,mask_ratio=0,embed_dim=decoder_embed_dim,
                                            num_heads=decoder_num_heads,mlp_ratio=mlp_ratio,depth=decoder_depth,position_flag=position_flag,norm_layer=norm_layer,xyz_brain_pth=xyz_brain,
                                            position_ablation=position_ablation,
                                            position_ablation_seed=position_ablation_seed,
                                            position_ablation_ref_index=position_ablation_ref_index,
                                            position_ablation_virtual_coord=position_ablation_virtual_coord,
                                            vit_order=vit_order)#解码器只参与第一阶段位置编码
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim),
        #                                       requires_grad=False)  # fixed sin-cos embedding
        # self.decoder_embed = nn.Linear(self.num_patches, decoder_embed_dim, bias=True)
        self.decoder_embed = nn.Linear(embed_dim, self.masked_channels*patch_size, bias=True)

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, time_len))#(1,1,time_len)后面还会有初始化。是为了保证，嵌入层各个维度不同。

        # self.decoder_blocks = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])
        # self.decoder_blocks = nn.ModuleList([
        #     ChannelBranch(
        #         num_blocks=3,
        #         embed_dim=decoder_embed_dim,
        #         num_heads=decoder_num_heads,
        #         mlp_ratio=mlp_ratio
        #     )
        # for _ in range(decoder_depth)])
        # channel_pos_encoding = self.encoder.brain_prior_encoding(in_chans, self.num_patches)  # 脑先验位置编码
        # self.register_buffer('channel_pos_encoding', channel_pos_encoding)  # 注册为缓冲区，方便在不同设备上移动

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans*patch_size, bias=True)  # encoder to decoder
        # --------------------------------------------------------------------------

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        print('自编码器输入电极通道数为:',self.in_chans)
        print('自编码器保留电极通道数为:',self.masked_channels)
        print('自编码器通道掩码率为:', self.mask_ratio)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        # decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches,
        #                                                cls_token=False)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        #torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def patchify(self, imgs):
        """
        imgs: [N, chan, T]
        x: (N, L, patch_size)
        x: [N, chan * 4, T/4]
        """
        p = self.patch_size

        # h = imgs.shape[2] // p
        assert imgs.shape[1] % p == 0
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1] // p, -1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, 1, num_voxels)
        """
        p = self.patch_size
        h = x.shape[1]

        imgs = x.reshape(shape=(x.shape[0], -1, x.shape[2] // p))
        return imgs.transpose(1, 2)

    def forward_encoder(self, x):
        #x(N,channels,T)
        # embed patches
        # x = self.patch_embed(x)
        # print('encoder embed')
        # print(x.shape)
        # masking: length -> length * mask_ratio
        #print('编码器部分输入维度',x.shape)
        if self.mask_ratio != 0:
            x , mask, ids_restore= self.encoder(x)
            return x, mask, ids_restore
        # apply Transformer blocks
        x,_,_ = self.encoder(x)
        #print('编码器部分输出维度', x.shape)
        return x #(N,T,embed_dim)

    def forward_decoder(self, x, ids_restore=None):
        #x(B,T,embed_dim)
        #print('编码器部分输入维度', x.shape)
        N,T,embed_dim = x.shape

        x = self.decoder_embed(x).view(N,T*self.patch_size,self.masked_channels)#(B,T,masked_channels*patch_size)--(B,T*4,masked_channels)time_len=T*4
        mask_tokens = self.mask_token.repeat(x.shape[0], self.droped_channels, 1).transpose(1,2)
        x_ = torch.cat([x, mask_tokens], dim=-1)  # (B,time_len,masked_channels+droped_channels)
        assert x_.shape[2] == self.in_chans
        x_ = torch.gather(x_, dim=-1, index=ids_restore.unsqueeze(1).repeat(1, x.shape[1], 1))  # unshuffle
        x,_,_ = self.decoder(x_.transpose(1,2))

        x = self.decoder_norm(x)
        # print(x.shape)
        # predictor projection
        x = self.decoder_pred(x)
        # print("x.shape",x.shape)

        return x.view(N,self.time_len, self.in_chans).transpose(1,2)#(N,channels,time_point)

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, chanels, timepoint]
        pred: [N,channels,timepoint]
        mask: [N, channels], 0 is keep, 1 is remove,
        """
        # imgs = imgs.transpose(1, 2)#(N,timepoint,channels)
        # target = self.patchify(imgs)#(N,timepoint//4,4*channels) T=timepoint//4
        N,C,T = pred.shape
        # pred = pred.view(N,T*4,C//4)#(N,timepoint,channels)
        # target = imgs.transpose(1,2)
        loss = (pred - imgs) ** 2#(N,channels,timepoint)

        loss = loss.mean(dim=-1)  # [N, channels], mean loss per channel
        # loss = loss.mean()
        loss = (loss * mask).sum() / mask.sum() if mask.sum() != 0 else (
                    loss * mask).sum()  # mean loss on removed patches
        return loss

    def forward(self, x):
        # latent = self.forward_encoder(imgs, mask_ratio)
        # x(N,channels,T)
        if self.mask_ratio != 0:
            latent, mask, ids_restore = self.forward_encoder(x)#latent(N,T,channels)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p]
            loss = self.forward_loss(x, pred, mask)
            return loss, pred, mask
        else:
            latent = self.forward_encoder(x)
            return latent

    def refresh_position_encoding(self):
        self.encoder.refresh_channel_pos_encoding()
        self.decoder.refresh_channel_pos_encoding()

class eeg_encoder_ours(nn.Module):
    """ Masked Channels Autoencoder with VisionTransformer backbone.
    """

    def __init__(self, time_len=400, patch_size=4, embed_dim=1024, in_chans=62,mask_ratio=0.25,
                 depth=4, num_heads=16,
                 mlp_ratio=1.,position_flag=False, norm_layer=nn.LayerNorm, xyz_brain=None):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.time_len = time_len
        self.in_chans = in_chans
        self.num_patches = int(time_len / patch_size)
        self.masked_channels = int(in_chans*(1-mask_ratio))#剩下的电极通道数
        self.droped_channels = in_chans - int(in_chans*(1-mask_ratio))#有多少电极通道被去除
        if position_flag:
            print('启用位置编码')
        else:
            print('关闭位置编码')
        # 将时间点维度按照patch_size打包成token,并且通过分组卷积来实现通道间信息分离。也就是说此操作仅仅是为了将时间步打包成token，对电极通道无影响。
        # self.patch_embed = PatchEmbed1D(time_len=time_len, patch_size=patch_size, in_chans=in_chans, embed_dim=in_chans,
        #                                 group=in_chans)
        # self.encoder = EEGTwoBranchModel(T=self.num_patches, channels=in_chans,mask_ratio=mask_ratio,embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
        #                                  num_stages=depth, norm_layer=norm_layer, xyz_brain_pth=xyz_brain)
        # --------------------------------------------------------------------------
        self.encoder = EEGHybridBranchModel(time_len=time_len,patch_size=patch_size,channels=in_chans,mask_ratio=mask_ratio,embed_dim=embed_dim,num_heads=num_heads,
                                            mlp_ratio=mlp_ratio,depth=depth,position_flag=position_flag,norm_layer=norm_layer,xyz_brain_pth=xyz_brain)

        # --------------------------------------------------------------------------

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.xyz_brain = xyz_brain
        print('单编码器输入电极通道数为:',self.in_chans)
        print('单编码器通道掩码率为:',self.mask_ratio)
        self.initialize_weights()
    def initialize_weights(self):
        # initialization

        # decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches,
        #                                                cls_token=False)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        #torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def patchify(self, imgs):
        """
        imgs: [N, chan, T]
        x: (N, L, patch_size)
        x: [N, chan * 4, T/4]
        """
        p = self.patch_size

        # h = imgs.shape[2] // p
        assert imgs.shape[1] % p == 0
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1] // p, -1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, 1, num_voxels)
        """
        p = self.patch_size
        h = x.shape[1]

        imgs = x.reshape(shape=(x.shape[0], -1, x.shape[2] // p))
        return imgs.transpose(1, 2)

    def forward_encoder(self, x):
        #x(N,channels,T)
        # embed patches
        # x = self.patch_embed(x)
        # print('encoder embed')
        # print(x.shape)
        # masking: length -> length * mask_ratio
        x,_,_ = self.encoder(x)
        return x #(N,T,embed_dim+C)

    def forward(self, x):
        latent = self.forward_encoder(x)
        return latent

    def load_checkpoint(self, state_dict):
        m, u = self.load_state_dict(state_dict, strict=False)
        print('missing keys:', u)
        print('unexpected keys:', m)
        return

class eeg_encoder_ours_V2(nn.Module):
    """ Masked Channels Autoencoder with VisionTransformer backbone.
    """

    def __init__(self, time_len=400, patch_size=4, embed_dim=1024, in_chans=62, mask_ratio=0.25,
                 depth=4, num_heads=16,
                 mlp_ratio=1., position_flag=False, norm_layer=nn.LayerNorm, xyz_brain=None):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.time_len = time_len
        self.in_chans = in_chans
        self.num_patches = int(time_len / patch_size)
        self.masked_channels = int(in_chans * (1 - mask_ratio))  # 剩下的电极通道数
        self.droped_channels = in_chans - int(in_chans * (1 - mask_ratio))  # 有多少电极通道被去除
        if position_flag:
            print('启用位置编码')
        else:
            print('关闭位置编码')
        # 将时间点维度按照patch_size打包成token,并且通过分组卷积来实现通道间信息分离。也就是说此操作仅仅是为了将时间步打包成token，对电极通道无影响。
        # self.patch_embed = PatchEmbed1D(time_len=time_len, patch_size=patch_size, in_chans=in_chans, embed_dim=in_chans,
        #                                 group=in_chans)
        # self.encoder = EEGTwoBranchModel(T=self.num_patches, channels=in_chans,mask_ratio=mask_ratio,embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
        #                                  num_stages=depth, norm_layer=norm_layer, xyz_brain_pth=xyz_brain)
        # --------------------------------------------------------------------------
        self.encoder = EEGHybridBranchModel_V2(time_len=time_len, patch_size=patch_size, channels=in_chans,
                                            mask_ratio=mask_ratio, embed_dim=embed_dim, num_heads=num_heads,
                                            mlp_ratio=mlp_ratio, depth=depth, position_flag=position_flag,
                                            norm_layer=norm_layer, xyz_brain_pth=xyz_brain)

        # --------------------------------------------------------------------------

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.xyz_brain = xyz_brain
        print('单编码器输入电极通道数为:', self.in_chans)
        print('单编码器通道掩码率为:', self.mask_ratio)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        # decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches,
        #                                                cls_token=False)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def patchify(self, imgs):
        """
        imgs: [N, chan, T]
        x: (N, L, patch_size)
        x: [N, chan * 4, T/4]
        """
        p = self.patch_size

        # h = imgs.shape[2] // p
        assert imgs.shape[1] % p == 0
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1] // p, -1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, 1, num_voxels)
        """
        p = self.patch_size
        h = x.shape[1]

        imgs = x.reshape(shape=(x.shape[0], -1, x.shape[2] // p))
        return imgs.transpose(1, 2)

    def forward_encoder(self, x):
        # x(N,channels,T)
        # embed patches
        # x = self.patch_embed(x)
        # print('encoder embed')
        # print(x.shape)
        # masking: length -> length * mask_ratio
        x, _, _ = self.encoder(x)
        return x  # (N,T,embed_dim+C)

    def forward(self, x):
        latent = self.forward_encoder(x)
        return latent

    def load_checkpoint(self, state_dict):
        m, u = self.load_state_dict(state_dict, strict=False)
        print('missing keys:', u)
        print('unexpected keys:', m)
        return

class eeg_encoder_ours_V3(nn.Module):
    """ Masked Channels Autoencoder with VisionTransformer backbone.
    """

    def __init__(self, time_len=400, patch_size=4, embed_dim=1024, in_chans=62, mask_ratio=0,
                 depth=4, num_heads=16,
                 mlp_ratio=1., position_flag=False, norm_layer=nn.LayerNorm, xyz_brain=None,
                 fixed_electrode_indices=None, fixed_electrode_index_base=0):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.time_len = time_len
        self.in_chans = in_chans
        self.num_patches = int(time_len / patch_size)
        self.masked_channels = int(in_chans * (1 - mask_ratio))  # 剩下的电极通道数
        self.droped_channels = in_chans - int(in_chans * (1 - mask_ratio))  # 有多少电极通道被去除
        if position_flag:
            print('启用位置编码')
        else:
            print('关闭位置编码')
        # 将时间点维度按照patch_size打包成token,并且通过分组卷积来实现通道间信息分离。也就是说此操作仅仅是为了将时间步打包成token，对电极通道无影响。
        # self.patch_embed = PatchEmbed1D(time_len=time_len, patch_size=patch_size, in_chans=in_chans, embed_dim=in_chans,
        #                                 group=in_chans)
        # self.encoder = EEGTwoBranchModel(T=self.num_patches, channels=in_chans,mask_ratio=mask_ratio,embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
        #                                  num_stages=depth, norm_layer=norm_layer, xyz_brain_pth=xyz_brain)
        # --------------------------------------------------------------------------
        self.encoder = EEGHybridBranchModel_V3(time_len=time_len, patch_size=patch_size, channels=in_chans,
                                               mask_ratio=mask_ratio, embed_dim=embed_dim, num_heads=num_heads,
                                               mlp_ratio=mlp_ratio, depth=depth, position_flag=position_flag,
                                               norm_layer=norm_layer, xyz_brain_pth=xyz_brain)

        # --------------------------------------------------------------------------

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.xyz_brain = xyz_brain
        print('单编码器输入电极通道数为:', self.in_chans)
        print('单编码器通道掩码率为:', self.mask_ratio)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        # decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches,
        #                                                cls_token=False)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def patchify(self, imgs):
        """
        imgs: [N, chan, T]
        x: (N, L, patch_size)
        x: [N, chan * 4, T/4]
        """
        p = self.patch_size

        # h = imgs.shape[2] // p
        assert imgs.shape[1] % p == 0
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1] // p, -1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, 1, num_voxels)
        """
        p = self.patch_size
        h = x.shape[1]

        imgs = x.reshape(shape=(x.shape[0], -1, x.shape[2] // p))
        return imgs.transpose(1, 2)

    def forward_encoder(self, x):
        # x(N,channels,T)
        # embed patches
        # x = self.patch_embed(x)
        # print('encoder embed')
        # print(x.shape)
        # masking: length -> length * mask_ratio
        x, _, _ = self.encoder(x)
        return x  # (N,T,embed_dim+C)

    def forward(self, x):
        latent = self.forward_encoder(x)
        return latent

    def load_checkpoint(self, state_dict):
        m, u = self.load_state_dict(state_dict, strict=False)
        print('missing keys:', u)
        print('unexpected keys:', m)
        return

class eeg_encoder_ours_V4(nn.Module):
    """ Masked Channels Autoencoder with VisionTransformer backbone.
    """

    def __init__(self, time_len=400, patch_size=4, embed_dim=1024, in_chans=62, mask_ratio=0,
                 depth=4, num_heads=16,
                 mlp_ratio=1., position_flag=False, norm_layer=nn.LayerNorm, xyz_brain=None,
                 fixed_electrode_indices=None, fixed_electrode_index_base=0,
                 position_ablation='none', position_ablation_seed=0,
                 position_ablation_ref_index=0,
                 position_ablation_virtual_coord=None,
                 spatial_depth=None, vit_order='spatial_temporal'):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.time_len = time_len
        self.in_chans = in_chans
        self.num_patches = int(time_len / patch_size)
        self.masked_channels = int(in_chans * (1 - mask_ratio))  # 剩下的电极通道数
        self.droped_channels = in_chans - int(in_chans * (1 - mask_ratio))  # 有多少电极通道被去除
        if position_flag:
            print('启用位置编码')
        else:
            print('关闭位置编码')
        # 将时间点维度按照patch_size打包成token,并且通过分组卷积来实现通道间信息分离。也就是说此操作仅仅是为了将时间步打包成token，对电极通道无影响。
        # self.patch_embed = PatchEmbed1D(time_len=time_len, patch_size=patch_size, in_chans=in_chans, embed_dim=in_chans,
        #                                 group=in_chans)
        # self.encoder = EEGTwoBranchModel(T=self.num_patches, channels=in_chans,mask_ratio=mask_ratio,embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
        #                                  num_stages=depth, norm_layer=norm_layer, xyz_brain_pth=xyz_brain)
        # --------------------------------------------------------------------------
        self.encoder = EEGHybridBranchModel_V4(time_len=time_len, patch_size=patch_size, channels=in_chans,
                                               mask_ratio=mask_ratio, embed_dim=embed_dim, num_heads=num_heads,
                                               mlp_ratio=mlp_ratio, depth=depth, position_flag=position_flag,
                                               norm_layer=norm_layer, xyz_brain_pth=xyz_brain,
                                               fixed_electrode_indices=fixed_electrode_indices,
                                               fixed_electrode_index_base=fixed_electrode_index_base,
                                               position_ablation=position_ablation,
                                               position_ablation_seed=position_ablation_seed,
                                               position_ablation_ref_index=position_ablation_ref_index,
                                               position_ablation_virtual_coord=position_ablation_virtual_coord,
                                               spatial_depth=spatial_depth,
                                               vit_order=vit_order)

        # --------------------------------------------------------------------------

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.xyz_brain = xyz_brain
        self.position_ablation = position_ablation
        self.position_ablation_seed = position_ablation_seed
        self.position_ablation_ref_index = position_ablation_ref_index
        self.position_ablation_virtual_coord = position_ablation_virtual_coord
        self.spatial_depth = spatial_depth
        self.vit_order = vit_order
        print('单编码器输入电极通道数为:', self.in_chans)
        print('单编码器通道掩码率为:', self.mask_ratio)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        # decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches,
        #                                                cls_token=False)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def patchify(self, imgs):
        """
        imgs: [N, chan, T]
        x: (N, L, patch_size)
        x: [N, chan * 4, T/4]
        """
        p = self.patch_size

        # h = imgs.shape[2] // p
        assert imgs.shape[1] % p == 0
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1] // p, -1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, 1, num_voxels)
        """
        p = self.patch_size
        h = x.shape[1]

        imgs = x.reshape(shape=(x.shape[0], -1, x.shape[2] // p))
        return imgs.transpose(1, 2)

    def forward_encoder(self, x):
        # x(N,channels,T)
        # embed patches
        # x = self.patch_embed(x)
        # print('encoder embed')
        # print(x.shape)
        # masking: length -> length * mask_ratio
        x, _, _ = self.encoder(x)
        return x  # (N,T,embed_dim)

    def forward(self, x):
        latent = self.forward_encoder(x)
        return latent

    def refresh_position_encoding(self):
        self.encoder.refresh_channel_pos_encoding()

    def load_checkpoint(self, state_dict):
        current_state = self.state_dict()
        compatible_state = {}
        skipped_keys = []
        for key, value in state_dict.items():
            if key in current_state and current_state[key].shape != value.shape:
                skipped_keys.append((key, tuple(value.shape), tuple(current_state[key].shape)))
            else:
                compatible_state[key] = value
        m, u = self.load_state_dict(compatible_state, strict=False)
        if skipped_keys:
            print('skipped incompatible keys:', skipped_keys)
        print('missing keys:', u)
        print('unexpected keys:', m)
        if self.position_ablation != 'none':
            self.refresh_position_encoding()
        return
    # class eeg_encoder_ours(nn.Module):
#     def __init__(self, time_len=400, patch_size=4, embed_dim=512, in_chans=62, mask_ratio=0.125,
#                  depth=4, num_heads=16,
#                  mlp_ratio=4., norm_layer=nn.LayerNorm,
#                  xyz_brain='/home/mahui/Dataset/EEG-ImageNet-Dataset/XYZ_Brain.pth'):
#         super().__init__()
#
#         # --------------------------------------------------------------------------
#         # MAE encoder specifics
#         self.num_patches = int(time_len / patch_size)
#         self.masked_channels = int(in_chans * (1 - mask_ratio))  # 剩下的电极通道数
#         self.droped_channels = in_chans - int(in_chans * (1 - mask_ratio))  # 有多少电极通道被去除
#         # 将时间点维度按照patch_size打包成token,并且通过分组卷积来实现通道间信息分离。也就是说此操作仅仅是为了将时间步打包成token，对电极通道无影响。
#         self.patch_embed = PatchEmbed1D(time_len=time_len, patch_size=patch_size, in_chans=in_chans, embed_dim=in_chans,
#                                         group=in_chans)
#         self.encoder = EEGTwoBranchModel(T=self.num_patches, channels=in_chans, mask_ratio=mask_ratio,
#                                          embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
#                                          num_stages=depth, norm_layer=norm_layer, xyz_brain_pth=xyz_brain)
#         # --------------------------------------------------------------------------
#
#         # --------------------------------------------------------------------------
#
#
#
#         # --------------------------------------------------------------------------
#
#         self.patch_size = patch_size
#         self.embed_dim = embed_dim
#         self.mask_ratio = mask_ratio
#         self.initialize_weights()
#
#     def initialize_weights(self):
#         # initialization
#
#         # decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches,
#         #                                                cls_token=False)
#         # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
#
#         # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
#         w = self.patch_embed.proj.weight.data
#         torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
#
#         # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
#         # torch.nn.init.normal_(self.cls_token, std=.02)
#         # torch.nn.init.normal_(self.mask_token, std=.02)
#
#         # initialize nn.Linear and nn.LayerNorm
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv1d):
#             torch.nn.init.normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#
#     def patchify(self, imgs):
#         """
#         imgs: (N, 1, num_voxels)
#         imgs: [N, chan, T]
#         x: (N, L, patch_size)
#         x: [N, chan * 4, T/4]
#         """
#         p = self.patch_embed.patch_size
#         assert imgs.ndim == 3 and imgs.shape[1] % p == 0
#
#         # h = imgs.shape[2] // p
#         x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1] // p, -1))
#         return x
#
#     def unpatchify(self, x):
#         """
#         x: (N, L, patch_size)
#         imgs: (N, 1, num_voxels)
#         """
#         p = self.patch_embed.patch_size
#         h = x.shape[1]
#
#         imgs = x.reshape(shape=(x.shape[0], -1, x.shape[2] // p))
#         return imgs.transpose(1, 2)
#
#     def forward_encoder(self, x):
#         # x(N,channels,T)
#         # embed patches
#         x = self.patch_embed(x)
#         # print('encoder embed')
#         # print(x.shape)
#         # masking: length -> length * mask_ratio
#         # apply Transformer blocks
#         x, _, _ = self.encoder(x)
#         return x  # (N,T,embed_dim+C)
#
#     def forward(self, x):
#         # latent = self.forward_encoder(imgs, mask_ratio)
#         # x(N,channels,T)
#
#         latent = self.forward_encoder(x)
#         return latent
#
#     def load_checkpoint(self, state_dict):
#         m, u = self.load_state_dict(state_dict, strict=False)
#         print('missing keys:', u)
#         print('unexpected keys:', m)
#         return


class eeg_encoder(nn.Module):
    def __init__(self, time_len=400, patch_size=4, embed_dim=1024, in_chans=62,
                 depth=24, num_heads=16, mlp_ratio=1., norm_layer=nn.LayerNorm, global_pool=False):
        super().__init__()
        self.patch_embed = PatchEmbed1D(time_len, patch_size, in_chans, embed_dim)

        num_patches = int(time_len / patch_size)

        self.num_patches = in_chans
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
    
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embed_dim = embed_dim

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.global_pool = global_pool
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = ut.get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        # print(x.shape)
        # print(self.pos_embed[:, 1:, :].shape)
        x = x + self.pos_embed[:, 1:, :]
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # print(x.shape)
        if self.global_pool:
            x = x.mean(dim=1, keepdim=True)
        # print(x.shape)
        x = self.norm(x)
        # print(x.shape)
        return x  

    def forward(self, imgs):
        if imgs.ndim == 2:
            imgs = torch.unsqueeze(imgs, dim=0)  # N, n_seq, embed_dim
        latent = self.forward_encoder(imgs) # N, n_seq, embed_dim
        return latent # N, n_seq, embed_dim
    
    def load_checkpoint(self, state_dict):
        if self.global_pool:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k and 'norm' not in k)}
        else:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k)}
        ut.interpolate_pos_embed(self, state_dict)
            
        m, u = self.load_state_dict(state_dict, strict=False)
        print('missing keys:', u)
        print('unexpected keys:', m)
        return 

class classify_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.Conv1d(128, 1, 1, stride=1)#nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(1024, 40)

    def forward(self, x):
        x = self.maxpool(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x


class mapping(nn.Module):
    def __init__(self, input_dim=1024, seq_len=128, output_dim=768):
        super().__init__()
        # self.maxpool = nn.Conv1d(128, 1, 1, stride=1)#nn.AdaptiveAvgPool1d((1))
        self.maxpool = nn.Conv1d(seq_len, 1, 1, stride=1)  #100=400/4 110=440/4 128=512/4
        self.fc = nn.Linear(input_dim, output_dim)#54=62*(1-0.125)也就是mask之后的通道数

    def forward(self, x):
        x = self.maxpool(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x


# if __name__ == '__main__':
#     # mae = MAEforEEG(time_len=512)
#     # mae.forward_encoder(input,0.5)
#     # print(encoder)
#     input = torch.randn(2,128,512)
#     # loss = mae(input)
#     # print(input[:,:,0:4])
#     # print(input.transpose(1,2)[:,0:4,:])
#     # print(mae.patchify(input.transpose(1,2))[:,0,:])
#     # print(loss)
#     encoder = eeg_encoder()
#     out = encoder(input)
#     print(out.shape)
#     clss = classify_network2()
#     pre_cls = clss(out)
#     print(pre_cls.shape)
#     # x, mask, ids_restore = mae.forward_encoder(input,0.75)
#     # # pred = mae.forward_decoder(latent, ids_restore)
#
#     # # print(x)
#     # print(x.shape)
#     # # print(mask)
#     # print(mask.shape)
#     # # print(ids_restore)
#     # print(ids_restore.shape)
#     # pred = mae.forward_decoder(x, ids_restore)
#
#     # # print(pred)
#     # print(pred.shape)






    # import sys
    # sys.path.append('..')
    # print(sys.path)
    # encoder = eeg_encoder2(num_voxels=440)
    # decoder = eeg_decoder2(num_voxels=440)
    # cond = cond_stage_model(encoder)
    # clss = classify_network2()
    
    # print(encoder)
    # lstm = Model()
    #现在数据家在上来就是128*1024的了 这样其实就更好做了
    # input = torch.randn(1,128,128)
    # # out = encoder(input)
    # out, latent_crossattn = cond(input)
    # print(out.shape)
    # print(latent_crossattn.shape)
    # pre_cls = clss(latent_crossattn)
    # print(pre_cls.shape)
    # recon = decoder(latent_crossattn)
    # print(recon.shape)
    # out = lstm(input)
