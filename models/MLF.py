__all__ = ['MLF.py']

# Cell
import time
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.MLF_backbone import MLF_backbone
from layers.RevIN import RevIN
from layers.duet_plugins import ChannelClustering

class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()

        c_in = configs.enc_in
        context_window=configs.context_window
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout

        individual = configs.individual

        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        self.configs=configs
        # === [新增] DUET 插件初始化 ===
        # 这里的 configs.enc_in 代表输入特征数量(Channel数)
        # 如果报错，可能是 configs.c_in 或 configs.n_vars
        self.duet_ccm = ChannelClustering(
            n_vars=configs.enc_in,
            d_model=configs.d_model,
            seq_len=configs.scal_all[0] # <--- 必须加这个！
        )

        self.model = MLF_backbone(configs=configs,c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                              max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                              n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                              dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                              attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                              pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                              pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                              subtract_last=subtract_last, verbose=verbose, **kwargs)

        self.D_norm=configs.D_norm
        if self.configs.revin_norm:
            self.configs.revin_layer = RevIN(self.configs.enc_in, affine=self.configs.affine,
                                                subtract_last=self.configs.subtract_last)
    def forward(self,x, x_mark_enc, x_dec, x_mark_dec):           # x: [Batch, Input length, Channel]
        channel_mask_dense = None  # default for use_duet=0; # identity_mask_force; channel_mask=torch.eye(channel_mask.shape[0], device=channel_mask.device, dtype=channel_mask.dtype)
        k_keep = 0  # default to avoid NameError
        scal_x_all=[]
        self.configs.bs=x.shape[0]

        if self.D_norm:
            seq_last = x[:, -1:, :].detach()
            self.configs.seq_last = seq_last
            x = x - seq_last
        elif self.configs.revin_norm:
            x=self.configs.revin_layer(x, 'norm')

        for i,s in enumerate(self.configs.scal_all):
        # for i, s in enumerate(self.scal_temp):
            if i==0:
                scal_x_all.append(x[:,-s:,:].permute(0,2,1))
            else:
                scal_x_all.append(x[:, -s:, :].permute(0,2,1))

        # === [新增] 计算 Channel Mask ===
        # scal_x_all[0] 的形状是 [Batch, Channel, Length]，正好符合 DUET 的口味
        x0 = scal_x_all[0]  # [B, C, L]
        C = x0.shape[1]
        if not getattr(self.configs, 'use_duet', 1):
            channel_mask = torch.zeros(C, C, device=x0.device, dtype=x0.dtype)
        else:
            cpu_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            channel_mask = self.duet_ccm(x0)
            torch.set_rng_state(cpu_state)
            if cuda_state is not None: torch.cuda.set_rng_state_all(cuda_state)
            channel_mask_dense = channel_mask; # identity_mask_force; channel_mask=torch.eye(channel_mask.shape[0], device=channel_mask.device, dtype=channel_mask.dtype)
            if not hasattr(self, 'DEBUG_MASK_ONCE'):
                self.DEBUG_MASK_ONCE = True
                with torch.no_grad():
                    nz = (channel_mask.abs() > 1e-12).float().mean().item()
                    # ---- FORCE top-k sparsify (guaranteed) ----
                    k_force = min(3, channel_mask.shape[0]-1)
                    if k_force > 0:
                        vals, inds = torch.topk(channel_mask, k=k_force, dim=1)
                        sparse = torch.zeros_like(channel_mask)
                        sparse.scatter_(1, inds, vals)
                        channel_mask = sparse
                    # print(f'[mask debug] min={channel_mask.min().item():.4f} max={channel_mask.max().item():.4f} nonzero_ratio={nz:.3f}')
            # ---- stabilize mask: remove self-loop + row-normalize ----
            eye = torch.eye(C, device=channel_mask.device, dtype=channel_mask.dtype)
            channel_mask = channel_mask * (1 - eye)
            channel_mask = channel_mask / (channel_mask.sum(dim=1, keepdim=True) + 1e-6)
            # ---- APPLY_TOPK_ALWAYS (disabled) ----
            # NOTE: sparsify is moved to backbone (before einsum) to avoid losing neighbors.
            if k_keep > 0:
                vals, inds = torch.topk(channel_mask, k=k_keep, dim=1)
                sparse = torch.zeros_like(channel_mask)
                sparse.scatter_(1, inds, vals)
                channel_mask = sparse
                channel_mask = channel_mask / (channel_mask.sum(dim=1, keepdim=True) + 1e-6)
            # ---- hard threshold to suppress weak edges ----
            tau = 0.0
            channel_mask = channel_mask * (channel_mask >= tau).float()
            # ---- sparsify mask: keep top-k per row ----
            k = min(3, C-1)
            if k > 0:
                vals, inds = torch.topk(channel_mask, k=k, dim=1)
                sparse = torch.zeros_like(channel_mask)
                sparse.scatter_(1, inds, vals)
                channel_mask = sparse
        # ==============================

        # === [修改] 把 Mask 传进去！ ===
        # 注意：这里我们强行加了一个参数 channel_mask
        # 下一步如果不改 Backbone，这里运行肯定会报错，但这正是我们计划中的！
        cm=(channel_mask_dense if channel_mask_dense is not None else channel_mask).clone(); cm.fill_diagonal_(-1e9); k=min(2, cm.shape[0]-1); vals,inds=torch.topk(cm,k=k,dim=1); sp=torch.zeros_like(cm); sp.scatter_(1,inds,vals); sp = sp * sp; sp = sp + torch.eye(sp.shape[0], device=sp.device, dtype=sp.dtype); channel_mask=sp/(sp.sum(dim=1,keepdim=True)+1e-6); # identity_mask_force; channel_mask=torch.eye(channel_mask.shape[0], device=channel_mask.device, dtype=channel_mask.dtype)
        x, scale_all_rec, scale_all_patch = self.model(scal_x_all, channel_mask=channel_mask)

        if self.D_norm: #DLinear Norm
            x = x + seq_last

        return x,scale_all_rec,scale_all_patch
