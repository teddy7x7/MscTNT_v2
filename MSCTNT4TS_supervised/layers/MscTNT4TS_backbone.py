# Copyright 2024 teddy7x7
# 
# This file is modified from the original PatchTST project 
# (https://github.com/yuqinie98/PatchTST), licensed under Apache License 2.0.
# 
# Modifications:
# - Refactored model structure to multi-scale stages
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# 
#     http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['MscTNT4TS_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np

#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN


# Cell
class MscTNT4TS_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, subpatch_len:int, 
                patch_stride:int, subpatch_stride:int,
                head_fc_dropout:float=0., head_dropout = 0, 
                pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                padding_patch = None,
                outer_dim=384, inner_dim=24,
                depth=6, outer_num_heads=12, inner_num_heads=4, qkv_bias=False, 
                inner_tcn_layers=4, outer_tcn_layers=4,
                early_inner_tcn_layers=3, early_outer_tcn_layers=6,
                pos_drop=0., se=0,
                inner_d_k=None, inner_d_v=None, inner_mlp_ratio=4.,
                outer_d_k=None, outer_d_v=None, outer_mlp_ratio=4.,
                norm='BatchNorm', dropout=0., attn_dropout=0.,
                activation='gelu', res_attention=False, pre_norm=False, store_attn=False,
                inner_tcn_drop=0.1, outer_tcn_drop=0.1,
                inner_attn_dropout = 0., inner_proj_dropout = 0.,
                outer_attn_dropout = 0., outer_proj_dropout = 0.,
                inner_repatching=True, outer_repatching=True
                ):
                        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.subpatch_len = subpatch_len
        self.patch_stride = patch_stride
        self.subpatch_stride = subpatch_stride
        self.padding_patch = padding_patch

        self.patch_num = int((context_window - patch_len)/patch_stride + 1)
        if padding_patch == 'end': 
            self.padding_patch_layer = nn.ReplicationPad1d((0, patch_stride)) 
            self.patch_num += 1
        
        self.subpatch_num = int(self.patch_len/self.subpatch_len)

        # projection to d-dim
        self.W_P_outer = nn.Linear(patch_len, outer_dim)            
        self.W_P_inner = nn.Linear(subpatch_len, inner_dim)         
 
        # position embedding
        self.outer_pos = nn.Parameter(torch.zeros(1, self.patch_num , outer_dim))
        self.inner_pos = nn.Parameter(torch.zeros(1, self.subpatch_num, inner_dim))
        self.pos_drop = nn.Dropout(p=pos_drop)

        self.encoder = MscTSTEncoder(patch_num=self.patch_num, subpatch_num=self.subpatch_num, outer_d_model=outer_dim, inner_d_model=inner_dim, 
                                     outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads, 
                                     inner_tcn_layers=inner_tcn_layers, outer_tcn_layers=outer_tcn_layers,
                                     early_inner_tcn_layers=early_inner_tcn_layers, early_outer_tcn_layers=early_outer_tcn_layers,
                                     inner_d_k=inner_d_k, inner_d_v=inner_d_v, inner_mlp_ratio=inner_mlp_ratio,
                                     outer_d_k=outer_d_k, outer_d_v=outer_d_v, outer_mlp_ratio=outer_mlp_ratio,
                                     norm=norm, attn_dropout=attn_dropout, dropout=dropout, 
                                     activation=activation, res_attention=res_attention, n_layers=depth, pre_norm=pre_norm, store_attn=store_attn,
                                     qkv_bias=qkv_bias, se=se,
                                     inner_tcn_drop=inner_tcn_drop, outer_tcn_drop=outer_tcn_drop,
                                     inner_attn_dropout = inner_attn_dropout, inner_proj_dropout = inner_proj_dropout,
                                     outer_attn_dropout = outer_attn_dropout, outer_proj_dropout = outer_proj_dropout,
                                     inner_repatching=inner_repatching, outer_repatching=outer_repatching
                                     )


        
        self.head_nf = outer_dim * self.patch_num
        
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, head_fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
             

    def forward(self, x):   # x: [bs x nvars x seq_len]

        # 1. RevIn
        if self.revin: 
            x = x.permute(0,2,1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0,2,1)

            
        # 2.  do patching
        # 2.1 padding
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)

        # 2.2 seperate out the patch_num dimension
        outer_token = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)   # outer_token: [bs x nvars x patch_num x patch_len]
        
        
        # 2.3 merge patch_num dim with Batch dim, let the inner transformer can doing function on the same subpatch under the same patch
        bs, nvr, pn, pl = outer_token.shape
        outer_token = outer_token.reshape(bs*nvr, pn, pl)                                                                           # outer_token: [bs*nvars x patch_num x patch_len]
        inner_token = outer_token.reshape(bs*nvr*pn, pl).unfold(dimension=-1, size=self.subpatch_len, step=self.subpatch_stride)    # [bs*nvars*patch_num x patch_len] > [bs*nvars*patch_num x subpatch_num x subpatch_len]

        # outer_token: [bs*nvars x patch_num x patch_len]
        # inner_token: [bs*nvars x subpatch_num x subpatch_len]


        # 3. project token last dim to transformer input's d-dim
        outer_token = self.W_P_outer(outer_token)
        inner_token = self.W_P_inner(inner_token)

        # 4. position embedding
        outer_token = self.pos_drop(outer_token + self.outer_pos)   # outer_token: [bs*nvars x patch_num x outer_dim]
        inner_token = self.pos_drop(inner_token + self.inner_pos)   # inner_token: [bs*nvars*patch_num x subpatch_num x inner_dim]
       
        # 5. Main block of MscTNT
        outer_token = self.encoder(n_vars=nvr, outer_token=outer_token, inner_token=inner_token)   # output_outer: [bs*nvars x patch_num x outer_dim]

        # 6. reshape the output for head
        out_d=outer_token.shape[2]
        outer_token = torch.reshape(outer_token, (bs, nvr, pn*out_d))            # [bs x nvars x patch_num*outer_dim]
        outer_token = torch.reshape(outer_token, (bs, nvr, pn, out_d))           # [bs x nvars  x patch_num x outer_dim]
        outer_token = outer_token.permute(0, 1, 3, 2)                            # [bs x nvars x outer_dim x patch_num]

        # 7. head
        outer_token = self.head(outer_token)                                     # z: [bs x nvars x target_window] 


        # 8. denorm
        if self.revin: 
            outer_token = outer_token.permute(0,2,1)
            outer_token = self.revin_layer(outer_token, 'denorm')
            outer_token = outer_token.permute(0,2,1)
            
        return outer_token
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)                           # x: [bs x nvars x d_model*patch_num]
            x = self.linear(x)                            # x: [bs x nvars x target_window]
            x = self.dropout(x)
        return x
        
    
def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v    


class Attention(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qk = nn.Linear(dim, hidden_dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def forward(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]   # make torchscript happy (cannot use tensor as tuple)
        v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MscTSTEncoder(nn.Module):
    def __init__(self, patch_num, subpatch_num, outer_d_model, inner_d_model, outer_num_heads, inner_num_heads, 
                 early_inner_tcn_layers, early_outer_tcn_layers,
                 inner_tcn_layers, outer_tcn_layers,
                 inner_d_k=None, inner_d_v=None, inner_mlp_ratio=4.,
                 outer_d_k=None, outer_d_v=None, outer_mlp_ratio=4.,
                 norm='BatchNorm', attn_dropout=0., dropout=0., bias=True, activation='gelu', 
                 res_attention=True, n_layers=1, pre_norm=False, store_attn=False, 
                 qkv_bias=False, drop_path=0., se=0,
                 inner_tcn_drop=0.1, outer_tcn_drop=0.1,
                 inner_attn_dropout = 0., inner_proj_dropout = 0.,
                 outer_attn_dropout = 0., outer_proj_dropout = 0.,
                inner_repatching=True, outer_repatching=True
                ):
                 
        super().__init__()

        # stacking how many blocks of MscTNT
        if n_layers == 1:
            self.layers = nn.ModuleList([MscTSTEncoderLayer(patch_num=patch_num, subpatch_num=subpatch_num, outer_d_model=outer_d_model,
                                                            inner_d_model=inner_d_model, 
                                                            outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads, 
                                                            inner_d_k=inner_d_k, inner_d_v=inner_d_v, inner_mlp_ratio=inner_mlp_ratio,
                                                            outer_d_k=outer_d_k, outer_d_v=outer_d_v, outer_mlp_ratio=outer_mlp_ratio,
                                                            inner_tcn_layers=early_inner_tcn_layers, outer_tcn_layers=early_outer_tcn_layers,
                                                            res_attention=res_attention,
                                                            store_attn=store_attn,
                                                            inner_tcn_drop=inner_tcn_drop, outer_tcn_drop=outer_tcn_drop,
                                                            inner_attn_dropout = inner_attn_dropout, inner_proj_dropout = inner_proj_dropout,
                                                            outer_attn_dropout = outer_attn_dropout, outer_proj_dropout = outer_proj_dropout,
                                                            inner_repatching=False, outer_repatching=outer_repatching
                                                            )])
        else:
            model_list=[MscTSTEncoderLayer(patch_num=patch_num, subpatch_num=subpatch_num, outer_d_model=outer_d_model,
                                                            inner_d_model=inner_d_model, 
                                                            outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads, 
                                                            inner_d_k=inner_d_k, inner_d_v=inner_d_v, inner_mlp_ratio=inner_mlp_ratio,
                                                            outer_d_k=outer_d_k, outer_d_v=outer_d_v, outer_mlp_ratio=outer_mlp_ratio,
                                                            inner_tcn_layers=early_inner_tcn_layers, outer_tcn_layers=early_outer_tcn_layers,
                                                            res_attention=res_attention,
                                                            store_attn=store_attn,
                                                            inner_tcn_drop=inner_tcn_drop, outer_tcn_drop=outer_tcn_drop,
                                                            inner_attn_dropout = inner_attn_dropout, inner_proj_dropout = inner_proj_dropout,
                                                            outer_attn_dropout = outer_attn_dropout, outer_proj_dropout = outer_proj_dropout)]

            model_remain_list = [MscTSTEncoderLayer(patch_num=patch_num, subpatch_num=subpatch_num, outer_d_model=outer_d_model,
                                                            inner_d_model=inner_d_model, 
                                                            outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads, 
                                                            inner_d_k=inner_d_k, inner_d_v=inner_d_v, inner_mlp_ratio=inner_mlp_ratio,
                                                            outer_d_k=outer_d_k, outer_d_v=outer_d_v, outer_mlp_ratio=outer_mlp_ratio,
                                                            inner_tcn_layers=inner_tcn_layers, outer_tcn_layers=outer_tcn_layers,
                                                            res_attention=res_attention,
                                                            store_attn=store_attn,
                                                            inner_tcn_drop=inner_tcn_drop, outer_tcn_drop=outer_tcn_drop,
                                                            inner_attn_dropout = inner_attn_dropout, inner_proj_dropout = inner_proj_dropout,
                                                            outer_attn_dropout = outer_attn_dropout, outer_proj_dropout = outer_proj_dropout
                                                            ) for i in range(n_layers-1)]
            model_list.extend(model_remain_list)
            self.layers = nn.ModuleList(model_list)


        self.res_attention = res_attention
        self.n_vars = None
        self.has_inner = inner_d_model > 0

    def forward(self, n_vars:int, outer_token:Tensor, inner_token:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        # inner_token: [bs*nvars*patch_num x subpatch_num x inner_dim]
        # outer_token: [bs*nvars x patch_num x outer_dim]

        output_inner = inner_token
        output_outer = outer_token

        self.n_vars = n_vars

        outer_scores = None
        inner_scores = None

        if self.res_attention:
            if self.has_inner:
                for mod in self.layers: output_inner, output_outer, outer_scores, inner_scores = mod(n_vars=self.n_vars, inner_tokens=output_inner, 
                                                            outer_tokens=output_outer,
                                                            outer_prev=outer_scores,
                                                            inner_prev=inner_scores,
                                                            key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            else:
                for mod in self.layers: output_outer, outer_scores = mod(n_vars=self.n_vars, inner_tokens=None, outer_token=output_outer,
                                                          outer_prev=outer_scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        else:
            if self.has_inner:
                for mod in self.layers: output_inner, output_outer = mod(n_vars=self.n_vars, inner_tokens=output_inner, outer_token=output_outer,
                                                                        key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            else:
                for mod in self.layers: output_outer = mod(n_vars=self.n_vars, inner_tokens=None, outer_token=output_outer,
                                                                        key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        return output_outer


class MscTSTEncoderLayer(nn.Module):
    def __init__(self, patch_num, subpatch_num, outer_d_model, inner_d_model,    
                 outer_num_heads, inner_num_heads,
                 inner_d_k, inner_d_v, inner_mlp_ratio,
                 outer_d_k, outer_d_v, outer_mlp_ratio,
                 inner_tcn_layers=4, outer_tcn_layers=4,
                 store_attn=False, res_attention=False, 
                 inner_tcn_drop = 0.1, outer_tcn_drop = 0.1,
                 inner_attn_dropout = 0., inner_proj_dropout = 0.,
                 outer_attn_dropout = 0., outer_proj_dropout = 0.,
                inner_repatching=True, outer_repatching=True
                 ):
        
        super().__init__()
        
        self.has_inner = inner_d_model > 0
        self.res_attention = res_attention
        self.inner_tcn_layers=inner_tcn_layers
        self.outer_tcn_layers=outer_tcn_layers
    
        if self.has_inner:

            self.subpatch_num = subpatch_num
            self.inner_d_model = inner_d_model
            self.inner_encoder = TSTEncoderLayer(q_len=subpatch_num, d_model=inner_d_model, n_heads=inner_num_heads, store_attn=store_attn, 
                                                 d_k=inner_d_k, d_v=inner_d_v, mlp_ratio=inner_mlp_ratio,
                                                attn_dropout=inner_attn_dropout, dropout=inner_proj_dropout, res_attention=res_attention)                                     
                                                
            if inner_tcn_layers>0:
                self.inner_tcn = TemporalConvNet(num_inputs=inner_d_model, num_channels=[inner_d_model for i in range(inner_tcn_layers)], kernel_size=2, dropout=inner_tcn_drop)
            
            self.tmpl_proj_ind_to_outd = nn.Linear(inner_d_model, outer_d_model)
            self.tmpl_proj1_norm = nn.BatchNorm1d(outer_d_model)
            self.tmpl_proj_sbptn_to_ptn = nn.Linear(subpatch_num, patch_num)
            self.tmpl_proj2_norm = nn.BatchNorm1d(outer_d_model)

            self.tmpl_proj = nn.Linear(subpatch_num*inner_d_model, outer_d_model)
            self.tmpl_proj_norm = nn.BatchNorm1d(outer_d_model)

            self.inner_repatching = inner_repatching

            if self.inner_repatching:
                self.inner_repatching_conv = nn.Conv1d(in_channels=inner_d_model, out_channels=inner_d_model, kernel_size=3, stride=1, padding=1)
        
        # Outer
        self.outer_encoder = TSTEncoderLayer(q_len=patch_num, d_model=outer_d_model, n_heads=outer_num_heads, 
                                             d_k=outer_d_k, d_v=outer_d_v, mlp_ratio=outer_mlp_ratio,
                                             store_attn=store_attn, 
                                             attn_dropout=outer_attn_dropout, dropout=outer_proj_dropout, res_attention=res_attention)

        if outer_tcn_layers>0:
            self.outer_tcn = TemporalConvNet(num_inputs=outer_d_model, num_channels=[outer_d_model for i in range(outer_tcn_layers)], kernel_size=2, dropout=outer_tcn_drop)

        self.outer_repatching = outer_repatching
        if self.outer_repatching:
            self.outer_repatching_conv = nn.Conv1d(in_channels=outer_d_model, out_channels=outer_d_model, kernel_size=3, stride=1, padding=1)
        

    def forward(self, n_vars:int, inner_tokens:Tensor, outer_tokens:Tensor, 
                outer_prev:Optional[Tensor]=None, inner_prev:Optional[Tensor]=None, 
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:
        if self.has_inner:
            # 1. 先抓Inner細節，inner token先經過 Inner tcn，再經過 Inner transformer
            if self.res_attention:
                inner_tokens, inner_scores = self.inner_encoder(inner_tokens, prev=inner_prev)
            else:
                # inner_token: [bs*nvars x subpatch_num x inner_dim]
                if self.inner_tcn_layers>0:
                    inner_tokens = inner_tokens.permute(0, 2, 1) # inner_token: [bs*nvars*patch_num x inner_dim x subpatch_num]
                    inner_tokens = self.inner_tcn(inner_tokens) 
                    inner_tokens = inner_tokens.permute(0, 2, 1) # inner_token: [bs*nvars*patch_num x subpatch_num x inner_dim]

                inner_tokens = self.inner_encoder(inner_tokens)  # inner_token: [bs*nvars*patch_num x subpatch_num x inner_dim]

            # 2. Outer token 在輸入Outer transformer 之前先經過 outer tcn 處理
            
            # TCN 應該把outer_dim當作輸入tcn中 channel 的維度、subpatch_num作為seq_len維度，所以應該做reshape，將patch_num的維度與outer_dim維度交換
            if self.outer_tcn_layers>0:
                outer_tokens = outer_tokens.permute(0, 2, 1) # outer_token: [bs*nvars x outer_dim x patch_num]
                outer_tokens = self.outer_tcn(outer_tokens)
                outer_tokens = outer_tokens.permute(0, 2, 1) # outer_token: [bs*nvars x patch_num x outer_dim]

            # 3. 將 inner token 的資訊疊加outer token 上
            #      inner_token : [bs*nvars x subpatch_num x inner_dim]
            #      outer_token : [bs*nvars x patch_num x outer_dim]

            # 3.1 先取出需要用的每個維度的分量個數 
            bsnvarptchnum, sbptch_num, in_dim = inner_tokens.shape  # [bs*nvars*patch_num x subpatch_num x inner_dim]
            bsnvar, ptch_num, out_dim = outer_tokens.shape
            
            # 3.2 將 inner token 疊加至 outer token 上，by 將inner_tokens reshape到對應outertoken的大小，再做投影轉換? 最後才疊加?
            # 1. 先將inner的num維度拆出來，以及 subpatch_num 與 inner_dim 兩個維度合併 by "inner_tokens.reshape(bsnvar, ptch_num, sbptch_num*in_dim)"
            # 2. 將其在投影 或 等大小的捲機 至 outer_token 的size，再疊加至 outertoken
            #    即 [bs*nvars*patch_num x subpatch_num x inner_dim] >reshape> [bs*nvars x patch_num x subpatch_num*inner_dim] >linear> [bs*nvars x patch_num x outer_dim] >permute>
            #    [bs*nvars x outer_dim x patch_num] >batchnorm1d> [bs*nvars x outer_dim x patch_num] >permute> [bs*nvars x patch_num x outer_dim]
            
            outer_tokens = outer_tokens + self.tmpl_proj_norm(self.tmpl_proj(inner_tokens.reshape(bsnvar, ptch_num, sbptch_num*in_dim)).permute(0, 2, 1)).permute(0, 2, 1)

            # TODO 檢查哪組是對的
            # inner_token: [bs*nvars*patch_num x subpatch_num x inner_dim]
            # outer_token: [bs*nvars x patch_num x outer_dim]
            # 此時 inner_token : [bs*nvars x sunpatch_num x inner_dim]
            #      outer_token : [bs*nvars x patch_num x outer_dim]       

        # 4. outer encoder
       
        if self.res_attention:
            outer_tokens, outer_scores = self.outer_encoder(outer_tokens, prev=outer_prev)
        else:
            outer_tokens = self.outer_encoder(outer_tokens)


        # 5. repatching
        # TODO inner跟outer分別做patching以及還原維度的投影
        # 5.1 repatching
        # 此時 inner_tokens : [bs*nvars, patch_num, outer_d_model]
        #      outer_token: [bs*nvars x patch_num x outer_dim]
        #      先將inner 疊加至outer，像是residual一樣補回因果資訊，再分別repatching+投影到下一個重複層可以吃的shape by 使用 conv1d
        #   即 inner_token: [bs*nvars x subpatch_num x inner_dim]
        #      outer_token: [bs*nvars x patch_num x outer_dim]
        if self.has_inner:

            # <20240302> 希望 inner矩陣大小保持小點，所以把inner做完casual疊加的部分直接改成投影直接加到outer上，不會存到inner上
            # 此時 inner_token : [bs*nvars x sunpatch_num x inner_dim]
            #      outer_token : [bs*nvars x patch_num x outer_dim]

            # <20240324 ver2 TNT ver>
            # inner_token: [bs*nvars*patch_num x subpatch_num x inner_dim]
            # outer_token: [bs*nvars x patch_num x outer_dim]
            # </20240324 ver2 TNT ver>


            # 5.2 inner repatching by conv1D
            # conv1D 的輸入為:[N, C, L]，所以要先將token permute，再做conv1D，最後再permute回來
            if self.inner_repatching:
                inner_tokens = self.inner_repatching_conv(inner_tokens.permute(0, 2, 1)).permute(0, 2, 1)
            if self.outer_repatching:
                outer_tokens = self.outer_repatching_conv(outer_tokens.permute(0, 2, 1)).permute(0, 2, 1)
            
            # 此時 inner_token: [bs*nvars x subpatch_num x inner_dim]
            #      outer_token: [bs*nvars x patch_num x outer_dim]

            # </20240302>
        else:
            if self.outer_repatching:
                outer_tokens = self.outer_repatching_conv(outer_tokens.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 5.3
        # TODO 最終要將 inner token 與 outer token 都reshape回以下形狀
        #      check # inner_token: [bs*nvars x subpatch_num x inner_dim]
        #            # outer_token: [bs*nvars x patch_num x outer_dim]
        if self.has_inner:
            if self.res_attention:
                return inner_tokens, outer_tokens, outer_scores, inner_scores
            else:
                return inner_tokens, outer_tokens
        else:
            if self.res_attention:
                return outer_tokens, outer_scores
            else:
                return outer_tokens # outer_token: [bs*nvars x patch_num x outer_dim]
    

class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, mlp_ratio=4, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_model*mlp_ratio, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_model*mlp_ratio, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # src : inner_token: [bs*nvars x subpatch_num x inner_dim] or outer_token: [bs*nvars x patch_num x outer_dim]
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)

        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        
        
        #  Q : inner_token: [bs*nvars x subpatch_num x inner_dim] or outer_token: [bs*nvars x patch_num x outer_dim]

        bs = Q.size(0)

        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

