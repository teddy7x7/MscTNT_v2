__all__ = ['MscTNT4TS']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.MscTNT4TS_backbone import MscTNT4TS_backbone, TemporalConvNet
from layers.PatchTST_layers import series_decomp

class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, 
                 inner_d_k=None, inner_d_v=None, outer_d_k=None, outer_d_v=None, 
                 norm:str='BatchNorm', act:str="gelu", 
                 key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, 
                 
                 qkv_bias=False, qk_scale=None, se=0, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in

        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        
        # n_heads = configs.n_heads                     # TODO 要在run_logExp中 拿掉 這項參數
        outer_num_heads=configs.outer_n_heads           # TODO default = 6 ，在run_longExp 裡要這樣設 
        inner_num_heads=configs.inner_n_heads           # TODO default = 4 ，在run_longExp 裡要這樣設

        # d_model = configs.d_model
        outer_dim=configs.outer_dim                     # TODO default = 384 ，在run_longExp 裡要這樣設
        inner_dim=configs.inner_dim                     # TODO default = 24 ，在run_longExp 裡要這樣設

        # d_ff = configs.d_ff
        inner_mlp_ratio=configs.inner_mlp_ratio         # TODO default = 2. ，在run_longExp 裡要這樣設
        outer_mlp_ratio=configs.outer_mlp_ratio         # TODO default = 4. ，在run_longExp 裡要這樣設
        

        # dropout = configs.dropout                     # patchtst在position embedding 後做的dropout
        pos_drop = configs.pos_drop
        # fc_dropout = configs.fc_dropout               # patchtst使用pretrain head 使用的 dropout
        
        head_dropout = configs.head_dropout             # flatten head 使用的dropout，patchtst中設為 0
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        subpatch_len = configs.subpatch_len             # TODO 要在run_logExp中 加入 這項參數，default=4*4
        
        # stride = configs.stride                       # TODO 要在run_logExp中 拿掉 這項參數
        patch_stride = configs.patch_stride             # TODO 要在run_logExp中 加入 這項參數
        subpatch_stride = configs.subpatch_stride       # TODO TODO default = 4 ，在run_longExp 裡要這樣設


        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        # <20240420> 讓tcn層數變為參數
        inner_tcn_layers=configs.inner_tcn_layers
        outer_tcn_layers=configs.outer_tcn_layers
        # </20240420>

        # <20240422> 讓 early conv的層數可以和後面的tcn的層數不同，可以獨立設置
        early_inner_tcn_layers=configs.early_inner_tcn_layers
        early_outer_tcn_layers=configs.early_outer_tcn_layers
        # </20240422>


        # <20240519>
        # attn_dropout = configs.attn_dropout
        inner_tcn_drop = configs.inner_tcn_drop
        outer_tcn_drop = configs.outer_tcn_drop
        inner_attn_dropout = configs.inner_attn_dropout
        outer_attn_dropout = configs.outer_attn_dropout
        inner_proj_dropout = configs.inner_proj_dropout
        outer_proj_dropout = configs.outer_proj_dropout
        # </20240519>

        # <20240731> Testing can existing LTSF-Transformers preserve temporal order well? by shuffling input sequence
        self.shuffle_method = configs.shuffle_method 
        # </20240731>

        # <20240803> 讓repatching conv 可以被拿掉
        inner_repatching = configs.inner_repatching
        outer_repatching = configs.outer_repatching
        # </20240803>



        # model
        self.decomposition = decomposition
        if self.decomposition:
            pass
            # Might can add decoposition function block into model,still not realize yet
        else:
            self.model = MscTNT4TS_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, patch_stride=patch_stride,
                                            subpatch_len=subpatch_len,
                                            head_dropout=head_dropout, padding_patch = padding_patch, # d_model=d_model,
                                            pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                            subtract_last=subtract_last, 
                                            ### TODO 以下就是新的，看前面呼叫這個 py 檔的其他檔案，要怎麼去改要輸的參數
                                            outer_dim=outer_dim, inner_dim=inner_dim, depth = n_layers, outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads, 
                                            qkv_bias=qkv_bias, pos_drop=pos_drop, 
                                            # attn_dropout=attn_dropout, 
                                            # d_k=d_k, d_v=d_v, d_ff=d_ff,
                                            inner_d_k=inner_d_k, inner_d_v=inner_d_v, inner_mlp_ratio=inner_mlp_ratio,
                                            outer_d_k=outer_d_k, outer_d_v=outer_d_v, outer_mlp_ratio=outer_mlp_ratio,
                                            norm=norm,
                                            subpatch_stride=subpatch_stride, se=0,
                                            inner_tcn_layers=inner_tcn_layers, outer_tcn_layers=outer_tcn_layers,
                                            early_inner_tcn_layers=early_inner_tcn_layers, early_outer_tcn_layers=early_outer_tcn_layers,
                                            activation=act, res_attention=res_attention, pre_norm=False, store_attn=False, 
                                            
                                            # <20240519>
                                            inner_tcn_drop=inner_tcn_drop, outer_tcn_drop=outer_tcn_drop,
                                            inner_attn_dropout = inner_attn_dropout, inner_proj_dropout = inner_proj_dropout,
                                            outer_attn_dropout = outer_attn_dropout, outer_proj_dropout = outer_proj_dropout,
                                            # # </20240519>
                                            # <20240803> 讓repatching conv 可以被拿掉
                                            inner_repatching=inner_repatching, outer_repatching=outer_repatching
                                            # </20240803> 讓repatching conv 可以被拿掉
                                            )

            ###########
    
    def forward(self, x):           # x: [Batch, Input length, Channel]

        ###
        # shuffle input sequence 
        if self.shuffle_method == 'RandShuf':
            x = x[:, torch.randperm(x.shape[1]), :]
        elif self.shuffle_method == 'HalfEx':
            x = torch.cat((x[:, x.shape[1]//2:, :], x[:, :x.shape[1]//2, :]), dim=1)
        ###

        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]

            # # <20240629> 降維用tcn
            # x=self.reduce_dim_tcn(x)
            # # </20240629>

            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x