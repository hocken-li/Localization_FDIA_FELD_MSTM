import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys, math
from copy import deepcopy as cp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def convert_to_complex(adj_df):
    return adj_df.applymap(lambda x: complex(x.strip("()")))


def symmetric_normalize(adj_matrix):

    adj_matrix = np.abs(adj_matrix) 
    degree_matrix = np.diag(np.power(np.sum(adj_matrix, axis=1), -0.5))  
    degree_matrix[np.isinf(degree_matrix)] = 0
    normalized_adj = degree_matrix @ adj_matrix @ degree_matrix
    return normalized_adj


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('nvlc,vw->nwlc',(x,A))  
        return x.contiguous()


class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,supports_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*supports_len+1)*c_in 
        self.mlp = nn.Linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=-1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class QKVAttention(nn.Module):
    """
    Assume input has shape B, N, T, C or B, T, N, C
    Note: Attention map will be B, N, T, T or B, T, N, N
        - Could be utilized for both spatial and temporal modeling
        - Able to get additional kv-input (for Time-Enhanced Attention)
    """  
    def __init__(self, in_dim, hidden_size, dropout, num_heads = 4):
        super(QKVAttention, self).__init__()
        self.query = nn.Linear(in_dim, hidden_size, bias = False)
        self.key = nn.Linear(in_dim, hidden_size, bias = False)
        self.value = nn.Linear(in_dim, hidden_size, bias = False)
        self.num_heads = num_heads
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        assert hidden_size % num_heads == 0

    def forward(self, x, kv = None):
        if kv is None:
            kv = x
        query = self.query(x)
        key = self.key(kv)
        value = self.value(kv)
        num_heads = self.num_heads
        if num_heads > 1:
            query = torch.cat(torch.chunk(query, num_heads, dim = -1), dim = 0)   # torch.chunk return a list. need a cat
            key = torch.cat(torch.chunk(key, num_heads, dim = -1), dim = 0)
            value = torch.cat(torch.chunk(value, num_heads, dim = -1), dim = 0)
        d = value.size(-1)
        energy = torch.matmul(query, key.transpose(-1,-2)) 
        energy = energy / (d ** 0.5)
        score = torch.softmax(energy, dim = -1)
        # visualize_last_time_step(score)

        head_out = torch.matmul(score, value)
        out = torch.cat(torch.chunk(head_out, num_heads, dim = 0), dim = -1)
        return self.dropout(self.proj(out))


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(*normalized_shape))
        self.beta = nn.Parameter(torch.zeros(*normalized_shape))

    def forward(self, x):
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]

        mean = x.mean(dim = dims, keepdims = True)
        std = x.std(dim = dims, keepdims = True, unbiased = False)
        x_norm = (x - mean) / (std + self.eps)
        out = x_norm * self.gamma + self.beta
        return out


class BatchNorm(nn.Module):
    def __init__(self, num_features, momentum = 0.1, eps = 1e-5, track_running_stats = True):
        super(BatchNorm, self).__init__()
        self.momentum = momentum
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)

    def forward(self, x):
        dims = [i for i in range(x.dim() - 1)]
        mean = x.mean(dim = dims)
        var = x.var(dim = dims, correction = 0)
        if (self.training) and (self.running_mean is not None):
            avg_factor = self.momentum
            moving_avg = lambda prev, cur: (1 - avg_factor) * prev + avg_factor * cur.detach()
            dims = [i for i in range(x.dim() - 1)]
            self.running_mean = moving_avg(self.running_mean, mean)
            self.running_var = moving_avg(self.running_var, var)
            mean, var = self.running_mean, self.running_var

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        out = x_norm * self.gamma + self.beta
        return out


class SkipConnection(nn.Module):
    """
    Helper Module to build skip connection
     - forward may get auxiliary input to handle multiple inputs (e.g., adjacency matrix or time-enhanced attention)
    """
    def __init__(self, module, norm):
        super(SkipConnection, self).__init__()
        self.module = module
        self.norm = norm

    def forward(self, x, aux = None):
        return self.norm(x + self.module(x, aux))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, in_dim, hidden_size, dropout, activation = nn.GELU()):
        super(PositionwiseFeedForward, self).__init__()
        self.act = activation
        self.l1 = nn.Linear(in_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, in_dim)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, kv = None):
        return self.dropout(self.l2(self.act(self.l1(x))))


class SwitchPositionwiseFeedForward(nn.Module):
    """
    Switch Positionwise Feed Forward module for the normal mixture-of-experts model
     - Note: not used for the TESTAM
    """
    def __init__(self, in_dim, hidden_size, dropout, activation = nn.ReLU(), n_experts = 4):
        super(SwitchPositionwiseFeedForward, self).__init__()
        self.n_experts = n_experts
        self.activation = activation
        self.dropout = nn.Dropout(p = dropout)
        expert = PositionwiseFeedForward(in_dim, hidden_size, dropout, activation)
        self.experts = nn.ModuleList([cp(expert) for _ in range(n_experts)])
        self.switch = nn.Linear(in_dim, n_experts)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x, kv = None):
        B, N, T, C = x.size()
        x = x.view(-1,C)

        route_prob = self.softmax(self.switch(x))
        route_prob_max, routes = torch.max(route_prob, dim = -1)

        # indices: (n_experts, B*T, N)
        indices = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]

        final_output = torch.zeros_like(x)

        for i in range(self.n_experts):
            expert_output = self.experts[i](x[indices[i]])
            final_output[indices[i]] = expert_output

        final_output = final_output * (route_prob_max).unsqueeze(dim = -1)
        final_output = final_output.view(B,N,T,C)

        return final_output



class STModel(nn.Module):
    """
    Input shape B, N, T, in_dim
    Output shape B, N, T, out_dim  & B, N_fdia, out_dim
    Arguments:
        - spatial: Flag that determine when spatial attention will be performed
            - True --> spatial first and then temporal attention will be performed
    """
    def __init__(self, hidden_size, supports_len, num_nodes, dropout, layers, out_dim = 1, in_dim = 2, spatial = 1, activation = nn.GELU(), num_heads=4, use_wpe = 0):
        super(STModel, self).__init__()
        self.spatial = spatial
        self.act = activation
        self.out_dim = out_dim
        self.use_wpe = use_wpe

        s_gcn = gcn(c_in = hidden_size, c_out = hidden_size, dropout = dropout, supports_len = supports_len, order = 2)
        t_attn = QKVAttention(in_dim = hidden_size, hidden_size = hidden_size, dropout = dropout, num_heads=num_heads)
        ff = PositionwiseFeedForward(in_dim = hidden_size, hidden_size = 4 * hidden_size, dropout = dropout)
        norm = LayerNorm(normalized_shape = (hidden_size, ))
        self.norm = norm
        
        self.start_linear = nn.Linear(in_dim, hidden_size)
        self.proj_class = nn.Linear(hidden_size*num_nodes, out_dim)

        self.temporal_layers = nn.ModuleList()
        self.spatial_layers = nn.ModuleList()
        self.ed_layers = nn.ModuleList()
        self.ff = nn.ModuleList()

        for _ in range(layers):
            self.temporal_layers.append(SkipConnection(cp(t_attn), cp(norm)))
            self.spatial_layers.append(SkipConnection(cp(s_gcn), cp(norm)))
            self.ed_layers.append(SkipConnection(cp(t_attn), cp(norm)))
            self.ff.append(SkipConnection(cp(ff), cp(norm)))

    def forward(self, x, prev_hidden, supports):
        if not self.use_wpe:
            x = self.start_linear(x.permute(0,2,3,1)) 
        x_start = x
        hiddens = []

        for i, (temporal_layer, spatial_layer, ed_layer, ff) in enumerate(zip(self.temporal_layers, self.spatial_layers, self.ed_layers, self.ff)):
            if not self.spatial:
                x1 = temporal_layer(x) 
                x_attn = spatial_layer(x1, supports) 
            else:
                x1 = spatial_layer(x, supports)
                x_attn = temporal_layer(x1)
            if prev_hidden is not None:
                x_attn = ed_layer(x_attn, prev_hidden[-1])
            x = ff(x_attn)
            hiddens.append(x)

            # hiddens.append(x)
            
        x_nemb = x[:,:,-1,:].reshape(x.shape[0],-1)     
        out = self.proj_class(self.act(x_nemb))           

        return x_start, out, hiddens
        

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):

        B, N, T, C = x.shape
        front = x[:, :, 0:1, :].repeat(1, 1, (self.kernel_size - 1) // 2, 1)
        end = x[:, :, -1:, :].repeat(1, 1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=2) 
        _, _, T_new, _ = x.shape
        x = self.avg(x.permute(0, 1, 3, 2).reshape(-1,C,T_new))     
        x = x.view(B, N, C, T)
        x = x.permute(0, 1, 3, 2)


        return x


class LD(nn.Module):
    def __init__(self,kernel_size=25):
        super(LD, self).__init__()
        # Define a shared convolution layers for all channels
        self.conv=nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=int(kernel_size//2), padding_mode='replicate', bias=True) 
        # Define the parameters for Gaussian initialization
        kernel_size_half = kernel_size // 2
        sigma = 1.0  # 1 for variance
        weights = torch.zeros(1, 1, kernel_size)
        for i in range(kernel_size):
            weights[0, 0, i] = math.exp(-((i - kernel_size_half) / (2 * sigma)) ** 2)

        # Set the weights of the convolution layer
        self.conv.weight.data = F.softmax(weights,dim=-1)
        self.conv.bias.data.fill_(0.0)
        
    def forward(self, inp):
        
        B, N, T, C = inp.shape
        inp = inp.reshape(B*N, T, C)

        self.conv = self.conv.to(inp.device)
        inp = inp.permute(0, 2, 1)
        input_channels = torch.split(inp, 1, dim=1)
        conv_outputs = [self.conv(input_channel) for input_channel in input_channels]
        
        out = torch.cat(conv_outputs, dim=1)
        out = out.permute(0, 2, 1)
        out = out.reshape(B, N, T, C)

        return out



class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size, use_LD=0, use_LDmoe=0):
        super(series_decomp_multi, self).__init__()
        self.use_LD = use_LD
        self.use_LDmoe = use_LDmoe
        if self.use_LD or self.use_LDmoe:
            self.LD = [LD(kernel) for kernel in kernel_size]
            self.layer = torch.nn.Linear(1, len(kernel_size))
        else:
            self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
            self.layer = torch.nn.Linear(1, len(kernel_size))


    def forward(self, x):

        if self.use_LDmoe:
            LD_trends=[]
            for func in self.LD:
                LD_trend = func(x)
                LD_trends.append(LD_trend.unsqueeze(-1))
            LD_trends=torch.cat(LD_trends,dim=-1)
            LD_trends = torch.sum(LD_trends*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
            res = x - LD_trends
            return res, LD_trends             
        
        elif self.use_LD:
            res = []
            LD_trends = []
            for func in self.LD:
                LD_trend = func(x)
                LD_trends.append(LD_trend)
                res.append(x-LD_trend)
            return res[0], LD_trends[0]
        else:
            res = []
            moving_mean=[]
            for func in self.moving_avg:
                moving_avg = func(x)
                moving_mean.append(moving_avg)
                res.append(x-moving_avg)

            return res, moving_mean


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index

# ########## fourier layer #############
class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes=0, num_heads=4, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        print('fourier enhanced block used!')
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        self.index = [1,4,5]
        print('modes={}, index={}'.format(modes, self.index))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(2000, num_heads, in_channels // num_heads, out_channels // num_heads, len(self.index), dtype=torch.float))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(2000, num_heads, in_channels // num_heads, out_channels // num_heads, len(self.index), dtype=torch.float))

    # Complex multiplication
    def compl_mul1d(self, order, x, weights):
        x_flag = True
        w_flag = True
        if not torch.is_complex(x):
            x_flag = False
            x = torch.complex(x, torch.zeros_like(x).to(x.device))
        if not torch.is_complex(weights):
            w_flag = False
            weights = torch.complex(weights, torch.zeros_like(weights).to(weights.device))
        if x_flag or w_flag:
            return torch.complex(torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag),
                                 torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real))
        else:
            return torch.einsum(order, x.real, weights.real)

    def forward(self, q, k, v, mask):

        B, N, L, H, E = q.shape
        x = q.permute(0, 1, 3, 4, 2) 

        x_ft = torch.fft.rfft(x, dim=-1)  
        out_ft = torch.zeros(B, N, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            if i >= x_ft.shape[4] or wi >= out_ft.shape[4]:
                continue
            out_ft[:, :, :, :, wi] = self.compl_mul1d("bnhi,nhio->bnho", x_ft[:, :, :, :, i],
                                                   torch.complex(self.weights1, self.weights2)[:, :, :, :, wi])

        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return (x, None)



class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, N, L, _ = queries.shape
        _, _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, N, L, H, -1)
        keys = self.key_projection(keys).view(B, N, S, H, -1)
        values = self.value_projection(values).view(B, N, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        # out = out.view(B, L, -1)
        out = out.view(B, N, L, -1)

        return self.out_projection(out), attn

class FFTDecompLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """

    def __init__(self, attention, d_model=None, d_ff=None, moving_avg=25, dropout=0.1, use_LD=0, use_LDmoe=0, activation="relu"):
        super(FFTDecompLayer, self).__init__()

        self.attention = attention
        self.use_LD = use_LD
        self.decomp1 = series_decomp_multi(moving_avg,use_LD=use_LD, use_LDmoe=use_LDmoe)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, trends = self.decomp1(x)
        
        if not self.use_LD:
            x = x[0]
            trends = trends[0]

        return x, trends


class FFTDecomp(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, activation = nn.GELU(), onlyFFT=0, use_wpe=1, hidden_size=32, out_dim=119):
        super(FFTDecomp, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.onlyFFT = onlyFFT
        self.use_wpe = use_wpe

        if onlyFFT:
            self.act = activation
            self.proj_class = nn.Linear(hidden_size*2000, out_dim)

        if not self.use_wpe:
            self.start_linear = nn.Linear(2, hidden_size)

    def forward(self, x, attn_mask=None):

        if not self.use_wpe:
            
            # x: [B,C,N,T]
            time_token, feat_token = x[:, -1, :, :], x[:, :-1, :, :]
            x = self.start_linear(feat_token.permute(0,2,3,1))

        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        # print(x.shape)
        if self.norm is not None:
            x = self.norm(x)
            # x = [self.norm(sub_x) for sub_x in x]

        if self.onlyFFT:
            x_c = x[:,:,-1,:].reshape(x.shape[0],-1)
            x = self.proj_class(self.act(x_c))

        return x, attn



class MemoryGate(nn.Module):
    """
    Input
     - input: B, N, T, in_dim, original input
     - hidden: hidden states from each expert, shape: E-length list of (B, N, T, C) tensors, where E is the number of experts
    Output
     - similarity score (i.e., routing probability before softmax function)
    Arguments
     - mem_hid, memory_size: hidden size and total number of memroy units
     - sim: similarity function to evaluate routing probability
     - nodewise: flag to determine routing level. Traffic forecasting could have a more fine-grained routing, because it has additional dimension for the roads
        - True: enables node-wise routing probability calculation, which is coarse-grained one
    """
    def __init__(self, hidden_size, num_nodes, mem_hid = 32, in_dim = 2, out_dim = 1, memory_size = 20, sim = nn.CosineSimilarity(dim = -1), nodewise = 0, metaloss = 0, ind_proj = True, attention_type = 'attention'):
        super(MemoryGate, self).__init__()
        self.attention_type = attention_type
        self.sim = sim
        self.nodewise = nodewise
        self.out_dim = out_dim
        # self.proclass = proclass
        self.metaloss = metaloss
        mem_hid = hidden_size

        self.memory = nn.Parameter(torch.empty(memory_size, mem_hid)) 
        
        self.hid_query = nn.ParameterList([nn.Parameter(torch.empty(hidden_size, mem_hid)) for _ in range(4)]) 
        self.key = nn.ParameterList([nn.Parameter(torch.empty(hidden_size, mem_hid)) for _ in range(4)])
        self.value = nn.ParameterList([nn.Parameter(torch.empty(hidden_size, mem_hid)) for _ in range(4)])

        self.input_query = nn.Parameter(torch.empty(hidden_size, mem_hid))   

        self.We1 = nn.Parameter(torch.empty(num_nodes, memory_size))    
        self.We2 = nn.Parameter(torch.empty(num_nodes, memory_size))    

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
    
    def forward(self, input, hidden):
        if self.attention_type == 'attention':
            attention = self.attention
        else:
            attention = self.topk_attention
        B, N, T, _ = input.size()

    def attention(self, x, i):
        B, N, T, _ = x.size()
        query = torch.matmul(x, self.hid_query[i])
        key = torch.matmul(x, self.key[i])
        value = torch.matmul(x, self.value[i])
        if self.nodewise:
            query = query.sum(dim = -2, keepdim = True)
        energy = torch.matmul(query, key.transpose(-1,-2))
        score = torch.softmax(energy, dim = -1)
        out = torch.matmul(score, value)
        return out.expand_as(value)

    def topk_attention(self, x, i, k = 3):
        B, N, T, _ = x.size()
        query = torch.matmul(x, self.hid_query[i])
        key = torch.matmul(x, self.key[i])
        value = torch.matmul(x, self.value[i])
        if self.nodewise:
            query = query.sum(dim = -2, keepdim = True)
        energy = torch.matmul(query, key.transpose(-1,-2))
        values, indices = torch.topk(energy, k = k, dim = -1)
        score = energy.zero_().scatter_(-1, indices, torch.relu(values))
        out = torch.matmul(score, value)
        return out.expand_as(value)

    def query_mem(self, input):
        B, N, T, _ = input.size()
        mem = self.memory
        query = torch.matmul(input, self.input_query) 
        energy = torch.matmul(query, mem.T)            
        score = torch.softmax(energy, dim = -1)      
        out = torch.matmul(score, mem)                
        return out

    def query_meta_memory(self, input): 
        B, N, T, _ = input.size()
        mem = self.memory
        query = torch.matmul(input, self.input_query)  
        energy = torch.matmul(query, mem.T)           
        score = torch.softmax(energy, dim = -1)        
        out = torch.matmul(score, mem)                

        _, ind = torch.topk(score, k=2, dim=-1)
        pos = self.memory[ind[:, :, :, 0]] # B, N, d
        neg = self.memory[ind[:, :, :, 1]] # B, N, d        

        out = out + input

        return out, query, pos, neg

    def reset_queries(self):
        with torch.no_grad():
            for p in self.hid_query:
                nn.init.xavier_uniform_(p)
            nn.init.xavier_uniform_(self.input_query)
    
    def reset_params(self):
        with torch.no_grad():
            for n, p in self.named_parameters():
                if n in "We1 We2 memory".split():
                    continue
                else:
                    nn.init.xavier_uniform_(p)


class AttnGate(nn.Module):
    def __init__(self, hidden_size, num_nodes, in_dim = 2, sim = nn.CosineSimilarity(dim = -1)):
        super(AttnGate, self).__init__()
        self.in_key = nn.Linear(in_dim, hidden_size, bias = False)
        self.hid_query = nn.Linear(hidden_size, hidden_size, bias = False)
        self.in_value = nn.Linear(in_dim, hidden_size, bias = False)
        sim = lambda x, y: nn.PairwiseDistance()(x, y) * -1
        self.sim = sim
        self.proj = nn.Linear(hidden_size, 1)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, input, hidden):
        num_heads = 1
        key = self.in_key(input)
        value = self.in_value(input)
        if num_heads > 1:
            key = torch.cat(torch.chunk(key, num_heads, dim = -1), dim = 0)
            value = torch.cat(torch.chunk(value, num_heads, dim = -1), dim = 0)
        scores = []
        for h in hidden:
            query = self.hid_query(h)
            if num_heads > 1:
                head_query = torch.cat(torch.chunk(query, num_heads, dim = -1), dim = 0)
                energy = torch.matmul(head_query, key.transpose(-1,-2)) / (head_query.size(-1) ** 0.5)
            else:
                energy = torch.matmul(query, key.transpose(-1,-2)) / (query.size(-1) ** 0.5)
            score = torch.softmax(energy, dim = -1)
            head_out = torch.matmul(score, value)
            out = torch.cat(torch.chunk(head_out, num_heads, dim = 0), dim = -1)
            scores.append(self.sim(query, out))
        return torch.stack(scores,dim = -1)


class WaveletEncoding(nn.Module):
    def __init__(self, hidden_size, in_dim, kernel_size=9):
        super().__init__()

        self.kernel_size = kernel_size
        self.start_linear = nn.Linear(in_dim, hidden_size)

        self.wave1 = torch.randn(2, hidden_size,1 )  # (2,128,1)  
        self.wave1[0]=torch.ones( hidden_size,1 )+ torch.randn( hidden_size,1 )  #make sure scale >0
        self.wave1 = nn.Parameter(self.wave1)
        
        self.wave2 = torch.zeros(2, hidden_size,1 )
        self.wave2[0]=torch.ones( hidden_size,1 )+ torch.randn( hidden_size,1 ) #make sure scale >0
        self.wave2 = nn.Parameter(self.wave2)
        
        self.wave3 = torch.zeros(2, hidden_size,1 )
        self.wave3[0]=torch.ones( hidden_size,1 )+ torch.randn( hidden_size,1 ) #make sure scale >0
        self.wave3 = nn.Parameter(self.wave3)  

        #n_w =3
        self.proj_1 = nn.Linear(hidden_size, hidden_size) 

    def mexican_hat_wavelet(self, size, scale, shift): #size :d*kernelsize  scale:d*1 shift:d*1
        """
        Generate a Mexican Hat wavelet kernel.

        Parameters:
        size (int): Size of the kernel.
        scale (float): Scale of the wavelet.
        shift (float): Shift of the wavelet.

        Returns:
        torch.Tensor: Mexican Hat wavelet kernel.
        """
    
        x = torch.linspace(-( size[1]-1)//2, ( size[1]-1)//2, size[1], device=self.wave1.device)
        x = x.reshape(1,-1).repeat(size[0],1)
        x = x - shift 
        C = 2 / ( 3**0.5 * torch.pi**0.25)
        wavelet = C * (1 - (x/scale)**2) * torch.exp(-(x/scale)**2 / 2)*1  /(torch.abs(scale)**0.5)

        return wavelet

    def forward(self, x):

        time_token, feat_token = x[:, -1, :, :], x[:, :-1, :, :]
        feat_token = self.start_linear(feat_token.permute(0,2,3,1))


        B, N, T, H = feat_token.shape
        feat_token = feat_token.view(B*N, T, H)
        
        x = feat_token.transpose(1, 2)
        
        D = x.shape[1]
        scale1, shift1 = self.wave1[0,:],self.wave1[1,:]
        wavelet_kernel1 = self.mexican_hat_wavelet(size=(D, self.kernel_size), scale=scale1, shift=shift1)
        scale2, shift2 = self.wave2[0,:],self.wave2[1,:]
        wavelet_kernel2 = self.mexican_hat_wavelet(size=(D, self.kernel_size), scale=scale2, shift=shift2)
        scale3, shift3 = self.wave3[0,:],self.wave3[1,:]
        wavelet_kernel3 = self.mexican_hat_wavelet(size=(D, self.kernel_size), scale=scale3, shift=shift3)
        
        pos1= torch.nn.functional.conv1d(x,wavelet_kernel1.unsqueeze(1),groups=D,padding ='same')
        pos2= torch.nn.functional.conv1d(x,wavelet_kernel2.unsqueeze(1),groups=D,padding ='same')
        pos3= torch.nn.functional.conv1d(x,wavelet_kernel3.unsqueeze(1),groups=D,padding ='same')
        x = x.transpose(1, 2) 
        
        x = x + self.proj_1(pos1.transpose(1, 2)+pos2.transpose(1, 2)+pos3.transpose(1, 2))
        
        x_restored = x.contiguous().view(B, N, T, H)

        return x_restored

class SparseDispatcher_TX(object):
    def __init__(self, num_experts):
        """
        Simplified dispatcher for cases where gates are [B, num_experts].
        
        Args:
            num_experts: Number of experts.
            gates: Tensor of shape [B, num_experts], where each element indicates
                   the weight of assigning a batch sample to an expert.
        """
        # self._gates = gates
        self._num_experts = num_experts
        self._original_input_shape = 10,12,2000,64


    def dispatch(self, inp, gates):
        """
        Dispatch inputs to experts based on gates.
        
        Args:
            inp: Tensor of shape [B, T, N, C].
        
        Returns:
            A list of tensors, each corresponding to the inputs for one expert.
            Each tensor has shape [num_samples_i, T, N, C], where num_samples_i is the number of inputs for expert i.
        """
        B, N, T, C = inp.shape
        self._original_input_shape = B, N, T, C

        expert_indices = torch.argmax(gates, dim=1)
        
        # Split inputs for each expert
        expert_inputs = []
        for i in range(self._num_experts):
            # Find batch samples assigned to expert i
            mask = expert_indices == i  # Shape: [B]
            
            # Select the corresponding inputs
            if mask.sum() > 0:
                expert_inputs.append(inp[mask])  # Shape: [num_samples_i, T, N, C]
            else:
                # Add an empty tensor if no samples are assigned to this expert
                expert_inputs.append(torch.empty(0, N, T, C, device=inp.device))
        
        return expert_inputs

    def combine(self, expert_out, gates):
        """
        Combine expert outputs into a single tensor matching the input shape.
        
        Args:
            expert_out: List of tensors, each with shape [num_samples_i, T, N, C].
        
        Returns:
            A tensor of shape [B, T, N, C], where outputs from experts are combined.
        """
        B, N, T, C = self._original_input_shape
        N = 119
        B = gates.shape[0]

        # Create an empty output tensor
        output = torch.zeros(B, N, device=gates.device)

        # Get the expert index for each batch sample
        expert_indices = torch.argmax(gates, dim=1)  # Shape: [B]

        # Fill the output tensor based on expert outputs
        offset = 0
        for i in range(self._num_experts):
            mask = expert_indices == i  # Shape: [B]
            # print(mask)
            num_samples = mask.sum()
            if num_samples > 0:
                output[mask] = expert_out[i]  # Assign outputs to the correct positions
        
        return output

class MoE_gate(nn.Module):

    def __init__(self, num_experts, hidden_size, noisy_gating=True, k=2, Tem=12, Nodes_num=2000):
        super(MoE_gate, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.k = k

        self.flan_size = Nodes_num*hidden_size
        
        self.w_gate = nn.Parameter(torch.empty(self.flan_size, num_experts))
        nn.init.normal_(self.w_gate, mean=0.0, std=0.02)
        self.w_noise = nn.Parameter(torch.empty(self.flan_size, num_experts))
        nn.init.normal_(self.w_noise, mean=0.0, std=0.02)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def noisy_top_k_gating(self, x, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        #[B,N,T,hidden_size]
        B,N,T,C = x.shape
        input_x = x[:,:,-1,:].contiguous().reshape(B,-1)    #[B,N,hidden_size]
        
        clean_logits = input_x @ self.w_gate
        if self.noisy_gating:
            raw_noise_stddev = input_x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(1, dim=1)
        zeros = torch.zeros_like(logits, requires_grad=True)  # Create zero tensor
        gates = zeros.scatter(1, top_indices, torch.ones_like(top_logits))  # Assign a gate of 1 to the top-1 expert
        
        return gates, logits



class FELDMSTM(nn.Module):
    """
    FELDMSTM model 
    """
    def __init__(self, num_nodes, dropout=0.3, in_dim=2, out_dim = 1, hidden_size = 32, layers = 1, use_adj=0, 
                 in_device=torch.device('cuda:0'), memory_size = 20, num_heads = 4, prob_mul = False, spatial = 1, 
                 use_wpe = 1, use_FFTdcom = 0, max_time_index = 168, modes=3, 
                 moving_avg=[7], use_LD=0, use_LDmoe=0, use_ada=1, metaLoss=0, moe_gate=0,wpe_kernel_size=9, **args):
        super(FELDMSTM, self).__init__()
        self.dropout = dropout
        self.prob_mul = prob_mul
        self.supports_len = 2
        self.max_time_index = max_time_index  #288
        self.use_adj = use_adj
        self.device = in_device
        self.memory_size = memory_size
        self.use_wpe = use_wpe
        self.use_FFTdcom = use_FFTdcom
        self.use_ada = use_ada
        self.metaLoss = metaLoss
        self.moe_gate = moe_gate
        self.onlyFFT = 0
        self.max_time_index = 24

        # IF proclass, out_dim need to set!!

        if self.use_wpe:
            self.waveposEmb = WaveletEncoding(hidden_size, in_dim = in_dim - 1, kernel_size=wpe_kernel_size)  # in_dim -1 because we do not use time feature here

        if self.use_FFTdcom:

            e_layers = 1
            encoder_self_att = FourierBlock(in_channels=hidden_size,
                                            out_channels=hidden_size,
                                            seq_len=12,
                                            modes=modes,
                                            num_heads = num_heads,
                                            mode_select_method='random')

            # Encoder
            self.fft_decomp = FFTDecomp(
                [
                    FFTDecompLayer(
                        AutoCorrelationLayer(
                            encoder_self_att,  # instead of multi-head attention in transformer
                            hidden_size, num_heads),
                        moving_avg=moving_avg,
                        dropout=dropout,
                        use_LD = use_LD,
                        use_LDmoe = use_LDmoe
                    ) for l in range(e_layers)
                ],
                norm_layer=LayerNorm(normalized_shape = (hidden_size, )),
                onlyFFT=self.onlyFFT,
                use_wpe=self.use_wpe,
                hidden_size=hidden_size,
                out_dim=out_dim
            )

            if not self.use_wpe:
                use_wpe = 1

        if self.use_adj:
            
            # Ybus matrix
            adj_m = pd.read_csv("/home/lihaoqin/DL_Code/Crossformer-master/cross_models/ACTIVSg2000_Ybus_Matrix.csv", header=None)
            adj_complex = convert_to_complex(adj_m)
            adj_complex_np = adj_complex.values 
            Ybus_real  = np.real(adj_complex_np)
            Ybus_imag = np.imag(adj_complex_np)
            Ybus_magnitude = np.abs(adj_complex_np)
            Ybus_phase = np.angle(adj_complex_np)
            Ybus_real  = np.array(Ybus_real, dtype=np.float32)
            Ybus_imag  = np.array(Ybus_imag, dtype=np.float32)


            # 对实部/虚部进行归一化
            normalized_real = symmetric_normalize(Ybus_real)
            normalized_imag = symmetric_normalize(Ybus_imag)
            normalized_magnitude = symmetric_normalize(Ybus_magnitude)
            normalized_phase = symmetric_normalize(Ybus_phase)        


            self.adj_0 = torch.tensor(normalized_real, dtype=torch.float32).to(self.device)
            self.adj_1 = torch.tensor(normalized_imag, dtype=torch.float32).to(self.device)
            self.adj_2 = torch.tensor(normalized_magnitude, dtype=torch.float32).to(self.device)
            self.adj_3 = torch.tensor(normalized_phase, dtype=torch.float32).to(self.device)

            self.identity_expert = STModel(hidden_size, self.supports_len, num_nodes, in_dim = in_dim, out_dim = out_dim, spatial = spatial, layers = layers, dropout = dropout, num_heads=num_heads, use_wpe=use_wpe)

        if self.use_ada:
            self.adaptive_expert = STModel(hidden_size, self.supports_len, num_nodes, in_dim = in_dim, out_dim = out_dim, spatial = spatial, layers = layers, dropout = dropout, num_heads=num_heads, use_wpe=use_wpe)


        self.gate_network = MemoryGate(hidden_size, num_nodes, in_dim = in_dim, out_dim = out_dim, memory_size=self.memory_size, metaloss=metaLoss)
        
        if self.moe_gate:
            self.spa_gate = MoE_gate(self.moe_gate, hidden_size, k=1, Tem=12, Nodes_num=num_nodes, noisy_gating=True)
            self.dispatcher = SparseDispatcher_TX(self.moe_gate)


        if not self.use_wpe:
            use_wpe = 0

    def forward(self, input, x_time, gate_out = False):
        """
        input: B, in_dim, N, T
         - Note: we assume that the last dimeions of in_dim is temporal feature, such as tod or dow (could be represented as integer)
        o_identity shape B, N, T, 1
        """
        
        B,T,N = input.shape                                    
        input = input.reshape(B, T, int(N/2), 2)     
        input = input.permute(0,3,2,1)
        _,_,N,_ = input.shape
        #cat input and x_daytime
        x_time_expanded = x_time.unsqueeze(1).unsqueeze(2)
        input = torch.cat([input, x_time_expanded.expand(-1, -1, N, -1)], dim=1)

        if self.use_adj:

            adj_supports = [self.adj_2, self.adj_2]

        # else:
        n1 = torch.matmul(self.gate_network.We1, self.gate_network.memory)   
        n2 = torch.matmul(self.gate_network.We2, self.gate_network.memory) 
        g1 = torch.softmax(torch.relu(torch.mm(n1, n2.T)), dim = -1)         
        g2 = torch.softmax(torch.relu(torch.mm(n2, n1.T)), dim = -1)
        new_supports = [g1, g2]

        if self.use_wpe:
            output_wave = self.waveposEmb(input) 

            input_fea = output_wave
            
        else:
            input_fea = input


        if self.use_FFTdcom:
            input_fea, trends = self.fft_decomp(input_fea, attn_mask=None)
            if self.onlyFFT:
                return input_fea

        if self.metaLoss:
            input_fea, query, pos, neg = self.gate_network.query_meta_memory(input_fea)

        if self.moe_gate:
            spar_gates, allgates = self.spa_gate.noisy_top_k_gating(input_fea)
            expert_inputs = self.dispatcher.dispatch(input_fea, spar_gates)
            h_future = None
            expert_outputs = []
            exp_0=[]
            exp_1=[]
            for i in range(self.moe_gate):

                if i==0 and expert_inputs[i].shape[0]>0:
                    _, exp_0, h_identity = self.identity_expert(expert_inputs[i], h_future, adj_supports)

                elif i==1 and expert_inputs[i].shape[0]>0:
                    _, exp_1, h_adaptive = self.adaptive_expert(expert_inputs[i], h_future, new_supports)


            expert_outputs.append(exp_0)
            expert_outputs.append(exp_1)    

            spaE_output = self.dispatcher.combine(expert_outputs, spar_gates)
            
            if self.metaLoss:
                return spaE_output, query, pos, neg, spar_gates, allgates
            else:
                return spaE_output, spar_gates, allgates

                


