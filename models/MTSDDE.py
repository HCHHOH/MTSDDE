import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import PositionalEmbedding
from layers.Graph_Constructor import graph_constructor
from utils.odegcn import ODEblock, ODEFunc

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.node_num = configs.node_num
        self.subgraph_size = configs.subgragh_size
        self.ode_t = configs.ode_t

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * self.period_len // 2,
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)

        # 动态图构建
        self.gc = graph_constructor(self.node_num, self.subgraph_size)

        # 初始化ODE
        self.odeblock = ODEblock(ODEFunc(self.seq_len, self.node_num), t=torch.tensor([0, self.ode_t]))

        self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)


    def forward(self, x):
        batch_size = x.shape[0]
        # normalization and permute     b,s,c -> b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        # 1D convolution aggregation
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.node_num, self.seq_len) + x

        # # TCN聚合
        # x = self.tcn(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

        idx = torch.arange(self.node_num).to(x.device)
        adj = self.gc(idx, x.device)

        # """
        # ODE网络
        self.odeblock.set_x0(x)
        self.odeblock.set_adj(adj)
        # self.odeblock.set_adj(self.i_adj) # adj
        x = self.odeblock(x)
        # """

        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # period_idx = torch.arange(self.period_len).to(x.device)

        # sparse forecasting
        y = self.linear(x)  # bc,w,m
        # y = self.linear_nop(x)

        # upsampling: bc,w,m -> bc,m,w -> b,c,s
        y = y.permute(0, 2, 1).reshape(batch_size, self.node_num, self.pred_len)

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_mean

        return y
