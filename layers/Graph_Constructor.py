import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F


class graph_constructor(nn.Module):
    def __init__(self, nnodes, topk, node_dim=40, gc_alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, node_dim)
            self.lin2 = nn.Linear(xd, node_dim)
        else:
            self.emb1 = nn.Embedding(nnodes, node_dim)
            self.emb2 = nn.Embedding(nnodes, node_dim)
            self.lin1 = nn.Linear(node_dim,node_dim)
            self.lin2 = nn.Linear(node_dim,node_dim)

        # self.device = device
        self.topk = topk
        self.node_dim = node_dim
        self.gc_alpha = gc_alpha
        self.static_feat = static_feat

    def forward(self, idx, device):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.gc_alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.gc_alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.gc_alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(device)
        mask.fill_(float('0'))
        # 向 adj 张量中添加一个与 adj 大小相同的随机噪声 torch.rand_like(adj)*0.01。这一步可以打破平局，避免在排序时出现相等的值。
        # 使用 topk(self.k, 1) 函数从每行中取出前 k 个最大值，s1 是这些最大值，t1 是对应的索引。
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.topk,1)
        # s1.fill_(1) 将 s1 中所有的值填充为 1
        # 使用 scatter_ 函数将 t1 中索引对应位置的值设置为 1
        mask.scatter_(1,t1,s1.fill_(1))
        # 不是 top k 索引位置的值置为 0，只保留 top k 索引位置的值
        adj = adj*mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.gc_alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.gc_alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.gc_alpha*a))
        return adj

class period_fullA(nn.Module):
    def __init__(self, period_len, node_dim=40, pa_alpha=3):
        super(period_fullA, self).__init__()
        self.period_len = period_len

        self.p_emb1 = nn.Embedding(period_len, node_dim)
        self.p_emb2 = nn.Embedding(period_len, node_dim)
        self.p_lin1 = nn.Linear(node_dim,node_dim)
        self.p_lin2 = nn.Linear(node_dim,node_dim)

        # self.device = device
        self.node_dim = node_dim
        self.pa_alpha = pa_alpha

    def forward(self, idx, device):
        nodevec1 = self.p_emb1(idx)
        nodevec2 = self.p_emb2(idx)

        nodevec1 = torch.tanh(self.pa_alpha*self.p_lin1(nodevec1))
        nodevec2 = torch.tanh(self.pa_alpha*self.p_lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.pa_alpha*a))
        return adj