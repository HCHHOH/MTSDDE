import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchdiffeq import odeint


# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class ODEFunc(nn.Module):
    def __init__(self, temporal_dim, node_num):
        super(ODEFunc, self).__init__()
        self.adj = None
        self.x0 = None
        # self.alpha = nn.Parameter(0.8 * torch.ones(adj.shape[0]))
        self.alpha = nn.Parameter(0.8 * torch.ones(node_num))
        self.beta = 0.6
        self.w = nn.Parameter(torch.eye(temporal_dim))
        self.d = nn.Parameter(torch.zeros(temporal_dim) + 1) # 一个可学习参数，初始化为全1的向量，长度为 temporal_dim

    def forward(self, t, x):
        # 将 self.adj 移动到与 x 相同的设备上
        self.adj = self.adj.to(x.device)
        alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(0)

        batch_size, num_nodes, temporal_dim = x.shape

        # 进行 einsum 操作
        xa = torch.einsum('ij, kjm->kim', self.adj, x)

        # 确保特征维度上的特征值小于 1
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        # 将输入张量 x 与权重矩阵 w 相乘，表示特征变换
        xw = torch.einsum('ijk, kl->ijl', x, w)

        # 计算 f
        f = alpha / 2 * xa - x + xw - x + self.x0
        return f

class ODEblock(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0, 1])):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def set_adj(self, adj):
        self.odefunc.adj = adj

    def forward(self, x):
        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t, method='euler')[1]
        return z


# Define the ODEGCN model.
class ODEG(nn.Module):
    def __init__(self, feature_dim, temporal_dim, adj, time):
        super(ODEG, self).__init__()
        self.odeblock = ODEblock(ODEFunc(feature_dim, temporal_dim, adj), t=torch.tensor([0, time]))

    def forward(self, x):
        self.odeblock.set_x0(x)
        z = self.odeblock(x)
        return F.relu(z)

