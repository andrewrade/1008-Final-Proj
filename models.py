from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)

class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output

####################################################################################
#                                      ENCODER                                     #
####################################################################################
class ResidualLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.ConvBlock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        # 1 x 1 Convolution for Residual whenever downsmapling (stride > 1)
        self.downsample_residual = None
        if stride != 1 or in_channels != out_channels:
            self.downsample_residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        if self.downsample_residual is not None:
            residual = self.downsample_residual(x)
        x = self.ConvBlock(x)
        return (F.relu(x + residual))


class EncoderBackbone(torch.nn.Module):

    def __init__(self, n_kernels):
        super().__init__()
        self.n_kernels = n_kernels
       
        # 2 x 64 x 64 --> n_kernels x 32 x 32
        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(2, self.n_kernels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(self.n_kernels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # n_kernels x 32 x 32
        )
        # n_kernels x 32 x 32 --> n_kernels x 32 x 32
        self.ResBlock1 = nn.Sequential(
            ResidualLayer(n_kernels, n_kernels),
            ResidualLayer(n_kernels, n_kernels)
        )
        # n_kernels x 32 x 32 --> n_kernels * 2 x 16 x 16
        self.ResBlock2 = nn.Sequential(
            ResidualLayer(n_kernels, n_kernels*2, stride=2),
            ResidualLayer(n_kernels*2, n_kernels*2)
        )
        # n_kernels * 2 x 16 x 16 --> n_kernels * 4 x 8 x 8
        self.ResBlock3 = nn.Sequential(
            ResidualLayer(n_kernels*2, n_kernels*4, stride=2),
            ResidualLayer(n_kernels*4, n_kernels*4)
        )
        # n_kernels * 4 x 8 x 8 --> n_kernels * 8 x 4 x 4
        self.ResBlock4 = nn.Sequential(
            ResidualLayer(n_kernels*4, n_kernels*8, stride=2),
            ResidualLayer(n_kernels*8, n_kernels*8)
        )

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2) # n_kernels * 8 x 1 x 1
    
    def forward(self, x):
        x = self.ConvLayer1(x) # 64x64 -> 16x16
        x = self.ResBlock1(x) # 16x16
        x = self.ResBlock2(x) # 16x16 -> 8x8
        x = self.ResBlock3(x) # 8x8 -> 4x4
        x = self.ResBlock4(x) # 4x4
        x = self.avgpool(x) # 4x4 -> 1x1
        return torch.flatten(x, 1) # (batch_size, n_kernels * 8)


class BarlowTwins(torch.nn.Module):

    def __init__(self, batch_size, repr_dim, projection_layers=3, lambd=5E-3):
        super().__init__()
        assert repr_dim % 8 == 0, 'Representation Size should be multiple of 8'
        
        self.backbone = EncoderBackbone(n_kernels=repr_dim // 8)
        self.batch_size = batch_size
        self.repr_dim = repr_dim
        self.projection_layers = projection_layers
        self.lambd = lambd

        layer_sizes = [self.repr_dim] + [(self.repr_dim * 4) for _ in range(self.projection_layers)]
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=False))
            layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.projector = nn.Sequential(*layers)
        
        self.batch_norm = nn.BatchNorm1d(layer_sizes[-1])
    
    def _off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m, 'Square Matrix Expected, Rectangular Instead'
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, Y_a, Y_b=None):

        if not self.training:
            return self.backbone(Y_a)
        
        Z_a = self.projector(self.backbone(Y_a))
        Z_b = self.projector(self.backbone(Y_b))

        print(f'Z_a size: {Z_a.size()}')
        print(f'Z_b size: {Z_a.size()}')

        # Cross-Correlation Matrix
        cc_mat = self.batch_norm(Z_a).T @ self.batch_norm(Z_b)
        cc_mat.div_(self.batch_size)

        diag = torch.diagonal(cc_mat).add_(-1).pow_(2).sum()
        off_diag = self._off_diagonal(cc_mat).pow_(2).sum()

        loss = diag + self.lambd * off_diag
        return loss
        







 
        

