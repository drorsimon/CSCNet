import torch
import numpy as np
from torch import nn
from torch.nn import functional
from collections import namedtuple
from model.utils import conv_power_method, calc_pad_sizes


class SoftThreshold(nn.Module):
    def __init__(self, size, init_threshold=1e-3):
        super(SoftThreshold, self).__init__()
        self.threshold = nn.Parameter(init_threshold * torch.ones(1,size,1,1))

    def forward(self, x):
        mask1 = (x > self.threshold).float()
        mask2 = (x < -self.threshold).float()
        out = mask1.float() * (x - self.threshold)
        out += mask2.float() * (x + self.threshold)
        return out


ListaParams = namedtuple('ListaParams', ['kernel_size', 'num_filters', 'stride', 'unfoldings'])


class ConvLista_T(nn.Module):
    def __init__(self, params: ListaParams, A=None, B=None, C=None, threshold=1e-2):
        super(ConvLista_T, self).__init__()
        if A is None:
            A = torch.randn(params.num_filters, 1, params.kernel_size, params.kernel_size)
            l = conv_power_method(A, [512, 512], num_iters=200, stride=params.stride)
            A /= torch.sqrt(l)
        if B is None:
            B = torch.clone(A)
        if C is None:
            C = torch.clone(A)
        self.apply_A = torch.nn.ConvTranspose2d(params.num_filters, 1, kernel_size=params.kernel_size,
                                                stride=params.stride, bias=False)
        self.apply_B = torch.nn.Conv2d(1, params.num_filters, kernel_size=params.kernel_size, stride=params.stride, bias=False)
        self.apply_C = torch.nn.ConvTranspose2d(params.num_filters, 1, kernel_size=params.kernel_size,
                                                stride=params.stride, bias=False)
        self.apply_A.weight.data = A
        self.apply_B.weight.data = B
        self.apply_C.weight.data = C
        self.soft_threshold = SoftThreshold(params.num_filters, threshold)
        self.params = params

    def _split_image(self, I):
        if self.params.stride == 1:
            return I, torch.ones_like(I)
        left_pad, right_pad, top_pad, bot_pad = calc_pad_sizes(I, self.params.kernel_size, self.params.stride)
        I_batched_padded = torch.zeros(I.shape[0], self.params.stride ** 2, I.shape[1], top_pad + I.shape[2] + bot_pad,
                                       left_pad + I.shape[3] + right_pad).type_as(I)
        valids_batched = torch.zeros_like(I_batched_padded)
        for num, (row_shift, col_shift) in enumerate([(i, j) for i in range(self.params.stride) for j in range(self.params.stride)]):
            I_padded = functional.pad(I, pad=(
            left_pad - col_shift, right_pad + col_shift, top_pad - row_shift, bot_pad + row_shift), mode='reflect')
            valids = functional.pad(torch.ones_like(I), pad=(
            left_pad - col_shift, right_pad + col_shift, top_pad - row_shift, bot_pad + row_shift), mode='constant')
            I_batched_padded[:, num, :, :, :] = I_padded
            valids_batched[:, num, :, :, :] = valids
        I_batched_padded = I_batched_padded.reshape(-1, *I_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return I_batched_padded, valids_batched

    def forward(self, I):
        I_batched_padded, valids_batched = self._split_image(I)
        conv_input = self.apply_B(I_batched_padded)
        gamma_k = self.soft_threshold(conv_input)
        for k in range(self.params.unfoldings - 1):
            x_k = self.apply_A(gamma_k)
            r_k = self.apply_B(x_k - I_batched_padded)
            gamma_k = self.soft_threshold(gamma_k - r_k)
        output_all = self.apply_C(gamma_k)
        output_cropped = torch.masked_select(output_all, valids_batched.byte()).reshape(I.shape[0], self.params.stride ** 2, *I.shape[1:])
        # if self.return_all:
        #     return output_cropped
        output = output_cropped.mean(dim=1, keepdim=False)
        return output