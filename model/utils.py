import torch
from torch.nn import functional


def conv_power_method(D, image_size, num_iters=100, stride=1):
    """
    Finds the maximal eigenvalue of D.T.dot(D) using the iterative power method
    :param D:
    :param num_needles:
    :param image_size:
    :param patch_size:
    :param num_iters:
    :return:
    """
    needles_shape = [int(((image_size[0] - D.shape[-2])/stride)+1), int(((image_size[1] - D.shape[-1])/stride)+1)]
    x = torch.randn(1, D.shape[0], *needles_shape).type_as(D)
    for _ in range(num_iters):
        c = torch.norm(x.reshape(-1))
        x = x / c
        y = functional.conv_transpose2d(x, D, stride=stride)
        x = functional.conv2d(y, D, stride=stride)
    return torch.norm(x.reshape(-1))


def calc_pad_sizes(I: torch.Tensor, kernel_size: int, stride: int):
    left_pad = stride
    right_pad = 0 if (I.shape[3] + left_pad - kernel_size) % stride == 0 else stride - ((I.shape[3] + left_pad - kernel_size) % stride)
    top_pad = stride
    bot_pad = 0 if (I.shape[2] + top_pad - kernel_size) % stride == 0 else stride - ((I.shape[2] + top_pad - kernel_size) % stride)
    right_pad += stride
    bot_pad += stride
    return left_pad, right_pad, top_pad, bot_pad