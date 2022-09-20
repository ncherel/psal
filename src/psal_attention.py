import torch

from torch.autograd import Function
from torch.nn.functional import pad, unfold, conv2d

from .patchmatch import backward, patchmatch


class PatchMatch(Function):
    @staticmethod
    def forward(ctx, a, b, n_iters=10):
        shift_map, cost_map = patchmatch(a, b, n_iters=n_iters)
        torch.cuda.synchronize()
        ctx.save_for_backward(a, b, shift_map)
        shift_map = shift_map.type(torch.int64)
        return shift_map, cost_map

    @staticmethod
    def backward(ctx, shift_map_grad, cost_map_grad):
        a, b, shift_map = ctx.saved_tensors
        grad_a, grad_b = backward(a, b, shift_map, cost_map_grad)
        torch.cuda.synchronize()
        return grad_a, grad_b


class PSAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        q = q[0]
        k = k[0]

        shift_map, cost_map = PatchMatch.apply(q, k)

        # Simple reconstruction using the central pixel and no weighting scheme
        cost_map = torch.softmax(-cost_map, dim=0)
        reconstruction = torch.sum(cost_map[None, None, :, :, :] * v[:,:,shift_map[0], shift_map[1]], dim=2)

        return reconstruction


def attention_layer(q, k, v):
    """Can only handle batch of size 1"""
    # PatchMatch layer takes C H W tensor
    q = q[0]
    k = k[0]

    shift_map, cost_map = PatchMatch.apply(q, k)

    # Simple reconstruction using the central pixel and no weighting scheme
    cost_map = torch.softmax(-cost_map, dim=0)
    reconstruction = torch.sum(cost_map[None, None, :, :, :] * v[:,:,shift_map[0], shift_map[1]], dim=2)

    return reconstruction
