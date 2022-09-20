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
    def __init__(self, n_iters=5, T=1.0):
        super().__init__()
        self.n_iters = n_iters
        self.T = T

    def forward(self, q, k, v):
        assert(q.shape[0] == k.shape[0])  # Same batch size
        assert(q.shape[1] == k.shape[1])  # Same number of channels (Q/K)
        assert(k.shape[2] == v.shape[2])  # Same spatial dimensions (K/V)
        assert(k.shape[3] == v.shape[3])

        output = torch.zeros(q.shape[0], v.shape[1], q.shape[2], q.shape[3], device=q.device)
        for i, (qi, ki, vi) in enumerate(zip(q, k, v)):
            output[i] = attention_layer(qi, ki, vi, self.n_iters, self.T)

        return output


def attention_layer(q, k, v, n_iters=5, T=1.0):
    shift_map, cost_map = PatchMatch.apply(q, k, n_iters)

    # Simple reconstruction using the central pixel and no weighting scheme
    cost_map = torch.softmax(-cost_map/T, dim=0)
    reconstruction = torch.sum(cost_map[None, None, :, :, :] * v[:,shift_map[0], shift_map[1]], dim=2)

    return reconstruction
