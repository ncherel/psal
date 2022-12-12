import torch

from torch.autograd import Function
from torch.nn.functional import pad, unfold, conv2d

from .patchmatch import backward, patchmatch
from .patchmatch_masked import backward_masked, patchmatch_masked


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
    def __init__(self, patch_size=3, n_iters=5, T=1.0, aggregation=False):
        super().__init__()
        self.patch_size = patch_size
        self.n_iters = n_iters
        self.T = T
        self.aggregation = aggregation

    def attention(self, q, k, v):
        shift_map, cost_map = PatchMatch.apply(q, k, self.n_iters)

        if not self.aggregation:
            # Simple reconstruction using the central pixel
            cost_map = torch.softmax(-cost_map/self.T, dim=0)
            reconstruction = torch.sum(cost_map[None, None, :, :, :] * v[:,shift_map[0], shift_map[1]], dim=2)

        else:
            reconstruction = self.aggregate(v, shift_map, cost_map)

        return reconstruction

    def aggregate(self, v, shift_map, cost_map):
        p = self.patch_size//2
        K, H, W = cost_map.shape
        padded_cost_map = pad(cost_map, (p,p,p,p), value=10)
        padded_shift_map = pad(shift_map, (p,p,p,p), value=10)  # Add a padding value in the valid region
        all_cost_map = torch.zeros((self.patch_size*self.patch_size*K, H, W), device="cuda")
        all_shift_map = torch.zeros((2, self.patch_size*self.patch_size*K, H, W), dtype=torch.int64, device="cuda")
        idx = 0
        # Complicated computation going on:
        for di in range(self.patch_size):
            for dj in range(self.patch_size):
                start_i, start_j = di, dj
                end_i, end_j = H + di, W + dj
                pi, pj = p - di, p - dj  # Relative to the patch position (reference is central pixel)
                all_cost_map[K*idx:K*(idx+1)] = padded_cost_map[:,start_i:end_i, start_j:end_j]
                all_shift_map[0, K*idx:K*(idx+1)] = padded_shift_map[0,:,start_i:end_i,start_j:end_j] + pi
                all_shift_map[1, K*idx:K*(idx+1)] = padded_shift_map[1,:,start_i:end_i,start_j:end_j] + pj
                idx += 1
        all_cost_map = torch.softmax(-all_cost_map / self.T, dim=0)
        all_shift_map[0] = torch.clamp(all_shift_map[0], 0, v.shape[1]-1)
        all_shift_map[1] = torch.clamp(all_shift_map[1], 0, v.shape[2]-1)

        return torch.sum(all_cost_map[None, None, :, :, :] * v[:,all_shift_map[0], all_shift_map[1]], dim=2)

    def forward(self, q, k, v):
        assert(q.shape[0] == k.shape[0])  # Same batch size
        assert(q.shape[1] == k.shape[1])  # Same number of channels (Q/K)
        assert(k.shape[2] == v.shape[2])  # Same spatial dimensions (K/V)
        assert(k.shape[3] == v.shape[3])

        output = torch.zeros(q.shape[0], v.shape[1], q.shape[2], q.shape[3], device=q.device)
        # Process batch elements one by one
        for i, (qi, ki, vi) in enumerate(zip(q, k, v)):
            output[i] = self.attention(qi, ki, vi)

        return output


class PatchMatchMasked(Function):
    @staticmethod
    def forward(ctx, a, b, n_iters=10):
        shift_map, cost_map = patchmatch_masked(a, b, n_iters=n_iters)
        torch.cuda.synchronize()
        ctx.save_for_backward(a, b, shift_map)
        shift_map = shift_map.type(torch.int64)
        return shift_map, cost_map

    @staticmethod
    def backward(ctx, shift_map_grad, cost_map_grad):
        a, b, shift_map = ctx.saved_tensors
        grad_a = backward_masked(a, b, shift_map, cost_map_grad)
        torch.cuda.synchronize()
        return grad_a, None, None


class PSAttentionMasked(torch.nn.Module):
    def __init__(self, patch_size=3, n_iters=5, T=1.0, aggregation=False):
        super().__init__()
        self.patch_size = patch_size
        self.n_iters = n_iters
        self.T = T
        self.aggregation = aggregation

    def attention(self, x, mask, v, T=1.0):
        shift_map, cost_map = PatchMatchMasked.apply(x, mask, self.n_iters)

        if not self.aggregation:
            # Simple reconstruction using the central pixel
            cost_map = torch.softmax(-cost_map/T, dim=0)
            reconstruction = torch.sum(cost_map[None, None, :, :, :] * v[:,shift_map[0], shift_map[1]], dim=2)

        else:
            reconstruction = self.aggregate(x, shift_map, cost_map)

        return reconstruction

    def aggregate(self, v, shift_map, cost_map):
        p = self.patch_size//2
        K, H, W = cost_map.shape
        padded_cost_map = pad(cost_map, (p,p,p,p), value=10)
        padded_shift_map = pad(shift_map, (p,p,p,p), value=10)  # Add a padding value in the valid region
        all_cost_map = torch.zeros((self.patch_size*self.patch_size*K, H, W), device="cuda")
        all_shift_map = torch.zeros((2, self.patch_size*self.patch_size*K, H, W), dtype=torch.int64, device="cuda")
        idx = 0
        # Complicated computation going on:
        for di in range(self.patch_size):
            for dj in range(self.patch_size):
                start_i, start_j = di, dj
                end_i, end_j = H + di, W + dj
                pi, pj = p - di, p - dj  # Relative to the patch position (reference is central pixel)
                all_cost_map[K*idx:K*(idx+1)] = padded_cost_map[:,start_i:end_i, start_j:end_j]
                all_shift_map[0, K*idx:K*(idx+1)] = padded_shift_map[0,:,start_i:end_i,start_j:end_j] + pi
                all_shift_map[1, K*idx:K*(idx+1)] = padded_shift_map[1,:,start_i:end_i,start_j:end_j] + pj
                idx += 1
        all_cost_map = torch.softmax(-all_cost_map / self.T, dim=0)
        all_shift_map[0] = torch.clamp(all_shift_map[0], 0, v.shape[1]-1)
        all_shift_map[1] = torch.clamp(all_shift_map[1], 0, v.shape[2]-1)

        return torch.sum(all_cost_map[None, None, :, :, :] * v[:,all_shift_map[0], all_shift_map[1]], dim=2)

    def forward(self, x, mask, v=None, T=1.0):
        output = torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)
        # Process batch elements one by one
        if v is None:
            v = x
        for i, (xi, maski, vi) in enumerate(zip(x, mask, v)):
            output[i] = self.attention(xi, maski, vi, T=T)

        return output
