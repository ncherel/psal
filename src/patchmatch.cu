#include "patchmatch.h"

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <cmath>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>


template <typename T, int PSZ>
__device__ T dist(at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t1,
		  at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t2,
		  int32_t x1, int32_t y1,
		  int32_t x2, int32_t y2,
		  T cutoff=1e10);

template <typename T, int PSZ>
__global__ void initialise_shift_map(at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t1,
				     at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t2,
				     at::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> shift_map,
				     at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> cost_map,
				     at::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> states);
template <typename T, int PSZ>
__global__ void propagation(at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t1,
			    at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t2,
			    at::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> shift_map,
			    at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> cost_map);

template <typename T, int PSZ>
__global__ void random_search(at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t1,
			      at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t2,
			      at::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> shift_map,
			      at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> cost_map,
			      at::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> states);

template <typename T, int PSZ>
__global__ void backward_kernel(at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t1,
				at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t2,
				at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> grad_t1,
				at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> grad_t2,
				at::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> shift_map,
				at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> cost_map);

std::vector<at::Tensor> patchmatch_cuda(const at::Tensor t1,
					const at::Tensor t2,
					int patch_size=3,
					int n_iters=10) {
  auto H = t1.size(1);
  auto W = t1.size(2);
  auto shift_map = torch::full({ 2, K, H, W }, -1, t1.options().dtype(torch::kInt32));
  auto cost_map = torch::full({ K, H, W }, 0.0, t1.options());

  // Must make the grid large enough to cover all pixels
  const dim3 blocks(4, 4);
  const dim3 grid(int(t1.size(1) / blocks.x) + 1, int(t1.size(2) / blocks.y) + 1);

  // Initialise the random states
  auto states = torch::randint(2 << 16, { H, W }, t1.options().dtype(torch::kInt64));
  
  AT_DISPATCH_FLOATING_TYPES(t1.scalar_type(), "patchmatch_cuda", ([&] {
	auto shift_map_accessor = shift_map.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>();
	auto cost_map_accessor = cost_map.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
	auto t1_accessor = t1.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
	auto t2_accessor = t2.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
	auto states_accessor = states.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>();

	// Alternative to macro is the cpp index sequence but this requires a templated lambda
	// only possible with recent c++ and gcc versions
	// We keep this version for the time being
        #define patch_match_n(psize) { \
	  initialise_shift_map<scalar_t, psize><<<grid, blocks>>>(t1_accessor, t2_accessor, shift_map_accessor, cost_map_accessor, states_accessor); \
	  for(int i = 0; i < n_iters; i++) { \
	    propagation<scalar_t, psize><<<grid, blocks>>>(t1_accessor, t2_accessor, shift_map_accessor, cost_map_accessor); \
	    random_search<scalar_t, psize><<<grid, blocks>>>(t1_accessor, t2_accessor, shift_map_accessor, cost_map_accessor, states_accessor); \
          } }

	// Dispatch to template
	if (patch_size == 1) patch_match_n(1)
	if (patch_size == 3) patch_match_n(3)
	if (patch_size == 5) patch_match_n(5)
	if (patch_size == 7) patch_match_n(7)
	if (patch_size == 9) patch_match_n(9)
	if (patch_size == 11) patch_match_n(11)
	if (patch_size == 12) patch_match_n(13)
	if (patch_size == 15) patch_match_n(15)
	if (patch_size == 17) patch_match_n(17)
	if (patch_size == 19) patch_match_n(19)
	if (patch_size == 21) patch_match_n(21)
	if (patch_size == 23) patch_match_n(23)

  }));

  return {shift_map, cost_map};
}


std::vector<at::Tensor> backward_cuda(const at::Tensor t1,
				      const at::Tensor t2,
				      const at::Tensor shift_map,
				      const at::Tensor cost_map_grad,
				      int patch_size=3) {
  auto grad_t1 = torch::zeros_like(t1);
  auto grad_t2 = torch::zeros_like(t2);

  // Must make the grid large enough to cover all pixels
  const dim3 blocks(4, 4);
  const dim3 grid(int(t1.size(1) / blocks.x) + 1, int(t1.size(2) / blocks.y) + 1);

  AT_DISPATCH_FLOATING_TYPES(t1.scalar_type(), "backward_cuda", ([&] {
	auto t1_accessor = t1.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
	auto t2_accessor = t2.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
	auto grad_t1_accessor = grad_t1.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
	auto grad_t2_accessor = grad_t2.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
	auto shift_map_accessor = shift_map.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>();
	auto cost_map_accessor = cost_map_grad.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();

        #define backward_n(psize) {backward_kernel<scalar_t, psize><<<grid, blocks>>>(t1_accessor, t2_accessor, grad_t1_accessor, grad_t2_accessor, shift_map_accessor, cost_map_accessor);}

	// Dispatch to template
	if (patch_size == 1) backward_n(1)
	if (patch_size == 3) backward_n(3)
	if (patch_size == 5) backward_n(5)
	if (patch_size == 7) backward_n(7)
	if (patch_size == 9) backward_n(9)
	if (patch_size == 11) backward_n(11)
	if (patch_size == 12) backward_n(13)
	if (patch_size == 15) backward_n(15)
	if (patch_size == 17) backward_n(17)
	if (patch_size == 19) backward_n(19)
	if (patch_size == 21) backward_n(21)
	if (patch_size == 23) backward_n(23)
  }));

  return {grad_t1, grad_t2};
}



template <typename T, int PSZ>
__global__ void initialise_shift_map(at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t1,
				     at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t2,
				     at::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> shift_map,
				     at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> cost_map,
				     at::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> states) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t j = blockDim.y * blockIdx.y + threadIdx.y;

  if (is_in_inner_boundaries<T, 3, PSZ>(t1, i, j)) {
    auto local_state = states[i][j];
    T local_heap[K];
    int32_t local_heap_shift[K*2];

    for(int k=0; k < K; k++) {
      auto ii = randint(PSZ/2, t2.size(1) - PSZ/2, &local_state);
      auto jj = randint(PSZ/2, t2.size(2) - PSZ/2, &local_state);

      while (!is_valid_match<T, PSZ>(t2, ii, jj)) {
	ii = randint(PSZ/2, t2.size(1) - PSZ/2, &local_state);
	jj = randint(PSZ/2, t2.size(2) - PSZ/2, &local_state);
      }

      auto distance = dist<T,PSZ>(t1, t2, i, j, ii, jj);
      add_to_heap(distance, local_heap, local_heap_shift, ii, jj, k+1);
    }

    // Write to memory
    for(int k=0; k < K; k++) {
      shift_map[0][k][i][j] = local_heap_shift[2*k];
      shift_map[1][k][i][j] = local_heap_shift[2*k+1];
      cost_map[k][i][j] = local_heap[k];
    }
    states[i][j] = local_state;
  }
  else if(i < t1.size(1) && j < t1.size(2)) {
    // For pixel at the borders, draw random shifts but do not compute
    // the distances as they are invalid
    auto local_state = states[i][j];
    for(int k=0; k < K; k++) {
      shift_map[0][k][i][j] = randint(PSZ/2, t2.size(1) - PSZ/2, &local_state);
      shift_map[1][k][i][j] = randint(PSZ/2, t2.size(2) - PSZ/2, &local_state);
      cost_map[k][i][j] = 0.0;
    }
    states[i][j] = local_state;
  }
}


/*
  Compute the distance between patches
 */
template <typename T, int PSZ>
__device__ T dist(at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t1,
		  at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t2,
		  int32_t x1, int32_t y1,
		  int32_t x2, int32_t y2,
		  T cutoff) {
  T dist = 0.0;
  for(int c=0; c < t1.size(0); c++) {
    for(int i = -PSZ/2; i < PSZ/2 + 1; i++) {
      for(int j = -PSZ/2; j < PSZ/2 + 1; j++) {
	auto diff = t1[c][x1+i][y1+j] - t2[c][x2+i][y2+j];
	dist += diff * diff;
      }

      // Early return if already worse than current
      if (dist > cutoff) {
	return dist;
      }
    }
  }
  return dist;
}


template <typename T, int PSZ>
__global__ void propagation(at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t1,
			    at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t2,
			    at::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> shift_map,
			    at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> cost_map) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t j = blockDim.y * blockIdx.y + threadIdx.y;

  const int dis[4] = { 0, 1, 0, -1 };
  const int djs[4] = { 1, 0, -1, 0 };

  if(is_in_inner_boundaries<T, 3, PSZ>(t1, i, j)) {
    T local_heap[K];
    int local_shift[2*K];
    
    // Read heap from global memory
    for(int k=0; k < K; k++) {
      local_heap[k] = cost_map[k][i][j];
      local_shift[2*k] = shift_map[0][k][i][j];
      local_shift[2*k+1] = shift_map[1][k][i][j];
    }

    auto worst_distance = local_heap[0];

    for(int step_length=1; step_length <= 8; step_length *= 2) {
      for(int index = 0; index < 4; index++) {
	auto di = dis[index] * step_length;
	auto dj = djs[index] * step_length;

	if (!is_in_inner_boundaries<T, 3, PSZ>(t1, i+di, j+dj)) {
	  continue;
	}

	auto ii = shift_map[0][K-1][i+di][j+dj] - di;
	auto jj = shift_map[1][K-1][i+di][j+dj] - dj;
      
	if(is_valid_match<T, PSZ>(t2, ii, jj) && !in_heap(local_shift, ii, jj)) {
	  auto distance = dist<T, PSZ>(t1, t2, i, j, ii, jj, worst_distance);
	  if(distance < worst_distance) {
	    insert_into_heap(local_heap, local_shift, distance, ii, jj);
	    worst_distance = local_heap[0];
	  }
	}
      }
    }

    // Write to memory
    for(int k=0; k < K; k++) {
      shift_map[0][k][i][j] = local_shift[2*k];
      shift_map[1][k][i][j] = local_shift[2*k+1];
      cost_map[k][i][j] = local_heap[k];
    }
  }
}

template <typename T, int PSZ>
__global__ void random_search(at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t1,
			      at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t2,
			      at::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> shift_map,
			      at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> cost_map,
			      at::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> states) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t j = blockDim.y * blockIdx.y + threadIdx.y;
  
  if(is_in_inner_boundaries<T, 3, PSZ>(t1, i, j)) {
    T local_heap[K];
    int local_shift[2*K];
    
    // Read heap from global memory
    for(int k=0; k < K; k++) {
      local_heap[k] = cost_map[k][i][j];
      local_shift[2*k] = shift_map[0][k][i][j];
      local_shift[2*k+1] = shift_map[1][k][i][j];
    }

    auto local_state = states[i][j];

    // Sample around current best
    auto best_ii = local_shift[2*(K-1)];
    auto best_jj = local_shift[2*(K-1)+1];

    // Worst match
    auto worst_distance = local_heap[0];

    const auto alpha = 0.5;
    auto wmax = max(t2.size(1), t2.size(2));
    int zmax = - logf(wmax) / logf(alpha);

    // Sample around the current match with a uniform window
    for(int z=0; z < zmax; z++) {
      int w = wmax * powf(alpha, z);
      int ii = randint(best_ii - w, best_ii + w, &local_state);
      int jj = randint(best_jj - w, best_jj + w, &local_state);
      
      if(is_valid_match<T,PSZ>(t2, ii, jj) && !in_heap(local_shift, ii, jj)) {
	auto distance = dist<T,PSZ>(t1, t2, i, j, ii, jj, worst_distance);
	if (distance < worst_distance) {
	  insert_into_heap(local_heap, local_shift, distance, ii, jj);
	  worst_distance = local_heap[0];
	}
      }
    }

    // Write to memory
    for(int k=0; k < K; k++) {
      shift_map[0][k][i][j] = local_shift[2*k];
      shift_map[1][k][i][j] = local_shift[2*k+1];
      cost_map[k][i][j] = local_heap[k];
    }
    states[i][j] = local_state;
  }
}

template <typename T, int PSZ>
__global__ void backward_kernel(at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t1,
			      at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t2,
			      at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> grad_t1,
			      at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> grad_t2,
			      at::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> shift_map,
			      at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> cost_map) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t j = blockDim.y * blockIdx.y + threadIdx.y;

  // Compute the gradients using the patches that contain the given pixel (to avoid race conditions)
  if(is_in_inner_boundaries<T,3,PSZ>(t1, i, j)) {
    for(int k=0; k < K; k++) {
      for(int di=-PSZ/2; di < PSZ/2 + 1; di++) {
	for(int dj=-PSZ/2; dj < PSZ/2 + 1; dj++) {
	  // Shifted positions
	  auto ii = shift_map[0][k][i+di][j+dj];
	  auto jj = shift_map[1][k][i+di][j+dj];

	  for(int c=0; c < t1.size(0); c++) {
	    grad_t1[c][i][j] += 2 * (t1[c][i][j] - t2[c][ii-di][jj-dj]) * cost_map[k][i+di][j+dj];

	    // Race condition for grad_t2
	    grad_t2[c][ii-di][jj-dj] += 2 * (t2[c][ii-di][jj-dj] - t1[c][i][j]) * cost_map[k][i+di][j+dj];
	  }
	}
      }
    }
  }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("patchmatch", &patchmatch_cuda, "PatchMatch implementation",
	pybind11::arg("t1"),
	pybind11::arg("t2"),
	pybind11::arg("patch_size") = 3,
	pybind11::arg("n_iters") = 10);
  m.def("backward", &backward_cuda, "Backward implementation",
	pybind11::arg("a"),
	pybind11::arg("b"),
	pybind11::arg("shift_map"),
	pybind11::arg("cost_map_grad"),
	pybind11::arg("patch_size") = 3);
}
