#pragma once

#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define K 3

/* Random integer generation on gpu [a, b] */
inline __device__ int randint(int a, int b, int64_t *seed) {
  *seed = (*seed ^ 61) ^ (*seed >> 16);
  *seed *= 9;
  *seed = *seed ^ (*seed >> 4);
  *seed *= 668265261;
  *seed = *seed ^ (*seed >> 15);
  return (*seed) % (b - a) + a;
}


template<typename T>
inline __device__ void swap(T* array, int idx1, int idx2) {
  auto temp = array[idx2];
  array[idx2] = array[idx1];
  array[idx1] = temp;
}

/*Implementation of a min-max heap for the best and worst neighbors*/
inline __device__ int level(int index) {
  // TODO: replace by __clz
  return static_cast<int>(logf(index + 1) / logf(2));
}

template <typename T>
inline __device__ void push_down(T* heap, int* shifts, int idx) {
  while (idx < K/2) {
    auto lvl = level(idx);
    // Min level
    if (lvl % 2 == 0) {
      // Check the children to see if we need to swap
      auto left = idx * 2 + 1;
      auto right = (idx+1) * 2;

      auto smallest = (heap[left] < heap[right]) ? left : right;

      // Assuming K = 2^N-1
      if (idx < K/4) {
	// Check grand children as well
	auto grand_left_left = left * 2 + 1;
	auto grand_left_right = (left+1) * 2;
	auto grand_right_left = right * 2 + 1;
	auto grand_right_right = (right+1) * 2;

	auto grand_left_min = (heap[grand_left_left] < heap[grand_left_right]) ? grand_left_left : grand_left_right;
	auto grand_right_min = (heap[grand_right_left] < heap[grand_right_right]) ? grand_right_left : grand_right_right;
	auto grand_smallest = (heap[grand_left_min] < heap[grand_right_min]) ? grand_left_min : grand_right_min;
	smallest = (heap[smallest] < heap[grand_smallest]) ? smallest : grand_smallest;
      }


      // Swap with child
      if (heap[smallest] < heap[idx]) {
	swap(heap, smallest, idx);
	swap(shifts, 2*smallest, 2*idx);
	swap(shifts, 2*smallest+1, 2*idx+1);

	// Grand child is the smallest but i is also larger than parent
	if(smallest > right) {
	  auto parent_smallest = static_cast<int>((smallest-1) / 2);
	  if (heap[smallest] > heap[parent_smallest]) {
	    swap(heap, smallest, parent_smallest);
	    swap(shifts, 2*smallest, 2*parent_smallest);
	    swap(shifts, 2*smallest+1, 2*parent_smallest+1);
	  }
	}

	// Change working index
	idx = smallest;
      }

      // Stop propagation
      else {
	break;
      }
    }

    // Max level
    else {
      // Check the children to see if we need to swap
      auto left = idx * 2 + 1;
      auto right = (idx+1) * 2;

      auto largest = (heap[left] > heap[right]) ? left : right;

      // Assuming K = 2^N-1
      if (idx < K/4) {
	// Check grand children as well
	auto grand_left_left = left * 2 + 1;
	auto grand_left_right = (left + 1) * 2;
	auto grand_right_left = right * 2 + 1;
	auto grand_right_right = (right + 1) * 2;

	auto grand_left_min = (heap[grand_left_left] > heap[grand_left_right]) ? grand_left_left : grand_left_right;
	auto grand_right_min = (heap[grand_right_left] > heap[grand_right_right]) ? grand_right_left : grand_right_right;
	auto grand_largest = (heap[grand_left_min] > heap[grand_right_min]) ? grand_left_min : grand_right_min;
	largest = (heap[largest] > heap[grand_largest]) ? largest : grand_largest;
      }


      // Swap with child
      if (heap[largest] > heap[idx]) {
	swap(heap, largest, idx);
	swap(shifts, 2*largest, 2*idx);
	swap(shifts, 2*largest+1, 2*idx+1);

	// Grand child is the largest but i is also smaller than parent
	if(largest > (idx+1) * 2) {
	  auto parent_largest = static_cast<int>((largest-1) / 2);
	  if (heap[largest] < heap[parent_largest]) {
	    swap(heap, largest, parent_largest);
	    swap(shifts, 2*largest, 2*parent_largest);
	    swap(shifts, 2*largest+1, 2*parent_largest+1);
	  }
	}

	// Change working index
	idx = largest;
      }

      // Stop propagation
      else {
	break;
      }
    }
  }
}


template <typename T>
inline __device__ void make_heap(T* heap, int* shifts) {
  for (int i = static_cast<int>(K/2) - 1; i >= 0; i--) {
    push_down(heap, shifts, i);
  }
}

template<typename T>
inline __device__ void insert_new_max(T* heap, int* shifts, T value, int ii, int jj) {
  auto largest = (heap[1] > heap[2]) ? 1 : 2;

  // If smaller than root put root in the largest node
  if(value < heap[0]) {
    heap[largest] = heap[0];
    shifts[2*largest] = shifts[2*0];
    shifts[2*largest + 1] = shifts[2*0+1];

    heap[0] = value;
    shifts[2*0] = ii;
    shifts[2*0+1] = jj;
  }

  else {
    heap[largest] = value;
    shifts[2*largest] = ii;
    shifts[2*largest + 1] = jj;
  }

  push_down(heap, shifts, largest);
}


inline __device__ bool in_heap(int* shifts, int ii, int jj) {
  for(int k=0; k < K; k++) {
    if(shifts[2*k] == ii && shifts[2*k+1] == jj) {
      return true;
    }
  }
  return false;
}



template <typename T, size_t N, int PSZ>
inline __device__ bool is_in_inner_boundaries(at::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits> t, int x, int y) {
  // TODO: Add assert for N > 2
  return (PSZ/2 <= x) && (x < t.size(1) - PSZ/2) && (PSZ/2 <= y) && (y < t.size(2) - PSZ/2);
}

template <typename T, int PSZ>
inline __device__ bool is_valid_match(at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t, int x, int y) {
  return is_in_inner_boundaries<T, 3, PSZ>(t, x, y);
}

template <typename T, size_t N>
inline __device__ bool is_in_boundaries(at::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits> t, int x, int y) {
  // TODO: Add assert for N > 2
  return (0 <= x) && (x < t.size(1)) && (0 <= y) && (y < t.size(2));
}

inline __device__ bool is_masked(at::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits> mask, int x, int y) {
  return mask[x][y];
}

template <typename T, int PSZ>
inline __device__ bool is_valid_match(at::PackedTensorAccessor32<T, 2, torch::RestrictPtrTraits> t, int x, int y) {
  return is_in_inner_boundaries<T, 2, PSZ>(t, x, y) && !is_masked(t, x, y);
}

