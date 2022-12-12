#pragma once

#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define K 15
#define PATCH_SIZE 7
#define H_PATCH_SIZE 3


/* Random integer generation on gpu [a, b] */
inline __device__ int randint(int a, int b, int64_t *seed) {
  *seed = (*seed ^ 61) ^ (*seed >> 16);
  *seed *= 9;
  *seed = *seed ^ (*seed >> 4);
  *seed *= 668265261;
  *seed = *seed ^ (*seed >> 15);
  return (*seed) % (b - a) + a;
}


template <typename T>
inline __device__ void add_to_heap(T value, T* heap, int* shifts, int ii, int jj, int size) {
  auto idx = size - 1;
  auto parent_idx = idx-1;

  while(idx > 0 && value > heap[parent_idx]) {
    // Swap
    heap[idx] = heap[parent_idx];
    shifts[2*idx] = shifts[2*parent_idx];
    shifts[2*idx+1] = shifts[2*parent_idx+1];
    
    // Go up
    parent_idx = parent_idx-1;
    idx = idx-1;
  }

  heap[idx] = value;
  shifts[2*idx] = ii;
  shifts[2*idx+1] = jj;
}


template <typename T>
inline __device__ void insert_into_heap(T* heap, int* shifts, T value, int ii, int jj, int size=K) {
  auto idx = 0;
  auto child = idx;
  auto has_changed = 1;
  
  while(has_changed) {
    has_changed = 0;
    auto left = idx+1;

    if (left < size && value < heap[left]) {
        child = left;
    }

    if(child != idx) {
      heap[idx] = heap[child];
      shifts[2*idx] = shifts[2*child];
      shifts[2*idx+1] = shifts[2*child+1];
      idx = child;
      has_changed = 1;
    }
  }

  // Insert new node at its right place
  heap[idx] = value;
  shifts[2*idx] = ii;
  shifts[2*idx+1] = jj;
}

inline __device__ bool in_heap(int* shifts, int ii, int jj) {
  for(int k=0; k < K; k++) {
    if(shifts[2*k] == ii && shifts[2*k+1] == jj) {
      return true;
    }
  }
  return false;
}



template <typename T, size_t N>
inline __device__ bool is_in_inner_boundaries(at::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits> t, int x, int y) {
  // TODO: Add assert for N > 2
  return (H_PATCH_SIZE <= x) && (x < t.size(1) - H_PATCH_SIZE) && (H_PATCH_SIZE <= y) && (y < t.size(2) - H_PATCH_SIZE);
}


template <typename T>
inline __device__ bool is_valid_match(at::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> t, int x, int y) {
  return is_in_inner_boundaries(t, x, y);
}



template <typename T, size_t N>
inline __device__ bool is_in_boundaries(at::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits> t, int x, int y) {
  // TODO: Add assert for N > 2
  return (0 <= x) && (x < t.size(1)) && (0 <= y) && (y < t.size(2));
}

inline __device__ bool is_masked(at::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits> mask, int x, int y) {
  return mask[x][y];
}

template <typename T>
inline __device__ bool is_valid_match(at::PackedTensorAccessor32<T, 2, torch::RestrictPtrTraits> t, int x, int y) {
  return is_in_inner_boundaries(t, x, y) && !is_masked(t, x, y);
}

