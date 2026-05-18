#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/irange.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <vector>

#ifndef USE_ROCM
#include <dlfcn.h>
// #define WARP_SIZE 64
#include "pytorch_extension_utils.h"
#else
#include "pytorch_extension_utils_rocm.h"
#include "utils.h"  // WARP_SIZE
#endif

// constexpr int WARP_SIZE = 32;
constexpr int32_t TOKEN_HIT = 0xFFFFFFFF;
constexpr int32_t HASH_EMPTY = -1;

#ifdef USE_ROCM
namespace {

void* get_rocm_kernel_accessible_ptr(const at::Tensor& tensor) {
  if (!tensor.defined()) {
    return nullptr;
  }

  void* ptr = tensor.data_ptr();
  if (tensor.is_cuda() || ptr == nullptr) {
    return ptr;
  }

  void* device_ptr = nullptr;
  cudaError_t err = cudaHostGetDevicePointer(&device_ptr, ptr, 0);
  TORCH_CHECK(
      err == cudaSuccess,
      "cudaHostGetDevicePointer failed for ROCm KV cache transfer host tensor: ",
      cudaGetErrorString(err));
  return device_ptr;
}

}  // namespace
#endif

__device__ __forceinline__ void
transfer_item_warp(int32_t lane_id, const void* src_addr, void* dst_addr, int64_t item_size_bytes) {
  const uint64_t* __restrict__ src = static_cast<const uint64_t*>(src_addr);
  uint64_t* __restrict__ dst = static_cast<uint64_t*>(dst_addr);
  const int total_chunks = item_size_bytes / sizeof(uint64_t);

#pragma unroll
  for (int j = lane_id; j < total_chunks; j += WARP_SIZE) {
#ifndef USE_ROCM
    uint64_t tmp;
    asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src + j) : "memory");
    asm volatile("st.global.cg.b64 [%0],%1;" ::"l"(dst + j), "l"(tmp) : "memory");

#else
    uint64_t tmp = __builtin_nontemporal_load(src + j);
    __builtin_nontemporal_store(tmp, dst + j);
#endif
  }
}

__device__ __forceinline__ void
transfer_item_warp_dcu(int32_t lane_id, const void* src_addr, void* dst_addr, int64_t item_size_bytes,int64_t total_threads) {
  const uint64_t* __restrict__ src = static_cast<const uint64_t*>(src_addr);
  uint64_t* __restrict__ dst = static_cast<uint64_t*>(dst_addr);
  const int total_chunks = item_size_bytes / sizeof(uint64_t);

#pragma unroll
  for (int j = lane_id; j < total_chunks; j += total_threads) {
#ifndef USE_ROCM
    uint64_t tmp;
    asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src + j) : "memory");
    asm volatile("st.global.cg.b64 [%0],%1;" ::"l"(dst + j), "l"(tmp) : "memory");

#else
    uint64_t tmp = __builtin_nontemporal_load(src + j);
    __builtin_nontemporal_store(tmp, dst + j);
#endif
  }
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_lf_tbl_dcu(
    T* /*unused*/,
    const uintptr_t* __restrict__ layer_base_tbl,
    int64_t layer_id,
    int64_t layer_dim,
    int64_t page_id,
    int64_t item_size_bytes) {
  // layer first
  return reinterpret_cast<T*>(layer_base_tbl[layer_id]) + page_id * item_size_bytes ;
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_pf_dcu(
    T* base,
    const uintptr_t* __restrict__ /*unused*/,
    int64_t layer_id,
    int64_t page_dim,
    int64_t page_id,
    int64_t item_size_bytes) {
  // layer first
  return base + page_id * page_dim + layer_id * item_size_bytes ;
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_lf_dcu(
    T* base,
    const uintptr_t* __restrict__ /*unused*/,
    int64_t layer_id,
    int64_t layer_dim,
    int64_t page_id,
    int64_t item_size_bytes) {
  // layer first
  return base + layer_id * layer_dim + page_id * item_size_bytes;
}


template <typename T>
__device__ __forceinline__ T* get_global_offset_lf(
    T* base,
    const uintptr_t* __restrict__ /*unused*/,
    int64_t layer_id,
    int64_t layer_dim,
    int64_t page_id,
    int64_t item_size_bytes) {
  // layer first
  return base + layer_id * layer_dim + page_id * item_size_bytes;
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_pf(
    T* base,
    const uintptr_t* __restrict__ /*unused*/,
    int64_t layer_id,
    int64_t page_dim,
    int64_t page_id,
    int64_t item_size_bytes) {
  // page first
  return base + page_id * page_dim + layer_id * item_size_bytes;
}

// get offset from layer base table when layers are not contiguous
template <typename T>
__device__ __forceinline__ T* get_global_offset_lf_tbl(
    T* /*unused*/,
    const uintptr_t* __restrict__ layer_base_tbl,
    int64_t layer_id,
    int64_t /*unused*/,
    int64_t page_id,
    int64_t item_size_bytes) {
  return reinterpret_cast<T*>(layer_base_tbl[layer_id]) + page_id * item_size_bytes;
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_per_head_lf(
    T* base,
    const uintptr_t* __restrict__ /*unused*/,
    int64_t layer_id,
    int64_t layer_dim,
    int64_t page_id,
    int64_t item_size_bytes,
    int64_t head_id,
    int64_t head_num,
    int64_t /*unused*/) {
  // layer first offset func per head
  return base + layer_id * layer_dim + page_id * item_size_bytes + item_size_bytes / head_num * head_id;
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_per_head_lf_tbl(
    T* /*unused*/,
    const uintptr_t* __restrict__ layer_base_tbl,
    int64_t layer_id,
    int64_t /*unused*/,
    int64_t page_id,
    int64_t item_size_bytes,
    int64_t head_id,
    int64_t head_num,
    int64_t /*unused*/) {
  return reinterpret_cast<T*>(layer_base_tbl[layer_id]) + page_id * item_size_bytes +
         item_size_bytes / head_num * head_id;
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_ph(
    T* base,
    const uintptr_t* __restrict__ /*unused*/,
    int64_t layer_id,
    int64_t page_dim,
    int64_t page_id,
    int64_t item_size_bytes,
    int64_t head_id,
    int64_t head_num,
    int64_t page_size) {
  // page head layout: [page_num, head_num, page_size, layer_num, head_dim]
  return base + page_id / page_size * page_size * page_dim +  // page_num dimension offset
         page_dim / head_num * head_id * page_size +          // head_num dimension offset
         page_id % page_size * page_dim / head_num +          // page_size dimension offset
         layer_id * item_size_bytes / head_num;               // layer_num dimension offset
}

//从deivce端(page_num, self.head_num, self.page_size, self.head_dim)中读取K
template <typename T>
__device__ __forceinline__ T* get_global_offset_per_head_dcu_k(
    T* base,
    const uintptr_t* __restrict__ layer_base_tbl,
    int64_t layer_id,
    int64_t page_dim,
    int64_t page_id,
    int64_t item_size_bytes,
    int64_t head_id,
    int64_t head_num,
    int64_t page_size) {
      //page_num, self.head_num, self.page_size, self.head_dim

  return  base + 
          page_id / page_size  * page_dim + 
          head_id * page_dim / head_num + 
          page_id % page_size * page_dim / head_num * page_size;
}

//从deivce端page_num, self.head_num, self.v_head_dim, self.page_size中读取v
template <typename T>
__device__ __forceinline__ T* get_global_offset_per_head_dcu_v(
    T* base,
    const uintptr_t* __restrict__ layer_base_tbl,
    int64_t layer_id,
    int64_t page_dim,
    int64_t page_id,
    int64_t item_size_bytes,
    int64_t head_id,
    int64_t head_dim_id,
    int64_t head_num,
    int64_t page_size) {
      //page_num, self.head_num, self.v_head_dim, self.page_size

  return  base + 
          page_id / page_size  * page_dim + 
          head_id * page_dim / head_num + 
          head_dim_id * page_dim / head_num * page_size +
          page_id % page_size;
}

//获取host端pf布局中k地址(head级)
template <typename T>
__device__ __forceinline__ T* get_global_offset_pf_dcu_k(
    T* base,
    const uintptr_t* __restrict__ layer_base_tbl,
    int64_t layer_id,
    int64_t layer_dim,
    int64_t page_id,
    int64_t item_size_bytes,
    int64_t head_id,
    int64_t head_num,
    int64_t page_size) {
  // page first  self.size, self.layer_num, self.head_num, self.head_dim
  return base + page_id * layer_dim + layer_id * item_size_bytes + item_size_bytes / head_num * head_id;
} 

//获取host端pf布局中v地址(head_dim级)
template <typename T>
__device__ __forceinline__ T* get_global_offset_pf_dcu_v(
    T* base,
    const uintptr_t* __restrict__ layer_base_tbl,
    int64_t layer_id,
    int64_t layer_dim,
    int64_t page_id,
    int64_t item_size_bytes,
    int64_t head_id,
    int64_t head_dim_id,
    int64_t head_num,
    int64_t page_size) {
  return  base + page_id * layer_dim + layer_id * item_size_bytes + item_size_bytes / head_num * head_id + head_dim_id;
    
}

template <auto SrcOffsetFn, auto DstOffsetFn>
__global__ void transfer_page_head_kernel_impl(
    const void* __restrict__ src_k,
    void* __restrict__ dst_k,
    const void* __restrict__ src_v,
    void* __restrict__ dst_v,
    const int64_t* __restrict__ src_indices,
    const int64_t* __restrict__ dst_indices,
    int64_t start_layer_id,
    int64_t num_layers_to_process,
    int64_t num_items,
    int64_t items_per_warp,
    int64_t item_size_bytes,
    int64_t src_layout_dim,
    int64_t dst_layout_dim,
    const uintptr_t* __restrict__ src_k_layer_tbl,
    const uintptr_t* __restrict__ dst_k_layer_tbl,
    const uintptr_t* __restrict__ src_v_layer_tbl,
    const uintptr_t* __restrict__ dst_v_layer_tbl,
    const int64_t page_size,
    const int64_t head_num) {
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t lane_id = tid % WARP_SIZE;
  int32_t warp_id = tid / WARP_SIZE;
  const int64_t head_size_bytes = item_size_bytes / head_num;

  for (int i = 0; i < items_per_warp; ++i) {
    int64_t item_id = warp_id * items_per_warp + i;
    if (item_id >= num_items) {
      break;
    }
    const int64_t src_page_id = src_indices[item_id];
    const int64_t dst_page_id = dst_indices[item_id];

    // Loop over layers if necessary
    for (int64_t layer_id = start_layer_id; layer_id < start_layer_id + num_layers_to_process; ++layer_id) {
      // For page head layout, the cache of each head in the token is discontinuous, need to loop
      for (int64_t head_id = 0; head_id < head_num; ++head_id) {
        const char* src_k_ptr = SrcOffsetFn(
            static_cast<const char*>(src_k),
            src_k_layer_tbl,
            layer_id,
            src_layout_dim,
            src_page_id,
            item_size_bytes,
            head_id,
            head_num,
            page_size);
        char* dst_k_ptr = DstOffsetFn(
            static_cast<char*>(dst_k),
            dst_k_layer_tbl,
            layer_id,
            dst_layout_dim,
            dst_page_id,
            item_size_bytes,
            head_id,
            head_num,
            page_size);
        transfer_item_warp(lane_id, src_k_ptr, dst_k_ptr, head_size_bytes);

        const char* src_v_ptr = SrcOffsetFn(
            static_cast<const char*>(src_v),
            src_v_layer_tbl,
            layer_id,
            src_layout_dim,
            src_page_id,
            item_size_bytes,
            head_id,
            head_num,
            page_size);
        char* dst_v_ptr = DstOffsetFn(
            static_cast<char*>(dst_v),
            dst_v_layer_tbl,
            layer_id,
            dst_layout_dim,
            dst_page_id,
            item_size_bytes,
            head_id,
            head_num,
            page_size);
        transfer_item_warp(lane_id, src_v_ptr, dst_v_ptr, head_size_bytes);
      }
    }
  }
}

template <auto SrcOffsetFn, auto DstOffsetFn, bool IsMLA>
__global__ void transfer_kernel_impl(
    const void* __restrict__ src_k,
    void* __restrict__ dst_k,
    const void* __restrict__ src_v,
    void* __restrict__ dst_v,
    const int64_t* __restrict__ src_indices,
    const int64_t* __restrict__ dst_indices,
    int64_t start_layer_id,
    int64_t num_layers_to_process,
    int64_t num_items,
    int64_t items_per_warp,
    int64_t item_size_bytes,
    int64_t src_layout_dim,
    int64_t dst_layout_dim,
    const uintptr_t* __restrict__ src_k_layer_tbl,
    const uintptr_t* __restrict__ dst_k_layer_tbl,
    const uintptr_t* __restrict__ src_v_layer_tbl,
    const uintptr_t* __restrict__ dst_v_layer_tbl) {
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t lane_id = tid % WARP_SIZE;
  int32_t warp_id = tid / WARP_SIZE;

  
  for (int i = 0; i < items_per_warp; ++i) {
    int64_t item_id = warp_id * items_per_warp + i;
    if (item_id >= num_items) {
      break;
    }
    const int64_t src_page_id = src_indices[item_id];
    const int64_t dst_page_id = dst_indices[item_id];

    
    // Loop over layers if necessary
    for (int64_t layer_id = start_layer_id; layer_id < start_layer_id + num_layers_to_process; ++layer_id) {
      const char* src_ptr = SrcOffsetFn(
          static_cast<const char*>(src_k), src_k_layer_tbl, layer_id, src_layout_dim, src_page_id, item_size_bytes);
      char* dst_ptr = DstOffsetFn(
          static_cast<char*>(dst_k), dst_k_layer_tbl, layer_id, dst_layout_dim, dst_page_id, item_size_bytes);
      transfer_item_warp(lane_id, src_ptr, dst_ptr, item_size_bytes);
      if constexpr (!IsMLA) {
        const char* src_v_ptr = SrcOffsetFn(
            static_cast<const char*>(src_v), src_v_layer_tbl, layer_id, src_layout_dim, src_page_id, item_size_bytes);
        char* dst_v_ptr = DstOffsetFn(
            static_cast<char*>(dst_v), dst_v_layer_tbl, layer_id, dst_layout_dim, dst_page_id, item_size_bytes);
        transfer_item_warp(lane_id, src_v_ptr, dst_v_ptr, item_size_bytes);
      }
    }
  }
}

template <auto SrcOffsetFn, auto DstOffsetFn>
__global__ void transfer_kernel_impl_dcu(
    const void* __restrict__ src_k,
    void* __restrict__ dst_k,
    const void* __restrict__ src_v,
    void* __restrict__ dst_v,
    const int64_t* __restrict__ src_indices,
    const int64_t* __restrict__ dst_indices,
    int64_t start_layer_id,
    int64_t num_layers_to_process,
    int64_t item_size_bytes,
    int64_t src_layout_dim,
    int64_t dst_layout_dim,
    const uintptr_t* __restrict__ src_k_layer_tbl,
    const uintptr_t* __restrict__ dst_k_layer_tbl,
    const uintptr_t* __restrict__ src_v_layer_tbl,
    const uintptr_t* __restrict__ dst_v_layer_tbl,
    int64_t page_size ) {

  int32_t page_index_id = blockIdx.x;
  // int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t lane_id = threadIdx.x;
  int32_t total_threads = blockDim.x;

  const int64_t s_page_id = src_indices[page_index_id * page_size] / page_size;
  const int64_t d_page_id = dst_indices[page_index_id * page_size] / page_size;    
  for (int64_t layer_id = start_layer_id; layer_id < start_layer_id + num_layers_to_process; ++layer_id) {
      const char* src_ptr = SrcOffsetFn(
          static_cast<const char*>(src_k), src_k_layer_tbl, layer_id, src_layout_dim, s_page_id, item_size_bytes);
      char* dst_ptr = DstOffsetFn(
          static_cast<char*>(dst_k), dst_k_layer_tbl, layer_id, dst_layout_dim, d_page_id, item_size_bytes);
      // if(page_index_id==0 && lane_id ==0 ){
      //   printf("DEBUG src_ptr:%p,src_ptr:%lu,dst_ptr:%p,dst_ptr:%lu ,layer_id:%d,s_page_id:%d,d_page_id:%d,src_layout_dim:%d,dst_layout_dim:%d,item_size_bytes:%d,total_threads:%d,num_layers_to_process:%d begin\n",(void*)src_ptr,(uintptr_t)src_ptr,(void*)dst_ptr,(uintptr_t)dst_ptr,layer_id,s_page_id,d_page_id,src_layout_dim,dst_layout_dim,item_size_bytes,total_threads,num_layers_to_process);
      // }
      transfer_item_warp_dcu(lane_id, src_ptr, dst_ptr, item_size_bytes,total_threads);
      // if(page_index_id==0 && lane_id ==0 ){
      //   printf("DEBUG src_ptr:%p,src_ptr:%lu,dst_ptr:%p,dst_ptr:%lu ,layer_id:%d,s_page_id:%d,d_page_id:%d,src_layout_dim:%d,dst_layout_dim:%d,item_size_bytes:%d finish\n",(void*)src_ptr,(uintptr_t)src_ptr,(void*)dst_ptr,(uintptr_t)dst_ptr,layer_id,s_page_id,d_page_id,src_layout_dim,dst_layout_dim,item_size_bytes);
      // }
      const char* src_v_ptr = SrcOffsetFn(
          static_cast<const char*>(src_v), src_v_layer_tbl, layer_id, src_layout_dim, s_page_id, item_size_bytes);
      char* dst_v_ptr = DstOffsetFn(
          static_cast<char*>(dst_v), dst_v_layer_tbl, layer_id, dst_layout_dim, d_page_id, item_size_bytes);
      transfer_item_warp_dcu(lane_id, src_v_ptr, dst_v_ptr, item_size_bytes,total_threads);
  }
  
}

template <auto SrcOffsetFn, auto DstOffsetFn, bool IsMLA, bool PageHeadLayout = false>
void transfer_kv_launcher(
    const at::Tensor& src_k,
    at::Tensor& dst_k,
    const at::Tensor& src_v,
    at::Tensor& dst_v,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t start_layer_id,
    int64_t num_layers_to_process,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t dst_layout_dim,
    const at::Tensor& src_k_layers,
    const at::Tensor& dst_k_layers,
    const at::Tensor& src_v_layers,
    const at::Tensor& dst_v_layers,
    int64_t block_quota,
    int64_t num_warps_per_block,
    const int64_t page_size = 16,
    const int64_t head_num = 1) {
  TORCH_CHECK(src_indices.is_cuda(), "Source indices must be a CUDA tensor");
  TORCH_CHECK(dst_indices.is_cuda(), "Destination indices must be a CUDA tensor");
  TORCH_CHECK(src_indices.scalar_type() == at::kLong, "Source indices must be of type long");
  TORCH_CHECK(dst_indices.scalar_type() == at::kLong, "Destination indices must be of type long");
  TORCH_CHECK(src_indices.numel() == dst_indices.numel(), "Source and destination indices must have the same length");
  TORCH_CHECK(item_size % 8 == 0, "Item byte size must be divisible by 8");

  auto div_up = [](int64_t x, int64_t y) { return (x + y - 1) / y; };
  const int64_t num_items = src_indices.numel();
  const int64_t items_per_warp = div_up(num_items, block_quota * num_warps_per_block);
  const int32_t num_blocks = div_up(num_items, items_per_warp * num_warps_per_block);
  dim3 grid_dim(num_blocks, 1, 1);
  const int32_t threads_per_block = num_warps_per_block * WARP_SIZE;

  const void* src_k_ptr = src_k.defined() ? src_k.data_ptr() : nullptr;
  void* dst_k_ptr = dst_k.defined() ? dst_k.data_ptr() : nullptr;
  const void* src_v_ptr = IsMLA || !src_v.defined() ? nullptr : src_v.data_ptr();
  void* dst_v_ptr = IsMLA || !dst_v.defined() ? nullptr : dst_v.data_ptr();
#ifdef USE_ROCM
  src_k_ptr = get_rocm_kernel_accessible_ptr(src_k);
  dst_k_ptr = get_rocm_kernel_accessible_ptr(dst_k);
  if constexpr (!IsMLA) {
    src_v_ptr = get_rocm_kernel_accessible_ptr(src_v);
    dst_v_ptr = get_rocm_kernel_accessible_ptr(dst_v);
  }
#endif
  const uintptr_t* src_k_tbl_ptr = src_k_layers.defined() ? src_k_layers.data_ptr<uintptr_t>() : nullptr;
  const uintptr_t* dst_k_tbl_ptr = dst_k_layers.defined() ? dst_k_layers.data_ptr<uintptr_t>() : nullptr;
  const uintptr_t* src_v_tbl_ptr = IsMLA || !src_v_layers.defined() ? nullptr : src_v_layers.data_ptr<uintptr_t>();
  const uintptr_t* dst_v_tbl_ptr = IsMLA || !dst_v_layers.defined() ? nullptr : dst_v_layers.data_ptr<uintptr_t>();

  cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();
  if constexpr (PageHeadLayout) {
    transfer_page_head_kernel_impl<SrcOffsetFn, DstOffsetFn><<<grid_dim, threads_per_block, 0, torch_current_stream>>>(
        src_k_ptr,
        dst_k_ptr,
        src_v_ptr,
        dst_v_ptr,
        src_indices.data_ptr<int64_t>(),
        dst_indices.data_ptr<int64_t>(),
        start_layer_id,
        num_layers_to_process,
        num_items,
        items_per_warp,
        item_size,
        src_layout_dim,
        dst_layout_dim,
        src_k_tbl_ptr,
        dst_k_tbl_ptr,
        src_v_tbl_ptr,
        dst_v_tbl_ptr,
        page_size,
        head_num);
  } else {

    transfer_kernel_impl<SrcOffsetFn, DstOffsetFn, IsMLA><<<grid_dim, threads_per_block, 0, torch_current_stream>>>(
        src_k_ptr,
        dst_k_ptr,
        src_v_ptr,
        dst_v_ptr,
        src_indices.data_ptr<int64_t>(),
        dst_indices.data_ptr<int64_t>(),
        start_layer_id,
        num_layers_to_process,
        num_items,
        items_per_warp,
        item_size,
        src_layout_dim,
        dst_layout_dim,
        src_k_tbl_ptr,
        dst_k_tbl_ptr,
        src_v_tbl_ptr,
        dst_v_tbl_ptr);

  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template <auto SrcOffsetFn, auto DstOffsetFn>
void transfer_kv_launcher_dcu(
    const at::Tensor& src_k,
    at::Tensor& dst_k,
    const at::Tensor& src_v,
    at::Tensor& dst_v,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t start_layer_id,
    int64_t num_layers_to_process,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t dst_layout_dim,
    const at::Tensor& src_k_layers,
    const at::Tensor& dst_k_layers,
    const at::Tensor& src_v_layers,
    const at::Tensor& dst_v_layers,
    int64_t page_size,
    int64_t num_warps_per_block) {

  TORCH_CHECK(src_indices.is_cuda(), "Source indices must be a CUDA tensor");
  TORCH_CHECK(dst_indices.is_cuda(), "Destination indices must be a CUDA tensor");
  TORCH_CHECK(src_indices.scalar_type() == at::kLong, "Source indices must be of type long");
  TORCH_CHECK(dst_indices.scalar_type() == at::kLong, "Destination indices must be of type long");
  TORCH_CHECK(src_indices.numel() == dst_indices.numel(), "Source and destination indices must have the same length");
  TORCH_CHECK(item_size % 8 == 0, "Item byte size must be divisible by 8");

  auto div_up = [](int64_t x, int64_t y) { return (x + y - 1) / y; };
  const int64_t num_items = src_indices.numel();
  const int64_t num_pages = num_items / page_size;
  dim3 grid_dim(num_pages, 1, 1);
  const int32_t threads_per_block = num_warps_per_block * WARP_SIZE;


  const void* src_k_ptr = src_k.defined() ? src_k.data_ptr() : nullptr;
  void* dst_k_ptr = dst_k.defined() ? dst_k.data_ptr() : nullptr;
  const void* src_v_ptr = !src_v.defined() ? nullptr : src_v.data_ptr();
  void* dst_v_ptr = !dst_v.defined() ? nullptr : dst_v.data_ptr();
#ifdef USE_ROCM
  src_k_ptr = get_rocm_kernel_accessible_ptr(src_k);
  dst_k_ptr = get_rocm_kernel_accessible_ptr(dst_k);
  src_v_ptr = get_rocm_kernel_accessible_ptr(src_v);
  dst_v_ptr = get_rocm_kernel_accessible_ptr(dst_v);
#endif
  const uintptr_t* src_k_tbl_ptr = src_k_layers.defined() ? src_k_layers.data_ptr<uintptr_t>() : nullptr;
  const uintptr_t* dst_k_tbl_ptr = dst_k_layers.defined() ? dst_k_layers.data_ptr<uintptr_t>() : nullptr;
  const uintptr_t* src_v_tbl_ptr = !src_v_layers.defined() ? nullptr : src_v_layers.data_ptr<uintptr_t>();
  const uintptr_t* dst_v_tbl_ptr = !dst_v_layers.defined() ? nullptr : dst_v_layers.data_ptr<uintptr_t>();

  cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();
  // printf("transfer_kv_launcher_dcu!!!\n");   
  transfer_kernel_impl_dcu<SrcOffsetFn, DstOffsetFn><<<grid_dim, threads_per_block, 0, torch_current_stream>>>(
      src_k_ptr,
      dst_k_ptr,
      src_v_ptr,
      dst_v_ptr,
      src_indices.data_ptr<int64_t>(),
      dst_indices.data_ptr<int64_t>(),
      start_layer_id,
      num_layers_to_process,
      item_size,
      src_layout_dim,
      dst_layout_dim,
      src_k_tbl_ptr,
      dst_k_tbl_ptr,
      src_v_tbl_ptr,
      dst_v_tbl_ptr,
      page_size);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void transfer_kv_per_layer(
    const at::Tensor src_k,
    at::Tensor dst_k,
    const at::Tensor src_v,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_lf<const char>, get_global_offset_lf<char>, false>(
      src_k,
      dst_k,
      src_v,
      dst_v,
      src_indices,
      dst_indices,
      0,
      1,
      item_size,
      0,
      0,
      empty,
      empty,
      empty,
      empty,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_per_layer_pf_lf(
    const at::Tensor src_k,
    at::Tensor dst_k,
    const at::Tensor src_v,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t layer_id,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_pf<const char>, get_global_offset_lf<char>, false>(
      src_k,
      dst_k,
      src_v,
      dst_v,
      src_indices,
      dst_indices,
      layer_id,
      1,
      item_size,
      src_layout_dim,
      0,
      empty,
      empty,
      empty,
      empty,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_per_layer_ph_lf(
    const at::Tensor src_k,
    at::Tensor dst_k,
    const at::Tensor src_v,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t layer_id,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t page_size,
    int64_t head_num,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_ph<const char>, get_global_offset_per_head_lf<char>, false, true>(
      src_k,
      dst_k,
      src_v,
      dst_v,
      src_indices,
      dst_indices,
      layer_id,
      1,
      item_size,
      src_layout_dim,
      0,
      empty,
      empty,
      empty,
      empty,
      block_quota,
      num_warps_per_block,
      page_size,
      head_num);
}

void transfer_kv_all_layer(
    const at::Tensor src_k_layers,
    const at::Tensor dst_k_layers,
    const at::Tensor src_v_layers,
    const at::Tensor dst_v_layers,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  TORCH_CHECK(num_layers == src_k_layers.size(0), "Number of layers in source k tensor does not match num_layers");
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_lf_tbl<const char>, get_global_offset_lf_tbl<char>, false>(
      empty,
      empty,
      empty,
      empty,
      src_indices,
      dst_indices,
      0,
      num_layers,
      item_size,
      0,
      0,
      src_k_layers,
      dst_k_layers,
      src_v_layers,
      dst_v_layers,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_all_layer_lf_pf(
    const at::Tensor src_k_layers,
    at::Tensor dst_k,
    const at::Tensor src_v_layers,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t dst_layout_dim,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  TORCH_CHECK(num_layers == src_k_layers.size(0), "Number of layers in source k tensor does not match num_layers");
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_lf_tbl<const char>, get_global_offset_pf<char>, false>(
      empty,
      dst_k,
      empty,
      dst_v,
      src_indices,
      dst_indices,
      0,
      num_layers,
      item_size,
      0,
      dst_layout_dim,
      src_k_layers,
      empty,
      src_v_layers,
      empty,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_all_layer_lf_ph(
    const at::Tensor src_k_layers,
    at::Tensor dst_k,
    const at::Tensor src_v_layers,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t dst_layout_dim,
    int64_t num_layers,
    int64_t page_size,
    int64_t head_num,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  TORCH_CHECK(num_layers == src_k_layers.size(0), "Number of layers in source k tensor does not match num_layers");
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_per_head_lf_tbl<const char>, get_global_offset_ph<char>, false, true>(
      empty,
      dst_k,
      empty,
      dst_v,
      src_indices,
      dst_indices,
      0,
      num_layers,
      item_size,
      0,
      dst_layout_dim,
      src_k_layers,
      empty,
      src_v_layers,
      empty,
      block_quota,
      num_warps_per_block,
      page_size,
      head_num);
}

void transfer_kv_per_layer_mla(
    const at::Tensor src,
    at::Tensor dst,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_lf<const char>, get_global_offset_lf<char>, true>(
      src,
      dst,
      empty,
      empty,
      src_indices,
      dst_indices,
      0,
      1,
      item_size,
      0,
      0,
      empty,
      empty,
      empty,
      empty,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_per_layer_mla_pf_lf(
    const at::Tensor src,
    at::Tensor dst,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t layer_id,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_pf<const char>, get_global_offset_lf<char>, true>(
      src,
      dst,
      empty,
      empty,
      src_indices,
      dst_indices,
      layer_id,
      1,
      item_size,
      src_layout_dim,
      0,
      empty,
      empty,
      empty,
      empty,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_all_layer_mla(
    const at::Tensor src_layers,
    const at::Tensor dst_layers,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  TORCH_CHECK(num_layers == src_layers.size(0), "Number of layers in source tensor does not match num_layers");
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_lf_tbl<const char>, get_global_offset_lf_tbl<char>, true>(
      empty,
      empty,
      empty,
      empty,
      src_indices,
      dst_indices,
      0,
      num_layers,
      item_size,
      0,
      0,
      src_layers,
      dst_layers,
      empty,
      empty,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_all_layer_mla_lf_pf(
    const at::Tensor src_layers,
    at::Tensor dst,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t dst_layout_dim,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  TORCH_CHECK(num_layers == src_layers.size(0), "Number of layers in source tensor does not match num_layers");
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_lf_tbl<const char>, get_global_offset_pf<char>, true>(
      empty,
      dst,
      empty,
      empty,
      src_indices,
      dst_indices,
      0,
      num_layers,
      item_size,
      0,
      dst_layout_dim,
      src_layers,
      empty,
      empty,
      empty,
      block_quota,
      num_warps_per_block);
}

inline void transfer_page_direct(
    const at::Tensor src_buffer,
    at::Tensor dst_buffer,
    int64_t src_page_index,
    int64_t dst_page_index,
    int64_t page_size) {
  dst_buffer.slice(0, dst_page_index, dst_page_index + page_size)
      .copy_(
          src_buffer.slice(0, src_page_index, src_page_index + page_size),
          /* non_blocking= */ true);
}

void transfer_kv_direct(
    const std::vector<at::Tensor>& src_layers,
    std::vector<at::Tensor> dst_layers,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t page_size) {
  TORCH_CHECK(
      src_layers.size() == dst_layers.size(), "Source and destination layers must have the same number of layers");
  TORCH_CHECK(src_indices.numel() == dst_indices.numel(), "Source and destination indices must have the same length");
  TORCH_CHECK(page_size > 0, "Page size must be positive");
  TORCH_CHECK(src_indices.numel() % page_size == 0, "Source indices size must be divisible by page size");

  auto src_indices_cpu = src_indices.cpu();
  auto dst_indices_cpu = dst_indices.cpu();

  const auto num_indices = src_indices_cpu.numel();
  const int64_t num_layers = src_layers.size();
  int64_t* src_indices_ptr = src_indices_cpu.data_ptr<int64_t>();
  int64_t* dst_indices_ptr = dst_indices_cpu.data_ptr<int64_t>();

  int64_t start_index = 0;
  int64_t end_index = 0;

  for (int64_t i = 0; i < num_indices; ++i) {
    if (i < num_indices - 1) {
      auto src_diff = src_indices_ptr[i + 1] - src_indices_ptr[i];
      auto dst_diff = dst_indices_ptr[i + 1] - dst_indices_ptr[i];

      if (src_diff == 1 && dst_diff == 1) {
        continue;
      }
      end_index = i + 1;
    } else {  // last batch
      end_index = num_indices;
    }
    auto src_index = src_indices_ptr[start_index];
    auto dst_index = dst_indices_ptr[start_index];
    auto num_tokens = end_index - start_index;

    for (int64_t j = 0; j < num_layers; ++j) {
      transfer_page_direct(src_layers[j], dst_layers[j], src_index, dst_index, num_tokens);
    }
    start_index = end_index;
  }
}

template <bool IsLf2Pf>
inline void transfer_kv_page_first_direct_impl(
    const std::vector<at::Tensor>& src_ptrs,
    std::vector<at::Tensor> dst_ptrs,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t start_layer_id,
    int64_t page_size) {
  TORCH_CHECK(src_indices.numel() == dst_indices.numel(), "Source and destination indices must have the same length");
  TORCH_CHECK(page_size > 0, "Page size must be positive");
  TORCH_CHECK(src_indices.numel() % page_size == 0, "Source indices size must be divisible by page size");

  auto src_indices_cpu = src_indices.cpu();
  auto dst_indices_cpu = dst_indices.cpu();
  const int64_t num_pages = src_indices_cpu.size(0) / page_size;
  int64_t* src_indices_ptr = src_indices_cpu.data_ptr<int64_t>();
  int64_t* dst_indices_ptr = dst_indices_cpu.data_ptr<int64_t>();
  printf("!!!!!src_ptrs size:%ld,dst_ptrs size:%ld \n",src_ptrs.size(),dst_ptrs.size());
  auto fallback_to_page_copy = [&]() {
    if constexpr (IsLf2Pf) {
      const bool is_mla = dst_ptrs.size() == 1;
      const int64_t num_layers = is_mla ? src_ptrs.size() : src_ptrs.size() / 2;
      for (const auto i : c10::irange(num_pages)) {
        const int64_t s_index = src_indices_ptr[i * page_size];
        const int64_t d_index = dst_indices_ptr[i * page_size] / page_size;
        for (int64_t j = 0; j < num_layers; ++j) {
          transfer_page_direct(
              src_ptrs[j], dst_ptrs[0].select(0, d_index).select(0, start_layer_id + j), s_index, 0, page_size);
          if (!is_mla) {
            transfer_page_direct(
                src_ptrs[j + num_layers],
                dst_ptrs[1].select(0, d_index).select(0, start_layer_id + j),
                s_index,
                0,
                page_size);
          }
        }
      }
      transfer_page_direct(src_ptrs[0], dst_ptrs[0].select(0, dst_indices_ptr[0] / page_size).select(0, 0), src_indices_ptr[0], 0, page_size);
    } else {
      const bool is_mla = src_ptrs.size() == 1;
      const int64_t num_layers = is_mla ? dst_ptrs.size() : dst_ptrs.size() / 2;
      printf("$$$size:%ld,num_layers:%ld,dst_ptrs.size():%ld \n",src_ptrs.size(),num_layers,dst_ptrs.size());
      for (const auto i : c10::irange(num_pages)) {
        const int64_t s_index = src_indices_ptr[i * page_size] / page_size;
        const int64_t d_index = dst_indices_ptr[i * page_size];
        for (int64_t j = 0; j < num_layers; ++j) {
          transfer_page_direct(
              src_ptrs[0].select(0, s_index).select(0, start_layer_id + j), dst_ptrs[j], 0, d_index, page_size);
          if (!is_mla) {
            transfer_page_direct(
                src_ptrs[1].select(0, s_index).select(0, start_layer_id + j),
                dst_ptrs[j + num_layers],
                0,
                d_index,
                page_size);
          }
        }
      }
    }
  };

#if defined(USE_ROCM) || !defined(CUDA_VERSION) || CUDA_VERSION < 12080
  printf("33333333333333333\n");
  fallback_to_page_copy();
  printf("4444444444444444444\n");
  return;

#else
  // Driver capability gate: only use cudaMemcpyBatchAsync on CUDA 12.8+ drivers.
  int driver_version = 0;
  cudaError_t driver_version_err = cudaDriverGetVersion(&driver_version);
  if (driver_version_err != cudaSuccess || driver_version < 12080) {
    fallback_to_page_copy();
    return;
  }

  // Symbol gate: runtime may not expose cudaMemcpyBatchAsync in some environments.
  using CudaMemcpyBatchAsyncFn =
      cudaError_t (*)(void**, void**, size_t*, size_t, cudaMemcpyAttributes*, size_t*, size_t, size_t*, cudaStream_t);
  static CudaMemcpyBatchAsyncFn cuda_memcpy_batch_async = []() {
    void* symbol = dlsym(RTLD_DEFAULT, "cudaMemcpyBatchAsync");
    return reinterpret_cast<CudaMemcpyBatchAsyncFn>(symbol);
  }();
  if (cuda_memcpy_batch_async == nullptr) {
    fallback_to_page_copy();
    return;
  }

  size_t num_copies = 0;
  std::vector<void*> batch_srcs;
  std::vector<void*> batch_dsts;
  std::vector<size_t> batch_sizes;
  std::vector<size_t> attrs_idxs(1, 0);
  cudaMemcpyAttributes attrs{};
  const int device_id = at::cuda::current_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto append_copy = [&](void* src, void* dst, size_t size_bytes) {
    batch_srcs.push_back(src);
    batch_dsts.push_back(dst);
    batch_sizes.push_back(size_bytes);
  };

  if constexpr (IsLf2Pf) {
    const bool is_mla = dst_ptrs.size() == 1;
    const int64_t num_layers = is_mla ? src_ptrs.size() : src_ptrs.size() / 2;

    const int64_t dst_stride0 = dst_ptrs[0].stride(0);
    const int64_t dst_stride1 = dst_ptrs[0].stride(1);
    const int64_t src_stride0 = src_ptrs[0].stride(0);
    const int64_t elem_size = dst_ptrs[0].element_size();
    const int64_t copy_size_bytes = page_size * src_stride0 * elem_size;
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attrs.srcLocHint.type = cudaMemLocationTypeDevice;
    attrs.srcLocHint.id = device_id;
    attrs.dstLocHint.type = cudaMemLocationTypeHost;
    attrs.dstLocHint.id = 0;
    attrs.flags = 0;

    num_copies = static_cast<size_t>(num_pages) * static_cast<size_t>(num_layers) * static_cast<size_t>(is_mla ? 1 : 2);
    batch_srcs.reserve(num_copies);
    batch_dsts.reserve(num_copies);
    batch_sizes.reserve(num_copies);

    for (const auto i : c10::irange(num_pages)) {
      auto s_index = src_indices_ptr[i * page_size];
      auto d_index = dst_indices_ptr[i * page_size] / page_size;

      for (int64_t j = 0; j < num_layers; ++j) {
        const char* src_k_ptr = static_cast<const char*>(src_ptrs[j].data_ptr()) + s_index * src_stride0 * elem_size;
        char* dst_k_ptr = static_cast<char*>(dst_ptrs[0].data_ptr()) + d_index * dst_stride0 * elem_size +
                          (start_layer_id + j) * dst_stride1 * elem_size;
        append_copy(const_cast<char*>(src_k_ptr), dst_k_ptr, copy_size_bytes);

        if (!is_mla) {
          const char* src_v_ptr =
              static_cast<const char*>(src_ptrs[j + num_layers].data_ptr()) + s_index * src_stride0 * elem_size;
          char* dst_v_ptr = static_cast<char*>(dst_ptrs[1].data_ptr()) + d_index * dst_stride0 * elem_size +
                            (start_layer_id + j) * dst_stride1 * elem_size;
          append_copy(const_cast<char*>(src_v_ptr), dst_v_ptr, copy_size_bytes);
        }
      }
    }

  } else {
    const bool is_mla = src_ptrs.size() == 1;
    const int64_t num_layers = is_mla ? dst_ptrs.size() : dst_ptrs.size() / 2;

    const int64_t src_stride0 = src_ptrs[0].stride(0);
    const int64_t src_stride1 = src_ptrs[0].stride(1);
    const int64_t dst_stride0 = dst_ptrs[0].stride(0);
    const int64_t elem_size = src_ptrs[0].element_size();
    const int64_t copy_size_bytes = page_size * dst_stride0 * elem_size;
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attrs.srcLocHint.type = cudaMemLocationTypeHost;
    attrs.srcLocHint.id = 0;
    attrs.dstLocHint.type = cudaMemLocationTypeDevice;
    attrs.dstLocHint.id = device_id;
    attrs.flags = 0;

    num_copies = static_cast<size_t>(num_pages) * static_cast<size_t>(num_layers) * static_cast<size_t>(is_mla ? 1 : 2);
    batch_srcs.reserve(num_copies);
    batch_dsts.reserve(num_copies);
    batch_sizes.reserve(num_copies);

    for (const auto i : c10::irange(num_pages)) {
      auto s_index = src_indices_ptr[i * page_size] / page_size;
      auto d_index = dst_indices_ptr[i * page_size];

      for (int64_t j = 0; j < num_layers; ++j) {
        const char* src_k_ptr = static_cast<const char*>(src_ptrs[0].data_ptr()) + s_index * src_stride0 * elem_size +
                                (start_layer_id + j) * src_stride1 * elem_size;
        char* dst_k_ptr = static_cast<char*>(dst_ptrs[j].data_ptr()) + d_index * dst_stride0 * elem_size;
        append_copy(const_cast<char*>(src_k_ptr), dst_k_ptr, copy_size_bytes);

        if (!is_mla) {
          const char* src_v_ptr = static_cast<const char*>(src_ptrs[1].data_ptr()) + s_index * src_stride0 * elem_size +
                                  (start_layer_id + j) * src_stride1 * elem_size;
          char* dst_v_ptr = static_cast<char*>(dst_ptrs[j + num_layers].data_ptr()) + d_index * dst_stride0 * elem_size;
          append_copy(const_cast<char*>(src_v_ptr), dst_v_ptr, copy_size_bytes);
        }
      }
    }
  }

  TORCH_CHECK(batch_srcs.size() == num_copies, "Batch memcpy count mismatch");
  if (num_copies > 0) {
    size_t fail_idx = std::numeric_limits<size_t>::max();
    cudaError_t err = cuda_memcpy_batch_async(
        batch_dsts.data(),
        batch_srcs.data(),
        batch_sizes.data(),
        num_copies,
        &attrs,
        attrs_idxs.data(),
        1,
        &fail_idx,
        stream);
    if (err == cudaErrorNotSupported || err == cudaErrorCallRequiresNewerDriver) {
      fallback_to_page_copy();
      return;
    }
    if (err != cudaSuccess) {
      TORCH_CHECK(false, "cudaMemcpyBatchAsync failed. failIdx=", fail_idx, " error=", cudaGetErrorString(err));
    }
  }
#endif
}

void transfer_kv_per_layer_direct_pf_lf(
    const std::vector<at::Tensor>& src_ptrs,
    std::vector<at::Tensor> dst_ptrs,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t layer_id,
    int64_t page_size) {
  transfer_kv_page_first_direct_impl<false>(src_ptrs, dst_ptrs, src_indices, dst_indices, layer_id, page_size);
}

void transfer_kv_all_layer_direct_lf_pf(
    const std::vector<at::Tensor>& src_ptrs,
    std::vector<at::Tensor> dst_ptrs,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t page_size) {
  transfer_kv_page_first_direct_impl<true>(src_ptrs, dst_ptrs, src_indices, dst_indices, 0, page_size);
}


// Knuth multiplicative hash for open-addressing table of size hash_size.
__device__ __forceinline__ int hash_slot(int32_t key, int hash_size) {
  return ((uint32_t)key * 2654435761u) % (uint32_t)hash_size;
}
__device__ __forceinline__ void
transfer_item_warp_hisparse(int32_t lane_id, const void* src_addr, void* dst_addr, int64_t item_size_bytes) {
  const uint64_t* __restrict__ src = static_cast<const uint64_t*>(src_addr);
  uint64_t* __restrict__ dst = static_cast<uint64_t*>(dst_addr);
  const int total_chunks = item_size_bytes / sizeof(uint64_t);
    #pragma unroll
    for (int j = lane_id; j < total_chunks; j += WARP_SIZE) {
        dst[j] = src[j];
    }
}
__device__ __forceinline__ int warp_inclusive_scan(int* s_data, int lane_id, int offset, int count, int accumulator) {
  int idx = lane_id + offset;
  int val = (idx < count) ? s_data[idx] : 0;
#pragma unroll
  for (int i = 1; i < 32; i *= 2) {
    int n = __shfl_up( val, i);
    if (lane_id >= i) val += n;
  }
  val += accumulator;
  if (idx < count) {
    s_data[idx] = val;
  }
  accumulator = __shfl( val, 31);
  return accumulator;
}

template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE, bool IsMLA>
__global__ void load_cache_to_device_buffer_kernel(
    const int32_t* __restrict__ top_k_tokens,
    int32_t* __restrict__ device_buffer_tokens,
    const int64_t* __restrict__ host_cache_locs,
    const int32_t* __restrict__ device_buffer_locs,
    const void* __restrict__ host_cache_k,
    const void* __restrict__ host_cache_v,
    void* __restrict__ device_buffer_k,
    void* __restrict__ device_buffer_v,
    int32_t* __restrict__ top_k_device_locs,
    const int64_t* __restrict__ req_pool_indices,
    const int32_t* __restrict__ seq_lens,
    int16_t* __restrict__ lru_slots,
    const int32_t* __restrict__ num_real_reqs,
    int64_t buffer_stride_0,
    int64_t host_stride,
    int64_t lru_slot_stride_0,
    int64_t top_k_tokens_stride,
    int64_t top_k_device_locs_stride,
    int64_t page_size,
    int64_t item_size_bytes) {
  // todo hisparse: support page wise sparsity
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  constexpr int NUM_TOKEN_CHUNKS = (NUM_TOP_K + WARP_SIZE - 1) / WARP_SIZE;
  constexpr int NUM_BUFFER_CHUNKS = (HOT_BUFFER_SIZE + WARP_SIZE - 1) / WARP_SIZE;

  const int bid = blockIdx.x;
  // Early exit for padded blocks (CUDA graph pads batch to a captured size)
  if (bid >= num_real_reqs[0]) return;

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const unsigned int lanes_before = ((unsigned int)1 << lane_id) - 1;

  const int64_t rid = req_pool_indices[bid];
  const int64_t seq_len = seq_lens[bid];

  // Calculate offsets for this request
  const int32_t* req_top_k_tokens = top_k_tokens + bid * top_k_tokens_stride;
  int32_t* req_top_k_device_locs = top_k_device_locs + bid * top_k_device_locs_stride;

  const int64_t buffer_offset = rid * buffer_stride_0;
  int32_t* req_device_buffer_tokens = device_buffer_tokens + buffer_offset;
  const int32_t* req_device_buffer_locs = device_buffer_locs + buffer_offset;
  const int64_t* req_host_cache_locs = host_cache_locs + rid * host_stride;
  int16_t* req_lru_slots = lru_slots + rid * lru_slot_stride_0;

  // Fast path: short sequences have all tokens in the device buffer in order.
  if (seq_len <= HOT_BUFFER_SIZE) {
    const int count = (seq_len < NUM_TOP_K) ? static_cast<int>(seq_len) : NUM_TOP_K;
    for (int i = tid; i < count; i += BLOCK_SIZE) {
      int32_t token_pos = req_top_k_tokens[i];
      if (token_pos >= 0) {
        req_top_k_device_locs[i] = req_device_buffer_locs[token_pos];
      }
    }
    return;
  }

  // Top-k token positions; reused as miss-token scratch in the copy phase
  __shared__ int32_t s_top_k_tokens[NUM_TOP_K];
  // Prefix-sum offsets for hit counting and miss counting
  __shared__ int32_t s_chunk_offset[NUM_BUFFER_CHUNKS + 1];
  // Prefix-sum offsets for evictable counting
  __shared__ int32_t s_evict_chunk_offset[NUM_BUFFER_CHUNKS + 1];
  // Compacted slot ordering: [hits fwd→  ...  ←evictables bwd]
  __shared__ int16_t s_lru_slots_out[HOT_BUFFER_SIZE];
  // Open-addressing hash table: top-k token_id → top-k index
  constexpr int HASH_SIZE = NUM_TOP_K * 2;
  __shared__ int32_t s_hash_keys[HASH_SIZE];
  __shared__ int16_t s_hash_vals[HASH_SIZE];

  __shared__ int32_t s_total_hits;
  __shared__ int32_t s_newest_hit;

  // Initialize shared memory: counters, hash table, prefix-sum offsets.
  if (tid == 0) {
    s_total_hits = 0;
    s_newest_hit = 0;
  }
  for (int i = tid; i < HASH_SIZE; i += BLOCK_SIZE) {
    s_hash_keys[i] = HASH_EMPTY;
  }
  for (int i = tid; i < NUM_BUFFER_CHUNKS + 1; i += BLOCK_SIZE) {
    s_chunk_offset[i] = 0;
    s_evict_chunk_offset[i] = 0;
  }
  __syncthreads();

  const int newest_slot = HOT_BUFFER_SIZE;
  const int32_t newest_token = seq_len - 1;

  // Insert top-k tokens into shared-memory hash table.
  for (int i = tid; i < NUM_TOP_K; i += BLOCK_SIZE) {
    int32_t token_idx = req_top_k_tokens[i];
    if (token_idx == newest_token) {
      // If topk includes the latest token, bind its canonical occurrence to newest_slot (at HOT_BUFFER_SIZE) and mark
      // it as a hit. newest_slot is at the first position of the extra page, excluded from LRU tracking.
      s_top_k_tokens[i] = TOKEN_HIT;
      req_top_k_device_locs[i] = req_device_buffer_locs[newest_slot];
      s_newest_hit = 1;
    } else {
      int slot = hash_slot(token_idx, HASH_SIZE);
      while (true) {
        int32_t old = atomicCAS(&s_hash_keys[slot], HASH_EMPTY, token_idx);
        if (old == HASH_EMPTY || old == token_idx) {
          s_hash_vals[slot] = static_cast<int16_t>(i);
          break;
        }
        slot = (slot + 1) % HASH_SIZE;
      }
      s_top_k_tokens[i] = token_idx;
    }
  }
  __syncthreads();

  constexpr int ITERATIONS_PER_WARP_BUFFER = (NUM_BUFFER_CHUNKS + NUM_WARPS - 1) / NUM_WARPS;
  int total_hit_count = 0;
  int total_evict_count = 0;
  for (int iter = 0; iter < ITERATIONS_PER_WARP_BUFFER; iter++) {
    int chunk_idx = warp_id + iter * NUM_WARPS;
    bool has_valid_chunk = chunk_idx < NUM_BUFFER_CHUNKS;

    const int slot_idx = chunk_idx * WARP_SIZE + lane_id;
    const bool has_valid_slot = has_valid_chunk && (slot_idx < HOT_BUFFER_SIZE);
    const int16_t buf_slot = has_valid_slot ? req_lru_slots[slot_idx] : -1;
    int32_t my_buffer_token = (buf_slot >= 0) ? req_device_buffer_tokens[buf_slot] : -1;
    int my_found_top_k_idx = -1;
    if (my_buffer_token >= 0) {
      int h = hash_slot(my_buffer_token, HASH_SIZE);
      while (true) {
        int32_t k = s_hash_keys[h];
        if (k == my_buffer_token) {
          my_found_top_k_idx = static_cast<int32_t>(s_hash_vals[h]);
          break;
        }
        if (k == HASH_EMPTY) break;
        h = (h + 1) % HASH_SIZE;
      }
    }
    bool is_hit = my_found_top_k_idx >= 0;
    bool is_evictable = has_valid_slot && !is_hit;

    // Record hits
    if (is_hit) {
      s_top_k_tokens[my_found_top_k_idx] = TOKEN_HIT;
      req_top_k_device_locs[my_found_top_k_idx] = req_device_buffer_locs[buf_slot];
    }

    int local_hit_offset = 0;
    int local_evict_offset = 0;
    if (has_valid_chunk) {
      const unsigned int hit_mask =  __ballot(is_hit);
      const unsigned int evict_mask =  __ballot(is_evictable);
      local_hit_offset = __popc(hit_mask & lanes_before);
      local_evict_offset = __popc(evict_mask & lanes_before);
      if (lane_id == 0) {
        s_chunk_offset[chunk_idx + 1] = __popc(hit_mask);
        s_evict_chunk_offset[chunk_idx + 1] = __popc(evict_mask);
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      total_hit_count =
          warp_inclusive_scan(s_chunk_offset, lane_id, chunk_idx + 1, NUM_BUFFER_CHUNKS + 1, total_hit_count);
      total_evict_count =
          warp_inclusive_scan(s_evict_chunk_offset, lane_id, chunk_idx + 1, NUM_BUFFER_CHUNKS + 1, total_evict_count);
      if (tid == 0) {
        s_total_hits = total_hit_count;
      }
    }
    __syncthreads();

    // Hits grow forward from index 0
    if (is_hit) {
      int hit_offset = s_chunk_offset[chunk_idx] + local_hit_offset;
      s_lru_slots_out[hit_offset] = buf_slot;
    }
    // Evictables grow backward from HOT_BUFFER_SIZE - 1
    if (is_evictable) {
      int evict_offset = s_evict_chunk_offset[chunk_idx] + local_evict_offset;
      s_lru_slots_out[HOT_BUFFER_SIZE - 1 - evict_offset] = buf_slot;
    }
  }
  __syncthreads();

  // Write back LRU order: evictables at front (LRU), hits at back (MRU).
  {
    const int total_evictable = HOT_BUFFER_SIZE - s_total_hits;
    for (int i = tid; i < HOT_BUFFER_SIZE; i += BLOCK_SIZE) {
      if (i < total_evictable) {
        // Evictables: source at backward end, dest at LRU front
        req_lru_slots[i] = s_lru_slots_out[HOT_BUFFER_SIZE - 1 - i];
      } else {
        // Hits: source at forward end, dest at MRU back
        req_lru_slots[i] = s_lru_slots_out[i - total_evictable];
      }
    }
  }

  // Reset offsets for the miss counting phase (only NUM_TOKEN_CHUNKS + 1 entries needed).
  for (int i = tid; i < NUM_TOKEN_CHUNKS + 1; i += BLOCK_SIZE) {
    s_chunk_offset[i] = 0;
  }
  __syncthreads();

  // Third pass to identify misses and their evictable slots
  int total_misses = 0;
  constexpr int ITERATIONS_PER_WARP_TOKEN = (NUM_TOKEN_CHUNKS + NUM_WARPS - 1) / NUM_WARPS;
  for (int iter = 0; iter < ITERATIONS_PER_WARP_TOKEN; iter++) {
    int chunk_idx = warp_id + iter * NUM_WARPS;
    bool has_valid_chunk = chunk_idx < NUM_TOKEN_CHUNKS;

    const int chunk_token_start = chunk_idx * WARP_SIZE;
    const int my_token_idx = chunk_token_start + lane_id;
    const bool has_valid_token = has_valid_chunk && (my_token_idx < NUM_TOP_K);

    int32_t my_token = 0;
    bool is_miss = false;
    int local_miss_offset = 0;

    if (has_valid_token) {
      is_miss = s_top_k_tokens[my_token_idx] != TOKEN_HIT;
      if (is_miss) {
        my_token = s_top_k_tokens[my_token_idx];
      }
    }

    if (has_valid_chunk) {
      const unsigned int miss_mask =  __ballot(is_miss);
      local_miss_offset = __popc(miss_mask & lanes_before);
      const int warp_miss_count = __popc(miss_mask);
      if (lane_id == 0) {
        s_chunk_offset[chunk_idx + 1] = warp_miss_count;
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      total_misses = warp_inclusive_scan(s_chunk_offset, lane_id, chunk_idx + 1, NUM_TOKEN_CHUNKS + 1, total_misses);
    }
    __syncthreads();

    if (is_miss) {
      int miss_offset = s_chunk_offset[chunk_idx] + local_miss_offset;
      int16_t evict_slot = s_lru_slots_out[HOT_BUFFER_SIZE - 1 - miss_offset];
      // Reuse s_top_k_tokens as miss scratch: miss_offset < my_token_idx always
      // holds (hits are skipped), so compacted writes never overrun pending reads.
      s_top_k_tokens[miss_offset] = my_token;
      req_top_k_device_locs[my_token_idx] = req_device_buffer_locs[evict_slot];
      req_device_buffer_tokens[evict_slot] = my_token;
    }
  }
  __syncthreads();

  total_misses = NUM_TOP_K - s_total_hits - s_newest_hit;
  // each warp copies one miss directly, can be separated into a new kernel if parallelism is a concern
  for (int miss_idx = warp_id; miss_idx < total_misses; miss_idx += NUM_WARPS) {
    const int32_t miss_token = s_top_k_tokens[miss_idx];
    const int16_t evict_slot = s_lru_slots_out[HOT_BUFFER_SIZE - 1 - miss_idx];

    const int64_t src_loc = req_host_cache_locs[miss_token];
    const int64_t dst_loc = static_cast<int64_t>(req_device_buffer_locs[evict_slot]);

    const auto src_k = static_cast<const char*>(host_cache_k) + src_loc * item_size_bytes;
    auto dst_k = static_cast<char*>(device_buffer_k) + dst_loc * item_size_bytes;
    transfer_item_warp_hisparse(lane_id, src_k, dst_k, item_size_bytes);

    if constexpr (!IsMLA) {
      const auto src_v = static_cast<const char*>(host_cache_v) + src_loc * item_size_bytes;
      auto dst_v = static_cast<char*>(device_buffer_v) + dst_loc * item_size_bytes;
      transfer_item_warp_hisparse(lane_id, src_v, dst_v, item_size_bytes);
    }
  }
}


template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE, bool IsMLA>
void load_cache_to_device_buffer_impl(
    at::Tensor top_k_tokens,
    at::Tensor device_buffer_tokens,
    at::Tensor host_cache_locs,
    at::Tensor device_buffer_locs,
    at::Tensor host_cache_k,
    at::Tensor host_cache_v,
    at::Tensor device_buffer_k,
    at::Tensor device_buffer_v,
    at::Tensor top_k_device_locs,
    at::Tensor req_pool_indices,
    at::Tensor seq_lens,
    at::Tensor lru_slots,
    at::Tensor num_real_reqs,
    int64_t page_size,
    int64_t item_size_bytes) {

    const int64_t bs = top_k_tokens.size(0);
    const int64_t host_stride = host_cache_locs.size(1);
    const int64_t buffer_stride_0 = device_buffer_tokens.stride(0);
    const int64_t lru_slot_stride_0 = lru_slots.stride(0);
    const int64_t top_k_tokens_stride = top_k_tokens.stride(0);
    const int64_t top_k_device_locs_stride = top_k_device_locs.stride(0);


    const int32_t* top_k_tokens_ptr = static_cast<const int32_t*>(top_k_tokens.data_ptr());
    int32_t* device_buffer_tokens_ptr = static_cast<int32_t*>(device_buffer_tokens.data_ptr());
    const int64_t* host_cache_locs_ptr = static_cast<const int64_t*>(host_cache_locs.data_ptr());
    const int32_t* device_buffer_locs_ptr = static_cast<const int32_t*>(device_buffer_locs.data_ptr());
    const void* host_cache_k_ptr = host_cache_k.data_ptr();
    const void* host_cache_v_ptr = (IsMLA || host_cache_v.dim() == 0) ? (const void*)nullptr : host_cache_v.data_ptr();
    void* device_buffer_k_ptr = device_buffer_k.data_ptr();
    void* device_buffer_v_ptr = (IsMLA || device_buffer_v.dim() == 0) ? (void*)nullptr : device_buffer_v.data_ptr();
    int32_t* top_k_device_locs_ptr = static_cast<int32_t*>(top_k_device_locs.data_ptr());
    const int64_t* req_pool_indices_ptr = static_cast<const int64_t*>(req_pool_indices.data_ptr());
    const int32_t* seq_lens_ptr = static_cast<const int32_t*>(seq_lens.data_ptr());
    int16_t* lru_slots_ptr = static_cast<int16_t*>(lru_slots.data_ptr());
    const int32_t* num_real_reqs_ptr = static_cast<const int32_t*>(num_real_reqs.data_ptr());   

    dim3 grid_dim(bs, 1, 1);
    dim3 Block_dim(BLOCK_SIZE, 1, 1);
    cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();
    load_cache_to_device_buffer_kernel<BLOCK_SIZE, NUM_TOP_K,HOT_BUFFER_SIZE,IsMLA><<<grid_dim, Block_dim, 0, torch_current_stream>>>(
        top_k_tokens_ptr,
        device_buffer_tokens_ptr,
        host_cache_locs_ptr, 
        device_buffer_locs_ptr,
        host_cache_k_ptr,
        host_cache_v_ptr,
        device_buffer_k_ptr,
        device_buffer_v_ptr,
        top_k_device_locs_ptr,
        req_pool_indices_ptr,
        seq_lens_ptr,
        lru_slots_ptr,
        num_real_reqs_ptr,
        buffer_stride_0,
        host_stride,
        lru_slot_stride_0,
        top_k_tokens_stride,
        top_k_device_locs_stride,
        page_size,
        item_size_bytes
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

}

void load_cache_to_device_buffer(
    at::Tensor top_k_tokens,
    at::Tensor device_buffer_tokens,
    at::Tensor host_cache_locs,
    at::Tensor device_buffer_locs,
    at::Tensor host_cache_k,
    at::Tensor host_cache_v,
    at::Tensor device_buffer_k,
    at::Tensor device_buffer_v,
    at::Tensor top_k_device_locs,
    at::Tensor req_pool_indices,
    at::Tensor seq_lens,
    at::Tensor lru_slots,
    at::Tensor num_real_reqs,
    int64_t page_size,
    int64_t item_size_bytes
){
    load_cache_to_device_buffer_impl<1024,2048,6144,true>(
      top_k_tokens,
      device_buffer_tokens,
      host_cache_locs,
      device_buffer_locs,
      host_cache_k,
      host_cache_v,
      device_buffer_k,
      device_buffer_v,
      top_k_device_locs,
      req_pool_indices,
      seq_lens,
      lru_slots,
      num_real_reqs,
      page_size,
      item_size_bytes
    );

}

void transfer_kv_all_direct_pf_lf_H2D_dcu( 
    const at::Tensor& src_ptrs_k,
    const at::Tensor& src_ptrs_v,
    std::vector<at::Tensor> dst_ptrs_k,
    std::vector<at::Tensor> dst_ptrs_v,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t start_layer_id,
    int64_t page_size){
    
    TORCH_CHECK(src_indices.numel() == dst_indices.numel(), "Source and destination indices must have the same length");
    TORCH_CHECK(page_size > 0, "Page size must be positive");
    TORCH_CHECK(src_indices.numel() % page_size == 0, "Source indices size must be divisible by page size");
    
    auto src_indices_cpu = src_indices.cpu();
    auto dst_indices_cpu = dst_indices.cpu();
    const int64_t num_pages = src_indices_cpu.size(0) / page_size;
    int64_t* src_indices_ptr = src_indices_cpu.data_ptr<int64_t>();
    int64_t* dst_indices_ptr = dst_indices_cpu.data_ptr<int64_t>();

    for (const auto i : c10::irange(num_pages)) {
      const int64_t s_index = src_indices_ptr[i * page_size] / page_size;
      const int64_t d_index = dst_indices_ptr[i * page_size] / page_size;
      dst_ptrs_k[start_layer_id][d_index].copy_(src_ptrs_k[s_index][start_layer_id],true);
      dst_ptrs_v[start_layer_id][d_index].copy_(src_ptrs_v[s_index][start_layer_id],true);
    }
  }

void transfer_kv_all_kernel_lf_pf_D2H_dcu(
    const at::Tensor& src_k,
    at::Tensor dst_k,
    const at::Tensor& src_v,
    at::Tensor dst_v,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t dst_layout_dim,
    int64_t page_size,
    int64_t layer_num,
    int64_t num_warps_per_block){

    TORCH_CHECK(layer_num == src_k.size(0), "Number of layers in source k tensor does not match num_layers");
    at::Tensor empty;
    transfer_kv_launcher_dcu<get_global_offset_lf_tbl_dcu<const char>, get_global_offset_pf_dcu<char>>(
      empty,
      dst_k,
      empty,
      dst_v,
      src_indices,
      dst_indices,
      0,
      layer_num,
      item_size,
      src_layout_dim,
      dst_layout_dim,
      src_k,
      empty,
      src_v,
      empty,
      page_size,
      num_warps_per_block);
}

void transfer_kv_per_layer_kernel_pf_lf_H2D_dcu(
    const at::Tensor& src_k,
    at::Tensor dst_k,
    const at::Tensor& src_v,
    at::Tensor dst_v,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t page_size,
    int64_t layer_id,
    int64_t num_warps_per_block){
    
    at::Tensor empty;
    transfer_kv_launcher_dcu<get_global_offset_pf_dcu<const char>, get_global_offset_lf_dcu<char>>(
      src_k,
      dst_k,
      src_v,
      dst_v,
      src_indices,
      dst_indices,
      layer_id,
      1,
      item_size,
      src_layout_dim,
      0,
      empty,
      empty,
      empty,
      empty,
      page_size,
      num_warps_per_block);


}

void transfer_kv_all_direct_lf_pf_D2H_dcu(
    const std::vector<at::Tensor>& src_ptrs_k,
    const std::vector<at::Tensor>& src_ptrs_v,
    at::Tensor dst_ptrs_k,
    at::Tensor dst_ptrs_v,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t start_layer_id,
    int64_t page_size) {
 
    TORCH_CHECK(src_indices.numel() == dst_indices.numel(), "Source and destination indices must have the same length");
    TORCH_CHECK(page_size > 0, "Page size must be positive");
    TORCH_CHECK(src_indices.numel() % page_size == 0, "Source indices size must be divisible by page size");

    auto src_indices_cpu = src_indices.cpu();
    auto dst_indices_cpu = dst_indices.cpu();
    const int64_t num_pages = src_indices_cpu.size(0) / page_size;
    const int64_t layer_num = src_ptrs_k.size();
    int64_t* src_indices_ptr = src_indices_cpu.data_ptr<int64_t>();
    int64_t* dst_indices_ptr = dst_indices_cpu.data_ptr<int64_t>();

    for (int64_t j = 0; j < layer_num; ++j) {
      for (const auto i : c10::irange(num_pages)) {
        const int64_t s_index = src_indices_ptr[i * page_size] / page_size;
        const int64_t d_index = dst_indices_ptr[i * page_size] / page_size;
        dst_ptrs_k[d_index][start_layer_id+j].copy_(src_ptrs_k[start_layer_id+j][s_index],true);
        dst_ptrs_v[d_index][start_layer_id+j].copy_(src_ptrs_v[start_layer_id+j][s_index],true);
    }
  }
}


__device__ int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

__device__ int64_t safe_min(int64_t a, int64_t b) {
    return a < b ? a : b;
}

__global__ void launch_alloc_decode_kernel(
    const int64_t* seq_lens_ptr,   
    const int32_t* last_loc_ptr,    
    const int64_t* free_page_ptr,   
    int64_t* out_indices,     
    int64_t bs,            
    int64_t page_size) {

  int64_t pid = blockIdx.x * blockDim.x + threadIdx.x;

  if (pid >= bs) return;
  
  int64_t seq_len = seq_lens_ptr[pid];
  int64_t pre_len = seq_len - 1;
  
  int64_t num_page_start_loc_self = ceil_div(seq_len, page_size) - ceil_div(pre_len, page_size);
  
  int64_t sum_num_new_pages = 0;
  for (int64_t i = 0; i <= pid; i++) {
      int64_t other_seq_len = seq_lens_ptr[i];
      int64_t other_pre_len = (i <= pid) ? (other_seq_len - 1) : other_seq_len;
      
      int64_t other_num_pages_after = ceil_div(other_seq_len, page_size);
      int64_t other_num_pages_before = ceil_div(other_pre_len, page_size);
      int64_t other_num_new_pages = other_num_pages_after - other_num_pages_before;
      
      sum_num_new_pages += other_num_new_pages;
  }
  int64_t new_page_start_loc = sum_num_new_pages - num_page_start_loc_self;

  if (num_page_start_loc_self == 0) {
      int32_t last_loc = last_loc_ptr[pid];
      out_indices[pid] = last_loc + 1;
  } else {
      int64_t page = free_page_ptr[new_page_start_loc];
      out_indices[pid] = page * page_size;
  }
}

__global__ void launch_alloc_extend_kernel(
    const int64_t* pre_lens_ptr,
    const int64_t* seq_lens_ptr,
    const int64_t* last_loc_ptr,
    const int64_t* free_page_ptr,
    int64_t* out_indices,
    int64_t bs,
    int64_t page_size) 
{
    int64_t pid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pid >= bs) return;
    
    int64_t seq_len = seq_lens_ptr[pid];
    int64_t pre_len = pre_lens_ptr[pid];
    int64_t extend_len = seq_len - pre_len;
    
    int64_t sum_extend_lens = 0;
    for (int64_t i = 0; i <= pid; i++) {
      int64_t other_seq_len = seq_lens_ptr[i];
      int64_t other_pre_len = pre_lens_ptr[i];
      int64_t other_extend_len = other_seq_len - other_pre_len;
      sum_extend_lens += other_extend_len;
    }
    
    int64_t output_start_loc = sum_extend_lens - extend_len;
    int64_t num_page_start_loc_self = ceil_div(seq_len, page_size) - ceil_div(pre_len, page_size);
    
    int64_t sum_num_new_pages = 0;
    for (int64_t i = 0; i <= pid; i++) {
      int64_t other_seq_len = seq_lens_ptr[i];
      int64_t other_pre_len = pre_lens_ptr[i];
      
      int64_t other_num_pages_after = ceil_div(other_seq_len, page_size);
      int64_t other_num_pages_before = ceil_div(other_pre_len, page_size);
      int64_t other_num_new_pages = other_num_pages_after - other_num_pages_before;
      
      sum_num_new_pages += other_num_new_pages;
    }
    int64_t new_page_start_loc = sum_num_new_pages - num_page_start_loc_self;
    
    int64_t last_loc = last_loc_ptr[pid];
    int64_t num_part1 = safe_min(seq_len, ceil_div(pre_len, page_size) * page_size) - pre_len;

    for (int64_t offset = 0; offset < num_part1 && offset < page_size; offset++) {
        int64_t output_idx = output_start_loc + offset;
        out_indices[output_idx] = last_loc + 1 + offset;
    }
    
    if (pre_len + num_part1 == seq_len) {
        return;
    }
    
    int64_t num_part2 = (seq_len / page_size) * page_size - ceil_div(pre_len, page_size) * page_size;
    for (int64_t offset = 0; offset < num_part2; offset++) {
      int64_t page_idx = new_page_start_loc + offset / page_size;
      int64_t page_start = free_page_ptr[page_idx];
      int64_t output_idx = output_start_loc + num_part1 + offset;
      out_indices[output_idx] = page_start * page_size + offset % page_size;
    }

    if (pre_len + num_part1 + num_part2 == seq_len) {
        return;
    }
    
    int64_t num_part3 = seq_len - (seq_len / page_size) * page_size;
    int64_t last_page_idx = new_page_start_loc + num_page_start_loc_self - 1;
    int64_t start_loc = free_page_ptr[last_page_idx];

    for (int64_t offset = 0; offset < num_part3 && offset < page_size; offset++) {
      int64_t output_idx = output_start_loc + num_part1 + num_part2 + offset;
      out_indices[output_idx] = start_loc * page_size + offset;
    }
}
__global__ void launch_create_extend_after_decode_spec_info_int32_kernel(
    const int32_t* verified_id_ptr,
    const int64_t* seq_lens_ptr,
    const int32_t* accept_lens_ptr,
    int64_t* positions_ptr,
    int32_t* new_verified_id_ptr,
    int64_t bs) {
    
    int64_t pid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pid >= bs) return;
    
    int64_t seq_length = seq_lens_ptr[pid];
    int32_t accept_length = accept_lens_ptr[pid];

    int32_t accept_len_cumsum = 0;
    for (int32_t offset = 0; offset < pid; offset++) {
        accept_len_cumsum += accept_lens_ptr[offset];
    }

    int64_t* positions_ptr1 = positions_ptr + accept_len_cumsum;
    for (int32_t offset = 0; offset < accept_length ; offset++) 
    {
      positions_ptr1[offset] = seq_length - accept_length + offset;
    }

    int32_t verified_idx = accept_len_cumsum + accept_length - 1;
    new_verified_id_ptr[pid] = verified_id_ptr[verified_idx];
}

__global__ void launch_create_extend_after_decode_spec_info_int64_kernel(
    const int32_t* verified_id_ptr,
    const int64_t* seq_lens_ptr,
    const int64_t* accept_lens_ptr,
    int64_t* positions_ptr,
    int32_t* new_verified_id_ptr,
    int64_t bs) {
    
    int64_t pid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pid >= bs) return;
    
    int64_t seq_length = seq_lens_ptr[pid];
    int64_t accept_length = accept_lens_ptr[pid];

    int64_t accept_len_cumsum = 0;
    for (int64_t offset = 0; offset < pid; offset++) {
        accept_len_cumsum += accept_lens_ptr[offset];
    }

    int64_t* positions_ptr1 = positions_ptr + accept_len_cumsum;
    for (int64_t offset = 0; offset < accept_length && offset < bs; offset++) 
    {
      positions_ptr1[offset] = seq_length - accept_length + offset;
    }

    int64_t verified_idx = accept_len_cumsum + accept_length - 1;
    new_verified_id_ptr[pid] = verified_id_ptr[verified_idx];
}

void dcu_alloc_decode_kernel(
  const at::Tensor seq_lens_ptr,   
  const at::Tensor last_loc_ptr,    
  const at::Tensor free_page_ptr,   
  at::Tensor out_indices, 
  int64_t bs,          
  int64_t page_size) {

    const int64_t* seq_lens_ptr1 = static_cast<const int64_t*>(seq_lens_ptr.data_ptr());
    const int32_t* last_loc_ptr1 = static_cast<const int32_t*>(last_loc_ptr.data_ptr());
    const int64_t* free_page_ptr1 = static_cast<const int64_t*>(free_page_ptr.data_ptr());
    int64_t* out_indices1 = static_cast<int64_t*>(out_indices.data_ptr());

    int64_t block_size = 64;
    int64_t grid_size = (bs + block_size - 1) / block_size;
    cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();
    launch_alloc_decode_kernel<<<grid_size, block_size, 0, torch_current_stream>>>(seq_lens_ptr1, last_loc_ptr1, free_page_ptr1, out_indices1, bs, page_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dcu_create_extend_after_decode_spec_info(
    const at::Tensor verified_id,
    const at::Tensor seq_lens,
    const at::Tensor accept_lens,
    at::Tensor positions,
    at::Tensor new_verified_id,
    int64_t bs) {

    const int32_t* verified_id_ptr;
    const int64_t* seq_lens_ptr;
    const int32_t* accept_lens_ptr_int32;
    const int64_t* accept_lens_ptr_int64;
    int64_t* positions_ptr;
    int32_t* new_verified_id_ptr;

    int64_t block_size = 64;
    int64_t grid_size = (bs + block_size - 1) / block_size;
    cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();

    if (accept_lens.dtype() == torch::kInt32)
    {
      verified_id_ptr       = static_cast<const int32_t*>(verified_id.data_ptr());
      seq_lens_ptr          = static_cast<const int64_t*>(seq_lens.data_ptr());
      accept_lens_ptr_int32 = static_cast<const int32_t*>(accept_lens.data_ptr());
      positions_ptr         = static_cast<int64_t*>(positions.data_ptr());
      new_verified_id_ptr   = static_cast<int32_t*>(new_verified_id.data_ptr());

      launch_create_extend_after_decode_spec_info_int32_kernel<<<grid_size, block_size, 0, torch_current_stream>>>(verified_id_ptr, seq_lens_ptr, accept_lens_ptr_int32, positions_ptr, new_verified_id_ptr, bs);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    else
    {
      verified_id_ptr       = static_cast<const int32_t*>(verified_id.data_ptr());
      seq_lens_ptr          = static_cast<const int64_t*>(seq_lens.data_ptr());
      accept_lens_ptr_int64 = static_cast<const int64_t*>(accept_lens.data_ptr());
      positions_ptr         = static_cast<int64_t*>(positions.data_ptr());
      new_verified_id_ptr   = static_cast<int32_t*>(new_verified_id.data_ptr());

      launch_create_extend_after_decode_spec_info_int64_kernel<<<grid_size, block_size, 0, torch_current_stream>>>(verified_id_ptr, seq_lens_ptr, accept_lens_ptr_int64, positions_ptr, new_verified_id_ptr, bs);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
};

void dcu_alloc_extend_kernel(
    const at::Tensor pre_lens_ptr,
    const at::Tensor seq_lens_ptr,
    const at::Tensor last_loc_ptr,
    const at::Tensor free_page_ptr,
    at::Tensor out_indices,
    int64_t bs,
    int64_t page_size) {

      const int64_t* pre_lens_ptr1 = static_cast<const int64_t*>(pre_lens_ptr.data_ptr());
      const int64_t* seq_lens_ptr1 = static_cast<const int64_t*>(seq_lens_ptr.data_ptr());
      const int64_t* last_loc_ptr1 = static_cast<const int64_t*>(last_loc_ptr.data_ptr());
      const int64_t* free_page_ptr1 = static_cast<const int64_t*>(free_page_ptr.data_ptr());
      int64_t* out_indices1 = static_cast<int64_t*>(out_indices.data_ptr());

      int64_t block_size = 64;
      int64_t grid_size = (bs + block_size - 1) / block_size;
      cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();
      launch_alloc_extend_kernel<<<grid_size, block_size, 0, torch_current_stream>>>(pre_lens_ptr1, seq_lens_ptr1, last_loc_ptr1, free_page_ptr1, out_indices1, bs, page_size);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void launch_assign_req_to_token_pool(
    const int64_t* req_pool_indices_ptr,
    int32_t* req_to_token_ptr,
    const int32_t* allocate_lens_ptr,
    const int32_t* new_allocate_lens,
    int64_t* out_cache_loc_ptr,
    int64_t shape,
    int64_t bs) 
{
   
    int64_t pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= bs) return;

    int32_t kv_start = allocate_lens_ptr[pid];
    int32_t kv_end = new_allocate_lens[pid];
    int64_t pool_idx = req_pool_indices_ptr[pid];  
    int32_t* token_pool = req_to_token_ptr + pool_idx * shape;
    
    int64_t sum_out_offset = 0;
    for(int64_t length_offset = 0; length_offset < pid;length_offset++){
        sum_out_offset +=
            (int64_t)(new_allocate_lens[length_offset] - allocate_lens_ptr[length_offset]);
    }
    int64_t* out_cache_ptr = out_cache_loc_ptr + sum_out_offset;

    int32_t copy_length = kv_end - kv_start; 
    #pragma unroll(32)
    for (int32_t out_cache_index = 0; out_cache_index < copy_length; out_cache_index++) {
        token_pool[kv_start + out_cache_index] =
          static_cast<int32_t>(out_cache_ptr[out_cache_index]);
    }

}


void dcu_assign_req_to_token_pool(
    const at::Tensor req_pool_indices_ptr,
    at::Tensor req_to_token_ptr,
    const at::Tensor allocate_lens_ptr,
    at::Tensor new_allocate_lens,
    at::Tensor out_cache_loc_ptr,
    int64_t shape,
    int64_t bs) {

      const int64_t* req_pool_indices_ptr1 = static_cast<const int64_t*>(req_pool_indices_ptr.data_ptr());
      int32_t* req_to_token_ptr1 = static_cast<int32_t*>(req_to_token_ptr.data_ptr());
      const int32_t* allocate_lens_ptr1 = static_cast<const int32_t*>(allocate_lens_ptr.data_ptr());
      int32_t* new_allocate_lens1 = static_cast<int32_t*>(new_allocate_lens.data_ptr());
      int64_t* out_cache_loc_ptr1 = static_cast<int64_t*>(out_cache_loc_ptr.data_ptr());

      int64_t block_size = 64;
      int64_t grid_size = (bs + block_size - 1) / block_size;
      cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();
      launch_assign_req_to_token_pool<<<grid_size, block_size, 0, torch_current_stream>>>(req_pool_indices_ptr1, req_to_token_ptr1, allocate_lens_ptr1, new_allocate_lens1, out_cache_loc_ptr1, shape, bs);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
}


__global__ void get_last_loc_kernel(
    const int32_t*  req_to_token,
    const int64_t*  req_pool_indices_tensor,
    const int32_t*  prefix_lens_tensor,
    int64_t*  result,
    int64_t num_tokens,
    int32_t req_to_token_stride){

    int64_t pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_tokens) return;

    int32_t pre_len = prefix_lens_tensor[pid];
    if (pre_len > 0) {
        int64_t req_idx = req_pool_indices_tensor[pid];
        int64_t token_idx = req_idx * req_to_token_stride + (pre_len - 1);
        result[pid] = static_cast<int64_t>(req_to_token[token_idx]);
    } else {
        result[pid] = static_cast<int64_t>(-1);
    }
}

at::Tensor dcu_get_last_loc(
    const at::Tensor req_to_token,     
    const at::Tensor req_pool_indices,  
    const at::Tensor prefix_lens) {
      
    // TORCH_CHECK(req_to_token.device().is_cuda(), "req_to_token must be CUDA tensor");
    // TORCH_CHECK(req_pool_indices.device().is_cuda(), "req_pool_indices must be CUDA tensor");
    // TORCH_CHECK(prefix_lens.device().is_cuda(), "prefix_lens must be CUDA tensor");

    // TORCH_CHECK(req_to_token.dim() == 2, "req_to_token must be 2D tensor [batch, seq_len]");
    // TORCH_CHECK(prefix_lens.dim() == 1, "prefix_lens must be 1D");
    // TORCH_CHECK(req_pool_indices.dim() == 1, "req_pool_indices must be 1D");

    int32_t num_tokens = prefix_lens.numel();
    // TORCH_CHECK(req_pool_indices.numel() == num_tokens, "req_pool_indices must have same length as prefix_lens");

    int32_t req_to_token_stride = req_to_token.stride(0);

    const int32_t* req_to_token_ptr = static_cast<const int32_t*>(req_to_token.data_ptr());
    const int64_t* req_pool_indices_ptr = static_cast<const int64_t*>(req_pool_indices.data_ptr());
    const int32_t* prefix_lens_ptr = static_cast<const int32_t*>(prefix_lens.data_ptr());

    // auto req_to_token_c = req_to_token.contiguous();
    // auto req_pool_indices_c = req_pool_indices.contiguous();
    // auto prefix_lens_c   = prefix_lens.contiguous();

    // const int32_t* req_to_token_ptr = req_to_token_c.data_ptr<int32_t>();
    // const int64_t* req_pool_indices_ptr = req_pool_indices_c.data_ptr<int64_t>();
    // const int32_t* prefix_lens_ptr  = prefix_lens_c.data_ptr<int32_t>();


    auto req_pool_indices_c = req_pool_indices.contiguous();
    auto result = at::empty_like(req_pool_indices_c);
    int64_t* result_ptr = static_cast<int64_t*>(result.data_ptr());

    const int64_t block_size = 64;
    const int64_t grid_size = (num_tokens + block_size - 1) / block_size;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    get_last_loc_kernel<<<grid_size, block_size, 0, stream>>>(
        req_to_token_ptr,
        req_pool_indices_ptr,
        prefix_lens_ptr,
        result_ptr,
        num_tokens,
        req_to_token_stride
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return result;
}


__global__ void launch_assign_extend_cache_locs_kernel(
    const int64_t* __restrict__ req_pool_indices,   // [bs]
    const int32_t* __restrict__ req_to_token,       // [max_num_req, pool_len]
    const int64_t* __restrict__ start_offset,       // [bs]
    const int64_t* __restrict__ end_offset,         // [bs]
    int64_t* __restrict__ out_cache_loc,            // [sum(draft_token_num)]
    int64_t pool_len,
    int64_t bs)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= bs) return;

    int64_t kv_start = start_offset[pid];
    int64_t kv_end   = end_offset[pid];
    int64_t req_id   = req_pool_indices[pid];

    int64_t out_offset = 0;
    for (int i = 0; i < pid; ++i) {
        out_offset += end_offset[i] - start_offset[i];
    }

    const int32_t* src = req_to_token + req_id * pool_len + kv_start;
    int64_t*       dst = out_cache_loc + out_offset;
    for (int64_t i = 0; i < kv_end - kv_start; ++i) {
        dst[i] = src[i];
    }
}

void dcu_assign_extend_cache_locs(
    const at::Tensor req_pool_indices,
    const at::Tensor req_to_token,
    const at::Tensor start_offset,
    const at::Tensor end_offset,
    at::Tensor out_cache_loc,
    int64_t pool_len,
    int64_t bs)
{
    const int64_t* req_pool_indices_ptr = req_pool_indices.data_ptr<int64_t>();
    const int32_t* req_to_token_ptr     = req_to_token.data_ptr<int32_t>();
    const int64_t* start_offset_ptr     = start_offset.data_ptr<int64_t>();
    const int64_t* end_offset_ptr       = end_offset.data_ptr<int64_t>();
    int64_t* out_cache_loc_ptr          = out_cache_loc.data_ptr<int64_t>();

    constexpr int64_t threads = 128;
    int64_t blocks = (bs + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_assign_extend_cache_locs_kernel<<<blocks, threads, 0, stream>>>(
        req_pool_indices_ptr,
        req_to_token_ptr,
        start_offset_ptr,
        end_offset_ptr,
        out_cache_loc_ptr,
        pool_len,
        bs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<int PAGED_SIZE>
__global__ void dcu_create_flashmla_kv_indices_kernel(
    const int32_t* __restrict__ req_to_token,
    const int32_t* __restrict__ req_pool_indices,
    const int32_t* __restrict__ page_kernel_lens,
    const int32_t* __restrict__ kv_start_idx,
    int32_t* __restrict__ kv_indices,
    int req_to_token_stride,
    int kv_indices_stride)
{
    int pid = blockIdx.x;  // batch index

    int req_pool_index = req_pool_indices[pid];

    int kv_start = 0;
    int kv_end = 0;

    if (kv_start_idx != nullptr) {
        kv_start = kv_start_idx[pid];
        kv_end = kv_start;
    }

    kv_end += page_kernel_lens[pid];

    int total_len = kv_end - kv_start;
    int num_pages = (total_len + PAGED_SIZE - 1) / PAGED_SIZE;

    for (int pg = 0; pg < num_pages; ++pg) {
        int offset = pg * PAGED_SIZE;

        // token id = req_to_token[req_pool_index][kv_start + offset]
        int64_t token =
            req_to_token[req_pool_index * req_to_token_stride + kv_start + offset];

        // 页索引
        kv_indices[pid * kv_indices_stride + pg] = token / PAGED_SIZE;
    }
}

void dcu_create_flashmla_kv_indices(
    const at::Tensor& req_to_token,
    const at::Tensor& req_pool_indices,
    const at::Tensor& page_kernel_lens,
    const c10::optional<at::Tensor>& kv_start_idx,
    at::Tensor& kv_indices,
    int64_t req_to_token_stride,
    int64_t kv_indices_stride,
    int64_t PAGED_SIZE)

{
    TORCH_CHECK(req_to_token.is_cuda(), "req_to_token must be CUDA tensor");
    TORCH_CHECK(kv_indices.is_cuda(), "kv_indices must be CUDA tensor");

    int bs = req_pool_indices.size(0);

    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(bs);
    dim3 block(1);

    const int32_t* kv_start_idx_ptr = nullptr;
    if (kv_start_idx.has_value()) {
        kv_start_idx_ptr = kv_start_idx.value().data_ptr<int32_t>();
    }
    if (PAGED_SIZE == 64) {
        dcu_create_flashmla_kv_indices_kernel<64><<<grid, block, 0, stream>>>(
            req_to_token.data_ptr<int32_t>(),
            req_pool_indices.data_ptr<int32_t>(),
            page_kernel_lens.data_ptr<int32_t>(),
            kv_start_idx_ptr,
            kv_indices.data_ptr<int32_t>(),
            req_to_token_stride,
            kv_indices_stride
        );
    } else {
        TORCH_CHECK(false, "Unsupported PAGED_SIZE");
    }
}



__global__ void launch_create_chunked_prefix_cache_kv_indices(
    int32_t* req_to_token_ptr,
    const int64_t* req_pool_indices_ptr,
    const int32_t* chunk_starts_ptr,
    const int32_t* chunk_seq_lens_ptr,
    const int32_t* chunk_cu_seq_lens_ptr,
    int32_t* chunk_kv_indices_ptr,
    int64_t col_num,
    int64_t bs) 
{
   
    int64_t pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= bs) return;

    int64_t req_pool_index = req_pool_indices_ptr[pid];
    int64_t chunk_kv_indices_offset = chunk_cu_seq_lens_ptr[pid];

    int32_t chunk_start_pos = chunk_starts_ptr[pid];
    int32_t chunk_seq_len = chunk_seq_lens_ptr[pid];
    #pragma unroll(32)
    for(int32_t offset = 0;offset < chunk_seq_len;offset++){
          chunk_kv_indices_ptr[chunk_kv_indices_offset+offset] = req_to_token_ptr[req_pool_index * col_num + chunk_start_pos + offset];
    }
   
}


void dcu_create_chunked_prefix_cache_kv_indices(
    at::Tensor req_to_token_ptr,
    const at::Tensor req_pool_indices_ptr,
    const at::Tensor chunk_starts_ptr,
    const at::Tensor chunk_seq_lens_ptr,
    const at::Tensor chunk_cu_seq_lens_ptr,
    at::Tensor chunk_kv_indices_ptr,
    int64_t col_num,
    int64_t bs) {
    
    int32_t* req_to_token_ptr1 = static_cast<int32_t*>(req_to_token_ptr.data_ptr());
    const int64_t* req_pool_indices_ptr1 = static_cast<const int64_t*>(req_pool_indices_ptr.data_ptr());
    const int32_t* chunk_starts_ptr1 = static_cast<const int32_t*>(chunk_starts_ptr.data_ptr());
    const int32_t* chunk_seq_lens_ptr1 = static_cast<const int32_t*>(chunk_seq_lens_ptr.data_ptr());
    const int32_t* chunk_cu_seq_lens_ptr1 = static_cast<const int32_t*>(chunk_cu_seq_lens_ptr.data_ptr());
    int32_t* chunk_kv_indices_ptr1 = static_cast<int32_t*>(chunk_kv_indices_ptr.data_ptr());

    int64_t block_size = 64;
    int64_t grid_size = (bs + block_size - 1) / block_size;
    cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();
    launch_create_chunked_prefix_cache_kv_indices<<<grid_size, block_size, 0, torch_current_stream>>>(req_to_token_ptr1, req_pool_indices_ptr1, chunk_starts_ptr1, chunk_seq_lens_ptr1, chunk_cu_seq_lens_ptr1,chunk_kv_indices_ptr1, col_num, bs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

}

__global__ void launch_align_evict_mask_to_page_size(
    const int64_t* seq_lens_ptr,
    uint8_t* evict_mask_ptr,
    int64_t page_size,
    int64_t num_draft_tokens,
    int64_t bs) 
{
    int64_t pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= bs) return;
    int64_t seq_lens = seq_lens_ptr[pid];

    int64_t num_trues = 0;
    for(int64_t i = 0; i < num_draft_tokens; i++){
      uint8_t evict_value = evict_mask_ptr[pid * num_draft_tokens + i];
      if(evict_value==1) num_trues++;
    }

    int64_t num_false = num_draft_tokens - num_trues;
    int64_t start = (seq_lens + num_false - 1) / page_size * page_size - seq_lens;
    for(int64_t i = max(start, 0); i < min(start + page_size, num_draft_tokens); i++){
      evict_mask_ptr[pid * num_draft_tokens + i] = 0;
    }
}


void dcu_align_evict_mask_to_page_size(
    const at::Tensor seq_lens_ptr,
    at::Tensor evict_mask_ptr,
    int64_t page_size,
    int64_t num_draft_tokens,
    int64_t bs){

    const int64_t* seq_lens_ptr1 = static_cast<const int64_t*>(seq_lens_ptr.data_ptr());  
    uint8_t* evict_mask_ptr1 = static_cast<uint8_t*>(evict_mask_ptr.data_ptr());
    int64_t block_size = 64;
    int64_t grid_size = (bs + block_size - 1) / block_size;
    cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();
    launch_align_evict_mask_to_page_size<<<grid_size, block_size, 0, torch_current_stream>>>(seq_lens_ptr1, evict_mask_ptr1, page_size, num_draft_tokens,bs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

}
