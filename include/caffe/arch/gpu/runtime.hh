#pragma once

#include <cassert>

#if defined(USE_CUDA)
#include "caffe/arch/gpu/nvidia/runtime.hh"
#elif defined(USE_HIP)
#include "caffe/arch/gpu/amd/runtime.hh"
#else
#error "No GPU backend is supported. Define either USE_CUDA or USE_HIP."
#endif

namespace caffe::gpu {

enum MemcpyKind {
  kHostToHost = 0,      ///< Host-to-Host copy.
  kHostToDevice = 1,    ///< Host-to-Device copy.
  kDeviceToHost = 2,    ///< Device-to-Host copy.
  kDeviceToDevice = 3,  ///< Device-to-Device copy.
  kDefault = 4,  ///< Runtime will automatically determine copy-kind based on
                 ///< virtual addresses.
  kDeviceToDeviceNoCU =
      5,  ///< Device-to-Device Copy without using compute units.
};

inline void caffe_memcpy(void* dst, const void* src, size_t count,
                         MemcpyKind kind = MemcpyKind::kDefault) {
#if defined(USE_CUDA)
  assert(kind != MemcpyKind::kDeviceToDeviceNoCU);

  caffe_memcpy(dst, src, count, static_cast<cudaMemcpyKind>(kind));

#elif defined(USE_HIP)

  caffe_memcpy(dst, src, count, static_cast<hipMemcpyKind>(kind));

#endif
}

}  // namespace caffe::gpu