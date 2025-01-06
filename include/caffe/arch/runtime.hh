#pragma once

#include <cstdlib>

#include "caffe/arch/cpu/runtime.hh"
#include "caffe/util/device_alternative.hh"
#include "caffe/util/macros.hh"

#ifndef CPU_ONLY
#include "caffe/arch/gpu/runtime.hh"
#endif

namespace caffe {

inline void caffe_malloc_host(void** ptr, size_t size, unsigned int flags = 0) {
#ifndef CPU_ONLY
  gpu::caffe_malloc_host(ptr, size, flags);
#else
  cpu::caffe_malloc_host(ptr, size);
#endif
}

inline void caffe_free_host(void* ptr) {
#ifndef CPU_ONLY
  gpu::caffe_free_host(ptr);
#elif defined(USE_HIP)
  cpu::caffe_free_host(ptr);
#endif
}

}  // namespace caffe