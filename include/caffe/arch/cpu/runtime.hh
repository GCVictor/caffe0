#pragma once

#include "caffe/util/macros.hh"

#ifdef USE_MKL
#include <mkl.h>
#endif

namespace caffe {
namespace cpu {

inline void caffe_malloc_host(void** ptr, size_t size) {
#ifdef USE_MKL
  *ptr = mkl_malloc(size ? size : 1, 64);
#else
  *ptr = malloc(size);
#endif

  CHECK(*ptr) << "Host allocation of size: " << size << " failed";
}

inline void caffe_free_host(void* ptr) {
#ifdef USE_MKL
  mkl_free(ptr);
#else
  free(ptr);
#endif
}

}  // namespace cpu
}  // namespace caffe