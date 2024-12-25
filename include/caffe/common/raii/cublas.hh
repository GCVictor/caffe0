#pragma once

#include <cublas_v2.h>

#include "caffe/common/macros.hh"

namespace caffe {

class CublasHandleRAII {
  DISALLOW_COPY_AND_ASSIGN(CublasHandleRAII);

  friend cublasHandle_t cublas_handle();

 private:
  CublasHandleRAII() {
    if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
    }
  }

  ~CublasHandleRAII() {
    if (cublas_handle_) {
      CUBLAS_CHECK(cublasDestroy(cublas_handle_));
    }
  }

  cublasHandle_t cublas_handle_;
};

inline cublasHandle_t cublas_handle() {
  static CublasHandleRAII singleton;

  return singleton.cublas_handle_;
}

}  // namespace caffe