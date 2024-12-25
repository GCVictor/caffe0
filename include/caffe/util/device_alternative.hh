#pragma once

#include <glog/logging.h>

#ifdef CPU_ONLY

#define REPORT_GPU_NOT_SUPPORTED() \
  LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode."

#elif defined(USE_CUDA)

/// https://stackoverflow.com/questions/6302695/difference-between-cuda-h-cuda-runtime-h-cuda-runtime-api-h

// clang-format off
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <driver_types.h>
// clang-format on

/// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition)                                           \
  do {                                                                  \
    cudaError_t status = condition;                                     \
    CHECK_EQ(status, cudaSuccess) << " " << cudaGetErrorString(status); \
  } while (0)

#define CUBLAS_CHECK(condition)                        \
  do {                                                 \
    cublasStatus_t status = condition;                 \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS)            \
        << " " << caffe::cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition)                        \
  do {                                                 \
    curandStatus_t status = condition;                 \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS)            \
        << " " << caffe::curandGetErrorString(status); \
  } while (0)

/// CUDA: grid stride looping.
#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

/// CUDA: check for error after kernel execution and exit loudly if there is
/// one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

namespace caffe {

/// CUDA: library error reporting.
const char* cublas_get_error_string(cublasStatus_t status);
const char* curand_get_error_string(curandStatus_t status);

/// CUDA: use 512 threads per block.
inline constexpr int kCaffeCudaThreadsPerBlock = 512;

/// CUDA: compute the number of blocks required.
inline int caffe_get_blocks(const int n) {
  return (n + kCaffeCudaThreadsPerBlock - 1) / kCaffeCudaThreadsPerBlock;
}

}  // namespace caffe

#elif defined(USE_HIP)

#define HIP_CHECK(condition)                                          \
  do {                                                                \
    hipError_t status = condition;                                    \
    CHECK_EQ(status, hipSuccess) << " " << hipGetErrorString(status); \
  } while (0)

#endif