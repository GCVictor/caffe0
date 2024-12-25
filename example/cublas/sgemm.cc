#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iostream>

#include "caffe/common/macros.hh"

/// Matrix Dimensions:
///
/// A: M x K
/// B: K x N
/// C: M x N
///
/// Function Parameters:
///     cublasSgemm computes C = alpha x A x B + beta x C
///     CUBLAS_OP_N specifies no transposition for the matrices.

const int M = 2;
const int N = 2;
const int K = 2;

void print_matrix(const float *data, const int M, const int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << data[i * N + j] << " ";
    }

    std::cout << '\n';
  }

  std::cout << std::endl;
}

int main() {
  float A[M * K] = {1.0f, 2.0f, 3.0f, 4.0f};  // 2x2 matrix
  float B[K * N] = {5.0f, 6.0f, 7.0f, 8.0f};  // 2x2 matrix
  float C[M * N] = {0.0f, 0.0f, 0.0f, 0.0f};  // 2x2 result matrix

  float *d_A, *d_B, *d_C;

  CUDA_CHECK(cudaMalloc((void **)&d_A, M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&d_B, K * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&d_C, M * N * sizeof(float)));

  // Copy data from host to device
  CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Perform matrix multiplication: C = alpha * A * B + beta * C
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
                           d_A, M, d_B, K, &beta, d_C, M));

  // Copy result back to host
  CUDA_CHECK(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  std::cout << "Matrix A:\n";
  print_matrix(A, M, K);

  std::cout << "Matrix B:\n";
  print_matrix(B, K, N);

  std::cout << "Matrix C = A x B:" << std::endl;
  print_matrix(C, M, N);

  // Clean up
  cublasDestroy(handle);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
