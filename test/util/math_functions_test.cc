#include "caffe/util/math_functions.hh"

#include <algorithm>

#include "caffe/common/macros.hh"
#include "test_caffe_main.hh"

template <typename Dtype>
static void expect_vector_eq(const Dtype* expect, size_t n,
                             const Dtype* actual) {
  EXPECT_TRUE(std::equal(expect, expect + n, actual));
}

template <typename Dtype>
static void gemm_on_cpu() {
  constexpr int m = 2;
  constexpr int n = 2;
  constexpr int k = 3;
  constexpr int mn = m * n;
  const Dtype alpha = 1;
  const Dtype beta = 0;
  const Dtype A[] = {1, 2, 3, 4, 5, 6};
  const Dtype B[] = {1, 2, 3, 4, 5, 6};
  Dtype C[mn]{};
  const Dtype Expect_C[] = {22, 28, 49, 64};

  caffe::cpu::caffe_gemm<Dtype>(CblasNoTrans, CblasNoTrans, m, n, k, alpha, A,
                                B, beta, C);

  expect_vector_eq<Dtype>(Expect_C, mn, C);
}

///     [1, 2, 3]
/// A = [4, 5, 6]
///
///     [1, 2]
/// B = [3, 4]
///     [5, 6]
///
/// C = A * B
///
///          [22, 28]
/// Expect = [49, 64]
TEST(Sgemm, OnCPU) {
  gemm_on_cpu<float>();
  gemm_on_cpu<double>();
}

// TEST(Sgemm, OnGPU) {
//   const int m = 2;
//   const int n = 2;
//   const int k = 3;
//   const float alpha = 1;
//   const float beta = 0;
//   const float a[] = {1, 2, 3, 4, 5, 6};
//   const float b[] = {1, 2, 3, 4, 5, 6};
//   float c[m * n]{};
//   const float expect[] = {22, 28, 49, 64};

//   float *d_a, *d_b, *d_c;
//   CUDA_CHECK(cudaMalloc((void **)&d_a, m * k * sizeof(float)));
//   CUDA_CHECK(cudaMalloc((void **)&d_b, k * n * sizeof(float)));
//   CUDA_CHECK(cudaMalloc((void **)&d_c, m * n * sizeof(float)));

//   // Copy data from host to device.
//   CUDA_CHECK(cudaMemcpy(d_a, a, m * k * sizeof(float),
//   cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_b, b, k * n *
//   sizeof(float), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_c, c, m *
//   n * sizeof(float), cudaMemcpyHostToDevice));

//   caffe::gpu::caffe_gemm<float>(CblasNoTrans, CblasNoTrans, m, n, k, alpha,
//   d_a,
//                                 d_b, beta, d_c);

//   // Copy result back to host.
//   CUDA_CHECK(cudaMemcpy(c, d_c, m * n * sizeof(float),
//   cudaMemcpyDeviceToHost));

//   expect_vector_eq(m * n, c, expect);

//   cudaFree(d_a);
//   cudaFree(d_b);
//   cudaFree(d_c);
// }

// ///     [1.0, 2.0, 3.0]
// /// A = [4.0, 5.0, 6.0]
// ///     [7.0, 8.0, 9.0]
// /// x = [1.0, 2.0, 3.0]
// TEST(Sgemv, OnCPU) {
//   const int n = 3;

//   float alpha = 1.0f;
//   float beta = 0.0f;
//   float a[n * n] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
//   float x[n] = {1.0f, 2.0f, 3.0f};
//   float y[n] = {0.0f, 0.0f, 0.0f};
//   const float expect[n] = {14, 32, 50};

//   caffe::cpu::caffe_gemv<float>(CblasNoTrans, n, n, alpha, a, x, beta, y);

//   expect_vector_eq(n, y, expect);
// }

// TEST(Sgemv, OnGPU) {
//   const int n = 3;

//   float alpha = 1.0f;
//   float beta = 0.0f;
//   float a[n * n] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
//   float x[n] = {1.0f, 2.0f, 3.0f};
//   float y[n] = {0.0f, 0.0f, 0.0f};
//   const float expect[n] = {14, 32, 50};

//   // Device variables.
//   float *d_a, *d_x, *d_y;

//   // Allocate device memory.
//   CUDA_CHECK(cudaMalloc((void **)&d_a, n * n * sizeof(float)));
//   CUDA_CHECK(cudaMalloc((void **)&d_x, n * sizeof(float)));
//   CUDA_CHECK(cudaMalloc((void **)&d_y, n * sizeof(float)));

//   // Copy host memory to device.
//   CUDA_CHECK(cudaMemcpy(d_a, a, n * n * sizeof(float),
//   cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(float),
//   cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_y, y, n * sizeof(float),
//   cudaMemcpyHostToDevice));

//   caffe::gpu::caffe_gemv<float>(CblasNoTrans, n, n, alpha, d_a, d_x, beta,
//   d_y);

//   // Copy result back to host.
//   CUDA_CHECK(cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost));

//   expect_vector_eq(n, y, expect);

//   CUDA_CHECK(cudaFree(d_a));
//   CUDA_CHECK(cudaFree(d_x));
//   CUDA_CHECK(cudaFree(d_y));
// }

// TEST(Aaxpy, OnCPU) {
//   const int n = 5;
//   float alpha = 2.0f;
//   float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
//   float y[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
//   const float expect[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

//   caffe::cpu::caffe_axpy<float>(n, alpha, x, y);

//   expect_vector_eq(n, y, expect);
// }

// TEST(Copy, OnCPU) {
//   const int n = 5;
//   float x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
//   float y[n];
//   const float expect[] = {1.0, 2.0, 3.0, 4.0, 5.0};

//   caffe::cpu::caffe_copy(n, x, y);

//   expect_vector_eq(n, y, expect);
// }

// TEST(Scale, OnCPU) {
//   const int n = 5;
//   float alpha = 2.0;
//   float x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
//   const float expect[] = {2.0, 4.0, 6.0, 8.0, 10.0};

//   caffe::cpu::caffe_scal(n, alpha, x);

//   expect_vector_eq(n, x, expect);
// }

// TEST(Square, OnCPU) {
//   const int n = 5;
//   float x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
//   float y[n];
//   const float expect[] = {1.0, 4.0, 9.0, 16.0, 25.0};

//   caffe::cpu::caffe_sqr(n, x, y);

//   expect_vector_eq(n, y, expect);
// }