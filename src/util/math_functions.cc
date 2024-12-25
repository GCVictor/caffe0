#include "caffe/util/math_functions.hh"

// #include "caffe/common/raii/cublas.hh"
// #include "caffe/common/raii/mkl.hh"

namespace caffe {
namespace cpu {

template <>
void caffe_gemm<float>(const CBLAS_TRANSPOSE transa,
                       const CBLAS_TRANSPOSE transb, const int m, const int n,
                       const int k, const float alpha, const float* A,
                       const float* B, const float beta, float* C) {
  int lda = (transa == CblasNoTrans) ? k : m;
  int ldb = (transb == CblasNoTrans) ? n : k;
  cblas_sgemm(CblasRowMajor, transa, transb, m, n, k, alpha, A, lda, B, ldb,
              beta, C, n);
}

template <>
void caffe_gemm<double>(const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const int M, const int N,
                        const int K, const double alpha, const double* A,
                        const double* B, const double beta, double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, N);
}

// template <>
// void caffe_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M, const int
// N,
//                        const float alpha, const float* A, const float* x,
//                        const float beta, float* y) {
//   cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
// }

// template <>
// void caffe_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M, const int
// N,
//                         const double alpha, const double* A, const double* x,
//                         const double beta, double* y) {
//   cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
// }

// template <>
// void caffe_axpy<float>(const int n, const float alpha, const float* x,
//                        float* y) {
//   cblas_saxpy(n, alpha, x, 1, y, 1);
// }

// template <>
// void caffe_axpy<double>(const int n, const double alpha, const double* x,
//                         double* y) {
//   cblas_daxpy(n, alpha, x, 1, y, 1);
// }

// template <>
// void caffe_axpby<float>(const int N, const float alpha, const float* X,
//                         const float beta, float* Y) {
//   cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
// }

// template <>
// void caffe_axpby<double>(const int N, const double alpha, const double* X,
//                          const double beta, double* Y) {
//   cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
// }

// template <>
// void caffe_copy<float>(const int N, const float* X, float* Y) {
//   cblas_scopy(N, X, 1, Y, 1);
// }

// template <>
// void caffe_copy<double>(const int N, const double* X, double* Y) {
//   cblas_dcopy(N, X, 1, Y, 1);
// }

// template <>
// void caffe_scal<float>(const int N, const float alpha, float* X) {
//   cblas_sscal(N, alpha, X, 1);
// }

// template <>
// void caffe_scal<double>(const int N, const double alpha, double* X) {
//   cblas_dscal(N, alpha, X, 1);
// }

// template <>
// void caffe_sqr<float>(const int n, const float* a, float* y) {
//   vsSqr(n, a, y);
// }

// template <>
// void caffe_sqr<double>(const int n, const double* a, double* y) {
//   vdSqr(n, a, y);
// }

// template <>
// void caffe_add<float>(const int n, const float* a, const float* b, float* y)
// {
//   vsAdd(n, a, b, y);
// }

// template <>
// void caffe_add<double>(const int n, const double* a, const double* b,
//                        double* y) {
//   vdAdd(n, a, b, y);
// }

// template <>
// void caffe_sub<float>(const int n, const float* a, const float* b, float* y)
// {
//   vsSub(n, a, b, y);
// }

// template <>
// void caffe_sub<double>(const int n, const double* a, const double* b,
//                        double* y) {
//   vdSub(n, a, b, y);
// }

// template <>
// void caffe_mul<float>(const int n, const float* a, const float* b, float* y)
// {
//   vsMul(n, a, b, y);
// }

// template <>
// void caffe_mul<double>(const int n, const double* a, const double* b,
//                        double* y) {
//   vdMul(n, a, b, y);
// }

// template <>
// void caffe_div<float>(const int n, const float* a, const float* b, float* y)
// {
//   vsDiv(n, a, b, y);
// }

// template <>
// void caffe_div<double>(const int n, const double* a, const double* b,
//                        double* y) {
//   vdDiv(n, a, b, y);
// }

// template <>
// void caffe_powx<float>(const int n, const float* a, const float b, float* y)
// {
//   vsPowx(n, a, b, y);
// }

// template <>
// void caffe_powx<double>(const int n, const double* a, const double b,
//                         double* y) {
//   vdPowx(n, a, b, y);
// }

// template <>
// void caffe_vrng_uniform<float>(const int n, float* r, const float a,
//                                const float b) {
//   VSL_CHECK(vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, caffe::vsl_stream(), n,
//   r,
//                          a, b));
// }

// template <>
// void caffe_vrng_uniform<double>(const int n, double* r, const double a,
//                                 const double b) {
//   VSL_CHECK(vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, caffe::vsl_stream(), n,
//   r,
//                          a, b));
// }

// template <>
// void caffe_vrng_gaussian<float>(const int n, float* r, const float a,
//                                 const float sigma) {
//   VSL_CHECK(vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER,
//                           caffe::vsl_stream(), n, r, a, sigma));
// }

// template <>
// void caffe_vrng_gaussian<double>(const int n, double* r, const double a,
//                                  const double sigma) {
//   VSL_CHECK(vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER,
//                           caffe::vsl_stream(), n, r, a, sigma));
// }

// template <>
// void caffe_exp<float>(const int n, const float* a, float* y) {
//   vsExp(n, a, y);
// }

// template <>
// void caffe_exp<double>(const int n, const double* a, double* y) {
//   vdExp(n, a, y);
// }

// template <>
// float caffe_dot<float>(const int n, const float* x, const float* y) {
//   return cblas_sdot(n, x, 1, y, 1);
// }

// template <>
// double caffe_dot<double>(const int n, const double* x, const double* y) {
//   return cblas_ddot(n, x, 1, y, 1);
// }

}  // namespace cpu

// namespace gpu {

// template <>
// void caffe_gemm<float>(const CBLAS_TRANSPOSE TransA,
//                        const CBLAS_TRANSPOSE TransB, const int M, const int
//                        N, const int K, const float alpha, const float* A,
//                        const float* B, const float beta, float* C) {
//   // Note that cublas follows fortran order.
//   int lda = (TransA == CblasNoTrans) ? K : M;
//   int ldb = (TransB == CblasNoTrans) ? N : K;
//   cublasOperation_t cuTransA =
//       (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
//   cublasOperation_t cuTransB =
//       (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

//   CUBLAS_CHECK(cublasSgemm(caffe::cublas_handle(), cuTransB, cuTransA, N, M,
//   K,
//                            &alpha, B, ldb, A, lda, &beta, C, N));
// }

// template <>
// void caffe_gemm<double>(const CBLAS_TRANSPOSE TransA,
//                         const CBLAS_TRANSPOSE TransB, const int M, const int
//                         N, const int K, const double alpha, const double* A,
//                         const double* B, const double beta, double* C) {
//   // Note that cublas follows fortran order.
//   int lda = (TransA == CblasNoTrans) ? K : M;
//   int ldb = (TransB == CblasNoTrans) ? N : K;
//   cublasOperation_t cuTransA =
//       (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
//   cublasOperation_t cuTransB =
//       (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

//   CUBLAS_CHECK(cublasDgemm(caffe::cublas_handle(), cuTransB, cuTransA, N, M,
//   K,
//                            &alpha, B, ldb, A, lda, &beta, C, N));
// }

// template <>
// void caffe_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M, const int
// N,
//                        const float alpha, const float* A, const float* x,
//                        const float beta, float* y) {
//   cublasOperation_t cuTransA =
//       (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;

//   CUBLAS_CHECK(cublasSgemv(caffe::cublas_handle(), cuTransA, N, M, &alpha, A,
//   N,
//                            x, 1, &beta, y, 1));
// }

// template <>
// void caffe_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M, const int
// N,
//                         const double alpha, const double* A, const double* x,
//                         const double beta, double* y) {
//   cublasOperation_t cuTransA =
//       (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;

//   CUBLAS_CHECK(cublasDgemv(caffe::cublas_handle(), cuTransA, N, M, &alpha, A,
//   N,
//                            x, 1, &beta, y, 1));
// }

// template <>
// void caffe_axpy<float>(const int n, const float alpha, const float* x,
//                        float* y) {
//   CUBLAS_CHECK(cublasSaxpy(caffe::cublas_handle(), n, &alpha, x, 1, y, 1));
// }

// template <>
// void caffe_axpy<double>(const int n, const double alpha, const double* x,
//                         double* y) {
//   CUBLAS_CHECK(cublasDaxpy(caffe::cublas_handle(), n, &alpha, x, 1, y, 1));
// }

// template <>
// void caffe_scal<float>(const int N, const float alpha, float* X) {
//   CUBLAS_CHECK(cublasSscal(caffe::cublas_handle(), N, &alpha, X, 1));
// }

// template <>
// void caffe_scal<double>(const int N, const double alpha, double* X) {
//   CUBLAS_CHECK(cublasDscal(caffe::cublas_handle(), N, &alpha, X, 1));
// }

// template <>
// void caffe_axpby<float>(const int N, const float alpha, const float* X,
//                         const float beta, float* Y) {
//   caffe_scal<float>(N, beta, Y);
//   caffe_axpy<float>(N, alpha, X, Y);
// }

// template <>
// void caffe_axpby<double>(const int N, const double alpha, const double* X,
//                          const double beta, double* Y) {
//   caffe_scal<double>(N, beta, Y);
//   caffe_axpy<double>(N, alpha, X, Y);
// }

// template <>
// void caffe_copy<float>(const int N, const float* X, float* Y) {
//   CUBLAS_CHECK(cublasScopy(caffe::cublas_handle(), N, X, 1, Y, 1));
// }

// template <>
// void caffe_copy<double>(const int N, const double* X, double* Y) {
//   CUBLAS_CHECK(cublasDcopy(caffe::cublas_handle(), N, X, 1, Y, 1));
// }

// template <>
// void caffe_dot<float>(const int n, const float* x, const float* y, float*
// out) {
//   CUBLAS_CHECK(cublasSdot(caffe::cublas_handle(), n, x, 1, y, 1, out));
// }

// template <>
// void caffe_dot<double>(const int n, const double* x, const double* y,
//                        double* out) {
//   CUBLAS_CHECK(cublasDdot(caffe::cublas_handle(), n, x, 1, y, 1, out));
// }

// }  // namespace gpu
}  // namespace caffe