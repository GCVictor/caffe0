#pragma once

// #include <cublas_v2.h>
#include <mkl.h>

namespace caffe {

namespace cpu {

/// Perform General Matrix-Matrix Multiplication (GEMM) on the CPU.
/// The operation is performed as:
///   C := alpha * A * B + beta * C
/// where A, B, and C are matrices, and alpha, beta are scalars.
///
/// \param transa The operation to apply to matrix A. It can be
///               - CUBLAS_OP_N (no transpose)
///               - CUBLAS_OP_T (transpose)
///               - CUBLAS_OP_C (conjugate transpose)
/// \param transb The operation to apply to matrix B, same options as transa.
/// \param m The number of rows in matrix A and matrix C.
/// \param n The number of columns in matrix B and matrix C.
/// \param k The number of columns in matrix A and rows in matrix B.
/// \param alpha Scalar multiplier for A * B.
/// \param A Pointer to the matrix A stored in row-major order.
/// \param B Pointer to the matrix B stored in row-major order
/// \param beta Scalar multiplier for matrix C.
/// \param C  Pointer to the matrix C (output).
template <typename Dtype>
void caffe_gemm(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                const int m, const int n, const int k, const Dtype alpha,
                const Dtype* A, const Dtype* B, const Dtype beta, Dtype* C);

// /// Perform General Matrix-Vector Multiplication (GEMV) on the CPU.
// /// The operation is performed as: y := alpha * A * x + beta * y
// /// where A is a matrix, x and y are vectors, and alpha, beta are scalars.
// ///
// /// \tparam Dtype
// /// \param TransA Specifies the operation to apply to matrix `A`:
// ///                - `CUBLAS_OP_N`: No transpose (A is used as is).
// ///                - `CUBLAS_OP_T`: Transpose matrix `A` (A^T).
// ///                - `CUBLAS_OP_C`: Conjugate transpose matrix `A` (A^H, for
// ///                                 complex numbers).
// /// \param M The number of rows in matrix `A`.
// /// \param N The number of columns in matrix `A`.
// /// \param alpha A scalar multiplier for the matrix-vector product `A * x`.
// /// \param A Pointer to the matrix `A`.
// /// \param x Pointer to the input vector `x`.
// /// \param beta A scalar multiplier for the vector `y`.
// /// \param y Pointer to the output vector `y`.
// template <typename Dtype>
// void caffe_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
//                 const Dtype alpha, const Dtype* A, const Dtype* x,
//                 const Dtype beta, Dtype* y);

// /// This function performs a vector-vector operation defined as
// /// y := a*x + y
// ///
// /// If incx = 1, incy = 1, <=> Y[i] = alpha * X[i] + Y[i], for i = 0 to N - 1
// ///
// /// Example:
// /// N     = 5
// /// alpha = 20
// /// X     = [1,  2,  3,  4,  5]
// /// Y     = [10, 20, 30, 40, 50]
// /// caffe_axpy(N, alpha, X, Y)
// /// Y     = [12, 24, 36, 48, 60]
// ///
// /// \param n The number of elements in vectors x and y.
// /// \param alpha Scalar used for multiplication.
// /// \param x Vector with n elements.
// /// \param y Vector with n elements
// template <typename Dtype>
// void caffe_axpy(const int n, const Dtype alpha, const Dtype* x, Dtype* y);

// template <typename Dtype>
// void caffe_axpby(const int N, const Dtype alpha, const Dtype* X,
//                  const Dtype beta, Dtype* Y);

// /// Copy a vector from one array to another.
// ///
// /// \param N The number of elements to copy.
// /// \param X Pointer to the source array.
// /// \param Y The pointer to the destination array
// template <typename Dtype>
// void caffe_copy(const int N, const Dtype* X, Dtype* Y);

// /// Scale a vector by a constant scalar.
// ///
// /// \param N The number of elements in the vector X to scale.
// /// \param alpha The scalar value that each element of the vector X will be
// ///              multiplied by.
// /// \param X Pointer to the vector to scale.
// template <typename Dtype>
// void caffe_scal(const int N, const Dtype alpha, Dtype* X);

// /// Performs element by element squaring of the vector.
// ///
// /// \param n The number of elements to be calculated.
// /// \param a Pointer to an array that contains the input vector a.
// /// \param y Pointer to an array that contains the output vector y.
// template <typename Dtype>
// void caffe_sqr(const int n, const Dtype* a, Dtype* y);

// /// Performs element by element addition of vector a and vector b.
// ///
// /// \param n The number of elements to be calculated.
// /// \param a Pointers to arrays that contain the input vectors a.
// /// \param b Pointers to arrays that contain the input vectors b
// /// \param y Pointer to an array that contains the output vector y.
// template <typename Dtype>
// void caffe_add(const int n, const Dtype* a, const Dtype* b, Dtype* y);

// /// Performs element by element subtraction of vector b from vector a.
// ///
// /// \param n The number of elements to be calculated.
// /// \param a Pointers to arrays that contain the input vectors a.
// /// \param b Pointers to arrays that contain the input vectors b
// /// \param y Pointer to an array that contains the output vector y.
// template <typename Dtype>
// void caffe_sub(const int n, const Dtype* a, const Dtype* b, Dtype* y);

// /// Performs element by element multiplication of vector a and vector b.
// ///
// /// \param n The number of elements to be calculated.
// /// \param a Pointers to arrays that contain the input vectors a.
// /// \param b Pointers to arrays that contain the input vectors b
// /// \param y Pointer to an array that contains the output vector y.
// template <typename Dtype>
// void caffe_mul(const int n, const Dtype* a, const Dtype* b, Dtype* y);

// /// Performs element by element division of vector a by vector b
// ///
// /// \param n The number of elements to be calculated.
// /// \param a Pointers to arrays that contain the input vectors a.
// /// \param b Pointers to arrays that contain the input vectors b
// /// \param y Pointer to an array that contains the output vector y.
// template <typename Dtype>
// void caffe_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

// /// Computes vector a to the scalar power b.
// ///
// /// \param n The number of elements to be calculated.
// /// \param a Pointers to arrays that contain the input vectors a.
// /// \param b Pointers to arrays that contain the input vectors b
// /// \param y Pointer to an array that contains the output vector y.
// template <typename Dtype>
// void caffe_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

// template <typename Dtype>
// void caffe_vrng_uniform(const int n, Dtype* r, const Dtype a, const Dtype b);

// template <typename Dtype>
// void caffe_vrng_gaussian(const int n, Dtype* r, const Dtype a,
//                          const Dtype sigma);

// template <typename Dtype>
// void caffe_exp(const int n, const Dtype* a, Dtype* y);

// template <typename Dtype>
// Dtype caffe_dot(const int n, const Dtype* x, const Dtype* y);

}  // namespace cpu

// namespace gpu {

// // Decaf gpu gemm provides an interface that is almost the same as the cpu
// // gemm function - following the c convention and calling the fortran-order
// // gpu code under the hood.
// template <typename Dtype>
// void caffe_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
//                 const int M, const int N, const int K, const Dtype alpha,
//                 const Dtype* A, const Dtype* B, const Dtype beta, Dtype* C);

// template <typename Dtype>
// void caffe_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
//                 const Dtype alpha, const Dtype* A, const Dtype* x,
//                 const Dtype beta, Dtype* y);

// template <typename Dtype>
// void caffe_axpy(const int n, const Dtype alpha, const Dtype* x, Dtype* y);

// template <typename Dtype>
// void caffe_axpby(const int N, const Dtype alpha, const Dtype* X,
//                  const Dtype beta, Dtype* Y);

// template <typename Dtype>
// void caffe_copy(const int N, const Dtype* X, Dtype* Y);

// template <typename Dtype>
// void caffe_scal(const int N, const Dtype alpha, Dtype* X);

// template <typename Dtype>
// void caffe_dot(const int n, const Dtype* x, const Dtype* y, Dtype* out);

// }  // namespace gpu
}  // namespace caffe