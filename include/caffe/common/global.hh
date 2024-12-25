#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <mkl_vsl.h>

#include "caffe/common/macros.hh"

namespace caffe {

class Caffe {
  DISALLOW_COPY_AND_ASSIGN(Caffe);

 public:
  enum Brew { kCpu, kGpu };
  enum Phase { kTrain, kTest };

  static Caffe& GetInstance() {
    static Caffe singleton;

    return singleton;
  }

  /// \return The cublas handle.
  static cublasHandle_t cublas_handle() { return GetInstance().cublas_handle_; }

  /// \return The curand generator.
  static curandGenerator_t curand_generator() {
    return GetInstance().curand_generator_;
  }

  /// \return The MKL random stream.
  static VSLStreamStatePtr vsl_stream() { return GetInstance().vsl_stream_; }

  /// \return The mode: running on CPU or GPU.
  static Brew mode() { return GetInstance().mode_; }

  /// \return The phase: TRAIN or TEST.
  static Phase phase() { return GetInstance().phase_; }

  /// Sets the mode.
  static void set_mode(Brew mode) { GetInstance().mode_ = mode; }

  /// Sets the phase.
  static void set_phase(Phase phase) { GetInstance().phase_ = phase; }

  // Sets the random seed of both MKL and curand
  static void set_random_seed(const unsigned int seed);

  ~Caffe();

 private:
  Caffe();

  cublasHandle_t cublas_handle_;
  curandGenerator_t curand_generator_;
  VSLStreamStatePtr vsl_stream_;
  Brew mode_;
  Phase phase_;
};

}  // namespace caffe