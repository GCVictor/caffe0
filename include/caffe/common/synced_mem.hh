#pragma once

#include <cstdlib>

#include "caffe/util/macros.hh"

namespace caffe {

/// A class for managing synchronized memory access between CPU and GPU.
class SyncedMemory {
  DISALLOW_COPY_AND_ASSIGN(SyncedMemory);

 public:
  enum SyncHead {
    kUninitialized,  ///< Memory is not initialized.
    kAtCpu,          ///< Memory is synchronized to the CPU.
    kAtGpu,          ///< Memory is synchronized to the GPU.
    kSynced,         ///< Memory is synchronized between CPU and GPU.
  };

  // Default constructor initializing SyncedMemory with size 0.
  SyncedMemory();

  /// Constructor initializing SyncedMemory with a specified size.
  ///
  /// \param size The size of the memory to allocate in bytes.
  explicit SyncedMemory(std::size_t size);

  /// Destructor releasing allocated memory.
  ~SyncedMemory();

  /// Ensures that the memory is available on the CPU.
  void ToCpu();

  /// Ensures that the memory is available on the GPU.
  void ToGpu();

  /// \return A constant pointer to the CPU memory.
  const void* cpu_ptr() {
    ToCpu();

    return cpu_ptr_;
  }

  /// \return A constant pointer to the GPU memory.
  const void* gpu_ptr() {
    ToGpu();

    return gpu_ptr_;
  }

  /// \return A mutable pointer to the CPU memory.
  void* mutable_cpu_ptr() {
    ToCpu();
    head_ = SyncHead::kAtCpu;

    return cpu_ptr_;
  }

  /// \return A mutable pointer to the GPU memory.
  void* mutable_gpu_ptr() {
    ToGpu();
    head_ = SyncHead::kAtGpu;

    return gpu_ptr_;
  }

  /// \return The current synchronization state as a SyncHead value.
  constexpr SyncHead head() const { return head_; }

  /// \return The size of the memory in bytes.
  constexpr std::size_t size() const { return size_; }

 private:
  /// Verifies that the memory is on the correct GPU device.
  void CheckDevice() const;

  void* cpu_ptr_;  ///< Pointer to the memory on the CPU.
  void* gpu_ptr_;  ///< Pointer to the memory on the GPU.
  size_t size_;    ///< Size of the allocated memory in bytes.
  int device_;     ///< Device ID associated with the memory.
  SyncHead head_;  ///< Current synchronization state of the memory.
};

}  // namespace caffe