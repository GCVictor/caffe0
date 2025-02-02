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
    kAtCPU,          ///< Memory is synchronized to the CPU.
    kAtGPU,          ///< Memory is synchronized to the GPU.
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
  void ToCPU();

  /// Ensures that the memory is available on the GPU.
  void ToGPU();

  /// \brief Checks if the data is currently on the GPU.
  ///
  /// This function determines whether the data is stored on the GPU by checking
  /// the current state of the `head_` flag. The data is considered to be on the
  /// GPU if the `head_` flag is either `kSynced` (data is synchronized between
  /// CPU and GPU) or `kAtGPU` (data is only on the GPU).
  ///
  /// \return `true` if the data is on the GPU; otherwise, `false`.
  constexpr bool IsOnGPU() const {
    return head_ == SyncedMemory::kSynced || head_ == SyncedMemory::kAtGPU;
  }

  /// \return A constant pointer to the CPU memory.
  const void* cpu_ptr() {
    ToCPU();

    return cpu_ptr_;
  }

  /// \return A constant pointer to the GPU memory.
  const void* gpu_ptr() {
    ToGPU();

    return gpu_ptr_;
  }

  /// \return A mutable pointer to the CPU memory.
  void* mutable_cpu_ptr() {
    ToCPU();
    head_ = SyncHead::kAtCPU;

    return cpu_ptr_;
  }

  /// \return A mutable pointer to the GPU memory.
  void* mutable_gpu_ptr() {
    ToGPU();
    head_ = SyncHead::kAtGPU;

    return gpu_ptr_;
  }

  /// \return The current synchronization state as a SyncHead value.
  constexpr SyncHead head() const { return head_; }

  /// \return The size of the memory in bytes.
  constexpr size_t size() const { return size_; }

  /// \brief Get the size of the memory block, expressed in bytes.
  ///
  /// \return The size of the memory in bytes.
  constexpr size_t size_in_bytes() const { return size_; }

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