#pragma once

#include <glog/logging.h>

#include <initializer_list>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#include "caffe.pb.h"
#include "caffe/common/synced_mem.hh"

namespace caffe {

template <typename Dtype>
class Blob {
  class Shape {
   public:
    /// \brief Initialize the shape using a list of dimensions.
    ///
    /// \param shape A list of integers representing the dimensions.
    explicit Shape(std::initializer_list<int> dims) : dims_{dims} {
      num_el_ = ComputeNumel();
    }

    /// \brief Attempts to reshape this shape to match another shape.
    ///
    /// If the shapes are compatible (i.e., they can have the same total
    /// number of dimensions, with one dimension possibly being -1, which will
    /// be inferred). If reshaping is not possible, the function returns false.
    ///
    /// \param other The target shape to reshape to.
    /// \return True if reshaping was successful; otherwise false.
    bool ReshapeTo(const Shape& other) {
      if (numel() == other.numel()) {
        dims_ = other.dims_;

        return true;
      }

      auto [quot, rem] = std::div(numel(), other.numel());

      // Either no -1 is found (more -1s are found) or size is not divisible.
      if (std::count(other.begin(), other.end(), -1) != 1 || rem != 0) {
        return false;
      }

      dims_ = other.dims_;
      auto it = std::find(begin(), end(), -1);
      *it = -quot;

      return true;
    }

    /// \brief Checks if the shape is valid (i.e., all dimensions are positive).
    ///
    /// \return True if all dimensions are positive; otherwise false.
    constexpr bool IsValid() const {
      return std::all_of(begin(), end(), [](int dim) { return dim > 0; });
    }

    /// \return The total number of elements in this shape.
    constexpr int numel() const { return num_el_; }

    /// \brief Returns an iterator to the beginning of the dimensions.
    ///
    /// \return An iterator to the beginning of the dimensions.
    auto begin() const { return dims_.begin(); }

    /// \brief Returns an iterator to the end of the dimensions.
    ///
    /// \return An iterator to the end of the dimensions.
    auto end() const { return dims_.end(); }

    /// \return The number of dimensions in this shape.
    constexpr int NDim() const { return dims_.size(); }

    /// \brief Get the dimension at the specified index.
    ///
    /// \param index The index of the dimension to retrieve.
    /// \return The dimension at the specified index.
    constexpr int DimAt(size_t index) const {
      CHECK_LT(index, NDim());

      return dims_[index];
    }

   private:
    /// \return The total product of the dimensions.
    constexpr int ComputeNumel() const {
      return std::accumulate(begin(), end(), 1, std::multiplies<int>{});
    }

    int num_el_;
    std::vector<int> dims_;
  };

 public:
  /// \brief Creates a Blob object from a protocol buffer message.
  ///
  /// \param proto The protocol buffer message containing the Blob data.
  /// \return A Blob object initialized with the data from the protocol buffer.
  static Blob<Dtype> FromProto(const BlobProto& proto) {
    auto requires_diff = proto.double_diff_size() > 0 || proto.diff_size() > 0;
    Blob<Dtype> blob{proto.shape(), requires_diff};

    // Copy data.
    auto numel = blob.NumElement();
    auto data = blob.GetMutableData();

    if (proto.double_data_size() > 0) {
      CHECK_EQ(numel, proto.double_data_size());

      std::copy(proto.double_data().begin(), proto.double_data().end(), data);
    } else {
      CHECK_EQ(numel, proto.data_size());

      std::copy(proto.data().begin(), proto.data().end(), data);
    }

    // Copy diff.
    auto diff = blob.GetMutableDiff();

    if (proto.double_diff_size()) {
      CHECK_EQ(numel, proto.double_diff_size());

      std::copy(proto.double_diff().begin(), proto.double_diff().end(), diff);
    } else if (proto.diff_size()) {
      CHECK_EQ(numel, proto.diff_size());

      std::copy(proto.diff().begin(), proto.diff().end(), diff);
    }

    return blob;
  }

  /// \brief Constructs a Blob object with the specified dimensions.
  ///
  /// \param dims A list of integers representing the dimensions.
  /// \param requires_diff Whether to allocate memory for diff (gradients).
  explicit Blob(std::initializer_list<int> dims, bool requires_diff = true)
      : shape_{dims} {
    CHECK(shape_.IsValid());

    auto numel = shape_.numel();
    auto nbytes = numel * ElementSize();
    data_ = std::make_shared<SyncedMemory>(nbytes);

    if (requires_diff) {
      diff_ = std::make_shared<SyncedMemory>(nbytes);
    }
  }

  /// \brief Returns a string representation of the shape.
  ///
  /// \return A string describing the shape.
  std::string ShapeString() const {
    std::ostringstream oss;

    for (int i = 0; i < NDim(); ++i) {
      oss << DimAt(i) << " ";
    }

    oss << "(" << NDim() << ")";

    return oss.str();
  }

  /// \brief Returns the size in bytes of an individual element.
  ///
  /// \return The size of an element in bytes.
  constexpr size_t ElementSize() const { return sizeof(Dtype); }

  /// \brief The dimension of the index-th axis (or the negative index-th axis
  ///        from the end, if index is negative).
  ///
  /// \param index The dimension index.
  /// \return The dimension at the specified index.
  constexpr int DimAt(int index) const {
    return shape_.DimAt(CanonicalDimIndex(index));
  }

  /// \brief Returns the number of dimensions in the Blob.
  ///
  /// \return The number of dimensions.
  constexpr int NDim() const { return shape_.NDim(); }

  /// \brief Returns the total number of elements in the Blob.
  ///
  /// \return The total number of elements.
  int NumElement() const { return shape_.numel(); }

  /// \brief Transfers the data and the diff to the GPU.
  void ToGPU() const {
    CHECK_NOTNULL(data_);

    data_->ToGPU();

    if (diff_) {
      diff_->ToGPU();
    }
  }

  /// \brief Serializes the Blob data and the diff into a protocol buffer
  ///        message.
  ///
  /// This method converts the Blob's data (and optionally the diff) into a
  /// protocol buffer message (`BlobProto`). This is typically used for saving
  /// the Blob's state to disk or transmitting it over a network.
  ///
  /// \param proto The protocol buffer message to store the serialized data.
  /// \param write_diff If true, the diff (gradients) will also be serialized;
  ///                   otherwise, only data is serialized.
  void ToProto(BlobProto& proto, bool write_diff = false) const;

 private:
  /// \brief Returns the 'canonical' version of a (usually) user-specified axis,
  ///        allowing for negative indexing (e.g., -1 for the last axis).
  ///
  /// \param dim_index The dimension index.
  /// \return The specified index if 0 <= index < NumDims().
  ///         If -NumDims() <= index <= -1, return (NumDims() - (-index)).
  int CanonicalDimIndex(int dim_index) const {
    const int max_dims = NDim();

    CHECK(-max_dims <= dim_index && dim_index < max_dims)
        << "axis " << dim_index << " out of range for " << max_dims
        << "-D Blob with shape " << ShapeString();

    return (dim_index + max_dims) % max_dims;
  }

  /// \brief Copies the data stored in the Blob to a destination buffer.
  ///
  /// \param dst A pointer to the destination buffer where the data will be
  ///            copied.
  void CopyDataTo(Dtype* dst) const {
    auto data = GetData();

    std::copy_n(data, NumElement(), dst);
  }

  /// \brief Copies the diff (gradients) stored in the Blob to a destination
  ///        buffer.
  ///
  /// \param dst A pointer to the destination buffer where the diff will be
  ///            copied.
  void CopyDiffTo(Dtype* dst) const {
    auto diff = GetDiff();

    std::copy_n(diff, NumElement(), dst);
  }

  /// \brief Returns a pointer to the data in SyncedMemory based on its current
  ///        state (CPU or GPU).
  ///
  /// \param sync_mem A pointer to the SyncedMemory object.
  /// \return A pointer to the data (CPU or GPU).
  static auto GetSyncedMemoryPtr(SyncedMemory* sync_mem) {
    CHECK_NOTNULL(sync_mem);

    if (sync_mem->IsOnGPU()) {
      return sync_mem->gpu_ptr();
    } else {
      return sync_mem->cpu_ptr();
    }
  }

  /// \brief Returns a const pointer to the data (read-only).
  ///
  /// \return A const pointer to the data.
  auto GetData() const {
    CHECK_NOTNULL(data_);

    return static_cast<const Dtype*>(GetSyncedMemoryPtr(data_.get()));
  }

  /// \brief Returns a mutable pointer to the data (read-write).
  ///
  /// \return A mutable pointer to the data.
  auto GetMutableData() {
    CHECK_NOTNULL(data_);

    return static_cast<Dtype*>(GetSyncedMemoryPtr(data_.get()));
  }

  /// \brief Returns a const pointer to the diff (gradients, read-only).
  ///
  /// \return A const pointer to the diff.
  auto GetDiff() const {
    CHECK_NOTNULL(diff_);

    return static_cast<const Dtype*>(GetSyncedMemoryPtr(diff_.get()));
  }

  /// \brief Returns a mutable pointer to the diff (gradients, read-write).
  ///
  /// \return A mutable pointer to the diff.
  auto GetMutableDiff() {
    CHECK_NOTNULL(diff_);

    return static_cast<Dtype*>(GetSyncedMemoryPtr(diff_.get()));
  }

  /// \brief Copies the shape dimensions of the Blob to a destination buffer.
  ///
  /// \param dst A pointer to the destination buffer where the shape dimensions
  ///            will be copied.
  void CopyShapeTo(int* dst) const { std::copy_n(shape_.begin(), NDim(), dst); }

  std::shared_ptr<SyncedMemory> data_;  ///< Pointer to the data memory.
  std::shared_ptr<SyncedMemory> diff_;  ///< Pointer to the diff memory.
  Shape shape_;                         ///< Shape of the blob.
};

}  // namespace caffe