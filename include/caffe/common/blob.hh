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

      // Case 2: no -1 is found (more -1s are found) or the remaining size is
      // not divisible.
      if (std::count(other.DimBegin(), other.DimEnd(), -1) != 1 || rem != 0) {
        return false;
      }

      dims_ = other.dims_;
      auto it = std::find(DimBegin(), DimEnd(), -1);
      *it = -quot;

      return true;
    }

    /// \brief Checks if the shape is valid (i.e., all dimensions are positive).
    ///
    /// \return True if all dimensions are positive; otherwise false.
    constexpr bool IsValid() const {
      return std::all_of(DimBegin(), DimEnd(), [](int dim) { return dim > 0; });
    }

    /// \return The total number of elements in this shape.
    constexpr int numel() const { return num_el_; }

    auto DimBegin() const { return dims_.begin(); }
    auto DimEnd() const { return dims_.end(); }

    /// \return The number of dimensions in this shape.
    constexpr int NDim() const { return dims_.size(); }

    /// \brief Get the dimension at the specified index.
    ///
    /// \param index The index of the dimension to retrieve.
    /// \return The dimension at the specified index.
    constexpr int DimAt(size_t index) const {
      CHECK_LE(index, NDim());

      return dims_[index];
    }

   private:
    /// \return The total product of the dimensions.
    constexpr int ComputeNumel() const {
      return std::accumulate(DimBegin(), DimEnd(), 1, std::multiplies<int>{});
    }

    int num_el_;
    std::vector<int> dims_;
  };

 public:
  static Blob<Dtype> FromProto(const BlobProto& proto) {
    auto requires_diff = proto.double_diff_size() > 0 || proto.diff_size() > 0;
    Blob<Dtype> blob{proto.shape(), requires_diff};

    // Copy data.
    auto count = blob.TotalDimProduct();
    auto dims_vec = static_cast<Dtype*>(blob.dims_->mutable_cpu_ptr());

    if (proto.double_data_size() > 0) {
      CHECK_EQ(count, proto.double_data_size());

      // TODO(gc): use simd to accelerate.
      for (int i = 0; i < count; ++i) {
        dims_vec[i] = proto.double_data(i);
      }
    } else {
      CHECK_EQ(count, proto.data_size());

      for (int i = 0; i < count; ++i) {
        dims_vec[i] = proto.data(i);
      };
    }

    // Copy diff.
    if (proto.double_diff_size()) {
      CHECK_EQ(count, proto.double_diff_size());

      auto diff_vec = static_cast<Dtype*>(blob.diff_->mutable_cpu_ptr());

      for (int i = 0; i < count; ++i) {
        diff_vec[i] = proto.double_diff(i);
      }
    } else if (proto.diff_size()) {
      CHECK_EQ(count, proto.diff_size());

      auto diff_vec = static_cast<Dtype*>(blob.diff_->mutable_cpu_ptr());

      for (int i = 0; i < count; ++i) {
        diff_vec[i] = proto.diff(i);
      }
    }

    return blob;
  }

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

  std::string ShapeString() const {
    std::ostringstream oss;

    for (int i = 0; i < NDim(); ++i) {
      oss << DimAt(i) << " ";
    }

    oss << "(" << NDim() << ")";

    return oss.str();
  }

  /// \return The size in bytes of an individual element.
  constexpr size_t ElementSize() const { return sizeof(Dtype); }

  /// \brief The dimension of the index-th axis (or the negative index-th axis
  ///        from the end, if index is negative).
  ///
  /// \param index The dimension index.
  /// \return The dimension at the specified index.
  constexpr int DimAt(int index) const {
    return shape_.DimAt(CanonicalDimIndex(index));
  }

  constexpr int NDim() const { return shape_.NDim(); }
  int NumElement() const { return shape_.numel(); }

  void ToProto(BlobProto& proto, bool write_diff = false) const {}

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

  void CopyDataTo(Dtype* dst) const {
    auto cpu_data = CpuData();

    std::copy_n(cpu_data, NumElement(), dst);
  }

  void CopyDiffTo(Dtype* dst) const {
    auto cpu_diff = CpuDiff();

    std::copy_n(cpu_diff, NumElement(), dst);
  }

  void CopyShapeTo(int* dst) const {
    std::copy_n(shape_.DimBegin(), NDim(), dst);
  }

  const Dtype* CpuData() const {
    CHECK(data_);

    return static_cast<const Dtype*>(data_->cpu_ptr());
  }

  Dtype* MutableCpuData() {
    CHECK(data_);

    return static_cast<Dtype*>(data_->mutable_cpu_ptr());
  }

  const Dtype* CpuDiff() const {
    CHECK(diff_);

    return static_cast<const Dtype*>(diff_->cpu_ptr());
  }

  Dtype* MutableCpuDiff() {
    CHECK(diff_);

    return static_cast<Dtype*>(diff_->mutable_cpu_ptr());
  }

  const Dtype* GpuData() const {
    CHECK(data_);

    return static_cast<const Dtype*>(data_->gpu_ptr());
  }

  Dtype* MutableGpuData() {
    CHECK(data_);

    return static_cast<Dtype*>(data_->mutable_gpu_ptr());
  }

  const Dtype* GpuDiff() const {
    CHECK(diff_);

    return static_cast<const Dtype*>(diff_->gpu_ptr());
  }

  Dtype* MutableGpuDiff() {
    CHECK(diff_);

    return static_cast<Dtype*>(diff_->mutable_gpu_ptr());
  }

  std::shared_ptr<SyncedMemory> data_;
  std::shared_ptr<SyncedMemory> diff_;
  Shape shape_;
};

}  // namespace caffe