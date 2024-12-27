#pragma once

#include <glog/logging.h>

#include <initializer_list>
#include <memory>
#include <numeric>
#include <vector>

#include "caffe.pb.h"
#include "caffe/common/synced_mem.hh"

namespace caffe {

template <typename Dtype>
class Blob {
 public:
  class Shape {
   public:
    explicit Shape(std::initializer_list<int> dims) : dims_{dims} {
      total_dim_ = ComputeTotalDimProduct();
    }

    explicit Shape(const std::vector<int>& dims) : dims_{dims} {
      total_dim_ = ComputeTotalDimProduct();
    }

    constexpr int DimAt(size_t index) const {
      CHECK_LE(index, dims_.size());

      return dims_[index];
    }

    /// \return The number of dimensions (axes) of the tensor.
    constexpr int NumDimensions() const { return dims_.size(); }

    /// \return The total product of dimensions.
    constexpr int TotalDimProduct() const { return total_dim_; }

    constexpr bool HasEqualSize(const Shape& other) const {
      return TotalDimProduct() == other.TotalDimProduct();
    }

    bool ReshapeTo(const Shape& other) {
      auto ssize = TotalDimProduct();
      auto osize = other.TotalDimProduct();

      // Case 1: self and other sizes are matched.
      if (ssize == osize) {
        dims_ = other.dims_;

        return true;
      }

      // Case 2: no -1 is found (more -1s are found) or the remaining size is
      // not
      //         divisible.
      if (std::count(other.dims_.begin(), other.dims_.end(), -1) != 1 ||
          ssize % osize != 0) {
        return false;
      }

      dims_ = other.dims_;
      auto it = std::find(dims_.begin(), dims_.end(), -1);
      *it = ssize / osize;

      return true;
    }

    constexpr bool IsValid() const {
      return std::all_of(dims_.begin(), dims_.end(),
                         [](int dim) { return dim > 0; });
    }

    const std::vector<int>& dims() const { return dims_; }

   private:
    constexpr int ComputeTotalDimProduct() const {
      return std::accumulate(dims_.begin(), dims_.end(), 1,
                             std::multiplies<int>{});
    }

    int total_dim_;
    std::vector<int> dims_;
  };

  /// \brief Reshape the tensor with the specified shape.
  ///
  /// Note the shape size must be matched with the original shape size. A
  /// single dimension may be -1, in which case it’s inferred from the
  /// remaining dimensions
  ///
  /// \param shape
  // void Reshape(const TensorShape& shape);

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

  explicit Blob(const Shape& shape, bool requires_diff = true) : shape_{shape} {
    CHECK(shape.IsValid());

    auto count = shape.TotalDimProduct();
    auto nbytes = count * sizeof(Dtype);
    data_ = std::make_shared<SyncedMemory>(nbytes);

    if (requires_diff) {
      diff_ = std::make_shared<SyncedMemory>(nbytes);
    }
  }

  constexpr int NumDims() const { return shape_.NumDimensions(); }
  constexpr int TotalDimProduct() const { return shape_.TotalDimProduct(); }

  void ToProto(BlobProto& proto, bool write_diff = false) const {}

 private:
  void CopyData(Dtype* dst) const {
    auto cpu_data = CpuData();

    std::copy_n(cpu_data, TotalDimProduct(), dst);
  }

  void CopyDiff(Dtype* dst) const {
    auto cpu_diff = CpuDiff();

    std::copy_n(cpu_diff, TotalDimProduct(), dst);
  }

  void CopyShape(int* dst) const {
    std::copy_n(shape_.dims().begin(), NumDims(), dst);
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