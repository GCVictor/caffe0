#pragma once

#include <initializer_list>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

#include "caffe.pb.h"
#include "caffe/common/synced_mem.hh"
#include "caffe/util/macros.hh"

namespace caffe {

template <bool... bs>
struct all_of {
  static constexpr bool value = (bs && ...);
};

template <typename Dtype>
class Blob {
 public:
  class Shape {
   public:
    template <typename... Dims,
              typename =
                  std::enable_if_t<all_of<std::is_same_v<int, Dims>...>::value>>
    Shape(Dims&&... dims) : data_{std::forward<Dims>(dims)...} {}

    explicit Shape(std::initializer_list<int> dims) : data_{dims} {}
    explicit Shape(const std::vector<int>& dims) : data_{dims} {}

    constexpr int DimAt(size_t index) const {
      CHECK_LE(index, data_.size());

      return data_[index];
    }

    /// \return The number of dimensions (axes) of the tensor.
    constexpr int NumDimensions() const { return data_.size(); }

    /// \return The total product of dimensions.
    constexpr int GetTotalDimProduct() const {
      return std::accumulate(data_.begin(), data_.end(), 1,
                             std::multiplies<int>{});
    }

    constexpr bool HasEqualSize(const Shape& other) const {
      return GetTotalSize() == other.GetTotalSize();
    }

    bool ReshapeTo(const Shape& other) {
      auto ssize = GetTotalDimProduct();
      auto osize = other.GetTotalDimProduct();

      // Case 1: self and other sizes are matched.
      if (ssize == osize) {
        data_ = other.data_;

        return true;
      }

      // Case 2: no -1 is found (more -1s are found) or the remaining size is
      // not
      //         divisible.
      if (std::count(other.data_.begin(), other.data_.end(), -1) != 1 ||
          ssize % osize != 0) {
        return false;
      }

      data_ = other.data_;
      auto it = std::find(data_.begin(), data_.end(), -1);
      *it = ssize / osize;

      return true;
    }

    constexpr bool IsValid() const {
      return std::all_of(data_.begin(), data_.end(),
                         [](int dim) { return dim > 0; });
    }

   private:
    std::vector<int> data_;
  };

  Blob(const Shape& shape, bool requires_diff = false) : shape_{shape} {
    CHECK(shape.IsValid());

    count_ = shape.GetTotalDimProduct();
    auto nbytes = count_ * sizeof(Dtype);
    data_ = std::make_shared<SyncedMemory>(nbytes);
    shape_data_ =
        std::make_shared<SyncedMemory>(shape.NumDimensions() * sizeof(int));

    if (requires_diff) {
      diff_ = std::make_shared<SyncedMemory>(nbytes);
    }
  }

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
    auto data_vec = static_cast<Dtype*>(blob.data_->mutable_cpu_ptr());

    if (proto.double_data_size() > 0) {
      CHECK_EQ(count_, proto.double_data_size());

      // TODO(gc): use simd to accelerate.
      for (int i = 0; i < count_; ++i) {
        data_vec[i] = proto.double_data(i);
      }
    } else {
      CHECK_EQ(count_, proto.data_size());

      for (int i = 0; i < count_; ++i) {
        data_vec[i] = proto.data(i)
      };
    }

    // Copy diff.
    if (proto.double_diff_size()) {
      CHECK_EQ(count_, proto.double_diff_size());

      auto diff_vec = static_cast<Dtype*>(blob.diff_->mutable_cpu_ptr());

      for (int i = 0; i < count_; ++i) {
        diff_vec[i] = proto.double_diff(i);
      }
    } else if (proto.diff_size()) {
      CHECK_EQ(count_, proto.diff_size());

      auto diff_vec = static_cast<Dtype*>(blob.diff_->mutable_cpu_ptr());

      for (int i = 0; i < count_; ++i) {
        diff_vec[i] = proto.diff(i);
      }
    }

    return blob;
  }

  void ToProto(BlobProto& proto, bool write_diff = false) const {}

 private:
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
  std::shared_ptr<SyncedMemory> shape_data_;
  Shape shape_;
  int count_;
  // int capacity_;
};

}  // namespace caffe