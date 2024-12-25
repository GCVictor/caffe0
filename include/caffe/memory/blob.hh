#pragma once

#include <memory>

#include "caffe/common/macros.hh"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

class SyncedMemory;

template <typename DType>
class Blob {
  DISALLOW_COPY_AND_ASSIGN(Blob);

 public:
  using DataType = DType;

  Blob() = default;
  Blob(const int num, const int channels, const int height, const int width);
  virtual ~Blob();

  constexpr int num() const { return num_; }
  constexpr int height() const { return height_; }
  constexpr int width() const { return width_; }
  constexpr int channels() const { return channels_; }
  constexpr int count() const { return count_; }

  const Dtype* CpuData() const;
  const Dtype* GpuData() const;
  const Dtype* CpuDiff() const;
  const Dtype* GpuDiff() const;
  Dtype* MutableCpuData();
  Dtype* MutableGpuData();
  Dtype* MutableCpuDiff();
  Dtype* MutableGpuDiff();

  constexpr int Offset(const int n, const int c = 0;
                       const int h = 0, const int w = 0) const {
    return ((n * channels_ + c) * height_ + h) * width_ + w;
  }

  Dtype DataAt(const int n, const int c, const int h, const int w) const {
    return *(CpuData() + Offset(n, c, h, w));
  }

  Dtype DiffAt(const int n, const int c, const int h, const int w) const {
    return *(CpuDiff() + Offset(n, c, h, w));
  }

  void Update();
  void FromProto(const BlobProto& proto);
  void ToProto(BlobProto& proto, bool write_diff = false) const;
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
                bool reshape = false);

 private:
  int num_{};
  int channels_{};
  int height_{};
  int width_{};
  int count_{};

  std::shared_ptr<SyncedMemory> data_;
  std::shared_ptr<SyncedMemory> diff_;
};

}  // namespace caffe