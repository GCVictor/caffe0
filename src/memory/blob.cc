// Copyright 2013 Yangqing Jia

#include "caffe/memory/blob.hh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cassert>

#include "caffe/memory/synced_mem.hh"
#include "caffe/util/math_functions.hh"

namespace caffe {

template <typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
                  const int width) {
  Reshape(num, channels, height, width);
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
                          const int width) {
  CHECK_GE(num, 0);
  CHECK_GE(channels, 0);
  CHECK_GE(height, 0);
  CHECK_GE(width, 0);

  num_ = num;
  channels_ = channels;
  height_ = height;
  width_ = width;
  count_ = num_ * channels_ * height_ * width_;

  if (count_) {
    auto size = count_ * sizeof(Dtype);
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
  } else {
    data_.reset(nullptr);
    diff_.reset(nullptr);
  }
}

template <typename Dtype>
const Dtype* Blob<Dtype>::CpuData() const {
  CHECK(data_);

  return static_cast<const Dtype*>(data_->CpuData());
}

template <typename Dtype>
const Dtype* Blob<Dtype>::GpuData() const {
  CHECK(data_);

  return static_cast<const Dtype*>(data_->gpu_data());
}

template <typename Dtype>
const Dtype* Blob<Dtype>::CpuDiff() const {
  CHECK(diff_);

  return static_cast<const Dtype*>(diff_->CpuData());
}

template <typename Dtype>
const Dtype* Blob<Dtype>::GpuDiff() const {
  CHECK(diff_);

  return static_cast<const Dtype*>(diff_->GpuData());
}

template <typename Dtype>
Dtype* Blob<Dtype>::MutableCpuData() {
  CHECK(data_);

  return static_cast<Dtype*>(data_->MutableCpuData());
}

template <typename Dtype>
Dtype* Blob<Dtype>::MutableGpuData() {
  CHECK(data_);

  return static_cast<Dtype*>(data_->MutableGpuData());
}

template <typename Dtype>
Dtype* Blob<Dtype>::MutableCpuDiff() {
  CHECK(diff_);

  return static_cast<Dtype*>(diff_->MutableCpuData());
}

template <typename Dtype>
Dtype* Blob<Dtype>::MutableGpuDiff() {
  CHECK(diff_);

  return static_cast<Dtype*>(diff_->MutableGpuData());
}

template <typename Dtype>
void Blob<Dtype>::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
    case SyncedMemory::HEAD_AT_CPU:
      // Perform computation on CPU.
      cpu::caffe_axpy<Dtype>(count_, static_cast<Dtype>(-1),
                             static_cast<const Dtype*>(diff_->CpuData()),
                             static_cast<Dtype*>(data_->MutableCpuData()));
      break;
    case SyncedMemory::HEAD_AT_GPU:
    case SyncedMemory::SYNCED:
      // Perform computation on GPU.
      gpu::caffe_axpy<Dtype>(count_, static_cast<Dtype>(-1),
                             static_cast<const Dtype*>(diff_->GpuData()),
                             static_cast<Dtype*>(data_->MutableGpuData()));
      break;
    default:
      LOG(FATAL) << "Syncedmem not initialized.";
  }
}

template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
  if (num_ != source.num() || channels_ != source.channels() ||
      height_ != source.height() || width_ != source.width()) {
    if (reshape) {
      Reshape(source.num(), source.channels(), source.height(), source.width());
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  switch (Caffe::mode()) {
    case Caffe::GPU:
      if (copy_diff) {
        CUDA_CHECK(cudaMemcpy(diff_->mutable_gpu_data(), source.gpu_diff(),
                              sizeof(Dtype) * count_,
                              cudaMemcpyDeviceToDevice));
      } else {
        CUDA_CHECK(cudaMemcpy(data_->mutable_gpu_data(), source.gpu_data(),
                              sizeof(Dtype) * count_,
                              cudaMemcpyDeviceToDevice));
      }
      break;
    case Caffe::CPU:
      if (copy_diff) {
        memcpy(diff_->mutable_cpu_data(), source.cpu_diff(),
               sizeof(Dtype) * count_);
      } else {
        memcpy(data_->mutable_cpu_data(), source.cpu_data(),
               sizeof(Dtype) * count_);
      }
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto) {
  Reshape(proto.num(), proto.channels(), proto.height(), proto.width());
  // copy data
  Dtype* data_vec = mutable_cpu_data();
  for (int i = 0; i < count_; ++i) {
    data_vec[i] = proto.data(i);
  }
  if (proto.diff_size() > 0) {
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.diff(i);
    }
  }
}

template <typename Dtype>
void Blob<Dtype>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->set_num(num_);
  proto->set_channels(channels_);
  proto->set_height(height_);
  proto->set_width(width_);
  proto->clear_data();
  proto->clear_diff();
  const Dtype* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }
  if (write_diff) {
    const Dtype* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}

INSTANTIATE_CLASS(Blob);

}  // namespace caffe