#include "caffe/common/blob.hh"

namespace caffe {

inline constexpr int kMaxTensorDims = 32;

// void Tensor::Reshape(const TensorShape& shape) {
//   CHECK_LE(shape.NumDimensions(), kMaxTensorDims);
//   CHECK(shape_.ReshapeTo(shape));
// }

template <>
void Blob<double>::ToProto(BlobProto& proto, bool write_diff) const {
  proto.clear_shape();

  for (int i = 0; i < shape_.size(); ++i) {
    proto.mutable_shape()->add_dim(shape_.DimAt(i));
  }

  proto.clear_double_data();
  proto.clear_double_diff();

  auto data_vec = CpuData();

  for (int i = 0; i < count_; ++i) {
    proto->add_double_data(data_vec[i]);
  }

  if (write_diff) {
    auto diff_vec = CpuDiff();

    for (int i = 0; i < count_; ++i) {
      proto->add_double_diff(diff_vec[i]);
    }
  }
}

template <>
void Blob<float>::ToProto(BlobProto& proto, bool write_diff) const {
  proto->clear_shape();

  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }

  proto->clear_data();
  proto->clear_diff();

  auto data_vec = CpuData();

  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }

  if (write_diff) {
    auto diff_vec = CpuDiff();

    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}

}  // namespace caffe