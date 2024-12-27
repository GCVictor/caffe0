#include "caffe/common/blob.hh"

namespace caffe {

inline constexpr int kMaxTensorDims = 32;

template <>
void Blob<double>::ToProto(BlobProto& proto, bool write_diff) const {
  auto dim_data = proto.mutable_shape()->mutable_dim();
  dim_data->Resize(NumDims(), {});
  CopyShape(dim_data->begin());

  auto count = TotalDimProduct();
  proto.mutable_double_data()->Resize(count, {});
  CopyData(proto.mutable_double_data()->begin());

  if (write_diff) {
    proto.mutable_double_diff()->Resize(count, {});
    CopyDiff(proto.mutable_double_diff()->begin());
  }
}

template <>
void Blob<float>::ToProto(BlobProto& proto, bool write_diff) const {
  auto dim_data = proto.mutable_shape()->mutable_dim();
  dim_data->Resize(NumDims(), {});
  CopyShape(dim_data->begin());

  auto count = TotalDimProduct();
  proto.mutable_data()->Resize(count, {});
  CopyData(proto.mutable_data()->begin());

  if (write_diff) {
    proto.mutable_diff()->Resize(count, {});
    CopyDiff(proto.mutable_diff()->begin());
  }
}

}  // namespace caffe