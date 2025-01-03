#include "caffe/common/blob.hh"

namespace caffe {

template <>
void Blob<double>::ToProto(BlobProto& proto, bool write_diff) const {
  auto dim_data = proto.mutable_shape()->mutable_dim();
  dim_data->Resize(NDim(), {});
  CopyShapeTo(dim_data->begin());

  auto numel = NumElement();
  proto.mutable_double_data()->Resize(numel, {});
  CopyDataTo(proto.mutable_double_data()->begin());

  if (write_diff) {
    proto.mutable_double_diff()->Resize(numel, {});
    CopyDiffTo(proto.mutable_double_diff()->begin());
  }
}

template <>
void Blob<float>::ToProto(BlobProto& proto, bool write_diff) const {
  auto dim_data = proto.mutable_shape()->mutable_dim();
  dim_data->Resize(NDim(), {});
  CopyShapeTo(dim_data->begin());

  auto numel = NumElement();
  proto.mutable_data()->Resize(numel, {});
  CopyDataTo(proto.mutable_data()->begin());

  if (write_diff) {
    proto.mutable_diff()->Resize(numel, {});
    CopyDiffTo(proto.mutable_diff()->begin());
  }
}

}  // namespace caffe