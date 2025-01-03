#pragma once

#include "caffe/common/blob.hh"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/// \brief An interface for the units of computation which can be composed into
///        a Net.
template <typename Dtype>
class Layer {
 public:
  explicit Layer(const LayerParameter& param) : layer_param_{param} {
    phase_ = param.phase();
  }

  virtual ~Layer() = default;

 protected:
  /// The protobuf that stores the layer parameters.
  LayerParameter layer_param_;
  /// The phase: TRAIN or TEST.
  Phase phase_;
  /// The vector that stores the learnable parameters as a set of blobs.
  std::vector<std::shared_ptr<Blob<Dtype> > > blobs_;
  /// Vector indicating whether to compute the diff of each param blob.
  std::vector<bool> param_propagate_down_;
  /// The vector that indicates whether each top blob has a non-zero weight in
  /// the objective function.
  std::vector<Dtype> loss_;
};

}  // namespace caffe