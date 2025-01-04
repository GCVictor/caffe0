#pragma once

#include "caffe.pb.h"
#include "caffe/common/blob.hh"

namespace caffe {

/// \brief An interface for the units of computation which can be composed into
///        a Net.
/// Layer%s must implement a Forward function, in which they take their input
/// (bottom) Blob%s (if any) and compute their output Blob%s (if any).
/// They may also implement a Backward function, in which they compute the error
/// gradients with respect to their input Blob%s, given the error gradients with
/// their output Blob%s.
template <typename Dtype>
class Layer {
 public:
  explicit Layer(const LayerParameter& param) : layer_param_{param} {
    phase_ = param.phase();
    auto size = param.blobs_size();

    if (size > 0) {
      blobs_.reserve(size);

      for (auto i = 0; i < size; ++i) {
        blobs_.push_back(Blob<Dtype>::FromProto(param.blobs(i)));
      }
    }
  }

  virtual ~Layer() = default;

  /// \brief Implements common layer setup functionality.
  ///
  /// Checks that the number of bottom and top blobs is correct.
  /// Calls LayerSetUp to do special layer setup for individual layer types,
  /// followed by Reshape to set up sizes of top blobs and internal buffers.
  /// Sets up the loss weight multiplier blobs for any non-zero loss weights.
  /// This method may not be overridden.
  ///
  /// \param bottom The preshaped input blobs.
  /// \param top The allocated but unshaped output blobs, to be shaped by
  ///            Reshape.
  void SetUp(const vector<Blob<Dtype>*>& bottom,
             const vector<Blob<Dtype>*>& top) {
    CheckBlobCounts(bottom, top);
  }

  /// \brief Returns the exact number of bottom blobs required by the layer,
  ///        or -1 if no exact number is required.
  ///
  /// This method should be overridden to return a non-negative value if your
  /// layer expects some exact number of bottom blobs.
  virtual inline int ExactNumBottomBlobs() const { return -1; }

  /// \brief Returns the minimum number of bottom blobs required by the layer,
  ///        or -1 if no minimum number is required.
  ///
  /// This method should be overridden to return a non-negative value if your
  /// layer expects some minimum number of bottom blobs.
  virtual inline int MinBottomBlobs() const { return -1; }

  /// \brief Returns the maximum number of bottom blobs required by the layer,
  ///        or -1 if no maximum number is required.
  ///
  /// This method should be overridden to return a non-negative value if your
  /// layer expects some maximum number of bottom blobs.
  virtual inline int MaxBottomBlobs() const { return -1; }

  /// \brief Returns the exact number of top blobs required by the layer,
  ///        or -1 if no exact number is required.
  ///
  /// This method should be overridden to return a non-negative value if your
  /// layer expects some exact number of top blobs.
  virtual inline int ExactNumTopBlobs() const { return -1; }

  /// \brief Returns the minimum number of top blobs required by the layer,
  ///        or -1 if no minimum number is required.
  ///
  /// This method should be overridden to return a non-negative value if your
  /// layer expects some minimum number of top blobs.
  virtual inline int MinTopBlobs() const { return -1; }

  /// \brief Returns the maximum number of top blobs required by the layer,
  ///        or -1 if no maximum number is required.
  ///
  /// This method should be overridden to return a non-negative value if your
  /// layer expects some maximum number of top blobs.
  virtual inline int MaxTopBlobs() const { return -1; }

  /// Called by the parent Layer's SetUp to check that the number of bottom and
  /// top Blobs provided as input match the expected numbers specified by the
  /// {ExactNum,Min,Max}{Bottom,Top}Blobs() functions.
  ///
  /// \param bottom The preshaped input blobs.
  /// \param top The allocated but unshaped output blobs, to be shaped by
  ///            Reshape.
  virtual void CheckBlobCounts(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top) {
    if (ExactNumBottomBlobs() >= 0) {
      CHECK_EQ(ExactNumBottomBlobs(), bottom.size())
          << type() << " Layer takes " << ExactNumBottomBlobs()
          << " bottom blob(s) as input.";
    }

    if (MinBottomBlobs() >= 0) {
      CHECK_LE(MinBottomBlobs(), bottom.size())
          << type() << " Layer takes at least " << MinBottomBlobs()
          << " bottom blob(s) as input.";
    }

    if (MaxBottomBlobs() >= 0) {
      CHECK_GE(MaxBottomBlobs(), bottom.size())
          << type() << " Layer takes at most " << MaxBottomBlobs()
          << " bottom blob(s) as input.";
    }

    if (ExactNumTopBlobs() >= 0) {
      CHECK_EQ(ExactNumTopBlobs(), top.size())
          << type() << " Layer produces " << ExactNumTopBlobs()
          << " top blob(s) as output.";
    }

    if (MinTopBlobs() >= 0) {
      CHECK_LE(MinTopBlobs(), top.size())
          << type() << " Layer produces at least " << MinTopBlobs()
          << " top blob(s) as output.";
    }

    if (MaxTopBlobs() >= 0) {
      CHECK_GE(MaxTopBlobs(), top.size())
          << type() << " Layer produces at most " << MaxTopBlobs()
          << " top blob(s) as output.";
    }

    if (EqualNumBottomTopBlobs()) {
      CHECK_EQ(bottom.size(), top.size())
          << type() << " Layer produces one top blob as output for each "
          << "bottom blob input.";
    }
  }

 protected:
  /// The protobuf that stores the layer parameters.
  LayerParameter layer_param_;
  /// The phase: TRAIN or TEST.
  Phase phase_;
  /// The vector that stores the learnable parameters as a set of blobs.
  std::vector<Blob<Dtype> > blobs_;
  /// Vector indicating whether to compute the diff of each param blob.
  std::vector<bool> param_propagate_down_;
  /// The vector that indicates whether each top blob has a non-zero weight in
  /// the objective function.
  std::vector<Dtype> loss_;
};

}  // namespace caffe