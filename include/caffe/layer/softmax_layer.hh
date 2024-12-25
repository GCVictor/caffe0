// Copyright 2013 Yangqing Jia

#pragma once

#include "caffe/nn/layer.hh"

namespace caffe {

template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxLayer(const LayerParameter& param) : Layer<Dtype>{param} {}

  virtual void SetUp(const std::vector<Blob<Dtype>*>& bottom,
                     std::vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
                             const bool propagate_down,
                             vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
                             const bool propagate_down,
                             vector<Blob<Dtype>*>* bottom);

  /// sum_multiplier is just used to carry out sum using blas
  Blob<Dtype> sum_multiplier_;
  /// scale is an intermediate blob to hold temporary results.
  Blob<Dtype> scale_;
};

}  // namespace caffe