#pragma once

#include <mkl_vsl.h>

#include "caffe/common/macros.hh"

namespace caffe {

class VSLStreamStatePtrRAII {
  DISALLOW_COPY_AND_ASSIGN(VSLStreamStatePtrRAII);

  friend VSLStreamStatePtr vsl_stream();

 private:
  VSLStreamStatePtrRAII() {
    if (vslNewStream(&vsl_stream_, VSL_BRNG_MT19937, 1701) != VSL_STATUS_OK) {
      LOG(ERROR) << "Cannot create vsl stream. VSL random number generator "
                 << "won't be available.";
    }
  }

  ~VSLStreamStatePtrRAII() {
    if (vsl_stream_) {
      VSL_CHECK(vslDeleteStream(&vsl_stream_));
    }
  }

  VSLStreamStatePtr vsl_stream_;
};

inline VSLStreamStatePtr vsl_stream() {
  VSLStreamStatePtrRAII singleton;

  return singleton.vsl_stream_;
}

}  // namespace caffe