#include "caffe/arch/runtime.hh"
#include "caffe/common/synced_mem.hh"

namespace caffe {

SyncedMemory::SyncedMemory() : SyncedMemory(0) {}

SyncedMemory::SyncedMemory(size_t size)
    : cpu_ptr_{}, gpu_ptr_{}, size_{size}, head_{kUninitialized} {
#ifndef CPU_ONLY
#ifdef DEBUG
  gpu::caffe_get_device(&device_);
#endif
#endif
}

SyncedMemory::~SyncedMemory() {
  CheckDevice();

  if (cpu_ptr_) {
    caffe_free_host(cpu_ptr_);
  }

#ifndef CPU_ONLY
  if (gpu_ptr_) {
    gpu::caffe_free_device(gpu_ptr_);
  }
#endif
}

void SyncedMemory::ToCPU() {
  CheckDevice();

  switch (head_) {
    case kUninitialized:
      caffe_malloc_host(&cpu_ptr_, size_);
      // TODO(gc): is it necessary to do memset?
      head_ = kAtCPU;
      break;

    case kAtGPU:
#ifndef CPU_ONLY
      if (nullptr == cpu_ptr_) {
        caffe_malloc_host(&cpu_ptr_, size_);
      }

      gpu::caffe_memcpy(cpu_ptr_, gpu_ptr_, size_);
      head_ = kSynced;
#else
      REPORT_GPU_NOT_SUPPORTED();
#endif

    case kAtCPU:
      break;

    case kSynced:
      break;

    default:
      break;
  }
}

void SyncedMemory::ToGPU() {
  CheckDevice();

#ifndef CPU_ONLY
  switch (head_) {
    case kUninitialized:
      gpu::caffe_malloc_device(&gpu_ptr_, size_);
      // TODO(gc): memset
      head_ = kAtGPU;
      break;

    case kAtCPU:
      if (nullptr == gpu_ptr_) {
        gpu::caffe_malloc_device(&gpu_ptr_, size_);
      }

      gpu::caffe_memcpy(gpu_ptr_, cpu_ptr_, size_);
      head_ = kSynced;
      break;

    case kAtGPU:
      break;

    case kSynced:
      break;

    default:
      break;
  }
#else
  REPORT_GPU_NOT_SUPPORTED();
#endif
}

void SyncedMemory::CheckDevice() const {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  gpu::caffe_get_device(&device);
  CHECK(device == device_);

  if (gpu_ptr_) {
    gpu::caffe_get_device_from_pointer(gpu_ptr_, &device);
    CHECK(device == device_);
  }
#endif
#endif
}

}  // namespace caffe