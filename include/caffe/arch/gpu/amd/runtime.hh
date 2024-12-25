#pragma once

#include <hip/hip_runtime.h>

namespace caffe::gpu {

/// Allocate device accessible page locked host memory [Deprecated].
///
/// Note: If size is 0, no memory is allocated, *ptr returns nullptr.
///
/// \param ptr Pointer to the allocated host pinned memory
/// \param size Requested memory size in bytes
/// \param flags Type of host memory allocation
void caffe_malloc_host(void **ptr, size_t size, unsigned int flags) {
  return HIP_CHECK(hipMallocHost(ptr, size, flags));
}

/// Copy data from src to dst.
///
/// It supports memory from host to device, device to host, device to device and
/// host to host The src and dst **must not overlap**.
///
/// For hipMemcpy, the copy is always performed by the current device (set by
/// hipSetDevice). For multi-gpu or peer-to-peer configurations, it is
/// recommended to set the current device to the device where the src data is
/// physically located. For optimal peer-to-peer copies, the copy device must be
/// able to access the src and dst pointers (by calling
/// hipDeviceEnablePeerAccess with copy agent as the current device and src/dest
/// as the peerDevice argument. if this is not done, the hipMemcpy will still
/// work, but will perform the copy using a staging buffer on the host. Calling
/// hipMemcpy with dst and src pointers that do not match the hipMemcpyKind
/// results in **undefined behavior**.
///
/// \param dst Data being copy to.
/// \param src Data being copy from.
/// \param count Data size in bytes.
/// \param kind Kind of transfer.
void caffe_memcpy(void *dst, const void *src, size_t count,
                  hipMemcpyKind kind) {
  return HIP_CHECK(hipMemcpy(dst, src, count, kind));
}

/// Fills the first sizeBytes bytes of the memory area pointed to by dest with
/// the constant byte value value.
///
/// \param dst Data being filled.
/// \param value Value to be set.
/// \param size_bytes Data size in bytes.
void caffe_memset(void *dst, int value, size_t size_bytes) {
  return HIP_CHECK(hipMemset(dst, value, size_bytes));
}

}  // namespace caffe::gpu