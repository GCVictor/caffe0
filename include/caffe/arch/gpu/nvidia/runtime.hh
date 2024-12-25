#pragma once

namespace caffe::gpu {

/// ============================================================================///
/// 6.11. Memory Management
/// ============================================================================///

/// Frees memory on the device.
///
/// Frees the memory space pointed to by devPtr, which must have been returned
/// by a previous call to one of the following memory allocation APIs -
/// cudaMalloc(), cudaMallocPitch(), cudaMallocManaged(), cudaMallocAsync(),
/// cudaMallocFromPoolAsync().
///
/// Note - This API will not perform any implicit synchronization when the
/// pointer was allocated with cudaMallocAsync or cudaMallocFromPoolAsync.
/// Callers must ensure that all accesses to these pointer have completed before
/// invoking cudaFree. For best performance and memory reuse, users should use
/// cudaFreeAsync to free memory allocated via the stream ordered memory
/// allocator. For all other pointers, this API may perform implicit
/// synchronization.
///
/// If cudaFree(devPtr) has already been called before, an error is returned. If
/// devPtr is 0, no operation is performed. cudaFree() returns cudaErrorValue in
/// case of failure.
///
/// The device version of cudaFree cannot be used with a *devPtr allocated using
/// the host API, and vice versa.
///
/// \param dev_ptr Device pointer to memory to free.
inline void caffe_free_device(void* dev_ptr) { CUDA_CHECK(cudaFree(dev_ptr)); }

inline void caffe_free_host(void* ptr) { CUDA_CHECK(cudaFreeHost(ptr)); }

/// Allocate memory on the device.
///
/// Allocates size bytes of linear memory on the device and returns in *dev_ptr
/// a pointer to the allocated memory. The allocated memory is suitably aligned
/// for any kind of variable. The memory is not cleared. cudaMalloc() returns
/// cudaErrorMemoryAllocation in case of failure.
/// The device version of cudaFree cannot be used with a *devPtr allocated using
/// the host API, and vice versa.
///
/// \param dev_ptr Pointer to allocated device memory.
/// \param size Requested allocation size in bytes.
inline void caffe_malloc_device(void** dev_ptr, size_t size) {
  CUDA_CHECK(cudaMalloc(dev_ptr, size));
}

/// Allocates page-locked memory on the host.
///
/// Allocates size bytes of host memory that is page-locked and accessible to
/// the device. The driver tracks the virtual memory ranges allocated with this
/// function and automatically accelerates calls to functions such as
/// cudaMemcpy*(). Since the memory can be accessed directly by the device, it
/// can be read or written with much higher bandwidth than pageable memory
/// obtained with functions such as malloc().
/// On systems where pageableMemoryAccessUsesHostPageTables is true,
/// cudaMallocHost may not page-lock the allocated memory.
/// Page-locking excessive amounts of memory with cudaMallocHost() may degrade
/// system performance, since it reduces the amount of memory available to the
/// system for paging. As a result, this function is best used sparingly to
/// allocate staging areas for data exchange between host and device.
///
/// \param ptr Pointer to allocated host memory.
/// \param size Requested allocation size in bytes.
inline void caffe_malloc_host(void** ptr, size_t size, unsigned int flags) {
  IGNORE_VALUE(flags);

  CUDA_CHECK(cudaMallocHost(ptr, size));
}

/// Copies data between host and device.
///
/// Description:
/// Copies count bytes from the memory area pointed to by src to the memory area
/// pointed to by dst, where kind specifies the direction of the copy, and must
/// be one of
/// - cudaMemcpyHostToHost
/// - cudaMemcpyHostToDevice
/// - cudaMemcpyDeviceToHost
/// - cudaMemcpyDeviceToDevice
/// - cudaMemcpyDefault
/// Passing cudaMemcpyDefault is recommended, in which case the type of transfer
/// is inferred from the pointer values. However, cudaMemcpyDefault is only
/// allowed on systems that support unified virtual addressing. Calling
/// cudaMemcpy() with dst and src pointers that do not match the direction of
/// the copy results in an **undefined behavior**.
///
/// \param dst Destination memory address.
/// \param src Source memory address.
/// \param count Size in bytes to copy.
/// \param kind Type of transfer.
inline void caffe_memcpy(void* dst, const void* src, size_t count,
                         cudaMemcpyKind kind) {
  CUDA_CHECK(cudaMemcpy(dst, src, count, kind));
}

/// Initializes or sets device memory to a value.
///
/// Fills the first count bytes of the memory area pointed to by dev_ptr with
/// the constant byte value value.
///
/// Note that this function is **asynchronous** with respect to the host unless
/// dev_ptr refers to pinned host memory.
///
/// \param dev_ptr Pointer to device memory.
/// \param value Value to set for each byte of specified memory.
/// \param count Size in bytes to set.
inline void caffe_memset(void* dev_ptr, int value, size_t count) {
  CUDA_CHECK(cudaMemset(dev_ptr, value, count));
}

/// Returns which device is currently being used.
///
/// \param device Returns the device on which the active host thread executes
///               the device code.
inline void caffe_get_device(int* device) { CUDA_CHECK(cudaGetDevice(device)); }

/// Returns which device is currently being used associated with the specified
/// pointer.
///
/// \param dev_ptr Pointer to get attributes for.
/// \param device The device ID corresponding to the pointer.
inline void caffe_get_device_from_pointer(void* dev_ptr, int* device) {
  cudaPointerAttributes attributes;
  CUDA_CHECK(cudaPointerGetAttributes(&attributes, dev_ptr));
  *device = attributes.device;
}

}  // namespace caffe::gpu