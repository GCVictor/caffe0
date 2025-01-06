#pragma once

#if defined(HAVE_AVX)
#include <immintrin.h>
#elif defined(HAVE_SSE)
#include <emmintrin.h>
#endif

namespace caffe {
namespace cpu {

#if defined(HAVE_AVX)

void* caffe_memset(void* dst, int value, size_t count) {
  if (NULL == dst || 0 == count) {
    return dst;
  }

  uint8_t byte_value = (uint8_t)value;
  __m256i avx_value = _mm256_set1_epi8(byte_value);

  uint8_t* p = (uint8_t*)dst;

  while (((uintptr_t)p % 32) != 0 && count > 0) {
    *p++ = byte_value;
    count--;
  }

  size_t avx_count = count / 32;

  for (size_t i = 0; i < avx_count; i++) {
    _mm256_store_si256((__m256i*)p, avx_value);
    p += 32;
  }

  count %= 32;

  while (count > 0) {
    *p++ = byte_value;
    count--;
  }

  return dst;
}

#elif defined(HAVE_SSE)

/// __m128i _mm_set1_epi8(char a)
/// Operation FOR j := 0 to 15
///     i := j*8
///     dst[i+7:i] := a[7:0] ENDFOR
///
/// Reference:
/// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_set1_epi8&ig_expand=5801
void* caffe_memset(void* dst, int value, size_t count) {
  if (NULL == dst || 0 == count) {
    return dst;
  }

  uint8_t byte_value = (uint8_t)value;
  __m128i simd_value = _mm_set1_epi8(byte_value);
  uint8_t* p = (uint8_t*)dst;

  while (((uintptr_t)p % 16) != 0 && count > 0) {
    *p++ = byte_value;
    count--;
  }

  size_t simd_count = count / 16;
  for (size_t i = 0; i < simd_count; i++) {
    _mm_store_si128((__m128i*)p, simd_value);
    p += 16;
  }

  count %= 16;

  while (count > 0) {
    *p++ = byte_value;
    count--;
  }

  return dst;
}

#else

#endif

}  // namespace cpu
}  // namespace caffe