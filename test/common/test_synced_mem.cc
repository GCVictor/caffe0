#include <gtest/gtest.h>

#include <cstring>
#include <memory>

#include "caffe/arch/runtime.hh"
#include "caffe/common/synced_mem.hh"

namespace caffe {

class SyncedMemoryTest : public ::testing::Test {};

TEST_F(SyncedMemoryTest, Initialization) {
  SyncedMemory mem(10);
  EXPECT_EQ(SyncedMemory::kUninitialized, mem.head());
  EXPECT_EQ(10, mem.size());

  auto sz = 10 * sizeof(float);
  auto p_mem = std::make_unique<SyncedMemory>(sz);
  EXPECT_EQ(sz, p_mem->size());
}

#ifndef CPU_ONLY

TEST_F(SyncedMemoryTest, AllocationToCPUAndGPU) {
  SyncedMemory mem(10);

  EXPECT_TRUE(nullptr != mem.cpu_ptr());
  EXPECT_TRUE(nullptr != mem.gpu_ptr());
  EXPECT_TRUE(nullptr != mem.mutable_cpu_ptr());
  EXPECT_TRUE(nullptr != mem.mutable_gpu_ptr());
}

#endif

TEST_F(SyncedMemoryTest, AllocationToCPU) {
  SyncedMemory mem(10);

  EXPECT_TRUE(mem.cpu_ptr());
  EXPECT_TRUE(mem.gpu_ptr());
}

#ifndef CPU_ONLY

TEST_F(SyncedMemoryTest, AllocationToGPU) {
  SyncedMemory mem(10);

  EXPECT_TRUE(mem.gpu_ptr());
  EXPECT_TRUE(mem.mutable_gpu_ptr());
}

#endif

TEST_F(SyncedMemoryTest, CPUWrite) {
  SyncedMemory mem(10);
  auto cpu_data = mem.mutable_cpu_ptr();

  EXPECT_EQ(SyncedMemory::kAtCPU, mem.head());

  memset(cpu_data, 1, mem.size());

  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ(((char*)cpu_data)[i], 1);
  }

  // Do another round.
  cpu_data = mem.mutable_cpu_ptr();

  EXPECT_EQ(SyncedMemory::kAtCPU, mem.head());

  memset(cpu_data, 2, mem.size());

  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ(((char*)cpu_data)[i], 2);
  }
}

#ifndef CPU_ONLY

TEST_F(SyncedMemoryTest, GPURead) {
  SyncedMemory mem(10);
  auto cpu_data = mem.mutable_cpu_ptr();

  EXPECT_EQ(SyncedMemory::kAtCPU, mem.head());

  memset(cpu_data, 1, mem.size());
  auto gpu_data = mem.gpu_ptr();

  EXPECT_EQ(SyncedMemory::kSynced, mem.head());

  // Check if values are the same.
  auto recovered_data = std::make_unique<char[]>(10);
  gpu::caffe_memcpy(recovered_data.get(), gpu_data, 10);

  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ(recovered_data[i], 1);
  }

  // Do another round.
  cpu_data = mem.mutable_cpu_ptr();

  EXPECT_EQ(SyncedMemory::kAtCPU, mem.head());

  memset(cpu_data, 2, mem.size());

  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char*>(cpu_data))[i], 2);
  }

  gpu_data = mem.gpu_ptr();

  EXPECT_EQ(SyncedMemory::kSynced, mem.head());

  // Check if values are the same.
  gpu::caffe_memcpy(recovered_data.get(), gpu_data, 10);

  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ(recovered_data[i], 2);
  }
}

TEST_F(SyncedMemoryTest, GPUWrite) {
  SyncedMemory mem(10);
  auto gpu_data = mem.mutable_gpu_ptr();

  EXPECT_EQ(SyncedMemory::kAtGPU, mem.head());

  gpu::caffe_memset(gpu_data, 1, mem.size());
  auto cpu_data = mem.cpu_ptr();

  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ(1, (static_cast<const char*>(cpu_data))[i]);
  }

  EXPECT_EQ(SyncedMemory::kSynced, mem.head());

  gpu_data = mem.mutable_gpu_ptr();

  EXPECT_EQ(SyncedMemory::kAtGPU, mem.head());

  gpu::caffe_memset(gpu_data, 2, mem.size());
  cpu_data = mem.cpu_ptr();

  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ(2, (static_cast<const char*>(cpu_data))[i]);
  }

  EXPECT_EQ(SyncedMemory::kSynced, mem.head());
}

#endif

}  // namespace caffe