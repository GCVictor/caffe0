#include "caffe/common/blob.hh"
#include "test_caffe_main.hh"

namespace caffe {

template <typename Dtype>
class BlobSimpleTest : public ::testing::Test {
 protected:
  static constexpr auto kShape = {2, 3, 4, 5};

  BlobSimpleTest() : blob_preshaped_{new Blob<Dtype>(kShape)} {}
  virtual ~BlobSimpleTest() { delete blob_preshaped_; }

  // const Blob<Dtype>* blob_;
  Blob<Dtype>* const blob_preshaped_;
};

TYPED_TEST_SUITE(BlobSimpleTest, TestDtypes);

TYPED_TEST(BlobSimpleTest, TestInitialization) {
  EXPECT_TRUE(this->blob_preshaped_);

  const auto& shape = BlobSimpleTest<TypeParam>::kShape;

  EXPECT_EQ(shape.size(), this->blob_preshaped_->NDim());
  EXPECT_EQ(
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()),
      this->blob_preshaped_->NumElement());

  for (size_t i = 0; i < shape.size(); ++i) {
    EXPECT_EQ(*(shape.begin() + i), this->blob_preshaped_->DimAt(i));
  }
}

}  // namespace caffe