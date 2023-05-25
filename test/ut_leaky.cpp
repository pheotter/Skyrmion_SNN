#include <gtest/gtest.h>
#include "../src/sky.h"
#include "../src/leaky.cpp"
#include <cassert>
#include <bitset>

TEST(testConstructor, case1)
{
  ASSERT_EQ(2, 1+1);
}

int main(int argc , char **argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
