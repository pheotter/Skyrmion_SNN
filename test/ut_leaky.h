#include "../src/include/sky.h"
#include "../src/leaky.cpp"
#include "../src/IEEE754.cpp"

TEST(testConstructor, case1)
{
  Leaky leaky(784, 1000);
  ASSERT_EQ(784, leaky.getInputSize());
  ASSERT_EQ(1000, leaky.getOutputSize());
  ASSERT_EQ(1000, leaky.getPreviousMemSize());
  ASSERT_EQ(1000, leaky.getNeuronSize());
}

TEST(testInitializeWeights, case1)
{
  Leaky leaky(4, 2);
  vector<float> w = {0.001, -0.23, 0.67, 0.94, -0.12, 0.44, 0.38, 0.87};
  vector<float> b = {0.51, 0.72};
  torch::Tensor weight = torch::from_blob(w.data(), {2, 4}, torch::kFloat32);
  torch::Tensor bias = torch::from_blob(b.data(), {2}, torch::kFloat32);
  leaky.initialize_weights(weight, bias);

  ASSERT_TRUE(leaky.neuronBitPosition(0, 0).empty());
  ASSERT_TRUE(leaky.neuronBitPosition(0, 1).empty());
  vector<int> res = {6, 31};
  ASSERT_EQ(res, leaky.neuronBitPosition(0, 2));
  res = {20};
  ASSERT_EQ(res, leaky.neuronBitPosition(0, 3));
  res = {28};
  ASSERT_EQ(res, leaky.neuronBitPosition(0, 4));
  res = {15};
  ASSERT_EQ(res, leaky.neuronBitPosition(0, 5));
  ASSERT_TRUE(leaky.neuronBitPosition(1, 0).empty());
  res = {3, 31};
  ASSERT_EQ(res, leaky.neuronBitPosition(1, 1));
  res = {13};
  ASSERT_EQ(res, leaky.neuronBitPosition(1, 2));
  res = {11};
  ASSERT_EQ(res, leaky.neuronBitPosition(1, 3));
  res = {26};
  ASSERT_EQ(res, leaky.neuronBitPosition(1, 4));
  res = {21};
  ASSERT_EQ(res, leaky.neuronBitPosition(1, 5));
}

TEST(testResetMechanism, case1)
{
  Leaky leaky(4, 2);
  vector<float> w = {0.001, -0.01, 0.67, 0.94, -0.015, 0.29, 0.26, 0.99};
  vector<float> b = {0.48, 0.016};
  torch::Tensor weight = torch::from_blob(w.data(), {2, 4}, torch::kFloat32);
  torch::Tensor bias = torch::from_blob(b.data(), {2}, torch::kFloat32);
  leaky.initialize_weights(weight, bias); // each is 32
  leaky.setPreviousMem(0, 33);
  leaky.setPreviousMem(1, 20);
  leaky.setPreviousNumShift(1, 20);
  leaky.reset_mechanism(0);
  leaky.reset_mechanism(1);
  ASSERT_EQ(32, leaky.getNeuron(0)->getSht_engy());
  ASSERT_TRUE(leaky.neuronBitPosition(0, 0).empty());
  vector<int> ans1 = {19};
  ASSERT_EQ(52, leaky.getNeuron(1)->getSht_engy());
  ASSERT_EQ(ans1, leaky.neuronBitPosition(1, 0));

  leaky.setPreviousMem(1, 25);
  leaky.setPreviousNumShift(1, 5);
  leaky.reset_mechanism(1);
  vector<int> ans2 = {24};
  ASSERT_EQ(57, leaky.getNeuron(1)->getSht_engy());
  ASSERT_EQ(ans2, leaky.neuronBitPosition(1, 0));

  leaky.setPreviousMem(1, 32);
  leaky.setPreviousNumShift(1, 7);
  leaky.reset_mechanism(1);
  ASSERT_EQ(82, leaky.getNeuron(1)->getSht_engy());
  ASSERT_TRUE(leaky.neuronBitPosition(1, 0).empty());
}

TEST(testFindZeros, case1)
{
  Leaky leaky(4, 2);
  vector<float> w = {0.001, -0.01, 0.67, 0.94, -0.015, 0.29, 0.26, 0.99};
  vector<float> b = {0.48, 0.016};
  torch::Tensor weight = torch::from_blob(w.data(), {2, 4}, torch::kFloat32);
  torch::Tensor bias = torch::from_blob(b.data(), {2}, torch::kFloat32);
  leaky.initialize_weights(weight, bias);
  unordered_set<int> input1 = {0,1,2,3,4,5};
  unordered_set<int> res1 = leaky.findZeros(input1, 0);
  unordered_set<int> res2 = leaky.findZeros(input1, 1);
  unordered_set<int> ans1 = {0,1,2};
  unordered_set<int> ans2 = {0,1,5};
  ASSERT_EQ(ans1, res1);
  ASSERT_EQ(ans2, res2);
}

TEST(testFindNegatives, case1)
{
  Leaky leaky(4, 2);
  vector<float> w = {0.001, -0.01, 0.67, -0.94, -0.015, 0.29, -0.26, -0.99};
  vector<float> b = {-0.48, 0.016};
  torch::Tensor weight = torch::from_blob(w.data(), {2, 4}, torch::kFloat32);
  torch::Tensor bias = torch::from_blob(b.data(), {2}, torch::kFloat32);
  leaky.initialize_weights(weight, bias);
  unordered_set<int> input1 = {0,1,2,3,4,5};
  unordered_set<int> zeros1 = leaky.findZeros(input1, 0);
  unordered_set<int> zeros2 = leaky.findZeros(input1, 1);
  unordered_map<int,int> res1 = leaky.findNegatives(input1, zeros1, 0);
  unordered_map<int,int> res2 = leaky.findNegatives(input1, zeros2, 1);
  ASSERT_TRUE(res1.count(4));
  ASSERT_TRUE(res1.count(5));
  ASSERT_TRUE(res2.count(3));
  ASSERT_TRUE(res2.count(4));
  ASSERT_EQ(-1, res1[4]);
  ASSERT_EQ(-1, res1[5]);
  ASSERT_EQ(-1, res2[3]);
  ASSERT_EQ(-1, res2[4]);
}

TEST(testCalculateMem, case1)
{
  Leaky leaky(4, 2);
  vector<float> w = {0.001, -0.01, 0.67, -0.94, -0.015, 0.29, -0.26, -0.99};
  vector<float> b = {-0.48, 0.016};
  torch::Tensor weight = torch::from_blob(w.data(), {2, 4}, torch::kFloat32);
  torch::Tensor bias = torch::from_blob(b.data(), {2}, torch::kFloat32);
  leaky.initialize_weights(weight, bias);
  unordered_set<int> input1 = {0,1,2,3,4,5};
  unordered_set<int> zeros1 = leaky.findZeros(input1, 0);
  unordered_set<int> zeros2 = leaky.findZeros(input1, 1);
  unordered_map<int,int> neg1 = leaky.findNegatives(input1, zeros1, 0);
  unordered_map<int,int> neg2 = leaky.findNegatives(input1, zeros2, 1);
  int res1 = leaky.calculateMem(neg1, input1, zeros1.size(), 0);
  int res2 = leaky.calculateMem(neg2, input1, zeros2.size(), 1);
  ASSERT_EQ(21-29-15, res1);
  ASSERT_EQ(9-8-31, res2);
}

TEST(testIEEE754Constructor, case1)
{
  IEEE754 ieee(784, 1000, 0.9);
  ASSERT_EQ(784, ieee.getInputSize());
  ASSERT_EQ(1000, ieee.getOutputSize());
  ASSERT_TRUE((ieee.getDecayRate()-0.9) < 0.0000001);
  ASSERT_EQ(1000, ieee.getPreviousMemSize());
  ASSERT_EQ(make_pair(784, 1), ieee.getWeightStride());
  ASSERT_EQ(make_pair(0, 1), ieee.getBiasStride());
  ASSERT_EQ(make_pair(0, 1), ieee.getMemStride());
  ASSERT_EQ(make_pair(0, 0), ieee.getWeightStart());
  ASSERT_EQ(make_pair(997, 358), ieee.getBiasStart());
  ASSERT_EQ(make_pair(998, 572), ieee.getMemStart());
}

TEST(testIEEE754InitializeWeights, case1)
{
  IEEE754 ieee(4, 2, 0.9);
  vector<float> w = {0.001, -0.23, 0.67, 0.94, -0.12, 0.44, 0.38, 0.87};
  vector<float> b = {0.51, 0.72};
  torch::Tensor weight = torch::from_blob(w.data(), {2, 4}, torch::kFloat32);
  torch::Tensor bias = torch::from_blob(b.data(), {2}, torch::kFloat32);
  ieee.initialize_weights(weight, bias);

  ASSERT_TRUE(ieee.neuronBitPosition(1, 4).empty()); // mem 0
  vector<int> res = {0,1,2,3,5,6,9,12,16,17,23,25,27,28,29};
  ASSERT_EQ(res, ieee.neuronBitPosition(0, 0)); // 0.001
  res = {0,1,2,3,4,8,10,15,16,17,19,21,22,25,26,27,28,29,31};
  ASSERT_EQ(res, ieee.neuronBitPosition(0, 1)); // -0.23
  res = {0,1,2,3,4,8,10,15,16,17,19,21,24,25,26,27,28,29};
  ASSERT_EQ(res, ieee.neuronBitPosition(0, 2)); // 0.67
  res = {0,1,2,4,6,7,8,9,13,15,20,21,22,24,25,26,27,28,29};
  ASSERT_EQ(res, ieee.neuronBitPosition(0, 3)); // 0.94
  res = {2,3,4,6,8,9,10,11,15,17,24,25,26,27,28,29};
  ASSERT_EQ(res, ieee.neuronBitPosition(1, 2)); // bias 0.51
  ASSERT_TRUE(ieee.neuronBitPosition(1, 5).empty()); // mem 0
  res = {0,1,2,3,7,9,14,15,16,18,20,21,22,23,24,26,27,28,29,31};
  ASSERT_EQ(res, ieee.neuronBitPosition(0, 4)); // -0.12
  res = {1,2,3,5,7,8,9,10,14,16,21,22,23,25,26,27,28,29};
  ASSERT_EQ(res, ieee.neuronBitPosition(0, 5)); // 0.44
  res = {2,3,4,6,8,9,10,11,15,17,22,23,25,26,27,28,29};
  ASSERT_EQ(res, ieee.neuronBitPosition(1, 0)); // 0.38
  res = {1,4,6,11,12,13,15,17,18,19,20,22,24,25,26,27,28,29};
  ASSERT_EQ(res, ieee.neuronBitPosition(1, 1)); // 0.87
  res = {2,3,5,6,7,8,12,14,19,20,21,24,25,26,27,28,29};
  ASSERT_EQ(res, ieee.neuronBitPosition(1, 3)); // bias 0.72
}

TEST(testIEEE754GetPlace, case1)
{
  IEEE754 ieee(784, 1000, 0.9);
  pair<int,int> weightStart = ieee.getWeightStart();
  pair<int,int> weightStride = ieee.getWeightStride();
  vector<int> res1 = ieee.getPlace(weightStart, weightStride, 2, 3);
  vector<int> ans1 = {1, 785};
  vector<int> res2 = ieee.getPlace(weightStart, weightStride, 800, 5);
  vector<int> ans2 = {797, 763};
  ASSERT_EQ(ans1, res1);
  ASSERT_EQ(ans2, res2);
}

TEST(testIEEE754FloatToBitSingle, case1)
{
  IEEE754 ieee(784, 1000, 0.9);
  sky_size_t *res1 = ieee.floatToBit_single(0.5);
  sky_size_t ans1[32] = {0};
  for (int i = 2; i < 8; i++) ans1[i] = 1;
  for (int i = 0; i < 32; i++)
    ASSERT_EQ(ans1[i], res1[i]);
}

TEST(testIEEE754BitToFloatSingle, case1)
{
  IEEE754 ieee(784, 1000, 0.9);
  sky_size_t input1[32] = {0};
  for (int i = 2; i < 8; i++) input1[i] = 1;
  sky_size_t input2[32] = {0,0,1,1,1,1,1,1,0,1,1,0,0,1,0,0,1,0,0,1,1,0,1,1,1,0,1,0,0,1,1,0};
  float res1 = ieee.bitToFloat_single(input1);
  float res2 = ieee.bitToFloat_single(input2);
  ASSERT_EQ(0.5, res1);
  ASSERT_TRUE(abs(res2-0.893)<0.000001);
}

TEST(testIEEE754ResetMechanism, case1)
{
  IEEE754 ieee(784, 1000, 0.9);
  ieee.setPreviousMem(0, 0.5);
  ieee.setPreviousMem(1, 1.5);
  ieee.reset_mechanism(0);
  ieee.reset_mechanism(1);
  vector<int> ans1 = {24,25,26,27,28,29};
  ASSERT_EQ(ans1, ieee.neuronBitPosition(998, 572));
  ASSERT_TRUE(ieee.neuronBitPosition(998, 573).empty());
}

TEST(testIEEE754InputIsOne, case1)
{
  IEEE754 ieee(784, 1000, 0.9);
  torch::Tensor t = torch::zeros(784);
  t[2] = t[5] = 1;
  unordered_set<int> ans1 = {0,3,6,785};
  ASSERT_EQ(ans1, ieee.inputIsOne(t));
}

TEST(testIEEE754PlaceToBeRead, case1)
{
  IEEE754 ieee(784, 1000, 0.9);
  unordered_set<int> input1 = {0,3,6,785};
  unordered_map<int, vector<int>> res1 = ieee.placeToBeRead(input1, 1);
  vector<int> ans1 = {573};
  vector<int> ans2 = {359};
  vector<int> ans3 = {3, 0};
  ASSERT_TRUE(res1.count(998));
  ASSERT_TRUE(res1.count(997));
  ASSERT_FALSE(res1.count(996));
  ASSERT_FALSE(res1.count(999));
  ASSERT_FALSE(res1.count(2));
  ASSERT_TRUE(res1.count(1));
  ASSERT_FALSE(res1.count(0));
  ASSERT_EQ(ans1, res1[998]);
  ASSERT_EQ(ans2, res1[997]);
  ASSERT_EQ(ans3, res1[1]);
}

TEST(testIEEE754ClaculateMem, case1)
{
  IEEE754 ieee(784, 1000, 0.9);
  unordered_set<int> input1 = {0,3,6,785};
  unordered_map<int, vector<int>> map1 = ieee.placeToBeRead(input1, 1);
  sky_size_t content1[32] = {0};
  for (int i = 2; i < 8; i++) content1[i] = 1;
  sky_size_t content2[32] = {0,0,1,1,1,1,1,1,0,1,1,0,0,1,0,0,1,0,0,1,1,0,1,1,1,0,1,0,0,1,1,0};
  ieee.setData(1, 0, content1);
  ieee.setData(1, 3, content1);
  ieee.setData(997, 359, content2);
  ieee.setData(998, 573, content1);
  float res1 = ieee.calculateMem(map1);
  ASSERT_TRUE(abs(res1-2.393)<0.000001);
}
