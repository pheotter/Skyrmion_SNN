#ifndef IEEE
#define IEEE

// #include <torch/extension.h>
// #include <ATen/ATen.h>
#include <torch/torch.h>
#include <cstdlib>
#include <iostream>
#include "sky.h"
#include "parent.h"

using namespace std;

union ufloat {
  float f;
  int u;
};

class IEEE754: public Parent {
protected:
  float _beta;
  pair<int,int> _weight_stride;
  pair<int,int> _bias_stride;
  pair<int,int> _mem_stride;
  // the first int to store which racetrack
  // the second int to store where to start
  pair<int,int> _weight_start;
  pair<int,int> _bias_start;
  pair<int,int> _mem_start;
public:
  IEEE754(int input_size, int output_size, float beta);
  float getDecayRate() const;
  pair<int,int> getWeightStride() const;
  pair<int,int> getBiasStride() const;
  pair<int,int> getMemStride() const;
  pair<int,int> getWeightStart() const;
  pair<int,int> getBiasStart() const;
  pair<int,int> getMemStart() const;
  vector<int> getPlace(pair<int,int> &start, pair<int,int> &stride, int row, int col);
  static sky_size_t *floatToBit_single(float f);
  static float bitToFloat_single(sky_size_t *v);
  void reset_mechanism(int outputIndex) override;
  unordered_map<int, vector<int>> placeToBeRead(unordered_set<int> &whichWeights, int outputIndex);
  float calculateMem(unordered_map<int, vector<int>> &readWhich);
  void initialize_weights(torch::Tensor weights, torch::Tensor bias) override;
  vector<torch::Tensor> forward(torch::Tensor input) override;
};

#endif
