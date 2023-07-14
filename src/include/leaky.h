#ifndef LEAKY
#define LEAKY
// #include <torch/extension.h>
// #include <ATen/ATen.h>
#include <torch/torch.h>
#include <cstdlib>
#include <iostream>
#include "sky.h"
#include "parent.h"

using namespace std;

class Leaky: public Parent {
protected:
  vector<int> _previous_numShift;
public:
  Leaky(int input_size, int output_size);
  void setPreviousNumShift(int outputIndex, int val);
  vector<double> constructWeightsTable();
  void reset_mechanism(int outputIndex) override;
  unordered_set<int> findZeros(unordered_set<int> &whichWeights, int outputIndex);
  unordered_map<int,int> findNegatives(unordered_set<int> &whichWeights, unordered_set<int> &zeros, int outputIndex);
  int calculateMem(unordered_map<int,int> &counters, unordered_set<int> whichWeights, int zerosSize, int outputIndex);
  void initialize_weights(torch::Tensor weights, torch::Tensor bias) override;
  vector<torch::Tensor> forward(torch::Tensor input) override;
};

#endif
