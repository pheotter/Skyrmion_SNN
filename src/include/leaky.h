#ifndef LEAKY
#define LEAKY
#include <torch/extension.h>
#include <ATen/ATen.h>
// #include <torch/torch.h>
#include <cstdlib>
#include <iostream>
#include "sky.h"

using namespace std;

class Leaky {
protected:
  vector<SkyrmionWord*> _neuron;
  int _input_size;
  int _output_size;
  vector<int> _previous_mem;
  vector<int> _previous_numShift;
  vector<float> _weights_tmp; // delete
  torch::Tensor _spike;
public:
  Leaky(int input_size, int output_size);
  ~Leaky();
  int getInputSize() const;
  int getOutputSize() const;
  int getPreviousMemSize() const;
  int getNeuronSize() const;
  SkyrmionWord *getNeuron(int outputIndex) const;
  void setPreviousMem(int outputIndex, int val);
  void setPreviousNumShift(int outputIndex, int val);
  vector<int> neuronBitPosition(int whichNeuron, int whichInterval) const;
  vector<double> constructWeightsTable();
  unordered_set<int> inputIsOne(torch::Tensor &input);
  void reset_mechanism(int outputIndex);
  unordered_set<int> findZeros(unordered_set<int> &whichWeights, int outputIndex);
  unordered_map<int,int> findNegatives(unordered_set<int> &whichWeights, unordered_set<int> &zeros, int outputIndex);
  void setData(int whichRaceTrack, int whichInterval, const sky_size_t *content);
  int calculateMem(unordered_map<int,int> &counters, unordered_set<int> &whichWeights, int zerosSize, int outputIndex);
  void initialize_weights(torch::Tensor weights, torch::Tensor bias);
  vector<torch::Tensor> forward(torch::Tensor input);
};

#endif
