#ifndef IEEE
#define IEEE

#include <torch/extension.h>
#include <ATen/ATen.h>
// #include <torch/torch.h>
#include <cstdlib>
#include <iostream>
#include "sky.h"

using namespace std;

union ufloat {
  float f;
  int u;
};

class IEEE754 {
protected:
  vector<SkyrmionWord*> _neuron;
  int _input_size;
  int _output_size;
  float _beta;
  vector<float> _previous_mem;
  vector<float> _weights_tmp; // delete
  torch::Tensor _spike;
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
  ~IEEE754();
  int getInputSize() const;
  int getOutputSize() const;
  float getDecayRate() const;
  int getPreviousMemSize() const;
  pair<int,int> getWeightStride() const;
  pair<int,int> getBiasStride() const;
  pair<int,int> getMemStride() const;
  pair<int,int> getWeightStart() const;
  pair<int,int> getBiasStart() const;
  pair<int,int> getMemStart() const;
  void setPreviousMem(int outputIndex, float f);
  vector<int> neuronBitPosition(int whichNeuron, int whichInterval) const;
  vector<int> getPlace(pair<int,int> &start, pair<int,int> &stride, int row, int col);
  static sky_size_t *floatToBit_single(float f);
  static float bitToFloat_single(sky_size_t *v);
  void reset_mechanism(int outputIndex);
  unordered_set<int> inputIsOne(torch::Tensor &input);
  unordered_map<int, vector<int>> placeToBeRead(unordered_set<int> &whichWeights, int outputIndex);
  void setData(int whichRaceTrack, int whichInterval, const sky_size_t *content);
  float calculateMem(unordered_map<int, vector<int>> &readWhich);
  void initialize_weights(torch::Tensor weights, torch::Tensor bias);
  vector<torch::Tensor> forward(torch::Tensor input);
};

#endif
