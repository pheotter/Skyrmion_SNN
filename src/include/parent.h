#ifndef PARENT
#define PARENT
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cstdlib>
#include <iostream>
#include "sky.h"

using namespace std;

class Parent {
protected:
  vector<SkyrmionWord*> _neuron;
  int _input_size;
  int _output_size;
  vector<float> _previous_mem;
  vector<float> _weights_tmp; // delete
  torch::Tensor _spike;
  vector<unsigned int> _previous_shift_latency;
  vector<unsigned int> _previous_insert_latency;
  vector<unsigned int> _previous_delete_latency;
  vector<unsigned int> _previous_detect_latency;
  vector<unsigned int> _previous_shiftVertcl_latency;
  Stat _shift_latency;
  Stat _insert_latency;
  Stat _delete_latency;
  Stat _detect_latency;
  Stat _shiftVertcl_latency;
  Stat _shift_energy;
  Stat _insert_energy;
  Stat _delete_energy;
  Stat _detect_energy;
  Stat _shiftVertcl_energy;
public:
  Parent(int input_size, int output_size);
  ~Parent();
  int getInputSize() const;
  int getOutputSize() const;
  int getPreviousMemSize() const;
  int getNeuronSize() const;
  SkyrmionWord *getNeuron(int outputIndex) const;
  void setPreviousMem(int outputIndex, float f);
  void setPreviousStat(int outputIndex, unsigned int val);
  void setData(int whichRaceTrack, int whichInterval, const sky_size_t *content);
  vector<int> neuronBitPosition(int whichNeuron, int whichInterval) const;
  unordered_set<int> inputIsOne(torch::Tensor &input);
  virtual void reset_mechanism(int outputIndex) = 0;
  virtual void initialize_weights(torch::Tensor weights, torch::Tensor bias) = 0;
  virtual vector<torch::Tensor> forward(torch::Tensor input) = 0;
  unsigned int getMaxLatency(int latency);
  void calculateLatency();
  void updateLatency();
  void output();
};

#endif
