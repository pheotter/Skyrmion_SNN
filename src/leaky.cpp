#include "include/leaky.h"

Leaky::Leaky(int input_size, int output_size): Parent(input_size, output_size){
  _previous_numShift.resize(output_size);
}

void Leaky::setPreviousNumShift(int outputIndex, int val) {
  _previous_numShift.at(outputIndex) = val;
}

void Leaky::reset_mechanism(int outputIndex){
  // so far we only implement reset-to-zero
  // Since we use one bit to represent the sign bit
  // precision downgrades from DISTANCE to DISTANCE-1
  int shiftLatency = 0;
  int detectLatency = 0;
  if (_previous_mem.at(outputIndex) >= DISTANCE-1){
    _neuron.at(outputIndex)->addDet_latcy(1, 0);
    while (!_neuron.at(outputIndex)->detect(1, 0)){
      //shift membrane potential to right, reset to zero
      _neuron.at(outputIndex)->shift(0, 2, 0);
      shiftLatency++;
      detectLatency++;
    }
  } else {
    for (int j = 0; j < _previous_numShift.at(outputIndex); j++){
      _neuron.at(outputIndex)->shift(2, 0, 0);
      shiftLatency++;
    }
  }
  // update latency
  _neuron.at(outputIndex)->addSht_latcy(shiftLatency, 0);
  _neuron.at(outputIndex)->addDet_latcy(detectLatency, 0);
}

unordered_set<int> Leaky::findZeros(unordered_set<int> &whichWeights, int outputIndex){
  unordered_set<int> zeros;
  for (auto& position:whichWeights){
    if (_neuron.at(outputIndex)->detect(position+1, 0) == 1)
      zeros.insert(position);
  }
  // update latency (detect parallel)
  _neuron.at(outputIndex)->addDet_latcy(1, 0);
  return zeros;
}

unordered_map<int,int> Leaky::findNegatives(unordered_set<int> &whichWeights, unordered_set<int> &zeros, int outputIndex){
  unordered_map<int,int> counters;
  // move position 31 left to access port to determine whether this value is negative
  _neuron.at(outputIndex)->shift(_input_size + 4, 0, 0);

  // if negative and not close to zero, set unordered_map's value to -1
  for (auto& position:whichWeights){
    if (_neuron.at(outputIndex)->detect(position, 0) == 1 && zeros.count(position) == 0){
      counters[position] = -1;
    }
  }
  // move backward
  _neuron.at(outputIndex)->shift(0, _input_size + 4, 0);

  // update latency (detect parallel)
  _neuron.at(outputIndex)->addSht_latcy(2, 0);
  _neuron.at(outputIndex)->addDet_latcy(1, 0);
  return counters;
}

int Leaky::calculateMem(unordered_map<int,int> &counters, unordered_set<int> whichWeights, int zerosSize, int outputIndex) {
  unsigned int num = 0;
  int count = 0;
  int n = whichWeights.size();

  // move right to get the weights, bias and membrane potential
  while (num != n - zerosSize && count < DISTANCE-1){
    count++;
    _neuron.at(outputIndex)->shift(0, _input_size + 4, 0);

    vector<int> toBeDeleted;
    for (auto position:whichWeights){
      if (_neuron.at(outputIndex)->detect(position+1, 0) == 1){
        toBeDeleted.push_back(position);
        num++;
        if (counters.count(position) > 0 && counters[position] == -1){
          counters[position] *= count;
        } else {
          counters[position] = count;
        }
      }
    }
    for (auto d:toBeDeleted) whichWeights.erase(d);
  }
  // move backwards
  for (int j = 0; j < count; j++)
    _neuron.at(outputIndex)->shift(_input_size + 4, 0, 0);

  // cumulate membrane potential
  int total_membrane = 0;
  for (auto it = counters.begin(); it != counters.end(); it++){
    total_membrane += it->second;
  }

  // update latency (detect parallel)
  _neuron.at(outputIndex)->addSht_latcy(count*2, 0);
  _neuron.at(outputIndex)->addDet_latcy(count, 0);
  return total_membrane;
}

vector<double> Leaky::constructWeightsTable(){
  vector<double> weightsTable(DISTANCE+1, 0.0);
  weightsTable.at(DISTANCE) = 1.0;
  weightsTable.at(DISTANCE-1) = (1+((double)DISTANCE-2)/(DISTANCE-1))*0.5;
  for (int k = 1; k < DISTANCE-1; k++){
    weightsTable.at(DISTANCE-1-k) = weightsTable.at(DISTANCE-k) - 1.0/(DISTANCE-1);
  }
  return weightsTable;
}

// weights.size() = [1000, 784]
// bias.size() = [1000]
void Leaky::initialize_weights(torch::Tensor weights, torch::Tensor bias){
  vector<double> weightsTable = constructWeightsTable();
  bool negKeep = false;

  // insert weight into member variable (delete after)-----------------------
  for (int k = 0; k < _output_size; k++){
    for (int j = 0; j < _input_size; j++){
      _weights_tmp.at(k*(_input_size+1)+j) = weights[k][j].item<double>();
    }
    _weights_tmp.at(k*(_input_size+1)+_input_size) = bias[k].item<double>();
  }
  // ------------------------------------------------------------------------

  for (int i = 0; i < _output_size; i++){
    // 1. ajust the weights and bias to the desired places
    // insert skyrmions representing negative
    for (int j = 0; j < _input_size; j++){
      if (weights[i][j].item<double>() < 0){
        _neuron.at(i)->insert(j+2, 1, 0);
        negKeep = true;
      }
    }
    if (bias[i].item<double>() < 0){
      _neuron.at(i)->insert(_input_size + 2, 1, 0);
      negKeep = true;
    }
    _neuron.at(i)->shift(_input_size + 4, 0, 0); // shift to left

    // insert skyrmions representing the values
    for (int k = 1; k < DISTANCE; k++){
      for(int j = 0; j < _input_size; j++){
        double value = abs(weights[i][j].item<double>());
        if (value >= weightsTable.at(DISTANCE-k) &&
          value < weightsTable.at(DISTANCE-k+1)){
          _neuron.at(i)->insert(j+2, 1, 0);
        }
      }

      if (abs(bias[i].item<double>()) >= weightsTable.at(DISTANCE-k) &&
        abs(bias[i].item<double>()) < weightsTable.at(DISTANCE-k+1)){
        _neuron.at(i)->insert(_input_size + 2, 1, 0);
      }
      _neuron.at(i)->shift(_input_size + 4, 0, 0); // shift to left
    }

    // 2. generate skyrmions representing 0
    // for membrane potential
    _neuron.at(i)->insert(1, 1, 0);

    // for weights
    for(int j = 0; j < _input_size; j++){
      double value = abs(weights[i][j].item<double>());
      if (value >= weightsTable.at(0) && value < weightsTable.at(1)){
        _neuron.at(i)->insert(j+2, 1, 0);
      }
    }

    // for bias
    if (abs(bias[i].item<double>()) >= weightsTable.at(0) &&
      abs(bias[i].item<double>()) < weightsTable.at(1)){
      _neuron.at(i)->insert(_input_size + 2, 1, 0);
    }

    // update latency
    // one is for weights, and the other is for membrane potential
    _neuron.at(i)->addIns_latcy(2, 0);
    if (negKeep) _neuron.at(i)->addIns_latcy(1, 0);
    _neuron.at(i)->addSht_latcy(DISTANCE, 0);
  }
}

// input.size() = [784]
vector<torch::Tensor> Leaky::forward(torch::Tensor input){
  // record which input is 1
  unordered_set<int> whichWeights = inputIsOne(input);

  for (int i = 0; i < _output_size; i++){
    // reset mechanism
    reset_mechanism(i);

    // First, look for the values that are close to zero
    unordered_set<int> zeros = findZeros(whichWeights, i);

    // Second, look for the values that is negative and is not close to zero
    unordered_map<int,int> counters = findNegatives(whichWeights, zeros, i);

    // claculate the membrance potential
    int total_membrane = calculateMem(counters, whichWeights, zeros.size(), i);

    // minus one to represent decay rate
    if (total_membrane > 0) total_membrane--;

    // generate spike if membrane potential exceeds threshold
    if (total_membrane >= DISTANCE-1) _spike[i] = 1;

    // use these to reset membrane potential next time
    _previous_mem.at(i) = total_membrane;
    int membrane = 0;
    if (counters.count(0) == 1) membrane = counters[0];
    _previous_numShift.at(i) = total_membrane - membrane;
  }
  calculateLatency();
  updateLatency();
  torch::Tensor mem = torch::from_blob(_previous_mem.data(), {_output_size}, torch::kInt32);

  return {_spike, mem};
}


namespace py = pybind11;

// PYBIND11_MODULE(leaky_cpp, m) {
//   py::class_<Leaky>(m, "Leaky")
//     .def(py::init<int, int>())
//     .def("initialize_weights", &Leaky::initialize_weights)
//     .def("getInputSize", &Leaky::getInputSize)
//     .def("getOutputSize", &Leaky::getOutputSize)
//     .def("leaky_forward", &Leaky::forward);
// }
