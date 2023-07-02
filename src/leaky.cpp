#include <torch/extension.h>
#include <ATen/ATen.h>
// #include <torch/torch.h>
#include <cstdlib>
#include <iostream>
#include "sky.h"

using namespace std;

class Leaky { // torch::nn::Module
protected:
  vector<SkyrmionWord*> _neuron;
  // int _batch_size; for inference, there is no need a batch size
  int _input_size;
  int _output_size;
  vector<int> _previous_mem;
  vector<float> _weights_tmp; // delete
  torch::Tensor _spike;
public:
  Leaky(int input_size, int output_size){
    // _batch_size = batch_size;
    _input_size = input_size;
    _output_size = output_size;
    for (int i = 0; i < output_size; i++){
      // one is for membrane potential, the other one is for bias
      _neuron.emplace_back(new SkyrmionWord(input_size + 2));
    }
    _previous_mem.resize(output_size);
    _weights_tmp.resize((input_size+1)*output_size);
    _spike = torch::zeros(output_size);
  }

  ~Leaky(){
    for (int i = 0; i < _output_size; i++){
      delete _neuron[i];
    }
  }

  // int getBatchSize() const {return _batch_size;}
  int getInputSize() const {return _input_size;}
  int getOutputSize() const {return _output_size;}

  // weights.size() = [1000, 784]
  // bias.size() = [1000]
  void initialize_weights(torch::Tensor weights, torch::Tensor bias){
    vector<double> weightsTable(DISTANCE+1, 0.0);
    weightsTable.at(DISTANCE) = 1.0;
    weightsTable.at(DISTANCE-1) = (1+((double)DISTANCE-2)/(DISTANCE-1))*0.5;
    for (int k = 1; k < DISTANCE-1; k++){
      weightsTable.at(DISTANCE-1-k) = weightsTable.at(DISTANCE-k) - 1.0/(DISTANCE-1);
    }

    // // check weightsTable
    // for (int k = 0; k <= DISTANCE; k++){
    //   cout << "k = " << k << " value = " << weightsTable.at(k) << endl;
    // }

    // insert weight into member variable (delete after)
    for (int k = 0; k < _output_size; k++){
      for (int j = 0; j < _input_size; j++){
        _weights_tmp.at(k*(_input_size+1)+j) = weights[k][j].item<double>();
      }

      _weights_tmp.at(k*(_input_size+1)+_input_size) = bias[k].item<double>();
    }

    for (int i = 0; i < _output_size; i++){
      // 1. ajust the weights and bias to the desired places
      // insert skyrmions representing positive or negative
      for (int j = 0; j < _input_size; j++){
        if (weights[i][j].item<double>() < 0){
          // cout << "weight[" << i << "][" << j << "] = " << weights[i][j].item<double>() << endl;
          _neuron.at(i)->insert(j+2, 1, 0);
        }
      }
      if (bias[i].item<double>() < 0)
        _neuron.at(i)->insert(_input_size + 2, 1, 0);
      _neuron.at(i)->shift(_input_size + 4, 0, 0); // shift to left

      // insert skyrmions representing the values
      for (int k = 1; k < DISTANCE; k++){
        for(int j = 0; j < _input_size; j++){
          double value = abs(weights[i][j].item<double>());
          if (value >= weightsTable.at(DISTANCE-k) &&
            value < weightsTable.at(DISTANCE-k+1)){
            // cout << "k = " << k;
            // cout << " weight value = " << value;
            // cout << " i = " << i << " j=" << j << endl;
            _neuron.at(i)->insert(j+2, 1, 0);
          }
        }

        if (abs(bias[i].item<double>()) >= weightsTable.at(DISTANCE-k) &&
          abs(bias[i].item<double>()) < weightsTable.at(DISTANCE-k+1)){
          // cout << "bias value = " << abs(bias[i].item<double>()) << endl;
          _neuron.at(i)->insert(_input_size + 2, 1, 0);
        }
        _neuron.at(i)->shift(_input_size + 4, 0, 0); // shift to left
      }

      // // check if setting is correct
      // for (int j = 0; j < _input_size; j++){
      //   cout << "weight[" << j << "] = " << weights[i][j];
      //   vector<int> positions = _neuron.at(i)->bitPositions(j+1);
      //   cout << " positions = ";
      //   for (unsigned int k = 0; k < positions.size(); k++){
      //     cout << positions.at(k) << " ";
      //   }
      //   cout << endl;
      // }
      // cout << "bias[" << i << "] = " << bias[i];
      // vector<int> positions1 = _neuron.at(i)->bitPositions(_input_size+1);
      // cout << " positions = ";
      // for (unsigned int k = 0; k < positions1.size(); k++){
      //   cout << positions1.at(k) << " ";
      // }
      // cout << endl;

      // 2. generate skyrmions representing 0

      // for membrane potential
      _neuron.at(i)->insert(1, 1, 0);

      // for other weights whose value is close to 0
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
    }
  }

  // input.size() = [784]
  vector<torch::Tensor> forward(torch::Tensor input){
    for (int i = 0; i < _output_size; i++){
      // reset mechanism
      // because use one bit to represent negative values
      // precesion downgrades from DISTANCE to DISTANCE-1

      // cout << "_previous_mem " << i << " " << _previous_mem[i] << endl;
      if (_previous_mem.at(i) >= DISTANCE-1){
          while (true){
            cout << "test\n";
            //shift membrane potential to right, reset to zero
            _neuron.at(i)->shift(0, 2, 0);

            // suppose we use set-to-zero mechanism and
            // then detect whether mem is set to 0
            if (_neuron.at(i)->detect(1, 0) == 1) break;
          }
      }

      // start to adjust membrane potential
      unordered_set<int> whichWeights;
      unordered_map<int,int> counters;
      whichWeights.insert(0); // for membrane potential
      for (int j = 0; j < _input_size; j++){
        if (input[j].item<int>() == 1)
          whichWeights.insert(j+1);
      }
      whichWeights.insert(_input_size+1); // for bias

      // check whichWeights
      cout << "whichWeights: ";
      for (auto& position:whichWeights){
        cout << position << " ";
      }
      cout << endl;

      // First, count the values that are close to zero
      unordered_set<int> zeros;
      // cout << "zero: " << endl;
      for (auto& position:whichWeights){
        if (_neuron.at(i)->detect(position+1, 0) == 1) {
          if (position > 0){
            cout << position << " " << _weights_tmp.at(i*(_input_size+1)+position-1);
          }
          zeros.insert(position);
        }
      }
      // check zeros
      cout << endl << "zeros: ";
      for (auto &zero:zeros){
        cout << zero << " ";
      }
      cout << endl;

      // Second, determine whose value is negative
      // shift to left to see if they are negative
      // if negative and whose value is not close to zero,
      // set unordered_map's value to -1
      _neuron[i]->shift(_input_size + 4, 0, 0);

      for (auto& position:whichWeights){
        if (_neuron.at(i)->detect(position, 0) == 1 && zeros.count(position) == 0){
          counters[position] = -1;
        }
      }

      // shift backwards
      _neuron.at(i)->shift(0, _input_size + 4, 0);

      cout << "negative:" << endl;
      for (auto it = counters.begin(); it != counters.end(); it++){
        cout << "first: " << it->first << " second: " << it->second << endl;
      }

      // check if setting is correct
      for (int j = 0; j < _input_size; j++){
        vector<int> positions = _neuron.at(i)->bitPositions(j+1);
        cout << " positions[" << j << "] = ";
        for (unsigned int k = 0; k < positions.size(); k++){
          cout << positions.at(k) << " ";
        }
        cout << endl;
      }

      unsigned int num = 0;
      int count = 0;
      // move right to get the weights, bias and membrane potential
      while (num != whichWeights.size() - zeros.size() && count < DISTANCE-1){
        // shift to right
        count++;
        _neuron.at(i)->shift(0, _input_size + 4, 0);

        for (auto& position:whichWeights){
          if (_neuron.at(i)->detect(position+1, 0) == 1){
            num++;
            cout << "num = " << num << endl;
            if (counters.count(position) > 0 && counters[position] == -1){
              counters[position] *= count;
              cout << "<0 count = " << count << " position = " << position << " val = " << counters[position]<< endl;
            } else {
              counters[position] = count;
              cout << "count = " << count << " position = " << position << " val = " << counters[position]<< endl;
            }
          }
        }
      }
      // move backwards
      for (int j = 0; j < count; j++)
        _neuron.at(i)->shift(_input_size + 4, 0, 0);

      // claculate whether the membrance exceeds the threshold
      int total_membrane = 0;
      for (auto it = counters.begin(); it != counters.end(); it++){
          total_membrane += it->second;
      }
      // cout << "total_membrane = " << total_membrane << endl;
      if (total_membrane > 0) total_membrane--; //represent the decay
      if (total_membrane >= DISTANCE-1){ // if exceeds threshold
        _spike[i] = 1;
      }
      _previous_mem.at(i) = total_membrane; // use this to reset membrane potential next time
    }

    torch::Tensor mem = torch::from_blob(_previous_mem.data(), {_output_size}, torch::kInt32);

    return {_spike, mem};
  }

};

// int main(){
//   torch::manual_seed(7);
//
//   int in = 36;
//   int out = 5;
//   Leaky neuron = Leaky(in, out);
//   cout << "neuron's input size = " << neuron.getInputSize() << endl;
//   cout << "neuron's output size = " << neuron.getOutputSize() << endl;
//   torch::Tensor w = torch::rand({out, in});
//   torch::Tensor b = torch::rand({out});
//   w[0][1] *= -1;
//   w[4][8] *= -1;
//   w[1][4] *= -1;
//   b[0] *= -1;
//
//   neuron.initialize_weights(w, b);
//
//   torch::Tensor input = torch::randint(0, 2, {in});
//
//   vector<torch::Tensor> result = neuron.forward(input);
//   torch::Tensor spk = result[0];
//   torch::Tensor mem = result[1];
//   cout << "spk = " << spk << endl;
//   cout << "mem = " << mem << endl;
//   return 0;
// }

namespace py = pybind11;

PYBIND11_MODULE(leaky_cpp, m) {
  py::class_<Leaky>(m, "Leaky")
    .def(py::init<int, int>())
    .def("initialize_weights", &Leaky::initialize_weights)
    .def("getInputSize", &Leaky::getInputSize)
    .def("getOutputSize", &Leaky::getOutputSize)
    .def("leaky_forward", &Leaky::forward);
}
