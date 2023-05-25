#include <torch/extension.h>
#include <ATen/ATen.h>
//#include <torch/torch.h>
//#include <ATen/Context.h>
#include <cstdlib>
#include <iostream>
#include "sky.h"

using namespace std;

class Leaky: torch::nn::Module{
private:
  SkyrmionWord* _neuron;
  int _batch_size;
  int _input_size;
  int _output_size;
  vector<int> _previous_mem;
public:
  Leaky(int batch_size, int input_size, int output_size){
    _batch_size = batch_size;
    _input_size = input_size;
    _output_size = output_size;
    _neuron = new SkyrmionWord [output_size];
    _previous_mem.resize(output_size);
  }
  int getBatchSize() const {return _batch_size;}
  int getInputSize() const {return _input_size;}
  int getOutputSize() const {return _output_size;}

  // weights.size() = [1000, 784]
  // bias.size() = [1000]
  void initialize_wieghts(torch::Tensor weights, torch::Tensor bias){
    double weigtsTable[DISTANCE] = {0.0};
    weigtsTable[DISTANCE-1] = (1+(DISTANCE-2)/(DISTANCE-1))*0.5;
    for (int k = 1; k < DISTANCE-1; k++){
      weigtsTable[DISTANCE-1-k] = weigtsTable[DISTANCE-k] - 1/(DISTANCE-1);
    }

    c10::IntArrayRef x = bias.sizes();
    cout << "bias[0] = " << bias[0] << endl;
    cout << "bias[1] = " << bias[1] << endl;
    cout << "bias.sizes()[0] = " << x[0] << endl;
    cout << "weights.size(1) = " << weights.size(1) << endl;

    // ajust the weights to the desired ones
    for (int i = 0; i < bias.sizes()[0]; i++){
      // insert skyrmions representing positive or negative
      for (int j = 0; j < weights.size(1); j++){
        if (weights[i][j].item<double>() < 0){
          _neuron[i].insert(j+2, 0, 1, 0);
        }
      }
      if (bias[i].item<double>() < 0) _neuron[i].insert(bias.sizes()[0] + 2, 0, 1, 0);
      _neuron[i].shift(bias.sizes()[0] + 2, 0, 0); // shift to left

      // insert skyrmions representing the values
      for (int k = 1; k < DISTANCE; k++){
        for(int j = 0; j < weights.size(1); j++){
          double value = abs(weights[i][j].item<double>());
          if (value >= weigtsTable[DISTANCE-1-k]
            && value < weigtsTable[DISTANCE-k]){
            _neuron[i].insert(j+2, 0, 1, 0);
          }
        }
        if (abs(bias[i].item<double>()) >= weigtsTable[DISTANCE-1-k]
          && abs(bias[i].item<double>()) < weigtsTable[DISTANCE-k])
          _neuron[i].insert(bias.sizes()[0] + 2, 0, 1, 0);
        _neuron[i].shift(bias.sizes()[0] + 2, 0, 0); // shift to left
      }

      // move back
      for (int k = 0; k < DISTANCE; k++){
        _neuron[i].shift(0, bias.sizes()[0] + 2, 0);
      }
    }
  }

  // input.view(-1, 784)
  // input.size() = [128, 784]
  // torch::Tensor leaky_forward(torch::Tensor input){
  void leaky_forward(torch::Tensor input){
  //   vector<int> spike(_output_size, 0);
  //   vector<int> membrance(_output_size, 0);
  //   for (int j = 0; j < output_size; j++){
  //     // reset mechanism
  //     if (_previous_mem[j] >= 32){
  //       while (true){
  //         //shift to right
  //         _neuron[j]->shift(0, 2, 0);
  //
  //         // suppose we use set-to-zero mechanism and
  //         // detect is to detect whether mem is set to 0
  //         if (_neuron[j]->detect(1, 0, 0) == 1) break;
  //       }
  //     }
  //
  //     // start to adjust membrane potential
  //     count = 0;
  //     vector<int> whichWeights;
  //     unordered_map<int,int> counters;
  //     whichWeights.push_back(0); //for membrane potential
  //     for (int i = 0; i < input.size(1); i++){
  //       if (input[i] == 1) whichWeights.push_back(i+1);
  //     }
  //     int num = 0;
  //     // move right to get the weights whose input value is 1
  //     while (num != whichWeights.sizes()){
  //       data[j]->shift(to right);
  //       for (int i = 0; i < whichWeights.sizes(); i++){
  //         if (data[j]->detect(whichWeights[i]) == 1){
  //           counters[whichWeights[i]] = count;
  //           num++;
  //         }
  //       }
  //       count++;
  //     }
  //     // move back
  //     for (int i = 0; i < count; i++) data[j]->shift(to left);
  //
  //     // claculate whether the membrance exceeds the threshold
  //     int total_membrane = 0;
  //     for (auto it = counters.begin(); it != counters.end(); it++){
  //       total_membrane += it->second;
  //       if (it->first == 0) membrane[j] = it->second;
  //     }
  //     if (total_membrane > 0) total_membrane--; //represent the decay rate
  //     if (total_membrane >= DISTANCE){ // if exceeds threshold
  //       spike[j] = 1;
  //     }
  //     previous_mem[j] = total_membrane; // use this to reset membrane potential next time
  //   }
  //   torch::tensor spk = convert_to_tensor(spike);
  //   torch::tensor mem = convert_to_tensor(membrance);
  //
  //   return spk, mem;
  }

  void leaky_backward(torch::Tensor input){
  }
};

int main(){
  torch::manual_seed(7);
  torch::Tensor features = torch::rand({2, 3});
  cout << "features = " << features << "\n";

  Leaky neuron = Leaky(128, 784, 1000);
  cout << "neuron's batch size = " << neuron.getBatchSize() << endl;
  cout << "neuron's input size = " << neuron.getInputSize() << endl;
  cout << "neuron's output size = " << neuron.getOutputSize() << endl;

  neuron.initialize_wieghts(torch::rand({1000, 784}), torch::rand({1000}));

  return 0;
}


// PYBIND11_MODULE(leaky_cpp, m) {
//   m.def("forward", &leaky_forward, "Leaky forward");
//   m.def("backward", &leaky_backward, "Leaky backward");
// }
