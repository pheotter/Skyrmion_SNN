#include <torch/extension.h>
#include <ATen/ATen.h>
// #include <torch/torch.h>
#include <cstdlib>
#include <iostream>
#include "sky.h"

using namespace std;

class IEEE754 { // torch::nn::Module
protected:
  vector<SkyrmionWord*> _neuron;
  // int _batch_size; for inference, there is no need a batch size
  int _input_size;
  int _output_size;
  vector<int> _previous_mem;
  vector<float> _weights_tmp; // delete
  torch::Tensor _spike;
  unordered_map<string, pair<int, int>> stride;
  // the first int to store which racetrack
  // the second int to store where to start
  unordered_map<string, pair<int, int>> start;
public:
  IEEE754(int input_size, int output_size){
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
    stride["weight"] = make_pair(input_size, 1);
    stride["bias"] = make_pair(0, 1);
    stride["mem"] = make_pair(0, 1);
    start["weight"] = make_pair(0, 0);
    int num_weights = output_size * input_size;
    int num_weights_bias = num_weights + output_size;
    start["bias"] = make_pair(num_weights/(input_size + 2), num_weights % (input_size + 2));
    cout << "bias which skyrmion: " << num_weights/(input_size + 2) << " bias start: " << num_weights % (input_size + 2) << endl;
    start["mem"] = make_pair(num_weights_bias/(input_size + 2), num_weights_bias % (input_size + 2));
    cout << "mem which skyrmion: " << num_weights_bias/(input_size + 2) << " mem start: " << num_weights_bias % (input_size + 2) << endl;
  }

  ~IEEE754(){
    for (int i = 0; i < _output_size; i++){
      delete _neuron[i];
    }
  }

  int getInputSize() const {return _input_size;}
  int getOutputSize() const {return _output_size;}

  vector<int> getPlace(pair<int,int> &start, int index){
    vector<int> result(2, 0);
    result[0] = start.first + (start.second + index) / (_input_size + 2);
    result[1] = (start.second + index) % (_input_size + 2);
    return result;
  }

  union ufloat {
    float f;
    int u;
  };

  static bitset<sizeof(float) * 8> FPTransform_single(float f){
    ufloat uf;
    uf.f = f;
    bitset<sizeof(float) * 8> bits(uf.u);
    return bits;
  }
  // weights.size() = [1000, 784]
  // bias.size() = [1000]
  void initialize_weights(torch::Tensor weights, torch::Tensor bias){
    for (int i = 0; i < _output_size; i++){
      bitset<sizeof(float) * 8> bias_transf = FPTransform_single(bias[i].item<double>());
      cout << "bias[" << i << "] = " << bias[i].item<double>() << endl;
      cout << "justify bias_transf: " << bias_transf.to_string() << endl;
      vector<int> bias_place = getPlace(start["bias"], i);
      for (int j = 0; j < _input_size; j++){
        bitset<sizeof(float) * 8> weights_transf = FPTransform_single(weights[i][j].item<double>());
        cout << "weight[" << i << "][" << j << "] = " << weights[i][j].item<double>() << endl;
        cout << "justify weights_transf: " << weights_transf.to_string() << endl;
        vector<int> weight_place = getPlace(start["weight"], i * _input_size + j);
        cout << "weight_place[0] = " << weight_place[0] << " weight_place[1] = " << weight_place[1] << endl;
        for (int k = 0; k < DISTANCE; k++){
          if (weights_transf[DISTANCE-k-1] == 1)
            _neuron.at(weight_place[0])->insert(weight_place[1] + 1, 1, 0);
          _neuron.at(weight_place[0])->shift(weight_place[1] + 2, weight_place[1] + 1, 0);
          if (bias_transf[DISTANCE-k-1] == 1)
            _neuron.at(bias_place[0])->insert(bias_place[1] + 1, 1, 0);
          _neuron.at(bias_place[0])->shift(bias_place[1] + 2, bias_place[1] + 1, 0);
        }
        vector<int> positions = _neuron.at(weight_place[0])->bitPositions(weight_place[1]);
        cout << " positions[" << j << "] = ";
        for (unsigned int m = 0; m < positions.size(); m++){
          cout << positions.at(m) << " ";
        }
        cout << endl;
      }
    }
  }

  // input.size() = [784]
  vector<torch::Tensor> forward(torch::Tensor input){
    for (int i = 0; i < _output_size; i++){
      // reset mechanism
      if (_previous_mem.at(i) >= 1){
        
      }
    }
  }

};

int main(){
  torch::manual_seed(7);

  int in = 12;
  int out = 3;
  IEEE754 neuron = IEEE754(in, out);
  double x = 8.5;
  bitset<sizeof(float) * 8> res = neuron.FPTransform_single(x);
  cout << "x: " << res.to_string() << endl;

  cout << "neuron's input size = " << neuron.getInputSize() << endl;
  cout << "neuron's output size = " << neuron.getOutputSize() << endl;
  torch::Tensor w = torch::rand({out, in});
  torch::Tensor b = torch::rand({out});
  w[0][1] *= -1;
  w[2][8] *= -1;
  w[1][4] *= -1;
  b[0] *= -1;

  neuron.initialize_weights(w, b);

  // torch::Tensor input = torch::randint(0, 2, {in});
  //
  // vector<torch::Tensor> result = neuron.forward(input);
  // torch::Tensor spk = result[0];
  // torch::Tensor mem = result[1];
  // cout << "spk = " << spk << endl;
  // cout << "mem = " << mem << endl;
  return 0;
}

// namespace py = pybind11;
//
// PYBIND11_MODULE(leaky_cpp, m) {
//   py::class_<Leaky>(m, "Leaky")
//     .def(py::init<int, int>())
//     .def("initialize_weights", &Leaky::initialize_weights)
//     .def("getInputSize", &Leaky::getInputSize)
//     .def("getOutputSize", &Leaky::getOutputSize)
//     .def("leaky_forward", &Leaky::forward);
// }
