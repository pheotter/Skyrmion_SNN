#include "include/IEEE754.h"

IEEE754::IEEE754(int input_size, int output_size, float beta){
  _input_size = input_size;
  _output_size = output_size;
  _beta = beta;
  for (int i = 0; i < output_size; i++){
    // one is for membrane potential, the other one is for bias
    _neuron.emplace_back(new SkyrmionWord(input_size + 2));
  }
  _previous_mem.resize(output_size);
  _weights_tmp.resize((input_size+1)*output_size);
  _spike = torch::zeros(output_size);
  _weight_stride = make_pair(input_size, 1);
  _bias_stride = make_pair(0, 1);
  _mem_stride = make_pair(0, 1);
  _weight_start = make_pair(0, 0);
  int num_weights = output_size * input_size;
  int num_weights_bias = num_weights + output_size;
  _bias_start = make_pair(num_weights/(input_size + 2), num_weights % (input_size + 2));
  _mem_start = make_pair(num_weights_bias/(input_size + 2), num_weights_bias % (input_size + 2));
}

IEEE754::~IEEE754(){
  for (int i = 0; i < _output_size; i++){
    delete _neuron[i];
  }
}

int IEEE754::getInputSize() const {return _input_size;}
int IEEE754::getOutputSize() const {return _output_size;}
float IEEE754::getDecayRate() const {return _beta;}
int IEEE754::getPreviousMemSize() const {return _previous_mem.size();}
pair<int,int> IEEE754::getWeightStride() const {return _weight_stride;}
pair<int,int> IEEE754::getBiasStride() const {return _bias_stride;}
pair<int,int> IEEE754::getMemStride() const {return _mem_stride;}
pair<int,int> IEEE754::getWeightStart() const {return _weight_start;}
pair<int,int> IEEE754::getBiasStart() const {return _bias_start;}
pair<int,int> IEEE754::getMemStart() const {return _mem_start;}
void IEEE754::setPreviousMem(int outputIndex, float f) {_previous_mem.at(outputIndex) = f;}
vector<int> IEEE754::neuronBitPosition(int whichNeuron, int whichInterval) const {
  return _neuron.at(whichNeuron)->bitPositions(whichInterval);
}

vector<int> IEEE754::getPlace(pair<int,int> &start, pair<int,int> &stride, int row, int col){
  if (row > _output_size -1 || row < 0 || col < 0 || col > _input_size -1){
    cout << "getPlace: row or column is out of range (row: 0~"
      << _output_size-1 << ", col: 0~" << _input_size-1 << ")" << endl;
    exit(1);
  }
  vector<int> result(2, 0);
  result[0] = start.first + (start.second + (stride.first*row + stride.second*col)) / (_input_size + 2);
  result[1] = (start.second + stride.first*row + stride.second*col) % (_input_size + 2);
  return result;
}

sky_size_t *IEEE754::floatToBit_single(float f){
  ufloat uf;
  uf.f = f;
  bitset<sizeof(float) * 8> bits(uf.u);
  sky_size_t *res = new sky_size_t [sizeof(float) * 8];
  for (int i = 0; i < sizeof(float) * 8; i++){
    res[i] = bits[sizeof(float) * 8 - i - 1];
  }
  return res;
}

float IEEE754::bitToFloat_single(sky_size_t *v){
  float res = 1;
  if (v[0] == 1) res = -1;
  sky_size_t *exp = Skyrmion::bitToByte(1, v+1);
  float fraction = 1;
  for (int i = 9; i < 32; i++){
    if (v[i] == 1){
      fraction += pow(2, 8-i);
    }
  }
  return res * pow(2, *exp - 127) * fraction;
}

void IEEE754::reset_mechanism(int outputIndex){ // so far we only reset to zero
  vector<int> mem_place = getPlace(_mem_start, _mem_stride, 0, outputIndex);
  if (_previous_mem.at(outputIndex) >= 1){
    // delete all skyrmions in the interval
    sky_size_t content[DISTANCE] = {0};
    _neuron.at(mem_place.at(0))->writeData(mem_place.at(1), DISTANCE/8, content, NAIVE, 0);
  } else {
    sky_size_t *content = floatToBit_single(_previous_mem.at(outputIndex));
    // need to change bit to bype first (for gem5's format)
    sky_size_t *contentByte = Skyrmion::bitToByte(DISTANCE/8, content);
    _neuron.at(mem_place.at(0))->writeData(mem_place.at(1)*DISTANCE/8, DISTANCE/8, contentByte, PERMUTATION_WRITE, 0);
  }
}

unordered_set<int> IEEE754::inputIsOne(torch::Tensor &input){
  unordered_set<int> whichWeights;
  whichWeights.insert(0); // for membrane potential
  for (int j = 0; j < _input_size; j++){
    if (input[j].item<int>() == 1)
      whichWeights.insert(j+1);
  }
  whichWeights.insert(_input_size+1); // for bias
  return whichWeights;
}

unordered_map<int, vector<int>> IEEE754::placeToBeRead(unordered_set<int> &whichWeights, int outputIndex){
  unordered_map<int,vector<int>> readWhich;
  for (auto which:whichWeights){
    if (which == 0){
      vector<int> place = getPlace(_mem_start, _mem_stride, 0, outputIndex);
      readWhich[place.at(0)].push_back(place.at(1));
    } else if (which == _input_size+1){
      vector<int> place = getPlace(_bias_start, _bias_stride, 0, outputIndex);
      readWhich[place.at(0)].push_back(place.at(1));
    } else {
      vector<int> place = getPlace(_weight_start, _weight_stride, outputIndex, which - 1);
      readWhich[place.at(0)].push_back(place.at(1));
    }
  }
  return readWhich;
}

void IEEE754::setData(int whichRaceTrack, int whichInterval, const sky_size_t *content){
  if (whichRaceTrack < 0 || whichRaceTrack >= _output_size){
    cout << "setData: whichRaceTrack out of range\n";
    exit(1);
  }
  sky_size_t *contentByte = Skyrmion::bitToByte(DISTANCE/8, content);
  _neuron.at(whichRaceTrack)->writeData(whichInterval*DISTANCE/8, DISTANCE/8, contentByte, NAIVE, 0);
}

float IEEE754::calculateMem(unordered_map<int, vector<int>> &readWhich){
  float new_mem = 0;
  for (auto it = readWhich.begin(); it != readWhich.end(); it++){
    sky_size_t *dataByte = _neuron.at(it->first)->readData(0, (_input_size+2)*DISTANCE/8, 1, 0);
    sky_size_t *dataBit = Skyrmion::byteToBit((_input_size+2)*DISTANCE/8, dataByte);
    for (int j = 0; j < it->second.size(); j++){
      sky_size_t *ptr = dataBit + readWhich[it->first].at(j) * DISTANCE;
      new_mem += bitToFloat_single(ptr);
    }
    delete [] dataByte;
    delete [] dataBit;
  }
  return new_mem;
}

// weights.size() = [1000, 784]
// bias.size() = [1000]
void IEEE754::initialize_weights(torch::Tensor weights, torch::Tensor bias){
  for (int i = 0; i < _output_size; i++){
    sky_size_t *buffer = new sky_size_t [(_input_size + 2) * DISTANCE];
    sky_size_t *bufferPtr = buffer;
    for (int j = 0; j < _input_size +2; j++){
      sky_size_t *ptr = nullptr;
      pair<int,int> index = make_pair(i, j);
      if (index >= _mem_start){
        ptr = floatToBit_single(0.0);
      } else if (index < _mem_start && index >= _bias_start){
        int bIndex = (i - _bias_start.first) * (_input_size + 2) + j - _bias_start.second;
        ptr = floatToBit_single(bias[bIndex].item<double>());
      } else if (index < _bias_start){
        int wI = (i * (_input_size + 2) + j) / _input_size;
        int wJ = (i * (_input_size + 2) + j) % _input_size;
        ptr = floatToBit_single(weights[wI][wJ].item<double>());
      }
      memcpy(bufferPtr, ptr, DISTANCE * sizeof(sky_size_t));
      bufferPtr += DISTANCE;
      delete [] ptr;
      ptr = nullptr;
    }
    sky_size_t *bufferByte = Skyrmion::bitToByte((_input_size + 2)*DISTANCE/8, buffer);
    _neuron.at(i)->writeData(0, (_input_size+2)*DISTANCE/8, bufferByte, NAIVE, 0);
    delete [] bufferByte;
    delete [] buffer;
  }
}

// input.size() = [784]
vector<torch::Tensor> IEEE754::forward(torch::Tensor input){
  // record which input is 1
  unordered_set<int> whichWeights = inputIsOne(input);

  for (int i = 0; i < _output_size; i++){
    // reset to zero or set to the new membrane potential
    reset_mechanism(i);

    // record which neuron and which interval to read
    unordered_map<int,vector<int>> readWhich = placeToBeRead(whichWeights, i);

    // calculate neuron's value to be added
    float new_mem = calculateMem(readWhich);
    _previous_mem[i] = new_mem;
  }
  torch::Tensor mem = torch::from_blob(_previous_mem.data(), {_output_size}, torch::kFloat32);

  return {_spike, mem};
}



namespace py = pybind11;

PYBIND11_MODULE(ieee754_cpp, m) {
  py::class_<IEEE754>(m, "IEEE754")
    .def(py::init<int, int, float>())
    .def("initialize_weights", &IEEE754::initialize_weights)
    .def("getInputSize", &IEEE754::getInputSize)
    .def("getOutputSize", &IEEE754::getOutputSize)
    .def("ieee754_forward", &IEEE754::forward);
}
