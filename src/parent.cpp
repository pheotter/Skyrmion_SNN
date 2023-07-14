#include "include/parent.h"

Parent::Parent(int input_size, int output_size){
  _input_size = input_size;
  _output_size = output_size;
  for (int i = 0; i < output_size; i++){
    // one is for membrane potential, the other one is for bias
    _neuron.emplace_back(new SkyrmionWord(input_size + 2));
  }
  _previous_mem.resize(output_size);
  _weights_tmp.resize((input_size+1)*output_size);
  _spike = torch::zeros(output_size);
  _previous_shift_latency.resize(output_size);
  _previous_insert_latency.resize(output_size);
  _previous_delete_latency.resize(output_size);
  _previous_detect_latency.resize(output_size);
  _previous_shiftVertcl_latency.resize(output_size);
  _shift_latency = 0;
  _insert_latency = 0;
  _delete_latency = 0;
  _detect_latency = 0;
  _shiftVertcl_latency = 0;
  _shift_energy = 0;
  _insert_energy = 0;
  _delete_energy = 0;
  _detect_energy = 0;
  _shiftVertcl_energy = 0;
}

Parent::~Parent(){
  for (int i = 0; i < _output_size; i++){
    delete _neuron.at(i);
  }
  output();
}

int Parent::getInputSize() const {return _input_size;}
int Parent::getOutputSize() const {return _output_size;}
int Parent::getPreviousMemSize() const {return _previous_mem.size();}
int Parent::getNeuronSize() const {return _neuron.size();}

SkyrmionWord *Parent::getNeuron(int outputIndex) const {
  return _neuron.at(outputIndex);
}

void Parent::setPreviousMem(int outputIndex, float f) {
  _previous_mem.at(outputIndex) = f;
}

void Parent::setPreviousStat(int outputIndex, unsigned int val){
  _previous_insert_latency.at(outputIndex) = val;
}

void Parent::setData(int whichRaceTrack, int whichInterval, const sky_size_t *content){
  if (whichRaceTrack < 0 || whichRaceTrack >= _output_size){
    cout << "setData: whichRaceTrack out of range\n";
    exit(1);
  }
  sky_size_t *contentByte = Skyrmion::bitToByte(DISTANCE/8, content);
  _neuron.at(whichRaceTrack)->writeData(whichInterval*DISTANCE/8, DISTANCE/8, contentByte, NAIVE, 0);
}

vector<int> Parent::neuronBitPosition(int whichNeuron, int whichInterval) const {
  return _neuron.at(whichNeuron)->bitPositions(whichInterval);
}

unordered_set<int> Parent::inputIsOne(torch::Tensor &input){
  unordered_set<int> whichWeights;
  whichWeights.insert(0); // for membrane potential
  for (int j = 0; j < _input_size; j++){
    if (input[j].item<int>() == 1)
      whichWeights.insert(j+1);
  }
  whichWeights.insert(_input_size+1); // for bias
  return whichWeights;
}

unsigned int Parent::getMaxLatency(int latency){
  unsigned int res = 0;
  switch(latency){
    case 0: //shift
    for (int i = 0; i < _output_size; i++)
      res = max(res, (unsigned int)_neuron.at(i)->getSht_latcy() - _previous_shift_latency.at(i));
    break;
    case 1: //insert
      for (int i = 0; i < _output_size; i++)
        res = max(res, (unsigned int)_neuron.at(i)->getIns_latcy() - _previous_insert_latency.at(i));
      break;
    case 2: //delete
      for (int i = 0; i < _output_size; i++)
        res = max(res, (unsigned int)_neuron.at(i)->getDel_latcy() - _previous_delete_latency.at(i));
      break;
    case 3: //detect
      for (int i = 0; i < _output_size; i++)
        res = max(res, (unsigned int)_neuron.at(i)->getDet_latcy() - _previous_detect_latency.at(i));
      break;
    case 4: //shiftVertical
      for (int i = 0; i < _output_size; i++)
        res = max(res, (unsigned int)_neuron.at(i)->getShtVrtcl_latcy() - _previous_shiftVertcl_latency.at(i));
      break;
    default:
      cout << "The valid latency includes 0(shift)/1(insert)/2(delete)/3(detect)/4(vertical shift)\n";
      exit(1);
  }
  return res;
}

void Parent::calculateLatency(){
  _shift_latency += getMaxLatency(0);
  _insert_latency += getMaxLatency(1);
  _delete_latency += getMaxLatency(2);
  _detect_latency += getMaxLatency(3);
  _shiftVertcl_latency += getMaxLatency(4);
}

void Parent::updateLatency(){
  for (int i = 0; i < _output_size; i++){
    _previous_shift_latency.at(i) = _neuron.at(i)->getSht_latcy();
    _previous_insert_latency.at(i) = _neuron.at(i)->getIns_latcy();
    _previous_delete_latency.at(i) = _neuron.at(i)->getDel_latcy();
    _previous_detect_latency.at(i) = _neuron.at(i)->getDet_latcy();
    _previous_shiftVertcl_latency.at(i) = _neuron.at(i)->getShtVrtcl_latcy();
  }
}

void Parent::output(){
  for (int i = 0; i < _output_size; i++){
    _shift_energy += _neuron.at(i)->getSht_engy();
    _insert_energy += _neuron.at(i)->getIns_engy();
    _delete_energy += _neuron.at(i)->getDel_engy();
    _detect_energy += _neuron.at(i)->getDet_engy();
    _shiftVertcl_energy += _neuron.at(i)->getShtVrtcl_engy();
  }

  fstream file;
	file.open("output.csv", ios::app);
	if (!file) {
		cout << "File not created!\n";
	} else {
		// cout << "File created successfully!\n";
		file << "shift energy," << _shift_energy << endl;
		file << "insert energy," << _insert_energy << endl;
		file << "delete energy," << _delete_energy << endl;
		file << "detect energy," << _detect_energy << endl;
		file << "shiftVertcl energy," << _shiftVertcl_energy << endl;
		file << "shift latency," << _shift_latency << endl;
		file << "insert latency," << _insert_latency << endl;
		file << "delete latency," << _delete_latency << endl;
		file << "detect latency," << _detect_latency << endl;
		file << "shiftVertcl latency," << _shiftVertcl_latency << endl << endl;
		file.close();
	}
}

// namespace py = pybind11;
//
// PYBIND11_MODULE(parent_cpp, m) {
//   py::class_<Parent>(m, "Parent")
//     .def(py::init<int, int>())
//     .def("getInputSize", &Parent::getInputSize)
//     .def("getOutputSize", &Parent::getOutputSize);
// }
