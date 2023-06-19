#include "sky.h"
//#include "base/trace.hh" //open
//#include "debug/Skyrmion.hh" //open
//#include "sim/core.hh" //open
//#include "sim/sim_exit.hh" //open
//#include "sky_run_def.hh" //open

Skyrmion::Skyrmion()
{
	if (OVER_HEAD != DISTANCE){
		cout << "Create: OVER_HEAD must be equal to DISTANCE!" << endl;
		exit(1);
	}
	//DPRINTF(Skyrmion, "Skyrmion is created!\n");
	//cout << "Skyrmion is created!\n";
	_id = 0;
	_intervalSize = 0;
	shift_energy = 0;
	insert_energy = 0;
	delete_energy = 0;
	detect_energy = 0;
	shiftVertcl_energy = 0;
	shift_latency = 0;
	insert_latency = 0;
	delete_latency = 0;
	detect_latency = 0;
	shiftVertcl_latency = 0;
	shift_energy_DMW = 0;
	insert_energy_DMW = 0;
	delete_energy_DMW = 0;
	detect_energy_DMW = 0;
	shiftVertcl_energy_DMW = 0;
	shift_latency_DMW = 0;
	insert_latency_DMW = 0;
	delete_latency_DMW = 0;
	detect_latency_DMW = 0;
	shiftVertcl_latency_DMW = 0;

	//gem5::registerExitCallback([this]() {outPut();}); //open
}

int Skyrmion::getIntervalSize() const
{
	return _intervalSize;
}

void Skyrmion::setId(int id)
{
	_id = id;
}

int Skyrmion::getId() const
{
	return _id;
}

void Skyrmion::setWriteType(enum Write_Type write_type)
{
	_write_type = write_type;
}

SkyrmionWord::SkyrmionWord(): Skyrmion()
{
	_type = WORD_BASED;
	entry = new sky_size_t [MAX_SIZE + 2 * OVER_HEAD + MAX_SIZE / DISTANCE + 1];
	for (int i = 0; i < MAX_SIZE + 2 * OVER_HEAD + MAX_SIZE / DISTANCE + 1; i++){
		entry[i] = 0;
	}
	checkFullArray = new int [MAX_SIZE/8];
	memset(checkFullArray, 0, MAX_SIZE / 8 * sizeof(int));
	n_checkFull = 0;
}

SkyrmionWord::SkyrmionWord(int intervalSize): Skyrmion()
{
	_type = WORD_BASED;
	_intervalSize = intervalSize;
	entry = new sky_size_t [intervalSize * DISTANCE + 2 * OVER_HEAD + intervalSize + 1];
	for (int i = 0; i < intervalSize * DISTANCE + 2 * OVER_HEAD + intervalSize + 1; i++){
		entry[i] = 0;
	}
	checkFullArray = new int [intervalSize * DISTANCE/8];
	memset(checkFullArray, 0, intervalSize * DISTANCE / 8 * sizeof(int));
	n_checkFull = 0;
}

sky_size_t SkyrmionWord::getEntry(int position) const
{
	return entry[position];
}

/**
 * Calculate the value between the accessPort.
 * Because the return type is unsigned long long,
 * max DISTANCE will be limited to 64.
 * If the DISTANCE is greater than 64, the return value will be incorrect.
 * @param whichInterval The interval on the racetrack	skyrmion memory
 *  for example, MAX_SIZE is 128, DISTANCE is 32, then interval will range 0~3
 * @return The value calculated from the whichInterval
 *  for example, if there is only bit stored with the rightmost,
 *  the return value will be 1.
*/
unsigned long long SkyrmionWord::checkValueBWPorts(int whichInterval)
{
	if (_intervalSize == 0 && whichInterval > MAX_SIZE / DISTANCE){
		cout << "checkValueBWPorts: whichInterval is invalid!" << endl;
		exit(1);
	}
	if (_intervalSize != 0 && whichInterval > _intervalSize){
		cout << "checkValueBWPorts: whichInterval is invalid!" << endl;
		exit(1);
	}
	unsigned long long result = 0;
	for (int i = 0; i < DISTANCE; i++){
		result += ((unsigned int)entry[OVER_HEAD + (whichInterval + 1) * (DISTANCE + 1) - i - 1] << i);
	}
	return result;
}

/**
 * Tell you which places contain skyrmions in the interval you request.
 * @param whichInterval The interval on the racetrack	skyrmion memory
 *  for example, MAX_SIZE is 128, DISTANCE is 32, then interval will range 0~3
 * @return The positions where contain skyrmions
 *  for example, if DISTANCE is 16(0000 0101 0001 0011)
 *  the return vector will be {0,1,4,8,10}
*/
vector<int> SkyrmionWord::bitPositions(int whichInterval)
{
	if (_intervalSize == 0 && whichInterval > MAX_SIZE / DISTANCE){
		cout << "bitPositions: whichInterval is invalid!" << endl;
		exit(1);
	}
	if (_intervalSize != 0 && whichInterval > _intervalSize){
		cout << "bitPositions: whichInterval is invalid!" << endl;
		exit(1);
	}
	vector<int> result;
	for (int i = 0; i < DISTANCE; i++){
		if(entry[OVER_HEAD + (whichInterval + 1) * (DISTANCE + 1) - i - 1] == 1)
			result.push_back(i);
	}
	return result;
}

int SkyrmionWord::getN_checkFull() const
{
	return n_checkFull;
}

void SkyrmionWord::setEntry(int position, sky_size_t value)
{
	entry[position] = value;
}

SkyrmionBit::SkyrmionBit(): Skyrmion()
{
	_type = BIT_INTERLEAVED;
	entries = new sky_size_t* [ROW];
	for (int i = 0; i < ROW; i++){
			entries[i] = new sky_size_t [MAX_SIZE + 2 * OVER_HEAD + MAX_SIZE / DISTANCE + 1];
	}
	for (int k = 0; k < ROW; k++){
		for (int j = 0; j < MAX_SIZE + 2 * OVER_HEAD + MAX_SIZE / DISTANCE + 1; j++){
			entries[k][j] = 0;
		}
	}
	buffer = new int [MAX_SIZE*ROW/BUFFER_LENGTH];
	for (int i = 0; i < ROW/BUFFER_LENGTH; i++){
		for (int j = 0; j < MAX_SIZE; j++)
			buffer[i * MAX_SIZE + j] = 0;
	}
	checkFullArray = new int [ROW / 8 * MAX_SIZE];
	memset(checkFullArray, 0, ROW / 8 * MAX_SIZE * sizeof(int));
	n_checkFull = 0;
	blockUsedNumber = new Stat [MAX_SIZE];
	memset(blockUsedNumber, 0, MAX_SIZE * sizeof(Stat));
}

SkyrmionBit::SkyrmionBit(int intervalSize): Skyrmion()
{
	_type = BIT_INTERLEAVED;
	_intervalSize = intervalSize;
	entries = new sky_size_t* [ROW];
	for (int i = 0; i < ROW; i++){
			entries[i] = new sky_size_t [intervalSize*DISTANCE + 2 * OVER_HEAD + intervalSize + 1];
	}
	for (int k = 0; k < ROW; k++){
		for (int j = 0; j < intervalSize*DISTANCE + 2 * OVER_HEAD + intervalSize + 1; j++){
			entries[k][j] = 0;
		}
	}
	buffer = new int [intervalSize*DISTANCE*ROW/BUFFER_LENGTH];
	for (int i = 0; i < ROW/BUFFER_LENGTH; i++){
		for (int j = 0; j < intervalSize*DISTANCE; j++)
			buffer[i * intervalSize*DISTANCE + j] = 0;
	}
	checkFullArray = new int [ROW / 8 * intervalSize*DISTANCE];
	memset(checkFullArray, 0, ROW / 8 * intervalSize*DISTANCE * sizeof(int));
	n_checkFull = 0;
	blockUsedNumber = new Stat [intervalSize*DISTANCE];
	memset(blockUsedNumber, 0, intervalSize*DISTANCE * sizeof(Stat));
}

sky_size_t SkyrmionBit::getEntries(int row, int col) const
{
	return entries[row][col];
}

int SkyrmionBit::getN_checkFull() const
{
	return n_checkFull;
}

Stat SkyrmionBit::getBlockUsedNumber(int index) const
{
	return blockUsedNumber[index];
}

void SkyrmionBit::setEntries(int row, int col, sky_size_t value)
{
	entries[row][col] = value;
}

Stat Skyrmion::getSht_engy() const
{
	return shift_energy;
}

Stat Skyrmion::getIns_engy() const
{
	return insert_energy;
}

Stat Skyrmion::getDel_engy() const
{
	return delete_energy;
}

Stat Skyrmion::getDet_engy() const
{
	return detect_energy;
}

Stat Skyrmion::getShtVrtcl_engy() const
{
	return shiftVertcl_energy;
}

Stat Skyrmion::getSht_latcy() const
{
	return shift_latency;
}

Stat Skyrmion::getIns_latcy() const
{
	return insert_latency;
}

Stat Skyrmion::getDel_latcy() const
{
	return delete_latency;
}

Stat Skyrmion::getDet_latcy() const
{
	return detect_latency;
}

Stat Skyrmion::getShtVrtcl_latcy() const
{
	return shiftVertcl_latency;
}

Stat Skyrmion::getSht_engy_DMW() const
{
	return shift_energy_DMW;
}

Stat Skyrmion::getIns_engy_DMW() const
{
	return insert_energy_DMW;
}

Stat Skyrmion::getDel_engy_DMW() const
{
	return delete_energy_DMW;
}

Stat Skyrmion::getDet_engy_DMW() const
{
	return detect_energy_DMW;
}

Stat Skyrmion::getShtVrtcl_engy_DMW() const
{
	return shiftVertcl_energy_DMW;
}

Stat Skyrmion::getSht_latcy_DMW() const
{
	return shift_latency_DMW;
}

Stat Skyrmion::getIns_latcy_DMW() const
{
	return insert_latency_DMW;
}

Stat Skyrmion::getDel_latcy_DMW() const
{
	return delete_latency_DMW;
}

Stat Skyrmion::getDet_latcy_DMW() const
{
	return detect_latency_DMW;
}

Stat Skyrmion::getShtVrtcl_latcy_DMW() const
{
	return shiftVertcl_latency_DMW;
}

/**
 * print out the word-based data
 * used for debug
*/
void SkyrmionWord::print() const
{
	for (int i = 32+33*249; i < 32+33*251; i++) //2 * OVER_HEAD + MAX_SIZE + MAX_SIZE / DISTANCE + 1
    cout << i << " " << (int)entry[i] << endl;
	cout << endl;
}

/**
 * print out the word-based data
 * used for debug
*/
void SkyrmionBit::print() const
{
	for (int k = 0; k < 80; k++){ // revise i to 0
		cout << "k = " << k << " ";
		//for (int i = 0; i < 8; i++){
		//	cout << (int)entries[k*8+i][35] << " ";
		//}
		for (int j = 32; j < 38; j++){// 2 * OVER_HEAD + MAX_SIZE + MAX_SIZE / DISTANCE + 1
			cout << "[" << j << "]" << " "<< (int)entries[k][j] << " ";
		}
		cout << endl;
	}
	//cout << endl;
}

/* open
void SkyrmionBit::print(int index) const
{
	for (int k = 0; k < ROW/8; k++){ // revise i to 0
  	cout << "k = " << k << " ";
    for (int i = 0; i < 8; i++){
    	//for (int j = 0; j < 68; j++){// 2 * OVER_HEAD + MAX_SIZE + MAX_SIZE / DISTANCE + 1
      		cout << (int)entries[k*8+i][index] << " ";
      //}
      //cout << endl;
    }
    cout << endl;
  }
}
*/

void Skyrmion::outPut(void)
{
	fstream file;
	//string fileName = "output" + to_string(WRITETYPE) + to_string(WRITETYPEUP) + to_string(AUTO_DETECTED_METHOD) + to_string(ORDERED) + ".csv"; //open
	//file.open(fileName.c_str(), ios::app); //open
	file.open("output.csv", ios::app);
	if (!file) {
		//cout << "File not created!\n";
	} else {
		//cout << "File created successfully!\n";
		//DPRINTF(Skyrmion, "File created successfully!\n");
		file << "shift energy," << shift_energy << ",";
		file << "insert energy," << insert_energy << ",";
		file << "delete energy," << delete_energy << ",";
		file << "detect energy," << detect_energy << ",";
		file << "shiftVertcl energy," << shiftVertcl_energy << ",";
		file << "shift latency," << shift_latency << ",";
		file << "insert latency," << insert_latency << ",";
		file << "delete latency," << delete_latency << ",";
		file << "detect latency," << detect_latency << ",";
		file << "shiftVertcl latency," << shiftVertcl_latency << ",";
		file << "shift energy DMW," << shift_energy_DMW << ",";
		file << "insert energy DMW," << insert_energy_DMW << ",";
		file << "delete energy DMW," << delete_energy_DMW << ",";
		file << "detect energy DMW," << detect_energy_DMW << ",";
		file << "shiftVertcl energy DMW," << shiftVertcl_energy_DMW << ",";
		file << "shift latency DMW," << shift_latency_DMW << ",";
		file << "insert latency DMW," << insert_latency_DMW << ",";
		file << "delete latency DMW," << delete_latency_DMW << ",";
		file << "detect latency DMW," << detect_latency_DMW << ",";
		file << "shiftVertcl latency DMW," << shiftVertcl_latency_DMW << endl;
		file.close();
	}
}

Skyrmion::~Skyrmion()
{
	//outPut();
}

SkyrmionWord::~SkyrmionWord()
{
	//outPut();
	delete [] entry;
	delete [] checkFullArray;
}

SkyrmionBit::~SkyrmionBit()
{
	//outPut();
	for (int i = 0; i < ROW; i++){
		delete [] entries[i];
	}
	delete [] entries;
	delete [] checkFullArray;
	delete [] blockUsedNumber;
}

/**
 * Determine the access port which current start from and
 * the access port which current end at
 * the most left end of Skyrmion track is access port 0,
 * while the most right end of Skyrmion track is access port block/DISTANCE + 2
 * e.g. If we set DISTANCE 32 and MAX_SIZE 2048, block 31 is close to the right access port,
 * current is start from access port 1 and end at access port MAX_SIZE/DISTANCE + 2
 * @param block The column of the entries the block belongs to (start from 0)
 * @param portsBuffer The start access port save at portsBuffer[0] and the end save at portsBuffer[1]
 * @return The block need to move such steps to the access port
*/
int SkyrmionBit::determinePorts(int block, int *portsBuffer) const
{
	int blk = block % DISTANCE;
	if (blk < (DISTANCE >> 1)){
		portsBuffer[0] = block/DISTANCE + 2;
		portsBuffer[1] = 0;
		return blk + 1;
	} else {
		portsBuffer[0] = block/DISTANCE + 1;
		portsBuffer[1] = MAX_SIZE/DISTANCE + 2;
		return DISTANCE - blk;
	}
}

/**
 * Determine the access port which current start from and
 * the access port which current end at
 * the most left end of Skyrmion track is access port 0,
 * while the most right end of Skyrmion track is access port block/DISTANCE + 2
 * e.g. If we set DISTANCE 32, size 3 and MAX_SIZE 2048,
 * If size >= 4(=32/8, which is 4 bytes for each DISTANCE), we always move to left
 * but if size < 4, we move to the closest access port
 * address 0 is close to the left access port,
 * current is start from access port 2 and end at access port 0
 * @param address The start address of the data
 * @param size The size(bytes) of the data
 * @param portsBuffer The start access port save at portsBuffer[0] and the end save at portsBuffer[1]
 * @return The address need to move such steps to the access port
*/
int SkyrmionWord::determinePorts(Addr address, data_size_t size, int *portsBuffer) const
{
	// buffer[0] saves startPort, buffer[1] endPort
	int numAddrBtwPort = DISTANCE / 8;
	int addr = address % numAddrBtwPort;
	int rightAddr = (address + size) % numAddrBtwPort;
	int rightShiftDis = numAddrBtwPort - rightAddr;
	if (size < numAddrBtwPort){
		if (addr == 0){
			portsBuffer[0] = (address + size) / numAddrBtwPort + 2;
			portsBuffer[1] = 0;
			return 0;
		} else if (rightAddr == 0){
			portsBuffer[0] = address / numAddrBtwPort + 1;
			portsBuffer[1] = MAX_SIZE/DISTANCE + 2;
			return 0;
		} else if (addr <= rightShiftDis){
			portsBuffer[0] = (address + size) / numAddrBtwPort + 2;
			portsBuffer[1] = 0;
			return addr * 8;
		} else if (addr > rightShiftDis){
			portsBuffer[0] = address / numAddrBtwPort + 1;
			portsBuffer[1] = MAX_SIZE/DISTANCE + 2;
			return rightShiftDis * 8;
		}
	} else { //decide always to move to left
		portsBuffer[0] = ceil(float(address + size) / numAddrBtwPort) + 1;
		portsBuffer[1] = 0;
		return 0;
	}
	return -1;
}

/**
 * Shift one step from the startPort and end at the endPort
 * e.g. It contains access port 0~6 if MAX_SIZE 128 and DISTANCE 32, including virtual ports
 * @param startPort The access port which current start from (determined by function determinePorts)
 * @param endPort The access port which current end at (determined by function determinePorts)
 * @param saveData If 0 save energy data to the first statics, 1 to the second
*/
void SkyrmionWord::shift(int startPort, int endPort, int saveData)
{
	if (_intervalSize == 0 && (startPort < 0 || startPort > MAX_SIZE / DISTANCE + 2 || endPort < 0 || endPort > MAX_SIZE / DISTANCE + 2 || startPort == endPort)){
			cout << "Shift: start/end index is invalid!" << endl;
			exit(1);
	}
	if (_intervalSize != 0 && (startPort < 0 || startPort > _intervalSize + 2 || endPort < 0 || endPort > _intervalSize + 2 || startPort == endPort)){
			cout << "Shift: start/end index is invalid!" << endl;
			exit(1);
	}
	if (saveData == 0){
		shift_energy++;
	} else if (saveData == 1){
		shift_energy_DMW++;
	}

	if (startPort < endPort){ // left to right
		if (endPort == MAX_SIZE / DISTANCE + 2){
			for (int i = endPort - 1; i > startPort - 1; i--){
				for (int j = 0; j <= DISTANCE; j++){
					entry[OVER_HEAD + i * (DISTANCE + 1) - j - 1] = entry[OVER_HEAD + i * (DISTANCE + 1) - j - 2];
				}
			}
		} else {
			for (int i = endPort; i > startPort; i--){
				for (int j = 0; j <= DISTANCE; j++){
					entry[OVER_HEAD + (i - 1) * (DISTANCE + 1) - j] = entry[OVER_HEAD + (i - 1) * (DISTANCE + 1) - j - 1];
				}
			}
		}
		if (startPort == 0)
			entry[0] = 0;//-1
		else
			entry[OVER_HEAD + (startPort - 1) * (DISTANCE + 1)] = 0;//-1
	} else if (startPort > endPort){ // right to left
		if (endPort == 0){
			for (int i = -1; i < startPort -1 ; i++){
				for (int j = 0; j <= DISTANCE; j++){
					entry[OVER_HEAD + 1 + i * (DISTANCE + 1) + j] = entry[OVER_HEAD + 1 + i * (DISTANCE + 1) + j + 1];
				}
			}
		} else {
			for (int i = endPort; i < startPort; i++){
				for (int j = 0; j <= DISTANCE; j++){
					entry[OVER_HEAD + (i - 1) * (DISTANCE + 1) + j] = entry[OVER_HEAD + (i - 1) * (DISTANCE + 1) + j + 1];
				}
			}
		}
		if (startPort == MAX_SIZE / DISTANCE + 2){
			entry[2 * OVER_HEAD + MAX_SIZE + MAX_SIZE/DISTANCE] = 0;//-1
		} else{
			entry[OVER_HEAD + (startPort-1) * (DISTANCE + 1)] = 0;//-1
		}
	}
}

/**
 * Shift one step from the startPort and end at the endPort
 * e.g. It contains access port 0~6 if MAX_SIZE 128 and DISTANCE 32, including virtual ports
 * @param startPort The access port which current start from (determined by function determinePorts)
 * @param endPort The access port which current end at (determined by function determinePorts)
 * @param address The address of the data
 * @param size The size(bytes) of the data
 * @param saveData If 0 save energy data to the first statics, 1 to the second
*/
void SkyrmionBit::shift(int startPort, int endPort, Addr address, data_size_t size, int saveData)
{
	if (_intervalSize == 0 && (startPort < 0 || startPort > MAX_SIZE / DISTANCE + 2 || endPort < 0 || endPort > MAX_SIZE / DISTANCE + 2 || startPort == endPort)){
			cout << "Shift: start/end index is invalid!" << endl;
			exit(1);
	}
	if (_intervalSize != 0 && (startPort < 0 || startPort > _intervalSize + 2 || endPort < 0 || endPort > _intervalSize + 2 || startPort == endPort)){
			cout << "Shift: start/end index is invalid!" << endl;
			exit(1);
	}
	// record energy of shift
	if (saveData == 0){
		shift_energy += size * 8;
	} else if (saveData == 1){
		shift_energy_DMW += size * 8;
	}

	if (startPort < endPort){ // left to right
		if (endPort == MAX_SIZE / DISTANCE + 2){
			for (int k = 0; k < size*8; k++){
				for (int i = endPort - 1; i > startPort - 1; i--){
					for (int j = 0; j <= DISTANCE; j++){
						entries[address * 8 + k][OVER_HEAD + i * (DISTANCE + 1) - j - 1] = entries[address * 8 + k][OVER_HEAD + i * (DISTANCE + 1) - j - 2];
					}
				}
			}
		} else {
			for (int k = 0; k < size*8; k++){
				for (int i = endPort; i > startPort; i--){
					for (int j = 0; j <= DISTANCE; j++){
						entries[address * 8 + k][OVER_HEAD + (i - 1) * (DISTANCE + 1) - j] = entries[address * 8 + k][OVER_HEAD + (i - 1) * (DISTANCE + 1) - j - 1];
					}
				}
			}
		}
		if (startPort == 0){
			for (int k = 0; k < size*8; k++){
				entries[address * 8 + k][0] = 0;//-1
			}
		}	else {
			for (int k = 0; k < size*8; k++){
				entries[address * 8 + k][OVER_HEAD + (startPort - 1) * (DISTANCE + 1)] = 0;//-1
			}
		}
	} else if (startPort > endPort){ // right to left
		if (endPort == 0){
			for (int k = 0; k < size*8; k++){
				for (int i = -1; i < startPort -1 ; i++){
					for (int j = 0; j <= DISTANCE; j++){
						entries[address * 8 + k][OVER_HEAD + 1 + i * (DISTANCE + 1) + j] = entries[address * 8 + k][OVER_HEAD + 1 + i * (DISTANCE + 1) + j + 1];
					}
				}
			}
		} else {
			for (int k = 0; k < size*8; k++){
				for (int i = endPort; i < startPort; i++){
					for (int j = 0; j <= DISTANCE; j++){
						entries[address * 8 + k][OVER_HEAD + (i - 1) * (DISTANCE + 1) + j] = entries[address * 8 + k][OVER_HEAD + (i - 1) * (DISTANCE + 1) + j + 1];
					}
				}
			}
		}
		if (startPort == MAX_SIZE / DISTANCE + 2){
			for (int k = 0; k < size*8; k++){
				entries[address * 8 + k][2 * OVER_HEAD + MAX_SIZE + MAX_SIZE/DISTANCE] = 0;//-1
			}
		} else {
			for (int k = 0; k < size*8; k++){
				entries[address * 8 + k][OVER_HEAD + (startPort-1) * (DISTANCE + 1)] = 0;//-1
			}
		}
	}
}

/**
 * Shift vertically one step up or down at the address
 * @param accessPort The access port which skyrmions could be injected, deleted or detected
 * @param address The address of the data
 * @param updown Move upwards is 1, and downwards 0
 * @param saveData If 0 save energy data to the first statics, 1 to the second
*/
void SkyrmionBit::shiftVertcl(int accessPort, Addr address, int updown, int saveData)
{
	//save energy data
	if (saveData == 0){
		shiftVertcl_energy++;
	} else if (saveData == 1){
		shiftVertcl_energy_DMW++;
	}
	//start to shift
	int bufferSet = BUFFER_LENGTH / 8;
	if (updown == 1){
		for (int j = 0; j < BUFFER_LENGTH-1; j++){
			entries[address / bufferSet * BUFFER_LENGTH + j][OVER_HEAD + accessPort * (DISTANCE + 1)] = entries[address / bufferSet * BUFFER_LENGTH + j + 1][OVER_HEAD + accessPort * (DISTANCE + 1)];
		}
		entries[address / bufferSet * BUFFER_LENGTH + BUFFER_LENGTH - 1][OVER_HEAD + accessPort * (DISTANCE + 1)] = 0;//-1
	} else {
		for (int j = BUFFER_LENGTH-1; j > 0; j--){
			entries[address / bufferSet * BUFFER_LENGTH + j][OVER_HEAD + accessPort * (DISTANCE + 1)] = entries[address / bufferSet * BUFFER_LENGTH + j - 1][OVER_HEAD + accessPort * (DISTANCE + 1)];
		}
		entries[address / bufferSet * BUFFER_LENGTH][OVER_HEAD + accessPort * (DISTANCE + 1)] = 0;//-1
	}
}

/**
 * Insert content(bit 0 or 1) at accessPort(excluding virtual access ports)
 * e.g. If set MAX_SIZE 128 and DISTANCE 32, accessPort includes 0~4
 * @param accessPort The access port which skyrmions could be injected
 * @param content The data bit 0 or bit 1 will be inserted
 * @param saveData If 0 save energy data to the first statics, 1 to the second
*/
	void SkyrmionWord::insert(int accessPort, sky_size_t content, int saveData)
{
	if (_intervalSize == 0 && (accessPort < 0 || accessPort > MAX_SIZE / DISTANCE)){
		cout << "Insert: accessPort is invalid!" << endl;
		exit(1);
	}
	if (_intervalSize != 0 && (accessPort < 0 || accessPort > _intervalSize)){
		cout << "Insert: accessPort is invalid!" << endl;
		exit(1);
	}
	//record number of insert
	if (content == 1){
		if (saveData == 0){
			insert_energy++;
		} else if (saveData == 1){
			insert_energy_DMW++;
		}
	}
	//start to insert
	entry[OVER_HEAD + accessPort * (DISTANCE + 1)] = content;
}

/**
 * Insert content(bit 0 or 1) at accessPort(excluding virtual access ports)
 * e.g. If set MAX_SIZE 128 and DISTANCE 32, accessPort includes 0~4
 * @param accessPort The access port which skyrmions could be injected
 * @param row The row of the entries
 * @param content The data bit 0 or bit 1 will be inserted
 *  if content = 2, do not want to add one to insert_energy but it still represents bit 1
 * @param saveData If 0 save energy data to the first statics, 1 to the second
*/
void SkyrmionBit::insert(int accessPort, int row, sky_size_t content, int saveData)
{
	if (_intervalSize == 0 && (accessPort < 0 || accessPort > MAX_SIZE / DISTANCE)){
		cout << "Insert: accessPort is invalid!" << endl;
		exit(1);
	}
	if (_intervalSize != 0 && (accessPort < 0 || accessPort > _intervalSize)){
		cout << "Insert: accessPort is invalid!" << endl;
		exit(1);
	}
	if (row < 0 || row >= ROW){
		cout << "Insert: row for bit-interleaved method needs to be 0~" << ROW-1 << "!" << endl;
		exit(1);
	}
	//record number of insert
	if (content == 1){
		if (saveData == 0){
			insert_energy++;
		} else if (saveData == 1){
			insert_energy_DMW++;
		}
	}
	//start to insert
	if (content == 2){
		entries[row][OVER_HEAD + accessPort * (DISTANCE + 1)] = 1;
	}	else {
		entries[row][OVER_HEAD + accessPort * (DISTANCE + 1)] = content;
	}
}

/**
 * Delete data at accessPort(excluding virtual access ports)
 * e.g. If set MAX_SIZE 128 and DISTANCE 32, accessPort includes 0~4
 * @param accessPort The access port which skyrmions could be deleted
 * @param saveData If 0 save energy data to the first statics, 1 to the second
*/
void SkyrmionWord::deleteSky(int accessPort, int saveData)
{
	if (_intervalSize == 0 && (accessPort < 0 || accessPort > MAX_SIZE / DISTANCE)){
		cout << "Delete: accessPort is invalid!" << endl;
		exit(1);
	}
	if (_intervalSize != 0 && (accessPort < 0 || accessPort > _intervalSize)){
		cout << "Delete: accessPort is invalid!" << endl;
		exit(1);
	}
	//record energy of delete
	if (saveData == 0){
		delete_energy++;
	} else if (saveData == 1){
		delete_energy_DMW++;
	}
	//start to delete
	entry[OVER_HEAD + accessPort * (DISTANCE + 1)] = 0;//-1

}

/**
 * Delete data at accessPort(excluding virtual access ports)
 * e.g. If set MAX_SIZE 128 and DISTANCE 32, accessPort includes 0~4
 * @param accessPort The access port which skyrmions could be deleted
 * @param row The row of the entries
 * @param saveData If 0 save energy data to the first statics, 1 to the second
*/
void SkyrmionBit::deleteSky(int accessPort, int row, int saveData)
{
	if (_intervalSize == 0 && (accessPort < 0 || accessPort > MAX_SIZE / DISTANCE)){
		cout << "Delete: accessPort is invalid!" << endl;
		exit(1);
	}
	if (_intervalSize != 0 && (accessPort < 0 || accessPort > _intervalSize)){
		cout << "Delete: accessPort is invalid!" << endl;
		exit(1);
	}
	if (row < 0 || row >= ROW){
		cout << "Delete: row for bit-interleaved method needs to be 0~" << ROW-1 << endl;
		exit(1);
	}
	//record energy of delete
	if (saveData == 0){
		delete_energy++;
	} else if (saveData == 1){
		delete_energy_DMW++;
	}
	//start to delete
	entries[row][OVER_HEAD + accessPort * (DISTANCE + 1)] = 0;//-1
}

/**
 * Detect data at accessPort(excluding virtual access ports)
 * e.g. If set MAX_SIZE 128 and DISTANCE 32, accessPort includes 0~4
 * @param accessPort The access port which skyrmions could be detected
 * @param saveData If 0 save energy data to the first statics, 1 to the second
 * @return If data is bit 0, return 0; If bit 1, return 1
*/
sky_size_t SkyrmionWord::detect(int accessPort, int saveData)
{
	if (_intervalSize == 0 && (accessPort < 0 || accessPort > MAX_SIZE / DISTANCE)){
		cout << "Detect: accessPort is invalid!" << endl;
		exit(1);
	}
	if (_intervalSize != 0 && (accessPort < 0 || accessPort > _intervalSize)){
		cout << "Detect: accessPort is invalid!" << endl;
		exit(1);
	}
	// record energy of detect
	if (saveData == 0){
		detect_energy++;
	} else if (saveData == 1){
		detect_energy_DMW++;
	}
	//start to detect
	if (entry[OVER_HEAD + accessPort * (DISTANCE + 1)] == 0)
		return 0;
	else if (entry[OVER_HEAD + accessPort * (DISTANCE + 1)] == 1) // fix
		return 1;
	else
		return entry[OVER_HEAD + accessPort * (DISTANCE + 1)]; // fix
}

/**
 * Detect data at accessPort(excluding virtual access ports)
 * e.g. If set MAX_SIZE 128 and DISTANCE 32, accessPort includes 0~4
 * @param accessPort The access port which skyrmions could be detected
 * @param row The row of the entries
 * @param saveData If 0 save energy data to the first statics, 1 to the second
 * @return If data is bit 0, return 0; If bit 1, return 1
*/
sky_size_t SkyrmionBit::detect(int accessPort, int row, int saveData)
{
	if (_intervalSize == 0 && (accessPort < 0 || accessPort > MAX_SIZE / DISTANCE)){
		cout << "Detect: accessPort is invalid!" << endl;
		exit(1);
	}
	if (_intervalSize != 0 && (accessPort < 0 || accessPort > _intervalSize)){
		cout << "Detect: accessPort is invalid!" << endl;
		exit(1);
	}
	if (row < 0 || row >= ROW){
		cout << "Delete: row for bit-interleaved method needs to be 0~" << ROW-1 << endl;
		exit(1);
	}
	// record energy of detect
	if (saveData == 0){
		detect_energy++;
	} else if (saveData == 1){
		detect_energy_DMW++;
	}
	//start to detect
	if (entries[row][OVER_HEAD + accessPort * (DISTANCE + 1)] == 0)
		return 0;
	else if (entries[row][OVER_HEAD + accessPort * (DISTANCE + 1)] == 1) // fix
		return 1;
	else
		return entries[row][OVER_HEAD + accessPort * (DISTANCE + 1)]; // fix
}

/**
 * Read data with size bytes at address
 * @param address The address of the data which will be read
 * @param size Byte(s) of the data
 * @param parallel parallel 0 if read one byte data at once, parallel 1 if read data parallel
 * @param saveData If 0 save energy data to the first statics, 1 to the second
 * @return Bytes of the data at address
*/
sky_size_t *SkyrmionWord::read(Addr address, data_size_t size, int parallel, int saveData)
{
	if (_intervalSize == 0 && address >= MAX_SIZE/8){
		cout << "read: address is invalid (0 ~ " << MAX_SIZE/8-1 << ")" << endl;
		exit(1);
	}
	if (_intervalSize != 0 && address >= MAX_SIZE/8){
		cout << "read: address is invalid (0 ~ " << MAX_SIZE/8-1 << ")" << endl;
		exit(1);
	}
	int ports[2];
	int numAddrBtwPort = DISTANCE / 8;
	int port = address/numAddrBtwPort;
	int temp = ceil((float)size/numAddrBtwPort);

	int moves = determinePorts(address, size, ports);
	for (int i = 0; i < moves; i++){
		shift(ports[0], ports[1], saveData);
		if (saveData == 0){
			shift_latency++;
		} else if (saveData == 1){
			shift_latency_DMW++;
		}
	}

	if (parallel == 0) {
		sky_size_t *data = new sky_size_t [size * 8];
		if (ports[0] > ports[1]){ // read from bit 0 to bit 31
			for (int i = 0; i < size * 8; i++){
				shift(ports[0], ports[1], saveData);
				data[i] = detect(port, saveData);
				if (saveData == 0){
					detect_latency++;
					shift_latency++;
				} else if (saveData == 1){
					detect_latency_DMW++;
					shift_latency_DMW++;
				}
			}
		} else { // read from bit 31 to 0
			for (int i = 0; i < size * 8; i++){
				shift(ports[0], ports[1], saveData);
				data[size * 8 - 1 - i] = detect(port + 1, saveData);
				if (saveData == 0){
					detect_latency++;
					shift_latency++;
				} else if (saveData == 1){
					detect_latency_DMW++;
					shift_latency_DMW++;
				}
			}
		}
		// move data backward
		for (int i = 0; i < size * 8 + moves; i++){
			shift(ports[1], ports[0], saveData);
			if (saveData == 0){
				shift_latency++;
			} else if (saveData == 1){
				shift_latency_DMW++;
			}
		}
		return data;
	} else if (parallel == 1) {
		if (address % numAddrBtwPort != 0 && size % numAddrBtwPort == 0)
			temp++;
		sky_size_t *tmpBuffer = new sky_size_t [temp * DISTANCE];
		for (int i = 0; i < DISTANCE; i++){ // read from bit 0 to bit 31
			shift(ports[0], ports[1], saveData);
			for (int j = 0; j < temp; j++){
				tmpBuffer[j * DISTANCE + i] = detect(port + j, saveData);
			}
		}
		if (saveData == 0){
			if (address % numAddrBtwPort == 0 && ((address + size) % numAddrBtwPort) != 0){
				detect_energy -= (numAddrBtwPort - (address + size) % numAddrBtwPort) * 8;
			} else if (address % numAddrBtwPort != 0 && ((address + size) % numAddrBtwPort) == 0){
				detect_energy -= (address % numAddrBtwPort) * 8;
			} else if (address % numAddrBtwPort != 0 && ((address + size) % numAddrBtwPort) != 0){
				detect_energy -= ((numAddrBtwPort - (address + size) % numAddrBtwPort) * 8 + (address % numAddrBtwPort) * 8);
			}
		} else if (saveData == 1){
			if (address % numAddrBtwPort == 0 && ((address + size) % numAddrBtwPort) != 0){
				detect_energy_DMW -= (numAddrBtwPort - (address + size) % numAddrBtwPort) * 8;
			} else if (address % numAddrBtwPort != 0 && ((address + size) % numAddrBtwPort) == 0){
				detect_energy_DMW -= (address % numAddrBtwPort) * 8;
			} else if (address % numAddrBtwPort != 0 && ((address + size) % numAddrBtwPort) != 0){
				detect_energy_DMW -= ((numAddrBtwPort - (address + size) % numAddrBtwPort) * 8 + (address % numAddrBtwPort) * 8);
			}
		}
		sky_size_t *data = new sky_size_t [size * 8];
		sky_size_t *ptr = tmpBuffer + (address % numAddrBtwPort) * 8;
		memcpy(data, ptr, size * 8 * sizeof(sky_size_t));
		delete [] tmpBuffer;

		//move data backward
		for (int i = 0; i < DISTANCE; i++){
			shift(ports[1], ports[0], saveData);
		}

		//update detect_latency & shift_latency
		if (saveData == 0){
			detect_latency += DISTANCE;
			shift_latency += 2 * DISTANCE;
		} else if (saveData == 1){
			detect_latency_DMW += DISTANCE;
			shift_latency_DMW += 2 * DISTANCE;
		}
		return data;
	}
	return nullptr;
}

sky_size_t *Skyrmion::bitToByte(data_size_t size, sky_size_t *read)
{
	sky_size_t *ptr = read;
	sky_size_t *byte = new sky_size_t [size];
	for (int i = 0; i < size; i++){
		sky_size_t result = 0;
		for (int j = 0; j < 8; j++){
			sky_size_t oneBit = *(ptr + 7 - j);
			result = result | (oneBit << j);
		}
		byte[i] = result;
		ptr += 8;
	}
	return byte;
}

sky_size_t *Skyrmion::byteToBit(data_size_t size, const sky_size_t *read)
{
	sky_size_t *bit = new sky_size_t [size * 8];
	for (int i = 0; i < size; i++){
		sky_size_t tmp = *(read + i);
		for (int j = 0; j < 8; j++){
			if (tmp & (1 << (7-j)))
				bit[i * 8 + j] = 1;
			else
				bit[i * 8 + j] = 0;
		}
	}
	return bit;
}

/**
 * Read data with size bytes at address
 * @param address The address of the data which will be read
 * @param size Byte(s) of the data
 * @param parallel parallel 0 if read one byte data at a once, parallel 1 if read data parallel
 * @param saveData If 0 save energy data to the first statics, 1 to the second
 * @return Bytes of the data at address
*/
sky_size_t *SkyrmionWord::readData(Addr address, data_size_t size, int parallel, int saveData)
{
	if (size > ROW/8){
		cout << "read: size is invalid (0 ~ " << ROW/8 << ")" << endl;
		exit(1);
	}

	sky_size_t *data = new sky_size_t [size];
	sky_size_t *readPtr;
	sky_size_t *ptr;
	sky_size_t *dataPtr = data;
	int numAddrBtwPort = DISTANCE / 8;
	if (parallel == 0){
		int times = ceil(float(size)/numAddrBtwPort);
		Addr newAddr = address;
		if (address % numAddrBtwPort == 0){
			for (int i = 0; i < times; i++){
				if (i == times-1 && ((address + size) % numAddrBtwPort) != 0){
					readPtr = read(newAddr, (address + size) % numAddrBtwPort, 0, saveData);
					ptr = bitToByte((address + size) % numAddrBtwPort, readPtr);
					memcpy(dataPtr, ptr, ((address + size) % numAddrBtwPort) * sizeof(sky_size_t));
					delete [] readPtr;
					delete [] ptr;
					readPtr = nullptr;
					ptr = nullptr;
				} else {
					readPtr = read(newAddr, numAddrBtwPort, 0, saveData);
					ptr = bitToByte(numAddrBtwPort, readPtr);
					memcpy(dataPtr, ptr, numAddrBtwPort * sizeof(sky_size_t));
					newAddr += numAddrBtwPort;
					dataPtr += numAddrBtwPort;
					delete [] readPtr;
					delete [] ptr;
					readPtr = nullptr;
					ptr = nullptr;
				}
			}
			return data;
		} else {
			if ((size < numAddrBtwPort) && ((address % numAddrBtwPort) + size <= numAddrBtwPort)){
				readPtr = read(address, size, 0, saveData);
				ptr = bitToByte(size, readPtr);
				memcpy(dataPtr, ptr, size * sizeof(sky_size_t));
				delete [] readPtr;
				delete [] ptr;
				readPtr = nullptr;
				ptr = nullptr;
				return data;
			} else {
				if (((size < numAddrBtwPort) && ((address % numAddrBtwPort) + size > numAddrBtwPort)) || (size % numAddrBtwPort == 0))
					times++;
				for (int i = 0; i < times; i++){
					if (i == 0){
						readPtr = read(newAddr, numAddrBtwPort - address % numAddrBtwPort, 0, saveData);
						ptr = bitToByte(numAddrBtwPort - address % numAddrBtwPort, readPtr);
						memcpy(dataPtr, ptr, (numAddrBtwPort - address % numAddrBtwPort) * sizeof(sky_size_t));
						newAddr += numAddrBtwPort - address % numAddrBtwPort;
						dataPtr += numAddrBtwPort - address % numAddrBtwPort;
						delete readPtr;
						delete [] ptr;
						readPtr = nullptr;
						ptr = nullptr;
					} else if (i == times-1 && ((address + size) % numAddrBtwPort) != 0) {
						readPtr = read(newAddr, (address + size) % numAddrBtwPort, 0, saveData);
						ptr = bitToByte((address + size) % numAddrBtwPort, readPtr);
						memcpy(dataPtr, ptr, ((address + size) % numAddrBtwPort) * sizeof(sky_size_t));
						delete [] readPtr;
						delete [] ptr;
						readPtr = nullptr;
						ptr = nullptr;
					} else {
						readPtr = read(newAddr, numAddrBtwPort, 0, saveData);
						ptr = bitToByte(numAddrBtwPort, readPtr);
						memcpy(dataPtr, ptr, numAddrBtwPort * sizeof(sky_size_t));
						newAddr += numAddrBtwPort;
						dataPtr += numAddrBtwPort;
						delete [] readPtr;
						delete [] ptr;
						readPtr = nullptr;
						ptr = nullptr;
					}
				}
				return data;
			}
		}
	} else {
		if (size <= numAddrBtwPort){
			return readData(address, size, 0, saveData);
		} else {
			readPtr = read(address, size, 1, saveData);
			ptr = bitToByte(size, readPtr);
			memcpy(dataPtr, ptr, size * sizeof(sky_size_t));
			delete [] readPtr;
			delete [] ptr;
			readPtr = nullptr;
			ptr = nullptr;
			return data;
		}
	}
	return nullptr;
}

/**
 * Read data with size bytes at address in block
 * @param block The block(cache line) of the data
 * @param address The address(offset) of the data which will be read
 * @param size Byte(s) of the data
 * @param saveData If 0 save energy data to the first statics, 1 to the second
 * @return Bytes of the data at address in block
*/
sky_size_t *SkyrmionBit::read(int block, Addr address, data_size_t size, int saveData)
{
	if (block < 0 || block >= MAX_SIZE){
		cout << "read: block is invalid (0 ~ " << MAX_SIZE-1 << ")" << endl;
		exit(1);
	} else if (address >= ROW/8){
		cout << "read: address is invalid (0 ~ " << ROW/8-1 << ")" << endl;
		exit(1);
	}
	blockUsedNumber[block]++;

	sky_size_t *buf = new sky_size_t [size * 8];
	int ports[2];
	int moves = determinePorts(block, ports);
	int port = block / DISTANCE;
	if (ports[0] < ports[1]){
		port++;
	}

	for (int i = 0; i < moves; i++){
		shift(ports[0], ports[1], address, size, saveData);
		if (saveData == 0){
			shift_latency++;
		} else if (saveData == 1){
			shift_latency_DMW++;
		}
	}

	int addr = address * 8;
	for (int k = 0; k < size * 8; k++){
		buf[k] = detect(port, addr + k, saveData);
	}
	if (saveData == 0){
		detect_latency++;
	} else if (saveData == 1){
		detect_latency_DMW++;
	}

	// move data backward
	for (int i = 0; i < moves; i++){
		shift(ports[1], ports[0], address, size, saveData);
		if (saveData == 0){
			shift_latency++;
		} else if (saveData == 1){
			shift_latency_DMW++;
		}
	}
	sky_size_t *ptr = bitToByte(size, buf);

	delete [] buf;
	return ptr;
}

/**
 * Write data with size byte(s) at address
 * @param address The address(offset) of the data which will be written
 * @param size Byte(s) of the data
 * @param content The data which will be written
 * @param type The write type, including NAIVE_TRADITIONAL, NAIVE, DCW_TRADITIONL,...
 * @param saveData If 0 save energy data to the first statics, 1 to the second
*/
void SkyrmionWord::write(Addr address, data_size_t size, const sky_size_t *content, enum Write_Type type, int saveData)
{
	if (_intervalSize == 0 && address >= MAX_SIZE/8){
		cout << "write: address is invalid (0 ~ " << MAX_SIZE/8-1 << ")" << endl;
		exit(1);
	}
	if (_intervalSize != 0 && address >= _intervalSize*4){
		cout << "write: address is invalid (0 ~ " << _intervalSize*4-1 << ")" << endl;
		exit(1);
	}
	int ports[2];
	int moves = determinePorts(address, size, ports);
	if (moves % 8 != 0){
		cout << "wrtie: determinePorts result is wrong\n";
		exit(1);
	}

	_write_type = type;
	bool insrtKeep = false;
	bool delKeep = false;
	int numAddrBtwPort = DISTANCE / 8;
	int port = address / numAddrBtwPort;
	int temp = ceil((float)size/numAddrBtwPort);

	switch(type){
		case NAIVE_TRADITIONAL:
			// move data to the access port
			for (int i = 0; i < moves; i++){
				shift(ports[0], ports[1], saveData);
				if (saveData == 0){
					shift_latency++;
				} else if (saveData == 1){
					shift_latency_DMW++;
				}
			}

			if (ports[0] > ports[1]){ // write from bit 31 to bit 0
				// 1. delete all
				for (int i = 0; i < size * 8; i++){
					shift(ports[0], ports[1], saveData);
					deleteSky(port, saveData);
					if (saveData == 0){
						shift_latency++;
						delete_latency++;
					} else if (saveData == 1){
						shift_latency_DMW++;
						delete_latency_DMW++;
					}
				}

				// 2. insert a skyrmion/non-skyrmion according to content
				for (int i = 0; i < size * 8; i++){
					if (*(content + size * 8 - 1 - i) == 1){
						insert(port, 1, saveData);
						if (saveData == 0){
							insert_latency++;
						} else if (saveData == 1){
							insert_latency_DMW++;
						}
					} else {
						insert(port, 0, saveData);
					}
					shift(ports[1], ports[0], saveData);
					if (saveData == 0){
						shift_latency++;
					} else if (saveData == 1){
						shift_latency_DMW++;
					}
				}
				// move back to its original place
				for (int i = 0; i < moves; i++){
					shift(ports[1], ports[0], saveData);
					if (saveData == 0){
						shift_latency++;
					} else if (saveData == 1){
						shift_latency_DMW++;
					}
				}
			} else { // write from bit 0 to 31
				//1. delete all
				for (int i = 0; i < size * 8; i++){
					shift(ports[0], ports[1], saveData);
					deleteSky(port + 1, saveData);
					if (saveData == 0){
						shift_latency++;
						delete_latency++;
					} else if (saveData == 1){
						shift_latency_DMW++;
						delete_latency_DMW++;
					}
				}

				//2. insert a skyrmion/non-skyrmion according to content
				for (int i = 0; i < size * 8; i++){
					if (*(content + i) == 1){
						insert(port + 1, 1, saveData);
						if (saveData == 0){
							insert_latency++;
						} else if (saveData == 1){
							insert_latency_DMW++;
						}
					} else {
						insert(port + 1, 0, saveData);
					}
					shift(ports[1], ports[0], saveData);
					if (saveData == 0){
						shift_latency++;
					} else if (saveData == 1){
						shift_latency_DMW++;
					}
				}
				//move back to its original place
				for (int i = 0; i < moves; i++){
					shift(ports[1], ports[0], saveData);
					if (saveData == 0){
						shift_latency++;
					} else if (saveData == 1){
						shift_latency_DMW++;
					}
				}
			}
			break;
		case NAIVE:
				if (address % numAddrBtwPort != 0 && size % numAddrBtwPort == 0)
					temp++;

				//A. left and right ends both align to the access ports
				if (address % numAddrBtwPort == 0 && ((address + size) % numAddrBtwPort) == 0){
					//1. delete all
					for (int i = 0; i < DISTANCE; i++){
						shift(ports[0], ports[1], saveData);
						for (int j = 0; j < temp; j++){
							deleteSky(port + j, saveData);
						}
					}

					//2. insert a skyrmion/non-skyrmion according to content
					for (int i = 1; i <= DISTANCE; i++){
						insrtKeep = false;
						for (int j = 1; j <= temp; j++){
							if (*(content + DISTANCE * j - i) == 1){
								insert(port + j - 1, 1, saveData);
								insrtKeep = true;
							} else {
								insert(port + j - 1, 0, saveData);
							}
						}
						shift(ports[1], ports[0], saveData);
						if (insrtKeep == 1){
							if (saveData == 0){
								insert_latency++;
							} else if (saveData == 1){
								insert_latency_DMW++;
							}
						}
					}

				//B. Only left end aligns to the access port
				} else if (address % numAddrBtwPort == 0 && ((address + size) % numAddrBtwPort) != 0){
					//1. delete all
					for (int i = 0; i < DISTANCE; i++){
						shift(ports[0], ports[1], saveData);
						for (int j = 0; j < temp; j++){
							if (!(j == temp -1 && i >= (address + size) % numAddrBtwPort * 8))
								deleteSky(port + j, saveData);
						}
					}
					//2. insert a skyrmion/non-skyrmion according to content
					for (int i = 1; i <= DISTANCE; i++){
						insrtKeep = false;
						for (int j = 1; j <= temp; j++){
							if (!(j == temp && i <= (numAddrBtwPort - (address + size) % numAddrBtwPort) * 8)){
								if (*(content + DISTANCE * j - i) == 1){
									insert(port + j - 1, 1, saveData);
									insrtKeep = true;
								} else {
									insert(port + j - 1, 0, saveData);
								}
							}
						}
						shift(ports[1], ports[0], saveData);
						if (insrtKeep == 1){
							if (saveData == 0){
								insert_latency++;
							} else if (saveData == 1){
								insert_latency_DMW++;
							}
						}
					}
				//C. Only right end aligns to the access port
				} else if (address % numAddrBtwPort != 0 && ((address + size) % numAddrBtwPort) == 0){
					//1. delete all
					for (int i = 0; i < DISTANCE; i++){
						shift(ports[0], ports[1], saveData);
						for (int j = 0; j < temp; j++){
							if (!(j == 0 && (i < (address % numAddrBtwPort) * 8)))
								deleteSky(port + j, saveData);
						}
					}
					//2. insert a skyrmion/non-skyrmion according to content
					for (int i = 1; i <= DISTANCE; i++){
						insrtKeep = false;
						for (int j = 0; j < temp; j++){
							if (!(j == 0 && i > (numAddrBtwPort - address % numAddrBtwPort) * 8)){
								if (*(content + (numAddrBtwPort - address % numAddrBtwPort) * 8 + DISTANCE * j - i) == 1){
									insert(port + j, 1, saveData);
									insrtKeep = true;
								} else {
									insert(port + j, 0, saveData);
								}
							}
						}
						shift(ports[1], ports[0], saveData);
						if (insrtKeep == 1){
							if (saveData == 0){
								insert_latency++;
							} else if (saveData == 1){
								insert_latency_DMW++;
							}
						}
					}

				//D. Both ends do not align to the access ports
				} else {
					//1. delete all
					for (int i = 0; i < DISTANCE; i++){
						shift(ports[0], ports[1], saveData);
						for (int j = 0; j < temp; j++){
							if (!((j == 0 && i < (address % numAddrBtwPort) * 8) || (j == temp -1 && i >= (address + size) % numAddrBtwPort * 8))){
								deleteSky(port + j, saveData);
							}
						}
					}
					//2. insert a skyrmion/non-skyrmion according to content
					for (int i = 1; i <= DISTANCE; i++){
						insrtKeep = false;
						for (int j = 0; j < temp; j++){
							if (!((j == 0 && i > (numAddrBtwPort - address % numAddrBtwPort) * 8) || (j == temp - 1 && i <= (numAddrBtwPort - (address + size) % numAddrBtwPort) * 8))){
								if (*(content + (numAddrBtwPort - address % numAddrBtwPort) * 8 + DISTANCE * j - i) == 1){
									insert(port + j, 1, saveData);
									insrtKeep = true;
								} else {
									insert(port + j, 0, saveData);
								}
							}
						}
						shift(ports[1], ports[0], saveData);
						if (insrtKeep == 1){
							if (saveData == 0){
								insert_latency++;
							} else if (saveData == 1){
								insert_latency_DMW++;
							}
						}
					}
				}
				//update latency
				if (saveData == 0){
					delete_latency += DISTANCE;
					shift_latency += 2*DISTANCE;
				} else if (saveData == 1){
					delete_latency_DMW += DISTANCE;
					shift_latency_DMW += 2*DISTANCE;
				}

			break;
		case DCW_TRADITIONAL:
			// move data to the access port
			for (int i = 0; i < moves; i++){
				shift(ports[0], ports[1], saveData);
				if (saveData == 0){
					shift_latency++;
				} else if (saveData == 1){
					shift_latency_DMW++;
				}
			}
			if (ports[0] > ports[1]){ // write from bit 0 to bit 31
				for (int i = 0; i < size * 8; i++){
					shift(ports[0], ports[1], saveData);
					if (saveData == 0){
						shift_latency++;
						detect_latency++;
					} else if (saveData == 1){
						shift_latency_DMW++;
						detect_latency_DMW++;
					}
					if (detect(port, saveData) != *(content + i)){
						// test whether data are identical. if not, delete and then insert
						if (*(content + i) == 0){
							deleteSky(port, saveData);
							insert(port, 0, saveData);
							if (saveData == 0){
								delete_latency++;
							} else if (saveData == 1){
								delete_latency_DMW++;
							}
						}	else {
							insert(port, 1, saveData);
							if (saveData == 0){
								insert_latency++;
							} else if (saveData == 1){
								insert_latency_DMW++;
							}
						}
					}
				}
			} else { // write from bit 31 to 0
				for (int i = 0; i < size * 8; i++){
					shift(ports[0], ports[1], saveData);
					if (saveData == 0){
						shift_latency++;
						detect_latency++;
					} else if (saveData == 1){
						shift_latency_DMW++;
						detect_latency_DMW++;
					}
					if (detect(port + 1, saveData) != *(content + size * 8 - 1 - i)){
						// test whether data are identical. if not, delete and then insert
						if (*(content + size * 8 - 1 - i) == 0){
							deleteSky(port + 1, saveData);
							insert(port + 1, 0, saveData);
							if (saveData == 0){
								delete_latency++;
							} else if (saveData == 1){
								delete_latency_DMW++;
							}
						}	else {
							insert(port + 1, 1, saveData);
							if (saveData == 0){
								insert_latency++;
							} else if (saveData == 1){
								insert_latency_DMW++;
							}
						}
					}
				}
			}
			// move data backward
			for (int i = 0; i < size * 8 + moves; i++){
				shift(ports[1], ports[0], saveData);
				if (saveData == 0){
					shift_latency++;
				} else if (saveData == 1){
					shift_latency_DMW++;
				}
			}
			break;

		case DCW:
				if (address % numAddrBtwPort != 0 && size % numAddrBtwPort == 0)
					temp++;
				// A. left and right ends both align to the access ports
				if (address % numAddrBtwPort == 0 && ((address + size) % numAddrBtwPort) == 0){
					for (int i = 0; i < DISTANCE; i++){
						shift(ports[0], ports[1], saveData);
						insrtKeep = false;
						delKeep = false;
						for (int j = 0; j < temp; j++){
							if (detect(port + j, saveData) != *(content + DISTANCE * j + i)){
								if (*(content + DISTANCE * j + i) == 0){
									deleteSky(port + j, saveData);
									insert(port + j, 0, saveData);
									delKeep = true;
								}	else {
									insert(port + j, 1, saveData);
									insrtKeep = true;
								}
							}
						}
						if (insrtKeep == 1){
							if (saveData == 0){
								insert_latency++;
							} else if (saveData == 1){
								insert_latency_DMW++;
							}
						}
						if (delKeep == 1){
							if (saveData == 0){
								delete_latency++;
							} else if (saveData == 1){
								delete_latency_DMW++;
							}
						}
					}
				// B. Only left end aligns to the access port
				} else if (address % numAddrBtwPort == 0 && ((address + size) % numAddrBtwPort) != 0){
					for (int i = 0; i < DISTANCE; i++){
						shift(ports[0], ports[1], saveData);
						insrtKeep = false;
						delKeep = false;
						for (int j = 0; j < temp; j++){
							if (!(j == temp -1 && i >= (address + size) % numAddrBtwPort * 8)){
								if (detect(port + j, saveData) != *(content + DISTANCE * j + i)){
									if (*(content + DISTANCE * j + i) == 0){
										deleteSky(port + j, saveData);
										insert(port + j, 0, saveData);
										delKeep = true;
									}	else {
										insert(port + j, 1, saveData);
										insrtKeep = true;
									}
								}
							}
						}
						if (insrtKeep == 1){
							if (saveData == 0){
								insert_latency++;
							} else if (saveData == 1){
								insert_latency_DMW++;
							}
						}
						if (delKeep == 1){
							if (saveData == 0){
								delete_latency++;
							} else if (saveData == 1){
								delete_latency_DMW++;
							}
						}
					}
				// C. Only right end aligns to the access port
				} else if (address % numAddrBtwPort != 0 && ((address + size) % numAddrBtwPort) == 0){
					for (int i = 0; i < DISTANCE; i++){
						shift(ports[0], ports[1], saveData);
						insrtKeep = false;
						delKeep = false;
						for (int j = 0; j < temp; j++){
							if (!(j == 0 && (i < (address % numAddrBtwPort) * 8))) {
								if (detect(port + j, saveData) != *(content - address % numAddrBtwPort * 8 + DISTANCE * j + i)){
									if (*(content - address % numAddrBtwPort * 8 + DISTANCE * j + i) == 0){
										deleteSky(port + j, saveData);
										insert(port + j, 0, saveData);
										delKeep = true;
									}	else {
										insert(port + j, 1, saveData);
										insrtKeep = true;
									}
								}
							}
						}
						if (insrtKeep == 1){
							if (saveData == 0){
								insert_latency++;
							} else if (saveData == 1){
								insert_latency_DMW++;
							}
						}
						if (delKeep == 1){
							if (saveData == 0){
								delete_latency++;
							} else if (saveData == 1){
								delete_latency_DMW++;
							}
						}
					}
				// D. Both ends do not align to the access ports
				} else {
					for (int i = 0; i < DISTANCE; i++){
						shift(ports[0], ports[1], saveData);
						insrtKeep = false;
						delKeep = false;
						for (int j = 0; j < temp; j++){
							if (!((j == 0 && (i < (address % numAddrBtwPort) * 8)) || (j == temp -1 && i >= (address + size) % numAddrBtwPort * 8))) {
								if (detect(port + j, saveData) != *(content - address % numAddrBtwPort * 8 + DISTANCE * j + i)){
									if (*(content - address % numAddrBtwPort * 8 + DISTANCE * j + i) == 0){
										deleteSky(port + j, saveData);
										insert(port + j, 0, saveData);
										delKeep = true;
									}	else {
										insert(port + j, 1, saveData);
										insrtKeep = true;
									}
								}
							}
						}
						if (insrtKeep == 1){
							if (saveData == 0){
								insert_latency++;
							} else if (saveData == 1){
								insert_latency_DMW++;
							}
						}
						if (delKeep == 1){
							if (saveData == 0){
								delete_latency++;
							} else if (saveData == 1){
								delete_latency_DMW++;
							}
						}
					}
				}
				// move data backward
				for (int i = 0; i < DISTANCE; i++){
					shift(ports[1], ports[0], saveData);
				}
				// update latency
				if (saveData == 0){
					detect_latency += DISTANCE;
					shift_latency += 2*DISTANCE;
				} else if (saveData == 1){
					detect_latency_DMW += DISTANCE;
					shift_latency_DMW += 2*DISTANCE;
				}

			break;

		case PERMUTATION_WRITE: //; labels can only be followed by statements, and declarations do not count as statements in C
		{
			int sky_count = 0;
			// move data to the access port
			for (int i = 0; i < moves; i++){
				shift(ports[0], ports[1], saveData);
				if (saveData == 0){
					shift_latency++;
				} else if (saveData == 1){
					shift_latency_DMW++;
				}
			}

			// 1. assemble the existing skyrmions
			if (ports[0] > ports[1]){ // assemble from bit 0 to bit 31
				// because it only deals with at most one word at a time for word-based, alter ports[1] to ports[0]-1
				if ((address % numAddrBtwPort != 0) && ((address % numAddrBtwPort) + size < numAddrBtwPort)){
					shift(ports[0], ports[1], saveData);
				} else {
					shift(ports[0], ports[0] - 1, saveData);
				}
				if (saveData == 0){
					shift_latency++;
				} else if (saveData == 1){
					shift_latency_DMW++;
				}
				for (int i = 0; i < size * 8; i++){
					if (saveData == 0){
						detect_latency++;
						shift_latency++;
					} else if (saveData == 1){
						detect_latency_DMW++;
						shift_latency_DMW++;
					}
					if (detect(port, saveData) == 0){ // no skyrmion
						shift(ports[0], ports[0] - 1, saveData);
					} else { // a skyrmion
						shift(ports[0], ports[1], saveData);
						sky_count++;
					}
				}
			} else { // asseble from bit 31 to 0
				if ((address % numAddrBtwPort != 0) && ((address % numAddrBtwPort) + size < numAddrBtwPort)){
					shift(ports[0], ports[1], saveData);
				} else {
					shift(ports[0], ports[0] + 1, saveData);
				}
				if (saveData == 0){
					shift_latency++;
				} else if (saveData == 1){
					shift_latency_DMW++;
				}
				for (int i = 0; i < size * 8; i++){
					if (saveData == 0){
						detect_latency++;
						shift_latency++;
					} else if (saveData == 1){
						detect_latency_DMW++;
						shift_latency_DMW++;
					}
					if (detect(port + 1, saveData) == 0){ // no skyrmion
						shift(ports[0], ports[0] + 1, saveData);
					} else {
						shift(ports[0], ports[1], saveData);
						sky_count++;
					}
				}
			}

			//2. re-permute the exising skyrmions & inject new skyrmions (if needed)
			if (ports[0] > ports[1]){ // write from bit 31 to bit 0
				for (int i = 0; i < size * 8; i++){
					if (saveData == 0){
						shift_latency++;
					} else if (saveData == 1){
						shift_latency_DMW++;
					}
					if (*(content + size * 8 - 1 - i) == 0) {
						shift(ports[0] - 1, ports[0], saveData);
						insert(port, 0, saveData);
					} else {
						if (sky_count > 0) {
							shift(ports[1], ports[0], saveData);
							sky_count--;
						} else {
							shift(ports[0] - 1, ports[0], saveData);
							insert(port, 1, saveData);
							if (saveData == 0){
								insert_latency++;
							} else if (saveData == 1){
								insert_latency_DMW++;
							}
						}
					}
				}
				if ((address % numAddrBtwPort != 0) && ((address % numAddrBtwPort) + size < numAddrBtwPort)){
					shift(ports[1], ports[0], saveData);
				} else {
					shift(ports[0] - 1, ports[0], saveData);
				}
				if (saveData == 0){
					shift_latency++;
				} else if (saveData == 1){
					shift_latency_DMW++;
				}
			} else { // write from bit 0 to bit 31
				for (int i = 0; i < size * 8; i++){
					if (saveData == 0){
						shift_latency++;
					} else if (saveData == 1){
						shift_latency_DMW++;
					}
					if (*(content + i) == 0){
						shift(ports[0] + 1, ports[0], saveData);
						insert(port + 1, 0, saveData);
					} else {
						if (sky_count > 0) {
							shift(ports[1], ports[0], saveData);
							sky_count--;
						} else {
							shift(ports[0] + 1, ports[0], saveData);
							insert(port + 1, 1, saveData);
							if (saveData == 0){
								insert_latency++;
							} else if (saveData == 1){
								insert_latency_DMW++;
							}
						}
					}
				}
				if ((address % numAddrBtwPort != 0) && ((address % numAddrBtwPort) + size < numAddrBtwPort)){
					shift(ports[1], ports[0], saveData);
				} else {
					shift(ports[0] + 1, ports[0], saveData);
				}
				if (saveData == 0){
					shift_latency++;
				} else if (saveData == 1){
					shift_latency_DMW++;
				}
			}

			// 3. delete excess skyrmions(if any)
			while (sky_count > 0){
				if (ports[0] > ports[1]){
					shift(ports[1], port + 1, saveData);
					deleteSky(port, saveData);
				}	else {
					shift(ports[1], port + 2, saveData);
					deleteSky(port + 1, saveData);
				}
				sky_count--;
				if (saveData == 0){
					shift_latency++;
					delete_latency++;
				} else if (saveData == 1){
					shift_latency_DMW++;
					delete_latency_DMW++;
				}
			}
			// move data backwards
			for (int i = 0; i < moves; i++){
				shift(ports[1], ports[0], saveData);
				if (saveData == 0){
					shift_latency++;
				} else if (saveData == 1){
					shift_latency_DMW++;
				}
			}
			break;
		}
		default:
		{
				cout << "No such word-based method.\n";
				exit(1);
				break;
		}
	}
}

/**
 * Write data with size byte(s) at address
 * @param address The address(offset) of the data which will be written
 * @param size Byte(s) of the data
 * @param content The data which will be written
 * @param type The write type, including NAIVE_TRADITIONAL, NAIVE, DCW_TRADITIONL,...
 * @param saveData If 0 save energy data to the first statics, 1 to the second
*/
void SkyrmionWord::writeData(Addr address, data_size_t size, const sky_size_t *content, enum Write_Type type, int saveData)
{
	//mark checkFullArray
	for (int i = 0; i < size; i++){
		if (checkFullArray[address + i] == 0) {
			n_checkFull++;
			checkFullArray[address + i] = 1;
		}
	}
	int numAddrBtwPort = DISTANCE / 8;
	sky_size_t *ptr;
	if (type == PERMUTATION_WRITE || type == NAIVE_TRADITIONAL || type == DCW_TRADITIONAL || size <= numAddrBtwPort){
		if (type == NAIVE)
			type = NAIVE_TRADITIONAL;
		else if (type == DCW)
			type = DCW_TRADITIONAL;
		int times = ceil(float(size)/numAddrBtwPort);
		Addr newAddr = address;
		if (address % numAddrBtwPort == 0){
			for (int i = 0; i < times; i++){
				if (i == times-1 && ((address + size) % numAddrBtwPort) != 0){
					ptr = byteToBit((address + size) % numAddrBtwPort, content);
					write(newAddr, (address + size) % numAddrBtwPort, ptr, type, saveData);
					delete [] ptr;
					ptr = nullptr;
				} else {
					ptr = byteToBit(numAddrBtwPort, content);
					write(newAddr, numAddrBtwPort, ptr, type, saveData);
					newAddr += numAddrBtwPort;
					content += numAddrBtwPort;
					delete [] ptr;
					ptr = nullptr;
				}
			}
		} else {
			if ((size < numAddrBtwPort) && ((address % numAddrBtwPort) + size <= numAddrBtwPort)){
				ptr = byteToBit(size, content);
				write(address, size, ptr, type, saveData);
				delete [] ptr;
				ptr = nullptr;
			} else {
				if (((size < numAddrBtwPort) && ((address % numAddrBtwPort) + size > numAddrBtwPort)) || (size % numAddrBtwPort == 0))
					times++;
				for (int i = 0; i < times; i++){
					if (i == 0){
						ptr = byteToBit(numAddrBtwPort - address % numAddrBtwPort, content);
						write(newAddr, numAddrBtwPort - address % numAddrBtwPort, ptr, type, saveData);
						content += numAddrBtwPort - address % numAddrBtwPort;
						newAddr += numAddrBtwPort - address % numAddrBtwPort;
						delete [] ptr;
						ptr = nullptr;
					} else if (i == times-1 && ((address + size) % numAddrBtwPort) != 0) {
						ptr = byteToBit((address + size) % numAddrBtwPort, content);
						write(newAddr, (address + size) % numAddrBtwPort, ptr, type, saveData);
						delete [] ptr;
						ptr = nullptr;
					} else {
						ptr = byteToBit(numAddrBtwPort, content);
						write(newAddr, numAddrBtwPort, ptr, type, saveData);
						content += numAddrBtwPort;
						newAddr += numAddrBtwPort;
						delete [] ptr;
						ptr = nullptr;
					}
				}
			}
		}
	} else {
		ptr = byteToBit(size, content);
		write(address, size, ptr, type, saveData);
		delete [] ptr;
		ptr = nullptr;
	}
}

/**
 * Determine whether the race track is full or not
*/
bool SkyrmionWord::isFull()
{
	return n_checkFull == MAX_SIZE / 8;
}

/**
 * Count the number of shift to repermute
 * @param data The data which will be examined
 * @param numReuse The number of bit 1 in the buffer
 * @return The number of shift to repermute the saved skyrmions
*/
int SkyrmionBit::countNumShift(sky_size_t *data, int &numReuse)
{
	int i = 0;
	int num = 0;
	while (numReuse > 0 && i < BUFFER_LENGTH){
		if (data[i] == 1){
				numReuse--;
				num = i + 1;
		}
		i++;
	}
	return num;
}

/**
 * Count the bit 1 in content and save the index of the first bit 1 in start
 * @param content The content to be counted
 * @param start The index of the first bit 1 in the content
 * @param order Foward order is 0, while reverse order is 1
 * @return The number of the bit 1 in the content
*/
int SkyrmionBit::countBit1(const sky_size_t *content, int &start, int order)
{
	int n = 0;
	bool first = false;
	if (order == 0){
		for (int i = 0; i < 8; i++){
			if (content[i] == 1){
				n++;
				if (!first){
					start = i;
					first = true;
				}
			}
		}
	} else {
		for (int i = 8; i > 0; i--){
			if (content[i - 1] == 1){
				n++;
				if (!first){
					start = i - 1;
					first = true;
				}
			}
		}
	}
	return n;
}

/**
 * Compare the bit pattern of the content1 and content2
 * @param content1 The old data will be updated
 * @param contetn2 The new data
 * @param order Foward order is 0, while reverse order is 1
 * @return vector[0] 0 if the bits are all 0 for content1 and content2
 * @return vector[0] 1 if the bits are all 0 for content1
 * @return vector[0] 2 if the bits are all 0 for content2
 * @return vector[0] 3 if the number of bit 1 are the same
 * @return vector[0] 4 if the number of bit 1 of content1 is longer than content2
 * @return vector[0] 5 if the number of bit 1 of content2 is longer than content1
 * @return vector[1] shift Compared with the content2, we need to shift vertically shift steps
 *  positive shift represents moving downwards, negative upwards
 * @return vector[2] nextStart1 We need to address the rest of the content1 and it record the start index
 * @return vector[3] nextStart2 We need to address the rest of the content2 and it record the start index
*/
vector<int> SkyrmionBit::cmpPattern(const sky_size_t *content1, const sky_size_t *content2, int order)
{
	vector<int> result(4);
	int start1 = 0;
	int start2 = 0;
	int n1 = 0;
	int n2 = 0;
	//forward order
	if (order == 0) {
		n1 = countBit1(content1, start1, 0);
		n2 = countBit1(content2, start2, 0);
		int end = min(8 - start1, 8 - start2);
		int i = 0;
		for (i = 0; i < end; ++i){
			if (content1[start1 + i] != content2[start2 + i]){
				break;
			}
		}
		result[2] = start1 + i;
		result[3] = start2 + i;
	//reverse order
	} else {
		start1 = BUFFER_LENGTH - 1;
		start2 = BUFFER_LENGTH - 1;
		n1 = countBit1(content1, start1, 1);
		n2 = countBit1(content2, start2, 1);
		int end = min(start1, start2);
		int i = 0;
		for (i = 0; i <= end; ++i){
			if (content1[start1 - i] != content2[start2 - i]){
				break;
			}
		}
		result[2] = start1 - i;
		result[3] = start2 - i;
	}
	result[1] = start2 - start1;
	if (n1 == 0 && n2 == 0){
		result[0] = 0;
	} else if (n1 == 0){
		result[0] = 1;
	} else if (n2 == 0){
		result[0] = 2;
	} else if (n1 == n2){
		result[0] = 3;
	} else if (n1 > n2){
		result[0] = 4;
	} else {
		result[0] = 5;
	}
	return result;
}

int SkyrmionBit::assessPureShift(const sky_size_t *content1, const sky_size_t *content2, vector<int> cmpPatternResult, int order, int buffer)
{
	int insertToVShift = INSERT / VSHIFT;
	int deleteToVShift = DELETE / VSHIFT;
	int result = 0;
	int shift = cmpPatternResult[1];
	int start1 = cmpPatternResult[2];
	int start2 = cmpPatternResult[3];

	//contetn2 = 0 or contetn1 = 0
	if (cmpPatternResult[0] == 2 || cmpPatternResult[0] == 1){
		return assessPW(content1, content2, buffer);
	}

	//forward order, equal to assessDCWs
	if (order == 0){
		int n1 = 8 - start1;
		int n2 = 8 - start2;

		if (n1 <= n2){
			for (int i = 0; i < n1; i++){
				if (content1[start1 + i] == 1 && content2[start2 + i] == 0){
					if (buffer >= 8){
						result += deleteToVShift;
					} else {
						result += deleteToVShift + insertToVShift;
					}
				} else if (content1[start1 + i] == 0 && content2[start2 + i] == 1){
					result += insertToVShift;
				}
			}
			for (int i = 0; i < n2 - n1; i++){
				if (content2[start2 + n1 + i] == 1){
					result += insertToVShift;
				}
			}
			return result + abs(shift);
		} else if (n1 > n2){ //D V
			for (int i = 0; i < n2; i++){
				if (content1[start1 + i] == 0 && content2[start2 + i] == 1){
					result += insertToVShift;
				} else if (content1[start1 + i] == 1 && content2[start2 + i] == 0){ //V
					if (buffer >= 8){
						result += deleteToVShift;
					} else {
						result += deleteToVShift + insertToVShift;
					}
				}
			}
			for (int i = 0; i < n1 - n2; i++){
				if (content1[start1 + n2 + i] == 1){
					if (buffer >= 8){
						result += deleteToVShift;
					} else {
						result += deleteToVShift + insertToVShift;
					}
				}
			}
			if (cmpPatternResult[0] == 1){
				return result;
			} else {
				return result + abs(shift);
			}
		}
	//reverse order
	} else {
		if (start1 <= start2){
			int shiftTemp = 0;
			for (int i = 0; i <= start1; i++){ //G0 V
				if (content1[i] == 1){
					if (buffer >= 8){
						result += deleteToVShift;
						//shiftTemp = (i + 1) << 1;
					} else {
						shiftTemp = (i + 1) << 1;
						buffer++;
					}
				}
			}
			for (int i = 0; i < start2 - start1; i++){ //G1 V
				if (content2[i] == 1){
					if (buffer <= 0){
						result += insertToVShift;
					} else {
						buffer--;
					}
				}
			}
			for (int i = 0; i <= start1; i++){
				if (content2[start2 - start1 + i] == 1){
					if (buffer > 0){
						shiftTemp = max(shiftTemp, (i + 1) << 1);
						buffer--;
					} else {
						result += insertToVShift;
					}
				}
			}
			result += shiftTemp;
			return result + abs(shift);
		} else if (start1 > start2){ //H V
			int shiftTemp = 0;
			for (int i = 0; i < start1 - start2; i++){ //H0 V
				if (content1[i] == 1){
					if (buffer >= 8){
						result += deleteToVShift;
					} else {
						buffer++;
					}
				}
			}
			for (int i = 0; i <= start2; i++){
				if (content1[start1 - start2 + i] == 1){ //H1 V
					if (buffer >= 8){
						result += deleteToVShift;
					} else {
						shiftTemp = (i + 1) << 1;
						buffer++;
					}
				}
			}
			for (int i = 0; i <= start2; i++){
				if (content2[i] == 1) {
					if (buffer > 0){
						shiftTemp = max(shiftTemp, (i + 1) << 1);
						buffer--;
					} else {
						result += insertToVShift;
					}
				}
			}
			result += shiftTemp;
			return result + abs(shift);
		}
	}
	return -1;
}

int SkyrmionBit::assessDCW(const sky_size_t *content1, const sky_size_t *content2, int buffer)
{
	int insertToVShift = INSERT / VSHIFT;
	int deleteToVShift = DELETE / VSHIFT;
	int result = 0;

	for (int i = 0; i < BUFFER_LENGTH; i++){
		if (content1[i] == 1 && content2[i] == 0){
			if (buffer >= BUFFER_LENGTH){
				result += deleteToVShift;
			} else {
				result += deleteToVShift + insertToVShift;
			}
		} else if (content1[i] == 0 && content2[i] == 1){
			result += insertToVShift;
		}
	}
	return result;
}

int SkyrmionBit::assessPW(const sky_size_t *content1, const sky_size_t *content2, int buffer)
{
	int insertToVShift = INSERT / VSHIFT;
	int deleteToVShift = DELETE / VSHIFT;
	int result = 0;
	int shiftTemp = 0;

	for (int i = 0; i < BUFFER_LENGTH; i++){
		if (content1[i] == 1){
			if (buffer >= BUFFER_LENGTH){
				result += deleteToVShift;
			} else {
				shiftTemp = i + 1;
				buffer++;
			}
		}
	}
	result += shiftTemp;
	shiftTemp = 0;
	for (int i = 0; i < BUFFER_LENGTH; i++){
		if (content2[i] == 1){
			if (buffer > 0){
				shiftTemp = i + 1;
				buffer--;
			} else {
				result += insertToVShift;
			}
		}
	}
	return result + shiftTemp;

}

void SkyrmionBit::bitDCW(int port, Addr address, sky_size_t *content1, sky_size_t *ptr, bool &delKeep, bool &insrtKeep, int saveData)
{
	for (int k = 0; k < 8; k++){
		if (content1[k] != ptr[k]){
			if (ptr[k] == 0){
				deleteSky(port, address * 8 + k, saveData);
				insert(port, address * 8 + k, 0, saveData);
				delKeep = true;
			}	else {
				insert(port, address * 8 + k, 1, saveData);
				insrtKeep = true;
			}
		}
	}
}

void SkyrmionBit::bitPW(int block, int port, Addr address, sky_size_t *content1, sky_size_t *ptr, bool &insrtKeep, int &assembleShiftNum, int &repermuteShiftNum, int saveData)
{
	// 1. assemble the existing skyrmions
	assembleShiftNum = 0;
	for (int i = 0; i < 8 ; i++){
		if (content1[i] == 1){
			buffer[address * MAX_SIZE + block]++;
			assembleShiftNum = i + 1;
		}
	}
	for (int i = 0; i < assembleShiftNum; i++){
		shiftVertcl(port, address, 1, saveData);
	}

	// 2. re-permute the exising skyrmions & inject new skyrmions (if needed)
	// when we exhaust all skyrmions in the buffer, then we don't need
	// to shift, since we can insert at each access port
	repermuteShiftNum = countNumShift(ptr, buffer[address * MAX_SIZE + block]);

	for (int j = 0; j < repermuteShiftNum; j++){
		if (j == 0){
			// imagine there is a buffer track where we stored skyrmions,
			// and we first shift one skyrmion to the race track
			if (saveData == 0){
				shiftVertcl_energy++;
			} else if (saveData == 1) {
				shiftVertcl_energy_DMW++;
			}
		} else shiftVertcl(port, address, 0, saveData);
		if (ptr[repermuteShiftNum - j - 1] == 0){
			insert(port, address * 8, 0, saveData);
		} else if (ptr[repermuteShiftNum - j - 1] == 1){
			insert(port, address * 8, 2, saveData);
		}
	}
	for (int j = repermuteShiftNum; j < 8; j++){
		if (ptr[j] == 0){
			insert(port, address * 8 + j, 0, saveData);
		} else if (ptr[j] == 1){
			insert(port, address * 8 + j, 1, saveData);
			insrtKeep = true;
		}
	}
}

void SkyrmionBit::bitPureShift(int block, int port, Addr address, sky_size_t *content1, sky_size_t *ptr, bool &delKeep, bool &insrtKeep, vector<int> cmpPatternResult, int order, int saveData)
{
	int ret = cmpPatternResult[0];
	int shift = cmpPatternResult[1];
	int start1 = cmpPatternResult[2];
	int start2 = cmpPatternResult[3];
	int assembleShiftNum = 0;
	int repermuteShiftNum = 0;

	//contetn2 = 0 or content1 = 0
	if (ret == 2 || ret == 1){
		return bitPW(block, port, address, content1, ptr, insrtKeep, assembleShiftNum, repermuteShiftNum, saveData);
	}

	//forward order, similar to assessDCW
	if (order == 0){
		int n1 = 8 - start1;
		int n2 = 8 - start2;
		if (n1 <= n2){
			for (int i = 0; i < n1; i++){
				if (content1[start1 + i] == 1 && ptr[start2 + i] == 0){
					deleteSky(port, address * 8 + start1 + i, saveData);
					delKeep = true;
				} else if (content1[start1 + i] == 0 && ptr[start2 + i] == 1){
					insert(port, address * 8 + start1 + i, 1, saveData);
					insrtKeep = true;
				}
			}

			for (int i = 0; i < abs(shift); i++){
				shiftVertcl(port, address, 1, saveData);
			}

			for (int i = 0; i < n2 - n1; i++){
				if (ptr[start2 + n1 + i] == 1){
					insert(port, address * 8 + start2 + n1 + i, 1, saveData);
					insrtKeep = true;
				}
			}
		} else if (n1 > n2){ //D V
			for (int i = 0; i < n2; i++){
				if (content1[start1 + i] == 0 && ptr[start2 + i] == 1){
					insert(port, address * 8 + start1 + i, 1, saveData);
					insrtKeep = true;
				} else if (content1[start1 + i] == 1 && ptr[start2 + i] == 0){ //V
					deleteSky(port, address * 8 + start1 + i, saveData);
					delKeep = true;
				}
			}
			for (int i = 0; i < n1 - n2; i++){
				if (content1[start1 + n2 + i] == 1){
					deleteSky(port, address * 8 + start1 + n2 + i, saveData);
					delKeep = true;
				}
			}
			for (int i = 0; i < shift; i++){
				shiftVertcl(port, address, 0, saveData);
			}
		}
	//reverse order
	} else {
		if (start1 <= start2){ //G V
			int shiftTemp = 0;
			for (int i = 0; i <= start1; i++){ //G0 V
				if (content1[i] == 1){
					if (buffer[address * MAX_SIZE + block] >= 8){
						deleteSky(port, address * 8 + i, saveData);
						delKeep = true;
					} else {
						shiftTemp = i + 1;
						buffer[address * MAX_SIZE + block]++;
					}
				}
			}

			int shiftTemp2 = 0;
			int buf = buffer[address * MAX_SIZE + block];
			for (int i = 0; i < start2 - start1; i++){ //G1 V
				if (ptr[i] == 1 && buf > 0){
						buf--;
				}
			}
			for (int i = 0; i <= start1; i++){
				if (ptr[start2 - start1 + i] == 1){
					if (buf > 0){
						shiftTemp2 = i + 1;
						buf--;
					}
				}
			}
			shiftTemp = max(shiftTemp, shiftTemp2);
			for (int i = 0; i < shiftTemp; i++){
				shiftVertcl(port, address, 1, saveData);
			}

			for (int i = shiftTemp - 1; i >= 0; i--){
				shiftVertcl(port, address, 0, saveData);
				if (i <= shiftTemp2 - 1 && ptr[start2 - start1 + i] == 1){
					insert(port, address * 8, 2, saveData);
					buffer[address * MAX_SIZE + block]--;
				}
			}
			for (int i = start2 - start1 - 1; i >= 0; i--){ //G1 V
				shiftVertcl(port, address, 0, saveData);
				if (ptr[i] == 1){
					if (buffer[address * MAX_SIZE + block] <= 0){
						insert(port, address * 8, 1, saveData);
						insrtKeep = true;
					} else {
						insert(port, address * 8, 2, saveData);
						buffer[address * MAX_SIZE + block]--;
					}
				}
			}
			for (int i = shiftTemp2; i <= start1; i++){
				if (ptr[start2 - start1 + i] == 1){
					insert(port, address * 8 + start2 - start1 + i, 1, saveData);
					insrtKeep = true;
				}
			}
		} else if (start1 > start2){ //H V
			int shiftTemp = 0;
			for (int i = 0; i < start1 - start2; i++){ //H0 V
				if (content1[i] == 1){
					if (buffer[address * MAX_SIZE + block] >= 8){
						deleteSky(port, address * 8, saveData);
						delKeep = true;
					} else {
						buffer[address * MAX_SIZE + block]++;
					}
				}
				shiftVertcl(port, address, 1, saveData);
			}
			for (int i = 0; i <= start2; i++){
				if (content1[start1 - start2 + i] == 1){ //H1 V
					if (buffer[address * MAX_SIZE + block] >= 8){
						deleteSky(port, address * 8 + i, saveData);
						delKeep = true;
					} else {
						shiftTemp = i + 1;
						buffer[address * MAX_SIZE + block]++;
					}
				}
			}

			int buf = buffer[address * MAX_SIZE + block];
			int shiftTemp2 = 0;
			for (int i = 0; i <= start2; i++){
				if (ptr[i] == 1) {
					if (buf > 0){
						shiftTemp2 = i + 1;
						buf--;
					}
				}
			}
			shiftTemp = max(shiftTemp, shiftTemp2);
			//move
			for (int i = 0; i < shiftTemp; i++){
				shiftVertcl(port, address, 1, saveData);
			}
			for (int i = shiftTemp - 1; i >= 0; i--){
				shiftVertcl(port, address, 0, saveData);
				if (i <= shiftTemp2 - 1 && ptr[i] == 1){
					insert(port, address * 8, 2, saveData);
					buffer[address * MAX_SIZE + block]--;
				}
			}
			for (int i = shiftTemp2; i <= start2; i++){
				if (ptr[i] == 1){
					insert(port, address * 8 + i, 1, saveData);
					insrtKeep = true;
				}
			}
		}
	}
}

/**
 * Write data with size byte(s) at address in block
 * @param block The block(cache line) of the data
 * @param address The address(offset) of the data which will be written
 * @param size Byte(s) of the data
 * @param content The data which will be written
 * @param type The write type, including NAIVE, DCW, PW, pureShift, PW_IMPROVED
 * @param saveData If 0 save energy data to the first statics, 1 to the second
*/
void SkyrmionBit::write(int block, Addr address, data_size_t size, const sky_size_t *content, enum Write_Type type, int saveData)
{
	if (block < 0 || block >= MAX_SIZE){
		cout << "write: block is invalid (0 ~ %d" << MAX_SIZE-1 << ")" << endl;
		exit(1);
	}
	if (address >= MAX_SIZE/8){
		cout << "write: address is invalid (0 ~ %d" << MAX_SIZE/8-1 << ")" << endl;
		exit(1);
	}
	//mark checkFullArray
	for (int i = 0; i < size; i++){
		if (checkFullArray[block * ROW / 8 + address + i] == 0){
			checkFullArray[block * ROW / 8 + address + i] = 1;
			n_checkFull++;
		}
	}
	//count the block
	blockUsedNumber[block]++;

	int ports[2];
	_write_type = type;
	sky_size_t *ptr = byteToBit(size, content);
	int moves = determinePorts(block, ports);
	int port = block / DISTANCE;
	bool delKeep = false;
	bool insrtKeep = false;
	if (ports[0] < ports[1]){
		port++;
	}
	//move to the access port
	for (int i = 0; i < moves; i++){
		shift(ports[0], ports[1], address, size, saveData);
		if (saveData == 0){
			shift_latency++;
		} else if (saveData == 1){
			shift_latency_DMW++;
		}
	}
	switch(type){
		case BIT:
			for (int k = 0; k < size * 8; k++){
				deleteSky(port, address * 8 + k, saveData);
				insert(port, address * 8 + k, *(ptr + k), saveData);
				if (ptr[k] == 1)
					insrtKeep = true;
			}

			if (saveData == 0){
				delete_latency++;
				if (insrtKeep != 0)
					insert_latency++;
			} else {
				delete_latency_DMW++;
				if (insrtKeep != 0)
					insert_latency_DMW++;
			}
			break;

		case BIT_DCW:
		{
			//start to write
			sky_size_t content1[8];
			for (int i = 0; i < size; i++){
				for (int j = 0; j < 8; j++){
					content1[j] = detect(port, (address + i) * 8 + j, saveData);
				}
				bitDCW(port, address + i, content1, ptr + i * 8, delKeep, insrtKeep, saveData);
			}

			if (saveData == 0){
				detect_latency++;
				if (insrtKeep != 0)
					insert_latency++;
				if (delKeep != 0)
					delete_latency++;
			} else {
				detect_latency_DMW++;
				if (insrtKeep != 0)
					insert_latency_DMW++;
				if (delKeep != 0)
					delete_latency_DMW++;
			}
			break;
		}
		case BIT_PW:
		{
			/*
			int bufferSet = BUFFER_LENGTH / 8;
			int newAddress = address / bufferSet;
			int times = ceil(float(size)/bufferSet);
			if ((size < bufferSet && (address % bufferSet + size > bufferSet)) || (address % bufferSet != 0 && size % bufferSet == 0)){
				times++;
			}
			//cout << "times = " << times << endl;
			for (int i = 0; i < moves; i++){
				shift(ports[0], ports[1], newAddress * bufferSet, times * bufferSet, saveData);
				if (saveData == 0){
					shift_latency++;
				} else if (saveData == 1){
					shift_latency_DMW++;
				}
			}
			int shiftNumAssemble = 0;
			int shiftNumRepermute = 0;
			int maxShiftLatencyAssemble = 0;
			int maxShiftLatencyRepermute = 0;

			for (int i = 0; i < times; i++){
				sky_size_t *ptrNew = new sky_size_t[bufferSet * 8];
				if (i == 0 && address % bufferSet != 0) {
					shiftNumAssemble = 0;
					shiftNumRepermute = 0;
					// 1. assemble the existing skyrmions
					memcpy(ptrNew + (address % bufferSet) * 8, ptr, sizeof(sky_size_t) * (bufferSet - address % bufferSet) * 8);

					for (int j = 0; j < BUFFER_LENGTH ; j++){
						if (j < (address % bufferSet) * 8){
							int input = detect(port, (newAddress + i) * BUFFER_LENGTH + j, saveData);
							ptrNew[j] = input;
							if (input == 1){
								buffer[(newAddress + i) * MAX_SIZE + block]++;
								shiftNumAssemble = j + 1;
							}
						} else {
							if (detect(port, (newAddress + i) * BUFFER_LENGTH + j, saveData) == 1){
								buffer[(newAddress + i) * MAX_SIZE + block]++;
								shiftNumAssemble = j + 1;
							}
						}
					}
					ptr += (bufferSet - address % bufferSet) * 8;
				} else if (i == times-1 && (address + size) % bufferSet != 0) {
					shiftNumAssemble = 0;
					shiftNumRepermute = 0;
					// 1. assemble the existing skyrmions
					memcpy(ptrNew, ptr, sizeof(sky_size_t) * ((address + size) % bufferSet) * 8);

					for (int j = 0; j < BUFFER_LENGTH ; j++){
						if (j >= ((address + size) % bufferSet) * 8){
							int input = detect(port, (newAddress + i) * BUFFER_LENGTH + j, saveData);
							ptrNew[j] = input;
							if (input == 1){
								buffer[(newAddress + i) * MAX_SIZE + block]++;
								shiftNumAssemble = j + 1;
							}
						} else {
							if (detect(port, (newAddress + i) * BUFFER_LENGTH + j, saveData) == 1){
								buffer[(newAddress + i) * MAX_SIZE + block]++;
								shiftNumAssemble = j + 1;
							}
						}
					}
				} else {
					shiftNumAssemble = 0;
					shiftNumRepermute = 0;
					// 1. assemble the existing skyrmions
					memcpy(ptrNew, ptr, sizeof(sky_size_t) * BUFFER_LENGTH);

					for (int j = 0; j < BUFFER_LENGTH ; j++){
						if (detect(port, (newAddress + i) * BUFFER_LENGTH + j, saveData) == 1){
							buffer[(newAddress + i) * MAX_SIZE + block]++;
							shiftNumAssemble = j + 1;
						}
					}
					ptr += BUFFER_LENGTH;
				}
				for (int j = 0; j < shiftNumAssemble; j++){
					shiftVertcl(port, (newAddress + i) * bufferSet, 1, saveData);
				}
				if (shiftNumAssemble > maxShiftLatencyAssemble){
					maxShiftLatencyAssemble = shiftNumAssemble;
				}

				// 2. re-permute the exising skyrmions & inject new skyrmions (if needed)
				// when we exhaust all skyrmions in the buffer, then we don't need
				// to shift, since we can insert at each access port
				shiftNumRepermute = countNumShift(ptrNew, buffer[(newAddress + i) * MAX_SIZE + block]);
				//cout << "Repermute. i = " << i << " shiftNumRepermute = " << shiftNumRepermute << endl;
				//cout << "after shiftNumRepermute, buffer[" << (newAddress + i) * MAX_SIZE + block << "] = " << buffer[(newAddress + i) * MAX_SIZE + block] << endl;
				for (int j = 0; j < shiftNumRepermute; j++){
					if (j == 0){
						// imagine there is a buffer track where we stored skyrmions,
						// and we first shift one skyrmion to the race track
						if (saveData == 0){
							shiftVertcl_energy++;
						} else if (saveData == 1){
							shiftVertcl_energy_DMW++;
						}
						insert(port, (newAddress + i) * BUFFER_LENGTH, 2, saveData);
					} else {
						shiftVertcl(port, (newAddress + i) * bufferSet, 0, saveData);
						if (ptrNew[shiftNumRepermute - j - 1] == 0){
							insert(port, (newAddress + i) * BUFFER_LENGTH, 0, saveData);
						} else if (ptrNew[shiftNumRepermute - j - 1] == 1){
							insert(port, (newAddress + i) * BUFFER_LENGTH, 2, saveData);
						}
					}
				}
				if (shiftNumRepermute > maxShiftLatencyRepermute){
					maxShiftLatencyRepermute = shiftNumRepermute;
				}

				if (buffer[(newAddress + i) * MAX_SIZE + block] == 0){
					for (int j = shiftNumRepermute; j < BUFFER_LENGTH; j++){
						if (ptrNew[j] == 0){
							insert(port, (newAddress + i) * BUFFER_LENGTH + j, 0, saveData);
						} else if (ptrNew[j] == 1){
							insert(port, (newAddress + i) * BUFFER_LENGTH + j, 1, saveData);
							insrtKeep = true;
						}
					}
				}
				//print();
				delete [] ptrNew;
			}
			if (saveData == 0){
				shiftVertcl_latency += maxShiftLatencyRepermute;
				shiftVertcl_latency += maxShiftLatencyAssemble;
				detect_latency++;
				if (insrtKeep == 1)
					insert_latency++;
			} else if (saveData == 1){
				shiftVertcl_latency_DMW += maxShiftLatencyRepermute;
				shiftVertcl_latency_DMW += maxShiftLatencyAssemble;
				detect_latency_DMW++;
				if (insrtKeep == 1)
					insert_latency_DMW++;
			}

			// move backward
			for (int i = 0; i < moves; i++){
				shift(ports[1], ports[0], newAddress * bufferSet, times * bufferSet, saveData);
				if (saveData == 0){
					shift_latency++;
				} else if (saveData == 1){
					shift_latency_DMW++;
				}
			}
			break;
			*/
			int assembleShiftNum = 0;
			int repermuteShiftNum = 0;
			int maxAssembleShiftLatency = 0;
			int maxRepermuteShiftLatency = 0;
			sky_size_t content1[8];
			for (int i = 0; i < size; i++){
				for (int j = 0; j < 8; j++){
					content1[j] = detect(port, (address + i) * 8 + j, saveData);
				}
				bitPW(block, port, address + i, content1, ptr + 8 * i, insrtKeep, assembleShiftNum, repermuteShiftNum, saveData);
				if (maxAssembleShiftLatency < assembleShiftNum){
					maxAssembleShiftLatency = assembleShiftNum;
				}
				if (maxRepermuteShiftLatency < repermuteShiftNum){
					maxRepermuteShiftLatency = repermuteShiftNum;
				}
			}

			if (saveData == 0){
				shiftVertcl_latency += (maxAssembleShiftLatency + maxRepermuteShiftLatency);
				detect_latency++;
				if (insrtKeep == 1)
					insert_latency++;
			} else {
				shiftVertcl_latency_DMW += (maxAssembleShiftLatency + maxRepermuteShiftLatency);
				detect_latency_DMW++;
				if (insrtKeep == 1)
					insert_latency_DMW++;
			}

			break;
		}
		case BIT_PURESHIFT:
		{
			//Read current data to content1
			sky_size_t *content1 = new sky_size_t[size * 8];
			for (int k = 0; k < size * 8; k++){
				content1[k] = detect(port, address * 8 + k, saveData);
			}

			//start to write
			int beforeVShift = 0;
			int maxVerticalShift = 0;
			for (int i = 0; i < size; i++){
				if (saveData == 0){
					beforeVShift = shiftVertcl_energy;
				} else if (saveData == 1){
					beforeVShift = shiftVertcl_energy_DMW;
				}
				vector<int> cmpPatternResult = cmpPattern(content1 + 8 * i, ptr + 8 * i, 1);
				bitPureShift(block, port, address + i, content1 + 8 * i, ptr + 8 * i, delKeep, insrtKeep, cmpPatternResult, 1, saveData);
				if (saveData == 0){
					if ((shiftVertcl_energy - beforeVShift) > maxVerticalShift){
						maxVerticalShift = shiftVertcl_energy - beforeVShift;
					}
				} else if (saveData == 1){
					if ((shiftVertcl_energy_DMW - beforeVShift) > maxVerticalShift){
						maxVerticalShift = shiftVertcl_energy_DMW - beforeVShift;
					}
				}
			}

			if (saveData == 0){
				if (delKeep != 0)
					delete_latency++;
				if (insrtKeep != 0)
					insert_latency++;
				detect_latency++;
				shiftVertcl_latency += maxVerticalShift;
			} else if (saveData == 1){
				if (delKeep != 0)
					delete_latency_DMW++;
				if (insrtKeep != 0)
					insert_latency_DMW++;
				detect_latency_DMW++;
				shiftVertcl_latency_DMW += maxVerticalShift;
			}
			break;
		}
		case BIT_PW_IMPROVED:
		{
			int assembleShiftNum = 0;
			int repermuteShiftNum = 0;
			int maxAssembleShiftLatency = 0;
			int maxRepermuteShiftLatency = 0;
			int beforeVShift = 0;
			int maxVerticalShift = 0;

			// start to write
			for (int i = 0; i < size; i++){
				//cout << "1. i = " << i << endl;
				sky_size_t content1[8];
				sky_size_t content2[8];
				for (int j = 0; j < 8; j++){
					content1[j] = detect(port, (address + i) * 8 + j, saveData);
					content2[j] = ptr[i * 8 + j];
				}

				vector<int> pattern0 = cmpPattern(content1, content2, 0);
				vector<int> pattern1 = cmpPattern(content1, content2, 1);
				if (pattern0[0] != 0){
					int pureShiftOrder0 = assessPureShift(content1, content2, pattern0, 0, buffer[(address + i) * MAX_SIZE + block]);
					int pureShiftOrder1 = assessPureShift(content1, content2, pattern1, 1, buffer[(address + i) * MAX_SIZE + block]);
					int dcw = assessDCW(content1, content2, buffer[(address + i) * MAX_SIZE + block]);
					int pw = assessPW(content1, content2, buffer[(address + i) * MAX_SIZE + block]);
					//cout << "2. pos = " << pureShiftOrder0 << " neg = " << pureShiftOrder1 << " dcw = " << dcw << " pw = " << pw << endl;
					//cout << "3. buf = " << buffer[(address + i) * MAX_SIZE + block] << endl;
					if (pureShiftOrder1 <= pw && pureShiftOrder1 <= dcw && pureShiftOrder1 <= pureShiftOrder0){
						//cout << "4. method: neg\n";
						if (saveData == 0){
							beforeVShift = shiftVertcl_energy;
							bitPureShift(block, port, address + i, content1, content2, delKeep, insrtKeep, pattern1, 1, saveData);
							if (shiftVertcl_energy - beforeVShift > maxVerticalShift){
								maxVerticalShift = shiftVertcl_energy - beforeVShift;
							}
						} else if (saveData == 1){
							beforeVShift = shiftVertcl_energy_DMW;
							bitPureShift(block, port, address + i, content1, content2, delKeep, insrtKeep, pattern1, 1, saveData);
							if (shiftVertcl_energy_DMW - beforeVShift > maxVerticalShift){
								maxVerticalShift = shiftVertcl_energy_DMW - beforeVShift;
							}
						}
					} else if (pw <= pureShiftOrder1 && pw <= dcw && pw <= pureShiftOrder0) {
						//cout << "4. method: pw\n";
						bitPW(block, port, address + i, content1, content2, insrtKeep, assembleShiftNum, repermuteShiftNum, saveData);
						maxAssembleShiftLatency = max(maxAssembleShiftLatency, assembleShiftNum);
						maxRepermuteShiftLatency = max(maxRepermuteShiftLatency, repermuteShiftNum);
					} else if (dcw <= pureShiftOrder1 && dcw <= pw && dcw <= pureShiftOrder0) {
						//cout << "4. method: dcw\n";
						bitDCW(port, address + i, content1, content2, delKeep, insrtKeep, saveData);
					} else if (pureShiftOrder0 <= pureShiftOrder1 && pureShiftOrder0 <= pw && pureShiftOrder0 <= dcw) {
						//cout << "4. method: pos\n";
						if (saveData == 0){
							beforeVShift = shiftVertcl_energy;
							bitPureShift(block, port, address + i, content1, content2, delKeep, insrtKeep, pattern0, 0, saveData);
							if (shiftVertcl_energy - beforeVShift > maxVerticalShift){
								maxVerticalShift = shiftVertcl_energy - beforeVShift;
							}
						} else if (saveData == 1){
							beforeVShift = shiftVertcl_energy_DMW;
							bitPureShift(block, port, address + i, content1, content2, delKeep, insrtKeep, pattern0, 0, saveData);
							if (shiftVertcl_energy_DMW - beforeVShift > maxVerticalShift){
								maxVerticalShift = shiftVertcl_energy_DMW - beforeVShift;
							}
						}
					}
				}
			}
			maxVerticalShift = max(maxVerticalShift, maxAssembleShiftLatency + maxRepermuteShiftLatency);

			if (saveData == 0){
				if (delKeep != 0)
					delete_latency++;
				if (insrtKeep != 0)
					insert_latency++;
				detect_latency++;
				shiftVertcl_latency += maxVerticalShift;
			} else if (saveData == 1){
				if (delKeep != 0)
					delete_latency_DMW++;
				if (insrtKeep != 0)
					insert_latency_DMW++;
				detect_latency_DMW++;
				shiftVertcl_latency_DMW += maxVerticalShift;
			}

			break;
		}
		case AUTO_DETECTED:
			break;
		default:
		{
			cout << "No such bit-interleaved method.\n";
			exit(1);
			break;
		}
	}
	// move backward
	for (int i = 0; i < moves; i++){
		shift(ports[1], ports[0], address, size, saveData);
		if (saveData == 0){
			shift_latency++;
		} else if (saveData == 1){
			shift_latency_DMW++;
		}
	}
}

/**
 * Determine whether the race track is full or not
*/
bool SkyrmionBit::isFull()
{
	return n_checkFull == ROW / 8 * MAX_SIZE;
}