#ifndef SKY
#define SKY

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>
#include <algorithm>

#define MAX_SIZE 2048
#define OVER_HEAD 32
#define DISTANCE 32
#define ROW 512 // block size (unit: bits)
#define BUFFER_LENGTH 8
#define SHIFT 20 //energy
#define INSERT 200 //energy
#define DELETE 20 //energy
#define DETECT 2 //energy
#define VSHIFT 20 //energy



typedef uint8_t 				sky_size_t;
typedef unsigned long 	Stat;
typedef uint64_t				Addr;
typedef unsigned				data_size_t;

using namespace std;

typedef enum Method_Type{
	WORD_BASED, // 0
	BIT_INTERLEAVED // 1
}Method_Type;

typedef enum Write_Type{
	//for word-based
	NAIVE_TRADITIONAL,
	NAIVE,
	DCW_TRADITIONAL,
	DCW,
	PERMUTATION_WRITE,
	// for inter-leaved
	BIT, //naive
	BIT_DCW,
	BIT_PW,
	BIT_PURESHIFT,
	BIT_PW_IMPROVED,
	AUTO_DETECTED
}Write_Type;

class Skyrmion {
protected:
	int _id;
	enum Write_Type _write_type;
	enum Method_Type _type;
	int _intervalSize;
	// down statics
	Stat shift_energy;
	Stat insert_energy;
	Stat delete_energy;
	Stat detect_energy;
	Stat shift_latency;
	Stat shiftVertcl_energy;
	Stat insert_latency;
	Stat delete_latency;
	Stat detect_latency;
	Stat shiftVertcl_latency;
	// up statics
	Stat shift_energy_DMW;
	Stat insert_energy_DMW;
	Stat delete_energy_DMW;
	Stat detect_energy_DMW;
	Stat shift_latency_DMW;
	Stat shiftVertcl_energy_DMW;
	Stat insert_latency_DMW;
	Stat delete_latency_DMW;
	Stat detect_latency_DMW;
	Stat shiftVertcl_latency_DMW;

public:
	Skyrmion();
	virtual ~Skyrmion();
	int getIntervalSize() const;
	Stat getSht_engy() const;
	Stat getIns_engy() const;
	Stat getDel_engy() const;
	Stat getDet_engy() const;
	Stat getShtVrtcl_engy() const;
	Stat getSht_latcy() const;
	Stat getIns_latcy() const;
	Stat getDel_latcy() const;
	Stat getDet_latcy() const;
	Stat getShtVrtcl_latcy() const;
	Stat getSht_engy_DMW() const;
	Stat getIns_engy_DMW() const;
	Stat getDel_engy_DMW() const;
	Stat getDet_engy_DMW() const;
	Stat getShtVrtcl_engy_DMW() const;
	Stat getSht_latcy_DMW() const;
	Stat getIns_latcy_DMW() const;
	Stat getDel_latcy_DMW() const;
	Stat getDet_latcy_DMW() const;
	Stat getShtVrtcl_latcy_DMW() const;
	void setId(int id);
	int getId() const;
	void setWriteType(enum Write_Type write_type);
	int getWriteType() const;
	virtual void print() const = 0;
	static sky_size_t *bitToByte(data_size_t size, const sky_size_t *read);
	static sky_size_t *byteToBit(data_size_t size, const sky_size_t *read);
	void outPut(void);
};

class SkyrmionWord: public Skyrmion {
protected:
	sky_size_t *entry;
	int *checkFullArray;
	int n_checkFull;
private:
	sky_size_t *read(Addr address, data_size_t size, int saveData);
	void write(Addr address, data_size_t size, const sky_size_t *content, enum Write_Type type, int saveData);
	void Naive(int port, int length, int minRedundant, vector<sky_size_t> &content, int saveData);
	void dcw(int port, int length, int frontRedundant, int backRedundant, vector<sky_size_t> &content, int saveData);
	void pw(int shiftStartPort, int port, int length, int frontRedundant, int backRedundant, vector<sky_size_t> &content, int saveData);
public:
	SkyrmionWord();
	SkyrmionWord(int weightSize);
	~SkyrmionWord() override;
	void print() const override;
	int determinePorts(Addr address, data_size_t size) const;
	vector<int> bitPositions(int whichInterval) const;
	sky_size_t getEntry(int position) const;
	void setEntry(int position, sky_size_t value);
	int getN_checkFull() const;
	void shift(int startPort, int endPort, int saveData); //saveData 1:save data to DMW
	void insert(int accessPort, sky_size_t content, int saveData); //content is bit 0 or 1
	void deleteSky(int accessPort, int saveData);
	sky_size_t detect(int accessPort, int saveData);
	sky_size_t *readData(Addr address, data_size_t size, int parallel, int saveData); // type 1: parallel
	void writeData(Addr address, data_size_t size, const sky_size_t *content, enum Write_Type type, int saveData);
	bool isFull() const;
	void clear(int whichInterval, int saveData);
};

class SkyrmionBit: public Skyrmion {
protected:
	sky_size_t **entries;
	int *buffer;
	int *checkFullArray;
	int n_checkFull;
	Stat *blockUsedNumber;

public:
	SkyrmionBit();
	SkyrmionBit(int size);
	~SkyrmionBit() override;
	void print() const override;
	sky_size_t getEntries(int row, int col) const;
	void setEntries(int row, int col, sky_size_t value);
	int getN_checkFull() const;
	Stat getBlockUsedNumber(int index) const;
	int determinePorts(int block, int *portsBuffer) const;
	void shift(int startPort, int endPort, Addr address, data_size_t size, int saveData);
	void shiftVertcl(int accessPort, Addr address, int updown, int saveData); //updown: up = 1, down = 0
	void insert(int accessPort, int row, sky_size_t content, int saveData); //content is bit 0 or 1
	void deleteSky(int accessPort, int row, int saveData);
	sky_size_t detect(int accessPort, int row, int saveData);
	void move(int startPort, int endPort, int moves, Addr address, data_size_t size, int saveData);
	sky_size_t *read(int block, Addr address, data_size_t size, int saveData);
	void write(int block, Addr address, data_size_t size, const sky_size_t *content, enum Write_Type type, int saveData);
	void bitDCW(int port, Addr address, sky_size_t *content1, sky_size_t *ptr, bool &delKeep, bool &insrtKeep, int saveData);
	void bitPW(int block, int port, Addr address, sky_size_t *content1, sky_size_t *ptr, bool &insrtKeep, int &assembleShiftNum, int &repermuteShiftNum, int saveData);
	void bitPureShift(int block, int port, Addr address, sky_size_t *content1, sky_size_t *content2, bool &delKeep, bool &insrtKeep, vector<int> cmpPatternResult, int order, int saveData);
	static int assessPureShift(const sky_size_t *content1, const sky_size_t *content2, vector<int> cmpPatternResult, int order, int buffer);
	static int assessDCW(const sky_size_t *content1, const sky_size_t *content2, int buffer);
	static int assessPW(const sky_size_t *content1, const sky_size_t *content2, int buffer);
	static int countNumShift(sky_size_t *data, int &numReuse);
	static int countBit1(const sky_size_t *content, int &start, int order); //forward: 0, reverse: 1
	static vector<int> cmpPattern(const sky_size_t *content1, const sky_size_t *content2, int order); //forward: 0, reverse: 1
	bool isFull();
};

#endif
