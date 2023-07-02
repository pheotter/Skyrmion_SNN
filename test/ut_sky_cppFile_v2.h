// this test file for C++ version
#include "../src/sky.h"
#include <cassert>
#include <bitset>
//#define DEBUG 1
#define BYTEBWPORTS 4 //DISTANCE/8
#define CACHELINESIZE 64

/**
 * for google test
 * @param address The address(offset) of the data which will be written
 * @param size Byte(s) of the data
 * @param type The write/read type, including DCW_TRADITIONAL, DCW, PW
 * @return detect latency
*/
int wordDetect(Addr add, data_size_t size, enum Write_Type type){
  if (size < BYTEBWPORTS && add % BYTEBWPORTS != 0 && (add+size) % BYTEBWPORTS != 0)
    type = DCW;
  int cnt = 0;
  if (type == DCW_TRADITIONAL || type == PERMUTATION_WRITE){
    int first = BYTEBWPORTS - add % BYTEBWPORTS;
    cnt += (first * 8);
    size -= first;
    while (size >  0){
      int num = min((int)size, BYTEBWPORTS);
      cnt += (num * 8);
      size -= num;
    }
  } else if (type == DCW){
    if (size >= BYTEBWPORTS) return DISTANCE;
    return size * 8;
  }
  return cnt;
}

/**
 * for google test
 * @param address The address(offset) of the data which will be written
 * @param size Byte(s) of the data
 * @param type The write type, including NAIVE, DCW, PW
 * @return shift energy/latency (shift energy and latency are equal for word-based )
*/
int wordShift(Addr add, data_size_t size, enum Write_Type type){
  int pr = 0;
  if (type == PERMUTATION_WRITE) pr = 1;
  if (size < BYTEBWPORTS && (add % BYTEBWPORTS)+size < BYTEBWPORTS)
    type = NAIVE;
  int cnt = 0;
  if (type == NAIVE_TRADITIONAL || type == DCW_TRADITIONAL || type == PERMUTATION_WRITE){
    int first = BYTEBWPORTS - add % BYTEBWPORTS;
    int mult8 = first * 8;
    if (type == PERMUTATION_WRITE) mult8++;
    cnt += (mult8 * 2);
    size -= first;
    while (size >  0){
      int num = min((int)size, BYTEBWPORTS);
      int mult8_ = num * 8;
      if (type == PERMUTATION_WRITE) mult8_++;
      cnt += (mult8_ * 2);
      size -= num;
    }
  } else if (type == NAIVE || type == DCW){
    if (size >= BYTEBWPORTS) return DISTANCE*2;
    if (add % BYTEBWPORTS == 0 || (add+size) % BYTEBWPORTS == 0)
      return size*8*2;
    int left = add % BYTEBWPORTS;
    int right = BYTEBWPORTS - (add + size) % BYTEBWPORTS;
    if (left <= right) return ((left+size)*8+pr)*2;
    return ((right+size)*8+pr)*2;
  }
  return cnt;
}

/**
 * for google test
 * calculate shift energy and latency for bit-interleaved layout
 * @param block The column of the entries the block belongs to (start from 0)
 * @param size Byte(s) of the data
 * @return int[0] shift energy, int[1] shift latency
*/
int* bitShift(int block, data_size_t size){
  int *ptr = new int[2];
  ptr[0] = ptr[1] = 0;
  int rem = block % DISTANCE;
  if (rem < DISTANCE/2) ptr[1] = (rem + 1) * 2;
  else ptr[1] = (DISTANCE - rem) * 2;
  ptr[0] = ptr[1] * size * 8;
  return ptr;
}

/**
 * for google test
 * convert data into a bitset
*/
bitset<CACHELINESIZE*8>* convBitset(sky_size_t *data, data_size_t size){
  bitset<CACHELINESIZE*8>* bs = new bitset<CACHELINESIZE*8>();
  for (int i = 0; i < size*8; i++){
    if (data[i] == 1) bs->set(i);
  }
  return bs;
}

/**
 * for google test
 * @param address The address(offset) of the data which will be written
 * @param size Byte(s) of the data
 * @param oldData The original data on the racetrack
 * @param newData The data which will be written
 * @param type The write type, including NAIVE_TRADITIONAL, NAIVE, DCW_TRADITIONAL, DCW, PW
 * @return contains 4 values,
 *   int[0] is insert energy,
 *   int[1] is insert latency,
 *   int[2] is delete energy,
 *   int[3] is delete latency.
*/
int* wordInsert(Addr add, data_size_t size, sky_size_t *oldData, sky_size_t *newData, enum Write_Type type){
  int *cnt = new int[4];
  cnt[0] = cnt[1] = cnt[2] = cnt[3] = 0;
  bitset<CACHELINESIZE*8>* newbs = convBitset(newData, size);
  bitset<CACHELINESIZE*8>* oldbs = convBitset(oldData, size);
  bitset<CACHELINESIZE*8> delbs = *oldbs;
  if (type == PERMUTATION_WRITE){
    if (size < BYTEBWPORTS && add % BYTEBWPORTS != 0 && (add+size) % BYTEBWPORTS != 0){
      if (newbs->count() > oldbs->count())
        cnt[0] = newbs->count() - oldbs->count();
      else
        cnt[2] = oldbs->count() - newbs->count();
    } else {
      int first = BYTEBWPORTS - add % BYTEBWPORTS;
      string bit1 = "";
      for (int i = 0; i < first; i++) bit1 += "11111111";
      bitset<CACHELINESIZE*8> bs1 (bit1);
      if ((bs1 & *oldbs).count() > (bs1 & *newbs).count())
        cnt[2] += ((bs1 & *oldbs).count() - (bs1 & *newbs).count());
      else
        cnt[0] += ((bs1 & *newbs).count() - (bs1 & *oldbs).count());
      *newbs >>= (first*8);
      *oldbs >>= (first*8);
      size -= first;
      while (size > 0){
        int num = min((int)size, BYTEBWPORTS);
        string bit2 = "";
        for (int i = 0; i < num; i++) bit2 += "11111111";
        bitset<CACHELINESIZE*8> bs2 (bit2);
        if ((bs2 & *oldbs).count() > (bs2 & *newbs).count())
          cnt[2] += ((bs2 & *oldbs).count() - (bs2 & *newbs).count());
        else
          cnt[0] += ((bs2 & *newbs).count() - (bs2 & *oldbs).count());
        *newbs >>= (num*8);
        *oldbs >>= (num*8);
        size -= num;
      }
    }
    cnt[1] = cnt[0];
    cnt[3] = cnt[2];
  } else {
    if (type == NAIVE || type == NAIVE_TRADITIONAL){
      cnt[0] = cnt[1] = newbs->count();
      cnt[2] = cnt[3] = size * 8;
      if (type == NAIVE) {
        cnt[1] = cnt[3] = 0;
        if (size < BYTEBWPORTS) cnt[3] = size*8;
        else cnt[3] = DISTANCE;
      }
    } else { // DCW, DCW_TRADITIONAL
      delbs ^= *newbs;
      delbs &= *oldbs;
      *oldbs ^= *newbs;
      *newbs &= *oldbs;
      cnt[0] = cnt[1] = newbs->count();
      cnt[2] = cnt[3] = delbs.count();
      if (type == DCW) cnt[1] = cnt[3] = 0;
    }
    for (int i = 0; i < BYTEBWPORTS; i++){
      bitset<CACHELINESIZE*8> cpNewbs = *newbs;
      bitset<CACHELINESIZE*8> cpDelbs = delbs;
      bitset<CACHELINESIZE*8> bs1;
      bitset<CACHELINESIZE*8> bs2;
      cpNewbs >>= i*8;
      cpDelbs >>= i*8;
      for (int j = i; j < size; j+=BYTEBWPORTS){
        bs1 |= (cpNewbs & bitset<CACHELINESIZE*8> (0xff));
        bs2 |= (cpDelbs & bitset<CACHELINESIZE*8> (0xff));

        cpNewbs >>= DISTANCE;
        cpDelbs >>= DISTANCE;
      }
      if (type == NAIVE || type == DCW) cnt[1] += bs1.count();
      if (type == DCW) cnt[3] += bs2.count();
    }
  }
  return cnt;
}

TEST(checkWordInsert, case1)
{
  sky_size_t data_bit[128]={0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,0}; // 61 bit 1
  sky_size_t data[16]={120, 182, 52, 136, 115, 92, 150, 0, 247, 33, 59, 152, 248, 213, 165, 98};
  sky_size_t data_test1[32] = {0};
  if (DISTANCE == 16){
    ASSERT_EQ(14, wordInsert(0,4, data_bit, data_bit, NAIVE_TRADITIONAL)[0]);
    ASSERT_EQ(14, wordInsert(0,4, data_bit, data_bit, NAIVE_TRADITIONAL)[1]);
    ASSERT_EQ(13, wordInsert(0,4, data_bit, data_bit+32, NAIVE_TRADITIONAL)[0]);
    ASSERT_EQ(13, wordInsert(0,4, data_bit, data_bit+32, NAIVE_TRADITIONAL)[1]);
    ASSERT_EQ(17, wordInsert(0,4, data_bit, data_bit+64, NAIVE_TRADITIONAL)[0]);
    ASSERT_EQ(14, wordInsert(0,4, data_bit, data_bit, NAIVE)[0]);
    ASSERT_EQ(11, wordInsert(0,4, data_bit, data_bit, NAIVE)[1]);
    ASSERT_EQ(13, wordInsert(0,4, data_bit, data_bit+32, NAIVE)[0]);
    ASSERT_EQ(11, wordInsert(0,4, data_bit, data_bit+32, NAIVE)[1]);
    ASSERT_EQ(17, wordInsert(0,4, data_bit, data_bit+64, NAIVE)[0]);
    ASSERT_EQ(13, wordInsert(0,4, data_bit, data_bit+64, NAIVE)[1]);
    ASSERT_EQ(17, wordInsert(0,4, data_bit, data_bit+96, NAIVE)[0]);
    ASSERT_EQ(14, wordInsert(0,4, data_bit, data_bit+96, NAIVE)[1]);
    ASSERT_EQ(4, wordInsert(0,1, data_bit, data_bit, NAIVE)[0]);
    ASSERT_EQ(4, wordInsert(0,1, data_bit, data_bit, NAIVE)[1]);
    ASSERT_EQ(12, wordInsert(0,3, data_bit, data_bit, NAIVE)[0]);
    ASSERT_EQ(10, wordInsert(0,3, data_bit, data_bit, NAIVE)[1]);
    ASSERT_EQ(14, wordInsert(0,4, data_test1, data_bit, DCW_TRADITIONAL)[0]);
    ASSERT_EQ(6, wordInsert(0,4, data_bit, data_bit+32, DCW_TRADITIONAL)[1]);
    ASSERT_EQ(10, wordInsert(0,4, data_bit+32, data_bit+64, DCW_TRADITIONAL)[0]);
    ASSERT_EQ(5, wordInsert(0,4, data_bit+96, data_bit, DCW_TRADITIONAL)[1]);
    ASSERT_EQ(14, wordInsert(0,4, data_test1, data_bit, DCW)[0]);
    ASSERT_EQ(11, wordInsert(0,4, data_test1, data_bit, DCW)[1]);
    ASSERT_EQ(10, wordInsert(0,4, data_bit+32, data_bit+64, DCW)[0]);
    ASSERT_EQ(10, wordInsert(0,4, data_bit+32, data_bit+64, DCW)[1]);
    ASSERT_EQ(15, wordInsert(0,7, data_bit, data_bit+64, DCW)[0]);
    ASSERT_EQ(9, wordInsert(0,7, data_bit, data_bit+64, DCW)[1]);
    ASSERT_EQ(0, wordInsert(0,4, data_bit, data_bit+32, PERMUTATION_WRITE)[0]);
    ASSERT_EQ(0, wordInsert(0,4, data_bit, data_bit+32, PERMUTATION_WRITE)[1]);
    ASSERT_EQ(17, wordInsert(0,4, data_test1, data_bit+64, PERMUTATION_WRITE)[0]);
    ASSERT_EQ(17, wordInsert(0,4, data_test1, data_bit+64, PERMUTATION_WRITE)[1]);
    ASSERT_EQ(4, wordInsert(0,4, data_bit+32, data_bit+64, PERMUTATION_WRITE)[0]);
    ASSERT_EQ(4, wordInsert(0,4, data_bit+32, data_bit+64, PERMUTATION_WRITE)[1]);
  }
}


TEST(checkSkyrmionWord, case1)
{
  SkyrmionWord test;
  #ifdef DEBUG
    test.print();
  #endif
  ASSERT_EQ(0, test.getSht_engy());
  for (int i = 0; i < 2 * OVER_HEAD + MAX_SIZE +  MAX_SIZE / DISTANCE + 1; i++){
    ASSERT_EQ(0, test.getEntry(i));
  }
  ASSERT_EQ(false, test.isFull());
}

TEST(checkSkyrmionWordWithSize, case1)
{
  SkyrmionWord test(100);
  ASSERT_EQ(0, test.getSht_engy());
  for (int i = 0; i < 2 * OVER_HEAD + test.getIntervalSize()*DISTANCE + test.getIntervalSize() + 1; i++){
    ASSERT_EQ(0, test.getEntry(i));
  }
  ASSERT_EQ(false, test.isFull());
}

TEST(checkSkyrmionBit, case1)
{
  SkyrmionBit test;
  ASSERT_EQ(0, test.getSht_latcy_DMW());
  for (int i = 0; i < DISTANCE; i++){
    for (int j = 0; j < 2 * OVER_HEAD + MAX_SIZE +  MAX_SIZE / DISTANCE + 1; j++){
      ASSERT_EQ(0, test.getEntries(i, j));
    }
  }
  ASSERT_EQ(false, test.isFull());
}

TEST(checkSkyrmionBitWithSize, case1)
{
  SkyrmionBit test(100);
  ASSERT_EQ(0, test.getSht_latcy_DMW());
  for (int i = 0; i < DISTANCE; i++){
    for (int j = 0; j < 2 * OVER_HEAD + test.getIntervalSize()*DISTANCE +  test.getIntervalSize() + 1; j++){
      ASSERT_EQ(0, test.getEntries(i, j));
    }
  }
  ASSERT_EQ(false, test.isFull());
}

TEST(checkDeterminePorts, case1)
{
  SkyrmionWord test;
  if (DISTANCE == 16){
    ASSERT_EQ(37, test.determinePorts(68, 3));
    ASSERT_EQ(37, test.determinePorts(69, 3));
    ASSERT_EQ(36, test.determinePorts(69, 1));
    ASSERT_EQ(40, test.determinePorts(69, 8));
  } else if (DISTANCE == 32){
    ASSERT_EQ(19, test.determinePorts(68, 3)); //Dis:32->19  37
    ASSERT_EQ(19, test.determinePorts(69, 3)); //Dis:32->18
    ASSERT_EQ(19, test.determinePorts(69, 1));
    ASSERT_EQ(21, test.determinePorts(69, 8));
  }
}

TEST(checkDeterminePortsBit, case1)
{
  SkyrmionBit test;
  int ports[2];
  if (DISTANCE == 16){
    int moves = test.determinePorts(0, ports);
    ASSERT_EQ(2, ports[0]);
    ASSERT_EQ(0, ports[1]);
    ASSERT_EQ(1, moves);
    moves = test.determinePorts(15, ports);
    ASSERT_EQ(1, ports[0]);
    ASSERT_EQ(MAX_SIZE/DISTANCE+2, ports[1]);
    ASSERT_EQ(1, moves);
    moves = test.determinePorts(16, ports);
    ASSERT_EQ(3, ports[0]);
    ASSERT_EQ(0, ports[1]);
    ASSERT_EQ(1, moves);
    moves = test.determinePorts(33, ports);
    ASSERT_EQ(4, ports[0]);
    ASSERT_EQ(0, ports[1]);
    ASSERT_EQ(2, moves);
    moves = test.determinePorts(34, ports);
    ASSERT_EQ(4, ports[0]);
    ASSERT_EQ(0, ports[1]);
    ASSERT_EQ(3, moves);
    moves = test.determinePorts(60, ports);
    ASSERT_EQ(4, ports[0]);
    ASSERT_EQ(MAX_SIZE/DISTANCE+2, ports[1]);
    ASSERT_EQ(4, moves);
    moves = test.determinePorts(64, ports);
    ASSERT_EQ(6, ports[0]);
    ASSERT_EQ(0, ports[1]);
    ASSERT_EQ(1, moves);
    moves = test.determinePorts(66, ports);
    ASSERT_EQ(6, ports[0]);
    ASSERT_EQ(0, ports[1]);
    ASSERT_EQ(3, moves);
    moves = test.determinePorts(79, ports);
    ASSERT_EQ(5, ports[0]);
    ASSERT_EQ(MAX_SIZE/DISTANCE+2, ports[1]);
    ASSERT_EQ(1, moves);
    moves = test.determinePorts(80, ports);
    ASSERT_EQ(7, ports[0]);
    ASSERT_EQ(0, ports[1]);
    ASSERT_EQ(1, moves);
    moves = test.determinePorts(96, ports);
    ASSERT_EQ(8, ports[0]);
    ASSERT_EQ(0, ports[1]);
    ASSERT_EQ(1, moves);
    moves = test.determinePorts(97, ports);
    ASSERT_EQ(8, ports[0]);
    ASSERT_EQ(0, ports[1]);
    ASSERT_EQ(2, moves);
    moves = test.determinePorts(126, ports);
    ASSERT_EQ(8, ports[0]);
    ASSERT_EQ(MAX_SIZE/DISTANCE+2, ports[1]);
    ASSERT_EQ(2, moves);
    moves = test.determinePorts(127, ports);
    ASSERT_EQ(8, ports[0]);
    ASSERT_EQ(MAX_SIZE/DISTANCE+2, ports[1]);
    ASSERT_EQ(1, moves);
  } else if (DISTANCE == 32){
    int moves = test.determinePorts(0, ports);
    ASSERT_EQ(2, ports[0]);
    ASSERT_EQ(0, ports[1]);
    ASSERT_EQ(1, moves);
    moves = test.determinePorts(15, ports);
    ASSERT_EQ(2, ports[0]);
    ASSERT_EQ(0, ports[1]);
    ASSERT_EQ(16, moves);
    moves = test.determinePorts(16, ports);
    ASSERT_EQ(1, ports[0]);
    ASSERT_EQ(MAX_SIZE/DISTANCE+2, ports[1]);
    ASSERT_EQ(16, moves);
    moves = test.determinePorts(33, ports);
    ASSERT_EQ(3, ports[0]);
    ASSERT_EQ(0, ports[1]);
    ASSERT_EQ(2, moves);
    moves = test.determinePorts(34, ports);
    ASSERT_EQ(3, ports[0]);
    ASSERT_EQ(0, ports[1]);
    ASSERT_EQ(3, moves);
    moves = test.determinePorts(60, ports);
    ASSERT_EQ(2, ports[0]);
    ASSERT_EQ(MAX_SIZE/DISTANCE+2, ports[1]);
    ASSERT_EQ(4, moves);
    moves = test.determinePorts(64, ports);
    ASSERT_EQ(4, ports[0]);
    ASSERT_EQ(0, ports[1]);
    ASSERT_EQ(1, moves);
    moves = test.determinePorts(66, ports);
    ASSERT_EQ(4, ports[0]);
    ASSERT_EQ(0, ports[1]);
    ASSERT_EQ(3, moves);
    moves = test.determinePorts(79, ports);
    ASSERT_EQ(4, ports[0]);
    ASSERT_EQ(0, ports[1]);
    ASSERT_EQ(16, moves);
    moves = test.determinePorts(80, ports);
    ASSERT_EQ(3, ports[0]);
    ASSERT_EQ(MAX_SIZE/DISTANCE+2, ports[1]);
    ASSERT_EQ(16, moves);
    moves = test.determinePorts(96, ports);
    ASSERT_EQ(5, ports[0]);
    ASSERT_EQ(0, ports[1]);
    ASSERT_EQ(1, moves);
    moves = test.determinePorts(97, ports);
    ASSERT_EQ(5, ports[0]);
    ASSERT_EQ(0, ports[1]);
    ASSERT_EQ(2, moves);
    moves = test.determinePorts(126, ports);
    ASSERT_EQ(4, ports[0]);
    ASSERT_EQ(MAX_SIZE/DISTANCE+2, ports[1]);
    ASSERT_EQ(2, moves);
    moves = test.determinePorts(127, ports);
    ASSERT_EQ(4, ports[0]);
    ASSERT_EQ(MAX_SIZE/DISTANCE+2, ports[1]);
    ASSERT_EQ(1, moves);
  }
}

TEST(checkBitPositions, case1)
{
  if (DISTANCE == 32){
    SkyrmionWord test(786);
    sky_size_t arr[4] = {171, 134, 0, 0};
    //10101011  10000110  0  0
    test.writeData(996, 4, arr, DCW, 0);
    vector<int> res = test.bitPositions(249);
    ASSERT_EQ(8, res.size());
    vector<int> output = {17,18,23,24,25,27,29,31};
    ASSERT_EQ(output, res);
  }
}

TEST(checkInsertRemoveDetect, case1)
{
  SkyrmionWord test;
  #ifdef DEBUG
    test.print();
  #endif
  for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
    for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
      test.setEntry(i, i);
    }
  }
  #ifdef DEBUG
    test.print();
  #endif
  test.insert(2, 1, 0); // here
  #ifdef DEBUG
    test.print();
  #endif
  ASSERT_EQ(1, test.detect(2, 0)); // here
  test.deleteSky(2, 0); // here
  #ifdef DEBUG
    test.print();
  #endif
  ASSERT_EQ(0, test.detect(2, 0)); // here
}


TEST(checkInsertRemoveDetectBit, case1)
{
  SkyrmionBit test;
  #ifdef DEBUG
    test.print();
  #endif
  for (int k = 0; k < DISTANCE; k++){
    for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
      for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
        test.setEntries(k, i, i);
      }
    }
  }
  #ifdef DEBUG
    test.print();
  #endif
  test.insert(2, 31, 1, 0); // here
  #ifdef DEBUG
    test.print();
  #endif
  ASSERT_EQ(1, test.detect(2, 31, 0)); // here
  test.deleteSky(2, 31, 0); // here
  #ifdef DEBUG
    test.print();
  #endif
  ASSERT_EQ(0, test.detect(2, 31, 0)); // here
}

TEST(checkShift, case1)
{
  SkyrmionWord test;
  #ifdef DEBUG
    test.print();
  #endif
  for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
    for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
      test.setEntry(i, i);
    }
  }
  #ifdef DEBUG
    test.print();
  #endif
  test.shift(0, 3, 0);
  #ifdef DEBUG
    test.print();
  #endif
  for (int i = 0; i < 2; i++){
    ASSERT_EQ(0, test.getEntry(OVER_HEAD + 1 + i * (DISTANCE + 1)));
  }
}

TEST(checkShift, case2)
{
  SkyrmionWord test;
  #ifdef DEBUG
    test.print();
  #endif
  for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
    for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
      test.setEntry(i, i);
    }
  }
  #ifdef DEBUG
    test.print();
  #endif
  test.shift(2, 3, 0);
  #ifdef DEBUG
    test.print();
  #endif
  ASSERT_EQ(OVER_HEAD+(3-1)*(DISTANCE + 1)-1, test.getEntry(OVER_HEAD+(3-1)*(DISTANCE + 1)));
}

TEST(checkShift, case3)
{
  SkyrmionWord test;
  #ifdef DEBUG
    test.print();
  #endif
  for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
    for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
      test.setEntry(i, i);
    }
  }
  #ifdef DEBUG
  test.print();
  #endif

  test.shift(4, 3, 0);
  ASSERT_EQ(OVER_HEAD + (3-1) * (DISTANCE + 1) + 1, test.getEntry(OVER_HEAD + (3-1) * (DISTANCE + 1)));
  test.shift(4, 3, 0);
  ASSERT_EQ(OVER_HEAD + (3-1) * (DISTANCE + 1) + 2, test.getEntry(OVER_HEAD + (3-1) * (DISTANCE + 1)));
  test.shift(3, 4, 0);
  ASSERT_EQ(0, test.getEntry(OVER_HEAD + (3-1) * (DISTANCE + 1)));
  #ifdef DEBUG
    test.print();
  #endif

}

TEST(checkShift, case4)
{
  SkyrmionWord test;
  #ifdef DEBUG
    test.print();
  #endif
  for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
    for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
      test.setEntry(i, i);
    }
  }
  #ifdef DEBUG
    test.print();
  #endif
  test.shift(6, 3, 0);
  #ifdef DEBUG
    test.print();
  #endif
  for (int i = 3; i < (6-1); i++){
    ASSERT_EQ(OVER_HEAD + (i-1) * (DISTANCE + 1) + 1, test.getEntry(OVER_HEAD + (i-1) * (DISTANCE + 1)));
  }
}

TEST(checkShift, case5)
{
  SkyrmionWord test;
  #ifdef DEBUG
    test.print();
  #endif
  for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
    for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
      test.setEntry(i, i);
    }
  }
  #ifdef DEBUG
    test.print();
  #endif
  for (int i = 0; i < DISTANCE; i++)
    test.shift(3, 6, 0);
  #ifdef DEBUG
    test.print();
  #endif
  for (int i = 3; i < (6-1); i++){
    ASSERT_EQ(OVER_HEAD + (i-1) * (DISTANCE + 1) + 1, test.getEntry(OVER_HEAD + i * (DISTANCE + 1)));
  }
}

TEST(checkShift, case6)
{
  SkyrmionWord test;
  #ifdef DEBUG
    test.print();
  #endif
  for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
    for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
      test.setEntry(i, i);
    }
  }
  #ifdef DEBUG
    test.print();
  #endif
  for (int i = 0; i < DISTANCE; i++)
    test.shift(3, 0, 0);
  #ifdef DEBUG
    test.print();
  #endif
  for (int i = 0; i < 3-1; i++){
    ASSERT_EQ(OVER_HEAD + (i+1) * (DISTANCE + 1) - 1, test.getEntry(OVER_HEAD + i * (DISTANCE + 1)));
  }
}

TEST(checkShiftBit, case1)
{
  SkyrmionBit test;
  #ifdef DEBUG
    test.print();
  #endif

  for (int k = 0; k < DISTANCE; k++){
    for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
      for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
        test.setEntries(k, i, i);
      }
    }
  }
  #ifdef DEBUG
    test.print();
  #endif
  test.shift(0, 3, 0, 64, 0);
  #ifdef DEBUG
    test.print();
  #endif
  for (int k = 0;  k < 64*8; k++){
    for (int i = 0; i < 2; i++){
      ASSERT_EQ(0, test.getEntries(k, OVER_HEAD + 1 + i * (DISTANCE + 1)));//-1
    }
  }
}

TEST(checkShiftBit, case2)
{
  SkyrmionBit test;
  #ifdef DEBUG
    test.print();
  #endif
  for (int k = 0; k < DISTANCE; k++){
    for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
      for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
        test.setEntries(k, i, i);
      }
    }
  }
  #ifdef DEBUG
    test.print();
  #endif
  test.shift(2, 3, 0, 64, 0);
  #ifdef DEBUG
    test.print();
  #endif
  for (int k = 0; k < DISTANCE; k++){
    ASSERT_EQ(OVER_HEAD+(3-1)*(DISTANCE + 1)-1, test.getEntries(k, OVER_HEAD+(3-1)*(DISTANCE + 1)));
  }
}

TEST(checkShiftBit, case3)
{
  SkyrmionBit test;
  #ifdef DEBUG
    test.print();
  #endif
  for (int k = 0; k < DISTANCE; k++){
    for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
      for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
        test.setEntries(k, i, i);
      }
    }
  }


  test.shift(4, 3, 0, 64, 0);
  for (int k = 0; k < DISTANCE; k++){
    ASSERT_EQ(OVER_HEAD + (3-1) * (DISTANCE + 1) + 1, test.getEntries(k, OVER_HEAD + (3-1) * (DISTANCE + 1)));
  }
  test.shift(4, 3, 0, 64, 0);
  for (int k = 0; k < DISTANCE; k++){
    ASSERT_EQ(OVER_HEAD + (3-1) * (DISTANCE + 1) + 2, test.getEntries(k, OVER_HEAD + (3-1) * (DISTANCE + 1)));
  }
  test.shift(3, 4, 0, 64, 0);
  for (int k = 0; k < DISTANCE; k++){
    ASSERT_EQ(0, test.getEntries(k, OVER_HEAD + (3-1) * (DISTANCE + 1)));
  }
  #ifdef DEBUG
    test.print();
  #endif

}

TEST(checkShiftBit, case4)
{
  SkyrmionBit test;
  #ifdef DEBUG
    test.print();
  #endif
  for (int k = 0; k < DISTANCE; k++){
    for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
      for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
        test.setEntries(k, i, i);
      }
    }
  }
  #ifdef DEBUG
    test.print();
  #endif
  test.shift(6, 3, 0, 64, 0);
  #ifdef DEBUG
    test.print();
  #endif
  for (int k = 0; k < DISTANCE; k++){
    for (int i = 3; i < (6-1); i++){
      ASSERT_EQ(OVER_HEAD + (i-1) * (DISTANCE + 1) + 1, test.getEntries(k, OVER_HEAD + (i-1) * (DISTANCE + 1)));
    }
  }
}

TEST(checkShiftBit, case5)
{
  SkyrmionBit test;
  #ifdef DEBUG
    test.print();
  #endif
  for (int k = 0; k < DISTANCE; k++){
    for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
      for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
        test.setEntries(k, i, i);
      }
    }
  }
  #ifdef DEBUG
    test.print();
  #endif
  for (int i = 0; i < DISTANCE; i++)
    test.shift(3, 6, 0, 64, 0);
  #ifdef DEBUG
    test.print();
  #endif
  for (int k = 0; k < DISTANCE; k++){
    for (int i = 3; i < (6-1); i++){
      ASSERT_EQ(OVER_HEAD + (i-1) * (DISTANCE + 1) + 1, test.getEntries(k, OVER_HEAD + i * (DISTANCE + 1)));
    }
  }
}

TEST(checkShiftBit, case6)
{
  SkyrmionBit test;
  #ifdef DEBUG
    test.print();
  #endif
  for (int k = 0; k < DISTANCE; k++){
    for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
      for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
        test.setEntries(k, i, i);
      }
    }
  }
  #ifdef DEBUG
    test.print();
  #endif
  for (int i = 0; i < DISTANCE; i++)
    test.shift(3, 0, 0, 64, 0);
  #ifdef DEBUG
    test.print();
  #endif
  for (int k = 0; k < DISTANCE; k++){
    for (int i = 0; i < 3-1; i++){
      ASSERT_EQ(OVER_HEAD + (i+1) * (DISTANCE + 1) - 1, test.getEntries(k, OVER_HEAD + i * (DISTANCE + 1)));
    }
  }
}

TEST(checkWriteNaiveTraditional, case1)
{
  SkyrmionWord testA;
  for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
    for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
      testA.setEntry(i, 0);
    }
  }
  #ifdef DEBUG
    testA.print();
  #endif

  int save = 1;
  sky_size_t data_bit[128]={0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,0}; // 61 bit 1
  sky_size_t data[16]={120, 182, 52, 136, 115, 92, 150, 0, 247, 33, 59, 152, 248, 213, 165, 98};
  sky_size_t data_bit_0[128]={0};

  // write 1
  int wAdd = 0;
  int wSize = 4;
  int start = 0;
  testA.writeData(wAdd, wSize, data+start, NAIVE_TRADITIONAL, save);
  int *ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, NAIVE_TRADITIONAL);
  int e1 = ptr[0];
  int l1 = ptr[1];
  int delE1 = ptr[2];
  int delL1 = ptr[3];
  int s1 = wordShift(wAdd,wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(4, testA.getN_checkFull());
  ASSERT_EQ(e1, testA.getIns_engy_DMW());
  ASSERT_EQ(l1, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1, testA.getSht_engy_DMW());
  ASSERT_EQ(s1, testA.getSht_latcy_DMW());
  delete [] ptr;

  int pos = wAdd;
  int len = wSize;
  int testPos = start;
  int rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 2
  wAdd = 0;
  wSize = 4;
  start = 0;
  testA.writeData(wAdd, wSize, data+start, NAIVE_TRADITIONAL, save);
  ptr = wordInsert(wAdd,wSize, data_bit, data_bit+start*8, NAIVE_TRADITIONAL);
  int e2 = ptr[0];
  int l2 = ptr[1];
  int delE2 = ptr[2];
  int delL2 = ptr[3];
  int s2 = wordShift(wAdd,wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(4, testA.getN_checkFull());
  ASSERT_EQ(e1+e2, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1*2, testA.getSht_engy_DMW());
  ASSERT_EQ(s1*2, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 3
  wAdd = 4;
  wSize = 4;
  start = 4;
  testA.writeData(wAdd, wSize, data+start, NAIVE_TRADITIONAL, save);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, NAIVE_TRADITIONAL);
  int e3 = ptr[0];
  int l3 = ptr[1];
  int delE3 = ptr[2];
  int delL3 = ptr[3];
  int s3 = wordShift(wAdd,wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(8, testA.getN_checkFull());
  ASSERT_EQ(e1+e2+e3, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1*2+s3, testA.getSht_engy_DMW());
  ASSERT_EQ(s1*2+s3, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 4
  wAdd = 4;
  wSize = 4;
  start = 8;
  testA.writeData(wAdd, wSize, data+start, NAIVE_TRADITIONAL, save);
  ptr = wordInsert(wAdd,wSize, data_bit+32, data_bit+start*8, NAIVE_TRADITIONAL);
  int e4 = ptr[0];
  int l4 = ptr[1];
  int delE4 = ptr[2];
  int delL4 = ptr[3];
  ASSERT_EQ(8, testA.getN_checkFull());
  ASSERT_EQ(e1+e2+e3+e4, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1*2+s3*2, testA.getSht_engy_DMW());
  ASSERT_EQ(s1*2+s3*2, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 5
  wAdd = 8;
  wSize = 4;
  start = 8;
  testA.writeData(wAdd, wSize, data+start, NAIVE_TRADITIONAL, save);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, NAIVE_TRADITIONAL);
  int e5 = ptr[0];
  int l5 = ptr[1];
  int delE5 = ptr[2];
  int delL5 = ptr[3];
  int s5 = wordShift(wAdd,wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(12, testA.getN_checkFull());
  ASSERT_EQ(e1+e2+e3+e4+e5, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5, testA.getSht_engy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 6
  wAdd = 8;
  wSize = 4;
  start = 12;
  testA.writeData(wAdd, wSize, data+start, NAIVE_TRADITIONAL, save);
  ptr = wordInsert(wAdd,wSize, data_bit+64, data_bit+start*8, NAIVE_TRADITIONAL);
  int e6 = ptr[0];
  int l6 = ptr[1];
  int delE6 = ptr[2];
  int delL6 = ptr[3];
  ASSERT_EQ(12, testA.getN_checkFull());
  ASSERT_EQ(e1+e2+e3+e4+e5+e6, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2, testA.getSht_engy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 7
  wAdd = 12;
  wSize = 4;
  start = 12;
  testA.writeData(wAdd, wSize, data+start, NAIVE_TRADITIONAL, save);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, NAIVE_TRADITIONAL);
  int e7 = ptr[0];
  int l7 = ptr[1];
  int delE7 = ptr[2];
  int delL7 = ptr[3];
  int s7 = wordShift(wAdd, wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(16, testA.getN_checkFull());
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7, testA.getSht_engy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 8
  wAdd = 12;
  wSize = 4;
  start = 0;
  testA.writeData(wAdd, wSize, data+start, NAIVE_TRADITIONAL, save);
  ptr = wordInsert(wAdd,wSize, data_bit+96, data_bit+start*8, NAIVE_TRADITIONAL);
  int e8 = ptr[0];
  int l8 = ptr[1];
  int delE8 = ptr[2];
  int delL8 = ptr[3];
  ASSERT_EQ(16, testA.getN_checkFull());
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7*2, testA.getSht_engy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7*2, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 9
  wAdd = 254;
  wSize = 1;
  start = 0;
  testA.writeData(wAdd, wSize, data+start, NAIVE_TRADITIONAL, save);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, NAIVE_TRADITIONAL);
  int e9 = ptr[0];
  int l9 = ptr[1];
  int delE9 = ptr[2];
  int delL9 = ptr[3];
  int s9 = wordShift(wAdd, wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(17, testA.getN_checkFull());
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7*2+s9, testA.getSht_engy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7*2+s9, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 10
  wAdd = 253;
  wSize = 2;
  start = 0;
  testA.writeData(wAdd, wSize, data+start, NAIVE_TRADITIONAL, save);
  sky_size_t data_bit2[16]={0,0,0,0,0,0,0,0,
                            0,1,1,1,1,0,0,0};
  ptr = wordInsert(wAdd,wSize, data_bit2, data_bit+start*8, NAIVE_TRADITIONAL);
  int e10 = ptr[0];
  int l10 = ptr[1];
  int delE10 = ptr[2];
  int delL10 = ptr[3];
  int s10 = wordShift(wAdd, wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(18, testA.getN_checkFull());
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7*2+s9+s10, testA.getSht_engy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7*2+s9+s10, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 11
  wAdd = 252;
  wSize = 3;
  start = 0;
  testA.writeData(wAdd, wSize, data+start, NAIVE_TRADITIONAL, save);
  sky_size_t data_bit3[24]={0,0,0,0,0,0,0,0,
                            0,1,1,1,1,0,0,0,
                            0,1,1,1,1,0,0,0};
  ptr = wordInsert(wAdd,wSize, data_bit3, data_bit+start*8, NAIVE_TRADITIONAL);
  int e11 = ptr[0];
  int l11 = ptr[1];
  int delE11 = ptr[2];
  int delL11 = ptr[3];
  int s11 = wordShift(wAdd, wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(19, testA.getN_checkFull());
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7*2+s9+s10+s11, testA.getSht_engy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7*2+s9+s10+s11, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 12 C
  wAdd = 251;
  wSize = 5;
  start = 0;
  testA.writeData(wAdd, wSize, data+start, NAIVE_TRADITIONAL, save);
  sky_size_t data_bit4[40]={0,0,0,0,0,0,0,0,
                            0,1,1,1,1,0,0,0,
                            0,1,1,1,1,0,0,0,
                            0,1,1,1,1,0,0,0,
                            0,0,0,0,0,0,0,0};
  ptr = wordInsert(wAdd,wSize, data_bit4, data_bit+start*8, NAIVE_TRADITIONAL);
  int e12 = ptr[0];
  int l12 = ptr[1];
  int delE12 = ptr[2];
  int delL12 = ptr[3];
  int s12 = wordShift(wAdd, wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(21, testA.getN_checkFull());
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7*2+s9+s10+s11+s12, testA.getSht_engy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7*2+s9+s10+s11+s12, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 13 D
  wAdd = 245;
  wSize = 4;
  start = 4;
  testA.writeData(wAdd, wSize, data+start, NAIVE_TRADITIONAL, save);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, NAIVE_TRADITIONAL);
  int e13 = ptr[0];
  int l13 = ptr[1];
  int delE13 = ptr[2];
  int delL13 = ptr[3];
  int s13 = wordShift(wAdd, wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(25, testA.getN_checkFull());
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12+delE13, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12+delL13, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7*2+s9+s10+s11+s12+s13, testA.getSht_engy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7*2+s9+s10+s11+s12+s13, testA.getSht_latcy_DMW());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }


  // write 14 B
  wAdd = 244;
  wSize = 6;
  start = 0;
  testA.writeData(wAdd, wSize, data+start, NAIVE_TRADITIONAL, save);
  sky_size_t data_bit5[48]={0,0,0,0,0,0,0,0,
                            0,1,1,1,0,0,1,1,
                            0,1,0,1,1,1,0,0,
                            1,0,0,1,0,1,1,0,
                            0,0,0,0,0,0,0,0};
  ptr = wordInsert(wAdd,wSize, data_bit5, data_bit+start*8, NAIVE_TRADITIONAL);
  int e14 = ptr[0];
  int l14 = ptr[1];
  int delE14 = ptr[2];
  int delL14 = ptr[3];
  int s14 = wordShift(wAdd, wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(27, testA.getN_checkFull());
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13+l14, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12+delE13+delE14, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12+delL13+delL14, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7*2+s9+s10+s11+s12+s13+s14, testA.getSht_engy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7*2+s9+s10+s11+s12+s13+s14, testA.getSht_latcy_DMW());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 15
  wAdd = 249;
  wSize = 3;
  start = 0;
  testA.writeData(wAdd, wSize, data+start, NAIVE_TRADITIONAL, save);
  sky_size_t data_bit6[24]={0,0,0,0,0,0,0,0,
                            0,0,0,0,0,0,0,0,
                            0,1,1,1,1,0,0,0};
  ptr = wordInsert(wAdd,wSize, data_bit6, data_bit+start*8, NAIVE_TRADITIONAL);
  int e15 = ptr[0];
  int l15 = ptr[1];
  int delE15 = ptr[2];
  int delL15 = ptr[3];
  int s15 = wordShift(wAdd, wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(28, testA.getN_checkFull());
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14+e15, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13+l14+l15, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12+delE13+delE14+delE15, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12+delL13+delL14+delL15, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7*2+s9+s10+s11+s12+s13+s14+s15, testA.getSht_engy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7*2+s9+s10+s11+s12+s13+s14+s15, testA.getSht_latcy_DMW());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 16
  wAdd = 249;
  wSize = 1;
  start = 0;
  testA.writeData(wAdd, wSize, data+start, NAIVE_TRADITIONAL, save);
  ptr = wordInsert(wAdd,wSize, data_bit, data_bit+start*8, NAIVE_TRADITIONAL);
  int e16 = ptr[0];
  int l16 = ptr[1];
  int delE16 = ptr[2];
  int delL16 = ptr[3];
  int s16 = wordShift(wAdd, wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(28, testA.getN_checkFull());
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14+e15+e16, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13+l14+l15+l16, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12+delE13+delE14+delE15+delE16, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12+delL13+delL14+delL15+delL16, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7*2+s9+s10+s11+s12+s13+s14+s15+s16, testA.getSht_engy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7*2+s9+s10+s11+s12+s13+s14+s15+s16, testA.getSht_latcy_DMW());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 17
  wAdd = 243;
  wSize = 4;
  start = 1;
  testA.writeData(wAdd, wSize, data+start, NAIVE_TRADITIONAL, save);
  sky_size_t data_bit7[32]={0,0,0,0,0,0,0,0,
                            0,1,1,1,1,0,0,0,
                            1,0,1,1,0,1,1,0,
                            0,0,1,1,0,1,0,0};
  ptr = wordInsert(wAdd,wSize, data_bit7, data_bit+start*8, NAIVE_TRADITIONAL);
  int e17 = ptr[0];
  int l17 = ptr[1];
  int delE17 = ptr[2];
  int delL17 = ptr[3];
  int s17 = wordShift(wAdd, wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(29, testA.getN_checkFull());
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14+e15+e16+e17, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13+l14+l15+l16+l17, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12+delE13+delE14+delE15+delE16+delE17, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12+delL13+delL14+delL15+delL16+delL17, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7*2+s9+s10+s11+s12+s13+s14+s15+s16+s17, testA.getSht_engy_DMW());
  ASSERT_EQ(s1*2+s3*2+s5*2+s7*2+s9+s10+s11+s12+s13+s14+s15+s16+s17, testA.getSht_latcy_DMW());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }
  //check full
  for (int i = 0; i < MAX_SIZE/8/16; i++){
    testA.writeData(i*16, 16, data+1, NAIVE_TRADITIONAL, save);
  }
  ASSERT_EQ(256, testA.getN_checkFull());
  ASSERT_EQ(true, testA.isFull());
}

TEST(checkWriteNaive, case1)
{
  SkyrmionWord testA;
  for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
    for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
      testA.setEntry(i, 0);
    }
  }
  #ifdef DEBUG
    testA.print();
  #endif
  int save = 1;
  sky_size_t data_bit[128]={0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,0}; // 61 bit 1
  sky_size_t data[16]={120, 182, 52, 136, 115, 92, 150, 0, 247, 33, 59, 152, 248, 213, 165, 98};
  sky_size_t data_bit_0[128]={0};

  // write 1
  int wAdd = 0;
  int wSize = 4;
  int start = 0;
  testA.writeData(wAdd, wSize, data+start, NAIVE, save);
  int* ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, NAIVE);
  int e1 = ptr[0];
  int l1 = ptr[1];
  int delE1 = ptr[2];
  int delL1 = ptr[3];
  int s1 = wordShift(wAdd, wSize, NAIVE);
  ASSERT_EQ(e1, testA.getIns_engy_DMW());
  ASSERT_EQ(l1, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1, testA.getSht_engy_DMW());
  ASSERT_EQ(s1, testA.getSht_latcy_DMW());
  delete [] ptr;

  int pos = wAdd;
  int len = wSize;
  int testPos = start;
  int rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 2
  wAdd = 0;
  wSize = 4;
  start = 4;
  testA.writeData(wAdd, wSize, data+start, NAIVE, save);
  ptr = wordInsert(wAdd,wSize, data_bit, data_bit+start*8, NAIVE);
  int e2 = ptr[0];
  int l2 = ptr[1];
  int delE2 = ptr[2];
  int delL2 = ptr[3];
  int s2 = wordShift(wAdd, wSize, NAIVE);
  ASSERT_EQ(e1+e2, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1+s2, testA.getSht_engy_DMW());
  ASSERT_EQ(s1+s2, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 3
  wAdd = 4;
  wSize = 4;
  start = 4;
  testA.writeData(wAdd, wSize, data+start, NAIVE, save);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, NAIVE);
  int e3 = ptr[0];
  int l3 = ptr[1];
  int delE3 = ptr[2];
  int delL3 = ptr[3];
  int s3 = wordShift(wAdd, wSize, NAIVE);
  ASSERT_EQ(e1+e2+e3, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1+s2+s3, testA.getSht_engy_DMW());
  ASSERT_EQ(s1+s2+s3, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 4
  wAdd = 4;
  wSize = 4;
  start = 8;
  testA.writeData(wAdd, wSize, data+start, NAIVE, save);
  ptr = wordInsert(wAdd,wSize, data_bit+32, data_bit+start*8, NAIVE);
  int e4 = ptr[0];
  int l4 = ptr[1];
  int delE4 = ptr[2];
  int delL4 = ptr[3];
  int s4 = wordShift(wAdd, wSize, NAIVE);
  ASSERT_EQ(e1+e2+e3+e4, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1+s2+s3+s4, testA.getSht_engy_DMW());
  ASSERT_EQ(s1+s2+s3+s4, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 5
  wAdd = 8;
  wSize = 4;
  start = 8;
  testA.writeData(wAdd, wSize, data+start, NAIVE, save);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, NAIVE);
  int e5 = ptr[0];
  int l5 = ptr[1];
  int delE5 = ptr[2];
  int delL5 = ptr[3];
  int s5 = wordShift(wAdd, wSize, NAIVE);
  ASSERT_EQ(e1+e2+e3+e4+e5, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5, testA.getSht_engy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 6
  wAdd = 8;
  wSize = 4;
  start = 12;
  testA.writeData(wAdd, wSize, data+start, NAIVE, save);
  ptr = wordInsert(wAdd,wSize, data_bit+64, data_bit+start*8, NAIVE);
  int e6 = ptr[0];
  int l6 = ptr[1];
  int delE6 = ptr[2];
  int delL6 = ptr[3];
  int s6 = wordShift(wAdd, wSize, NAIVE);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6, testA.getSht_engy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 7
  wAdd = 12;
  wSize = 4;
  start = 12;
  testA.writeData(wAdd, wSize, data+start, NAIVE, save);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, NAIVE);
  int e7 = ptr[0];
  int l7 = ptr[1];
  int delE7 = ptr[2];
  int delL7 = ptr[3];
  int s7 = wordShift(wAdd, wSize, NAIVE);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7, testA.getSht_engy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 8
  wAdd = 12;
  wSize = 4;
  start = 0;
  testA.writeData(wAdd, wSize, data+start, NAIVE, save);
  ptr = wordInsert(wAdd,wSize, data_bit+96, data_bit+start*8, NAIVE);
  int e8 = ptr[0];
  int l8 = ptr[1];
  int delE8 = ptr[2];
  int delL8 = ptr[3];
  int s8 = wordShift(wAdd, wSize, NAIVE);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8, testA.getSht_engy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 9
  wAdd = 254;
  wSize = 1;
  start = 0;
  testA.writeData(wAdd, wSize, data+start, NAIVE, save);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, NAIVE);
  int e9 = ptr[0];
  int l9 = ptr[1];
  int delE9 = ptr[2];
  int delL9 = ptr[3];
  int s9 = wordShift(wAdd, wSize, NAIVE);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9, testA.getSht_engy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 10
  wAdd = 253;
  wSize = 2;
  start = 0;
  testA.writeData(wAdd, wSize, data+start, NAIVE, save);
  sky_size_t data_bit2[16]={0,0,0,0,0,0,0,0,
                            0,1,1,1,1,0,0,0};
  ptr = wordInsert(wAdd,wSize, data_bit2, data_bit+start*8, NAIVE);
  int e10 = ptr[0];
  int l10 = ptr[1];
  int delE10 = ptr[2];
  int delL10 = ptr[3];
  int s10 = wordShift(wAdd, wSize, NAIVE);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10, testA.getSht_engy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 11
  wAdd = 252;
  wSize = 3;
  start = 0;
  testA.writeData(wAdd, wSize, data+start, NAIVE, save);
  sky_size_t data_bit3[24]={0,0,0,0,0,0,0,0,
                            0,1,1,1,1,0,0,0,
                            0,1,1,1,1,0,0,0};
  ptr = wordInsert(wAdd,wSize, data_bit3, data_bit+start*8, NAIVE);
  int e11 = ptr[0];
  int l11 = ptr[1];
  int delE11 = ptr[2];
  int delL11 = ptr[3];
  int s11 = wordShift(wAdd, wSize, NAIVE);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11, testA.getSht_engy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 12 C
  wAdd = 251;
  wSize = 5;
  start = 0;
  testA.writeData(wAdd, wSize, data+start, NAIVE, save);
  sky_size_t data_bit4[40]={0,0,0,0,0,0,0,0,
                            0,1,1,1,1,0,0,0,
                            0,1,1,1,1,0,0,0,
                            0,1,1,1,1,0,0,0,
                            0,0,0,0,0,0,0,0};
  ptr = wordInsert(wAdd,wSize, data_bit4, data_bit+start*8, NAIVE);
  int e12 = ptr[0];
  int l12 = ptr[1];
  int delE12 = ptr[2];
  int delL12 = ptr[3];
  int s12 = wordShift(wAdd, wSize, NAIVE);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12, testA.getSht_engy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12, testA.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
    testA.print();
  #endif
  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 13 D
  wAdd = 245;
  wSize = 4;
  start = 4;
  testA.writeData(wAdd, wSize, data+start, NAIVE, save);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, NAIVE);
  int e13 = ptr[0];
  int l13 = ptr[1];
  int delE13 = ptr[2];
  int delL13 = ptr[3];
  int s13 = wordShift(wAdd, wSize, NAIVE);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12+delE13, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12+delL13, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13, testA.getSht_engy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13, testA.getSht_latcy_DMW());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 14 B
  wAdd = 244;
  wSize = 6;
  start = 0;
  testA.writeData(wAdd, wSize, data+start, NAIVE, save);
  sky_size_t data_bit5[48]={0,0,0,0,0,0,0,0,
                            0,1,1,1,0,0,1,1,
                            0,1,0,1,1,1,0,0,
                            1,0,0,1,0,1,1,0,
                            0,0,0,0,0,0,0,0};
   ptr= wordInsert(wAdd,wSize, data_bit5, data_bit+start*8, NAIVE);
  int e14 = ptr[0];
  int l14 = ptr[1];
  int delE14 = ptr[2];
  int delL14 = ptr[3];
  int s14 = wordShift(wAdd, wSize, NAIVE);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13+l14, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12+delE13+delE14, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12+delL13+delL14, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14, testA.getSht_engy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14, testA.getSht_latcy_DMW());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 15
  wAdd = 249;
  wSize = 3;
  start = 0;
  testA.writeData(wAdd, wSize, data+start, NAIVE, save);
  sky_size_t data_bit6[24]={0,0,0,0,0,0,0,0,
                            0,0,0,0,0,0,0,0,
                            0,1,1,1,1,0,0,0};
  ptr = wordInsert(wAdd,wSize, data_bit6, data_bit+start*8, NAIVE);
  int e15 = ptr[0];
  int l15 = ptr[1];
  int delE15 = ptr[2];
  int delL15 = ptr[3];
  int s15 = wordShift(wAdd, wSize, NAIVE);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14+e15, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13+l14+l15, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12+delE13+delE14+delE15, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12+delL13+delL14+delL15, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15, testA.getSht_engy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15, testA.getSht_latcy_DMW());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 16
  wAdd = 249;
  wSize = 1;
  start = 0;
  testA.writeData(wAdd, wSize, data+start, NAIVE, save);
  ptr = wordInsert(wAdd,wSize, data_bit, data_bit+start*8, NAIVE);
  int e16 = ptr[0];
  int l16 = ptr[1];
  int delE16 = ptr[2];
  int delL16 = ptr[3];
  int s16 = wordShift(wAdd, wSize, NAIVE);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14+e15+e16, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13+l14+l15+l16, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12+delE13+delE14+delE15+delE16, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12+delL13+delL14+delL15+delL16, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16, testA.getSht_engy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16, testA.getSht_latcy_DMW());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 17
  wAdd = 243;
  wSize = 4;
  start = 1;
  testA.writeData(wAdd, wSize, data+start, NAIVE, save);
  sky_size_t data_bit7[32]={0,0,0,0,0,0,0,0,
                            0,1,1,1,1,0,0,0,
                            1,0,1,1,0,1,1,0,
                            0,0,1,1,0,1,0,0};
  ptr = wordInsert(wAdd,wSize, data_bit7, data_bit+start*8, NAIVE);
  int e17 = ptr[0];
  int l17 = ptr[1];
  int delE17 = ptr[2];
  int delL17 = ptr[3];
  int s17 = wordShift(wAdd, wSize, NAIVE);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14+e15+e16+e17, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13+l14+l15+l16+l17, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12+delE13+delE14+delE15+delE16+delE17, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12+delL13+delL14+delL15+delL16+delL17, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW());
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17, testA.getSht_engy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17, testA.getSht_latcy_DMW());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 18
  wAdd = 239;
  wSize = 16;
  start = 0;
  testA.writeData(wAdd, wSize, data+start, NAIVE, save);
  sky_size_t data_bit8[128]={0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             1,0,1,1,0,1,1,0,
                             0,0,1,1,0,1,0,0,
                             1,0,0,0,1,0,0,0,
                             0,1,1,1,0,0,1,1,
                             1,0,0,0,1,0,0,0,
                             0,1,1,1,0,0,1,1,
                             0,1,1,1,1,0,0,0,
                             1,0,1,1,0,1,1,0,
                             0,0,1,1,0,1,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0};
  ptr = wordInsert(wAdd,wSize, data_bit8, data_bit+start*8, NAIVE);
  int e18 = ptr[0];
  int l18 = ptr[1];
  int delE18 = ptr[2];
  int delL18 = ptr[3];
  int s18 = wordShift(wAdd, wSize, NAIVE);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14+e15+e16+e17+e18, testA.getIns_engy_DMW());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13+l14+l15+l16+l17+l18, testA.getIns_latcy_DMW());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12+delE13+delE14+delE15+delE16+delE17+delE18, testA.getDel_engy_DMW());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12+delL13+delL14+delL15+delL16+delL17+delL18, testA.getDel_latcy_DMW());
  ASSERT_EQ(0, testA.getDet_engy_DMW()); //
  ASSERT_EQ(0, testA.getDet_latcy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18, testA.getSht_engy_DMW());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18, testA.getSht_latcy_DMW());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(testA.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

}

TEST(checkWriteDCWTRADITIONAL, case1)
{
  SkyrmionWord test;
  for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
    for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
      test.setEntry(i, 0);
    }
  }
  #ifdef DEBUG
    test.print();
  #endif

  sky_size_t data_bit[128]={0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,0}; // 61 bit 1
  sky_size_t data[16]={120, 182, 52, 136, 115, 92, 150, 0, 247, 33, 59, 152, 248, 213, 165, 98};
  sky_size_t data_bit_0[128]={0};
  // write 1
  int wAdd = 0;
  int wSize = 4;
  int start = 0;
  test.writeData(wAdd, wSize, data+start, DCW_TRADITIONAL, 0);
  int* ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, DCW_TRADITIONAL);
  int e1 = ptr[0];
  int l1 = ptr[1];
  int d1 = wordDetect(wAdd,wSize, DCW_TRADITIONAL);
  int delE1 = ptr[2];
  int delL1 = ptr[3];
  int s1 = wordShift(wAdd, wSize, DCW_TRADITIONAL);
  ASSERT_EQ(e1, test.getIns_engy());
  ASSERT_EQ(l1, test.getIns_latcy());
  ASSERT_EQ(delE1, test.getDel_engy());
  ASSERT_EQ(delL1, test.getDel_latcy());
  ASSERT_EQ(4*8, test.getDet_engy());
  ASSERT_EQ(d1, test.getDet_latcy());
  ASSERT_EQ(s1, test.getSht_engy());
  ASSERT_EQ(s1, test.getSht_latcy());
  delete [] ptr;

  int pos = wAdd;
  int len = wSize;
  int testPos = start;
  int rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 2
  wAdd = 0;
  wSize = 4;
  start = 4;
  test.writeData(wAdd, wSize, data+start, DCW_TRADITIONAL, 0);
  ptr = wordInsert(wAdd,wSize, data_bit, data_bit+start*8, DCW_TRADITIONAL);
  int e2 = ptr[0];
  int l2 = ptr[1];
  int d2 = wordDetect(wAdd,wSize, DCW_TRADITIONAL);
  int delE2 = ptr[2];
  int delL2 = ptr[3];
  int s2 = wordShift(wAdd, wSize, DCW_TRADITIONAL);
  ASSERT_EQ(e1+e2, test.getIns_engy());
  ASSERT_EQ(l1+l2, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2, test.getDel_engy());
  ASSERT_EQ(delL1+delL2, test.getDel_latcy());
  ASSERT_EQ(4*8*2, test.getDet_engy());
  ASSERT_EQ(d1+d2, test.getDet_latcy());
  ASSERT_EQ(s1+s2, test.getSht_engy());
  ASSERT_EQ(s1+s2, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 3
  wAdd = 4;
  wSize = 4;
  start = 4;
  test.writeData(wAdd, wSize, data+start, DCW_TRADITIONAL, 0);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, DCW_TRADITIONAL);
  int e3 = ptr[0];
  int l3 = ptr[1];
  int d3 = wordDetect(wAdd,wSize, DCW_TRADITIONAL);
  int delE3 = ptr[2];
  int delL3 = ptr[3];
  int s3 = wordShift(wAdd, wSize, DCW_TRADITIONAL);
  ASSERT_EQ(e1+e2+e3, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3, test.getDel_latcy());
  ASSERT_EQ(4*8*3, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 4
  wAdd = 4;
  wSize = 4;
  start = 8;
  test.writeData(wAdd, wSize, data+start, DCW_TRADITIONAL, 0);
  ptr = wordInsert(wAdd,wSize, data_bit+32, data_bit+start*8, DCW_TRADITIONAL);
  int e4 = ptr[0];
  int l4 = ptr[1];
  int d4 = wordDetect(wAdd,wSize, DCW_TRADITIONAL);
  int delE4 = ptr[2];
  int delL4 = ptr[3];
  int s4 = wordShift(wAdd, wSize, DCW_TRADITIONAL);
  ASSERT_EQ(e1+e2+e3+e4, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4, test.getDel_latcy());
  ASSERT_EQ(4*8*4, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 5
  wAdd = 8;
  wSize = 4;
  start = 8;
  test.writeData(wAdd, wSize, data+start, DCW_TRADITIONAL, 0);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, DCW_TRADITIONAL);
  int e5 = ptr[0];
  int l5 = ptr[1];
  int d5 = wordDetect(wAdd,wSize, DCW_TRADITIONAL);
  int delE5 = ptr[2];
  int delL5 = ptr[3];
  int s5 = wordShift(wAdd, wSize, DCW_TRADITIONAL);
  ASSERT_EQ(e1+e2+e3+e4+e5, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5, test.getDel_latcy());
  ASSERT_EQ(4*8*5, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 6
  wAdd = 8;
  wSize = 4;
  start = 12;
  test.writeData(wAdd, wSize, data+start, DCW_TRADITIONAL, 0);
  ptr = wordInsert(wAdd,wSize, data_bit+64, data_bit+start*8, DCW_TRADITIONAL);
  int e6 = ptr[0];
  int l6 = ptr[1];
  int d6 = wordDetect(wAdd,wSize, DCW_TRADITIONAL);
  int delE6 = ptr[2];
  int delL6 = ptr[3];
  int s6 = wordShift(wAdd, wSize, DCW_TRADITIONAL);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6, test.getDel_latcy());
  ASSERT_EQ(4*8*6, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 7
  wAdd = 12;
  wSize = 4;
  start = 12;
  test.writeData(wAdd, wSize, data+start, DCW_TRADITIONAL, 0);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, DCW_TRADITIONAL);
  int e7 = ptr[0];
  int l7 = ptr[1];
  int d7 = wordDetect(wAdd,wSize, DCW_TRADITIONAL);
  int delE7 = ptr[2];
  int delL7 = ptr[3];
  int s7 = wordShift(wAdd, wSize, DCW_TRADITIONAL);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7, test.getDel_latcy());
  ASSERT_EQ(4*8*7, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 8
  wAdd = 12;
  wSize = 4;
  start = 0;
  test.writeData(wAdd, wSize, data+start, DCW_TRADITIONAL, 0);
  ptr = wordInsert(wAdd,wSize, data_bit+96, data_bit+start*8, DCW_TRADITIONAL);
  int e8 = ptr[0];
  int l8 = ptr[1];
  int d8 = wordDetect(wAdd,wSize, DCW_TRADITIONAL);
  int delE8 = ptr[2];
  int delL8 = ptr[3];
  int s8 = wordShift(wAdd, wSize, DCW_TRADITIONAL);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8, test.getDel_latcy());
  ASSERT_EQ(4*8*8, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7+d8, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 9 C
  wAdd = 251;
  wSize = 5;
  start = 0;
  test.writeData(wAdd, wSize, data+start, DCW_TRADITIONAL, 0);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, DCW_TRADITIONAL);
  int e9 = ptr[0];
  int l9 = ptr[1];
  int d9 = wordDetect(wAdd,wSize, DCW_TRADITIONAL);
  int delE9 = ptr[2];
  int delL9 = ptr[3];
  int s9 = wordShift(wAdd, wSize, DCW_TRADITIONAL);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9, test.getDel_latcy());
  ASSERT_EQ(4*8*8+40, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7+d8+d9, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 10 D
  wAdd = 239;
  wSize = 16;
  start = 0;
  test.writeData(wAdd, wSize, data+start, DCW_TRADITIONAL, 0);
  sky_size_t data_bit2[128]={0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,1,1,1,1,0,0,0,
                             1,0,1,1,0,1,1,0,
                             0,0,1,1,0,1,0,0,
                             1,0,0,0,1,0,0,0};
  ptr = wordInsert(wAdd,wSize, data_bit2, data_bit+start*8, DCW_TRADITIONAL);
  int e10 = ptr[0];
  int l10 = ptr[1];
  int d10 = wordDetect(wAdd,wSize, DCW_TRADITIONAL);
  int delE10 = ptr[2];
  int delL10 = ptr[3];
  int s10 = wordShift(wAdd, wSize, DCW_TRADITIONAL);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10, test.getDel_latcy());
  ASSERT_EQ(4*8*8+40+16*8, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7+d8+d9+d10, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 11 D
  wAdd = 241;
  wSize = 12;
  start = 0;
  test.writeData(wAdd, wSize, data+start, DCW_TRADITIONAL, 0);
  sky_size_t data_bit3[96]={0,0,1,1,0,1,0,0,
                             1,0,0,0,1,0,0,0,
                             0,1,1,1,0,0,1,1,
                             0,1,0,1,1,1,0,0,
                             1,0,0,1,0,1,1,0,
                             0,0,0,0,0,0,0,0,
                             1,1,1,1,0,1,1,1,
                             0,0,1,0,0,0,0,1,
                             0,0,1,1,1,0,1,1,
                             1,0,0,1,1,0,0,0,
                             1,1,1,1,1,0,0,0,
                             1,1,0,1,0,1,0,1};
  ptr = wordInsert(wAdd,wSize, data_bit3, data_bit+start*8, DCW_TRADITIONAL);
  int e11 = ptr[0];
  int l11 = ptr[1];
  int d11 = wordDetect(wAdd,wSize, DCW_TRADITIONAL);
  int delE11 = ptr[2];
  int delL11 = ptr[3];
  int s11 = wordShift(wAdd, wSize, DCW_TRADITIONAL);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11, test.getDel_latcy());
  ASSERT_EQ(4*8*8+40+16*8+12*8, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7+d8+d9+d10+d11, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 12
  wAdd = 249;
  wSize = 2;
  start = 0;
  test.writeData(wAdd, wSize, data+start, DCW_TRADITIONAL, 0);
  sky_size_t data_bit4[16]={1,1,1,1,0,1,1,1,
                            0,0,1,0,0,0,0,1,};
  ptr = wordInsert(wAdd,wSize, data_bit4, data_bit+start*8, DCW_TRADITIONAL);
  int e12 = ptr[0];
  int l12 = ptr[1];
  int d12 = wordDetect(wAdd,wSize, DCW_TRADITIONAL);
  int delE12 = ptr[2];
  int delL12 = ptr[3];
  int s12 = wordShift(wAdd, wSize, DCW_TRADITIONAL);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12, test.getDel_latcy());
  ASSERT_EQ(4*8*8+40+16*8+12*8+16, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7+d8+d9+d10+d11+d12, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 13 B
  wAdd = 244;
  wSize = 5;
  start = 0;
  test.writeData(wAdd, wSize, data+start, DCW_TRADITIONAL, 0);
  sky_size_t data_bit5[40]={1,0,0,0,1,0,0,0,
                            0,1,1,1,0,0,1,1,
                            0,1,0,1,1,1,0,0,
                            1,0,0,1,0,1,1,0,
                            0,0,0,0,0,0,0,0};
  ptr = wordInsert(wAdd,wSize, data_bit5, data_bit+start*8, DCW_TRADITIONAL);
  int e13 = ptr[0];
  int l13 = ptr[1];
  int d13 = wordDetect(wAdd,wSize, DCW_TRADITIONAL);
  int delE13 = ptr[2];
  int delL13 = ptr[3];
  int s13 = wordShift(wAdd, wSize, DCW_TRADITIONAL);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12+delE13, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12+delL13, test.getDel_latcy());
  ASSERT_EQ(4*8*8+40+16*8+12*8+16+40, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7+d8+d9+d10+d11+d12+d13, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 14 D
  wAdd = 245;
  wSize = 4;
  start = 2;
  test.writeData(wAdd, wSize, data+start, DCW_TRADITIONAL, 0);
  sky_size_t data_bit6[32]={1,0,1,1,0,1,1,0,
                            0,0,1,1,0,1,0,0,
                            1,0,0,0,1,0,0,0,
                            0,1,1,1,0,0,1,1};
  ptr = wordInsert(wAdd,wSize, data_bit6, data_bit+start*8, DCW_TRADITIONAL);
  int e14 = ptr[0];
  int l14 = ptr[1];
  int d14 = wordDetect(wAdd,wSize, DCW_TRADITIONAL);
  int delE14 = ptr[2];
  int delL14 = ptr[3];
  int s14 = wordShift(wAdd, wSize, DCW_TRADITIONAL);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13+l14, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12+delE13+delE14, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12+delL13+delL14, test.getDel_latcy());
  ASSERT_EQ(4*8*8+40+16*8+12*8+16+40+32, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7+d8+d9+d10+d11+d12+d13+d14, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 15
  wAdd = 245;
  wSize = 1;
  start = 8;
  test.writeData(wAdd, wSize, data+start, DCW_TRADITIONAL, 0);
  ptr = wordInsert(wAdd,wSize, data_bit+16, data_bit+start*8, DCW_TRADITIONAL);
  int e15 = ptr[0];
  int l15 = ptr[1];
  int d15 = wordDetect(wAdd,wSize, DCW_TRADITIONAL);
  int delE15 = ptr[2];
  int delL15 = ptr[3];
  int s15 = wordShift(wAdd, wSize, DCW_TRADITIONAL);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14+e15, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13+l14+l15, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12+delE13+delE14+delE15, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12+delL13+delL14+delL15, test.getDel_latcy());
  ASSERT_EQ(4*8*8+40+16*8+12*8+16+40+32+8, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7+d8+d9+d10+d11+d12+d13+d14+d15, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 16
  wAdd = 245;
  wSize = 3;
  start = 12;
  test.writeData(wAdd, wSize, data+start, DCW_TRADITIONAL, 0);
  sky_size_t data_bit7[24]={1,1,1,1,0,1,1,1,
                            1,0,0,0,1,0,0,0,
                            0,1,1,1,0,0,1,1};
  ptr = wordInsert(wAdd,wSize, data_bit7, data_bit+start*8, DCW_TRADITIONAL);
  int e16 = ptr[0];
  int l16 = ptr[1];
  int d16 = wordDetect(wAdd,wSize, DCW_TRADITIONAL);
  int delE16 = ptr[2];
  int delL16 = ptr[3];
  int s16 = wordShift(wAdd, wSize, DCW_TRADITIONAL);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14+e15+e16, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13+l14+l15+l16, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12+delE13+delE14+delE15+delE16, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12+delL13+delL14+delL15+delL16, test.getDel_latcy());
  ASSERT_EQ(4*8*8+40+16*8+12*8+16+40+32+8+24, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7+d8+d9+d10+d11+d12+d13+d14+d15+d16, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = start;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }
}

TEST(checkWriteDCW, case1)
{
  SkyrmionWord test;
  for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
    for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
      test.setEntry(i, 0);
    }
  }
  #ifdef DEBUG
    test.print();
  #endif

  sky_size_t data_bit[128]={0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,0}; // 61 bit 1
  sky_size_t data[16]={120, 182, 52, 136, 115, 92, 150, 0, 247, 33, 59, 152, 248, 213, 165, 98};
  sky_size_t data_bit_0[128]={0};

  // write 1
  int wAdd = 0;
  int wSize = 4;
  int start = 0;
  test.writeData(wAdd, wSize, data+start, DCW, 0);
  int* ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, DCW);
  int e1 = ptr[0];
  int l1 = ptr[1];
  int delE1 = ptr[2];
  int delL1 = ptr[3];
  int d1 = wordDetect(wAdd,wSize, DCW);
  int s1 = wordShift(wAdd, wSize, DCW);
  ASSERT_EQ(e1, test.getIns_engy());
  ASSERT_EQ(l1, test.getIns_latcy());
  ASSERT_EQ(delE1, test.getDel_engy());
  ASSERT_EQ(delL1, test.getDel_latcy());
  ASSERT_EQ(4*8, test.getDet_engy());
  ASSERT_EQ(d1, test.getDet_latcy());
  ASSERT_EQ(s1, test.getSht_engy());
  ASSERT_EQ(s1, test.getSht_latcy());
  delete [] ptr;

  int pos = wAdd;
  int len = wSize;
  int testPos = 0;
  int rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 2
  wAdd = 0;
  wSize = 4;
  start = 4;
  test.writeData(wAdd, wSize, data+start, DCW, 0);
  ptr = wordInsert(wAdd,wSize, data_bit, data_bit+start*8, DCW);
  int e2 = ptr[0];
  int l2 = ptr[1];
  int delE2 = ptr[2];
  int delL2 = ptr[3];
  int d2 = wordDetect(wAdd,wSize, DCW);
  int s2 = wordShift(wAdd, wSize, DCW);
  ASSERT_EQ(e1+e2, test.getIns_engy());
  ASSERT_EQ(l1+l2, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2, test.getDel_engy());
  ASSERT_EQ(delL1+delL2, test.getDel_latcy());
  ASSERT_EQ(4*8*2, test.getDet_engy());
  ASSERT_EQ(d1+d2, test.getDet_latcy());
  ASSERT_EQ(s1+s2, test.getSht_engy());
  ASSERT_EQ(s1+s2, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 4;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+ testPos*8]); // here
  }

  // write 3
  wAdd = 4;
  wSize = 4;
  start = 4;
  test.writeData(wAdd, wSize, data+start, DCW, 0);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, DCW);
  int e3 = ptr[0];
  int l3 = ptr[1];
  int delE3 = ptr[2];
  int delL3 = ptr[3];
  int d3 = wordDetect(wAdd,wSize, DCW);
  int s3 = wordShift(wAdd, wSize, DCW);
  ASSERT_EQ(e1+e2+e3, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3, test.getDel_latcy());
  ASSERT_EQ(4*8*3, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 4;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }

  // write 4
  wAdd = 4;
  wSize = 4;
  start = 8;
  test.writeData(wAdd, wSize, data+start, DCW, 0);
  ptr = wordInsert(wAdd,wSize, data_bit+32, data_bit+start*8, DCW);
  int e4 = ptr[0];
  int l4 = ptr[1];
  int delE4 = ptr[2];
  int delL4 = ptr[3];
  int d4 = wordDetect(wAdd,wSize, DCW);
  int s4 = wordShift(wAdd, wSize, DCW);
  ASSERT_EQ(e1+e2+e3+e4, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4, test.getDel_latcy());
  ASSERT_EQ(4*8*4, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 8;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }

  // write 5
  wAdd = 8;
  wSize = 4;
  start = 8;
  test.writeData(wAdd, wSize, data+start, DCW, 0);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, DCW);
  int e5 = ptr[0];
  int l5 = ptr[1];
  int delE5 = ptr[2];
  int delL5 = ptr[3];
  int d5 = wordDetect(wAdd,wSize, DCW);
  int s5 = wordShift(wAdd, wSize, DCW);
  ASSERT_EQ(e1+e2+e3+e4+e5, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5, test.getDel_latcy());
  ASSERT_EQ(4*8*5, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 8;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }

  // write 6
  wAdd = 8;
  wSize = 4;
  start = 12;
  test.writeData(wAdd, wSize, data+start, DCW, 0);
  ptr = wordInsert(wAdd,wSize, data_bit+64, data_bit+start*8, DCW);
  int e6 = ptr[0];
  int l6 = ptr[1];
  int delE6 = ptr[2];
  int delL6 = ptr[3];
  int d6 = wordDetect(wAdd,wSize, DCW);
  int s6 = wordShift(wAdd, wSize, DCW);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6, test.getDel_latcy());
  ASSERT_EQ(4*8*6, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 12;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }

  // write 7
  wAdd = 12;
  wSize = 4;
  start = 12;
  test.writeData(wAdd, wSize, data+start, DCW, 0);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, DCW);
  int e7 = ptr[0];
  int l7 = ptr[1];
  int delE7 = ptr[2];
  int delL7 = ptr[3];
  int d7 = wordDetect(wAdd, wSize, DCW);
  int s7 = wordShift(wAdd, wSize, DCW);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7, test.getDel_latcy());
  ASSERT_EQ(4*8*7, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 12;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }

  // write 8
  wAdd = 12;
  wSize = 4;
  start = 0;
  test.writeData(wAdd, wSize, data+start, DCW, 0);
  ptr = wordInsert(wAdd, wSize, data_bit+96, data_bit+start*8, DCW);
  int e8 = ptr[0];
  int l8 = ptr[1];
  int delE8 = ptr[2];
  int delL8 = ptr[3];
  int d8 = wordDetect(wAdd,wSize, DCW);
  int s8 = wordShift(wAdd, wSize, DCW);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8, test.getDel_latcy());
  ASSERT_EQ(4*8*8, test.getDet_engy()); //
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7+d8, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 0;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 9 C
  wAdd = 251;
  wSize = 5;
  start = 0;
  test.writeData(wAdd, wSize, data+start, DCW, 0);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, DCW);
  int e9 = ptr[0];
  int l9 = ptr[1];
  int delE9 = ptr[2];
  int delL9 = ptr[3];
  int d9 = wordDetect(wAdd,wSize, DCW);
  int s9 = wordShift(wAdd, wSize, DCW);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9, test.getDel_latcy());
  ASSERT_EQ(4*8*8+40, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7+d8+d9, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 0;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }

  // write 10 D
  wAdd = 239;
  wSize = 16;
  start = 0;
  test.writeData(wAdd, wSize, data+start, DCW, 0);
  sky_size_t data_bit2[128]={0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,1,1,1,1,0,0,0,
                             1,0,1,1,0,1,1,0,
                             0,0,1,1,0,1,0,0,
                             1,0,0,0,1,0,0,0};
  ptr = wordInsert(wAdd,wSize, data_bit2, data_bit+start*8, DCW);
  int e10 = ptr[0];
  int l10 = ptr[1];
  int delE10 = ptr[2];
  int delL10 = ptr[3];
  int d10 = wordDetect(wAdd,wSize, DCW);
  int s10 = wordShift(wAdd, wSize, DCW);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10, test.getDel_latcy());
  ASSERT_EQ(4*8*8+40+16*8, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7+d8+d9+d10, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 0;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }

  // write 11 D
  wAdd = 241;
  wSize = 12;
  start = 0;
  test.writeData(wAdd, wSize, data+start, DCW, 0);
  sky_size_t data_bit3[96]={0,0,1,1,0,1,0,0,
                             1,0,0,0,1,0,0,0,
                             0,1,1,1,0,0,1,1,
                             0,1,0,1,1,1,0,0,
                             1,0,0,1,0,1,1,0,
                             0,0,0,0,0,0,0,0,
                             1,1,1,1,0,1,1,1,
                             0,0,1,0,0,0,0,1,
                             0,0,1,1,1,0,1,1,
                             1,0,0,1,1,0,0,0,
                             1,1,1,1,1,0,0,0,
                             1,1,0,1,0,1,0,1};
  ptr = wordInsert(wAdd,12, data_bit3, data_bit+start*8, DCW);
  int e11 = ptr[0];
  int l11 = ptr[1];
  int delE11 = ptr[2];
  int delL11 = ptr[3];
  int d11 = wordDetect(wAdd,wSize, DCW);
  int s11 = wordShift(wAdd, wSize, DCW);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11, test.getDel_latcy());
  ASSERT_EQ(4*8*8+40+16*8+12*8, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7+d8+d9+d10+d11, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 0;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }

  // write 12
  wAdd = 249;
  wSize = 2;
  start = 0;
  test.writeData(wAdd, wSize, data+start, DCW, 0);
  sky_size_t data_bit4[16]={1,1,1,1,0,1,1,1,
                            0,0,1,0,0,0,0,1,};
  ptr = wordInsert(wAdd,wSize, data_bit4, data_bit+start*8, DCW);
  int e12 = ptr[0];
  int l12 = ptr[1];
  int delE12 = ptr[2];
  int delL12 = ptr[3];
  int d12 = wordDetect(wAdd,wSize, DCW);
  int s12 = wordShift(wAdd, wSize, DCW);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12, test.getDel_latcy());
  ASSERT_EQ(4*8*8+40+16*8+12*8+2*8, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7+d8+d9+d10+d11+d12, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 0;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (249*8 + 249/BYTEBWPORTS) + j + 1), data_bit[j]); // here
  }

  // write 13 B
  wAdd = 244;
  wSize = 5;
  start = 0;
  test.writeData(wAdd, 5, data+start, DCW, 0);
  sky_size_t data_bit5[40]={1,0,0,0,1,0,0,0,
                            0,1,1,1,0,0,1,1,
                            0,1,0,1,1,1,0,0,
                            1,0,0,1,0,1,1,0,
                            0,0,0,0,0,0,0,0};
  ptr = wordInsert(wAdd,wSize, data_bit5, data_bit+start*8, DCW);
  int e13 = ptr[0];
  int l13 = ptr[1];
  int delE13 = ptr[2];
  int delL13 = ptr[3];
  int d13 = wordDetect(wAdd,wSize, DCW);
  int s13 = wordShift(wAdd, wSize, DCW);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12+delE13, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12+delL13, test.getDel_latcy());
  ASSERT_EQ(4*8*8+40+16*8+12*8+2*8+5*8, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7+d8+d9+d10+d11+d12+d13, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 0;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j]); // here
  }

  // write 14 D
  wAdd = 245;
  wSize = 4;
  start = 2;
  test.writeData(245, 4, data+start, DCW, 0);
  sky_size_t data_bit6[32]={1,0,1,1,0,1,1,0,
                            0,0,1,1,0,1,0,0,
                            1,0,0,0,1,0,0,0,
                            0,1,1,1,0,0,1,1};
  ptr = wordInsert(0,4, data_bit6, data_bit+start*8, DCW);
  int e14 = ptr[0];
  int l14 = ptr[1];
  int delE14 = ptr[2];
  int delL14 = ptr[3];
  int d14 = wordDetect(245,4, DCW);
  int s14 = wordShift(245, 4, DCW);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13+l14, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12+delE13+delE14, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12+delL13+delL14, test.getDel_latcy());
  ASSERT_EQ(4*8*8+40+16*8+12*8+2*8+5*8+4*8, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7+d8+d9+d10+d11+d12+d13+d14, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 2;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }

  // write 15
  wAdd = 245;
  wSize = 1;
  start = 8;
  test.writeData(245, 1, data+start, DCW, 0);
  ptr = wordInsert(0,1, data_bit+16, data_bit+start*8, DCW);
  int e15 = ptr[0];
  int l15 = ptr[1];
  int delE15 = ptr[2];
  int delL15 = ptr[3];
  int d15 = wordDetect(245,1, DCW);
  int s15 = wordShift(245, 1, DCW);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14+e15, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13+l14+l15, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12+delE13+delE14+delE15, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12+delL13+delL14+delL15, test.getDel_latcy());
  ASSERT_EQ(4*8*8+40+16*8+12*8+2*8+5*8+4*8+8, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7+d8+d9+d10+d11+d12+d13+d14+d15, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 8;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }

  // write 16
  wAdd = 245;
  wSize = 3;
  start = 12;
  test.writeData(245, 3, data+start, DCW, 0);
  sky_size_t data_bit7[24]={1,1,1,1,0,1,1,1,
                            1,0,0,0,1,0,0,0,
                            0,1,1,1,0,0,1,1};
  ptr = wordInsert(0,3, data_bit7, data_bit+start*8, DCW);
  int e16 = ptr[0];
  int l16 = ptr[1];
  int delE16 = ptr[2];
  int delL16 = ptr[3];
  int d16 = wordDetect(245,3, DCW);
  int s16 = wordShift(245, 3, DCW);
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13+e14+e15+e16, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13+l14+l15+l16, test.getIns_latcy());
  ASSERT_EQ(delE1+delE2+delE3+delE4+delE5+delE6+delE7+delE8+delE9+delE10+delE11+delE12+delE13+delE14+delE15+delE16, test.getDel_engy());
  ASSERT_EQ(delL1+delL2+delL3+delL4+delL5+delL6+delL7+delL8+delL9+delL10+delL11+delL12+delL13+delL14+delL15+delL16, test.getDel_latcy());
  ASSERT_EQ(4*8*8+40+16*8+12*8+2*8+5*8+4*8+8+24, test.getDet_engy());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7+d8+d9+d10+d11+d12+d13+d14+d15+d16, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 12;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }
}

TEST(checkWritePermutationWrite, case1)
{
  SkyrmionWord test;
  for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
    for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
      test.setEntry(i, 0);
    }
  }
  #ifdef DEBUG
    test.print();
  #endif

  sky_size_t data_bit[128]={0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,0}; // 61 bit 1
  sky_size_t data[16]={120, 182, 52, 136, 115, 92, 150, 0, 247, 33, 59, 152, 248, 213, 165, 98};
  sky_size_t data_bit_0[128]={0};

  // write 1
  int wAdd = 0;
  int wSize = 4;
  int start = 0;
  test.writeData(wAdd, wSize, data+start, PERMUTATION_WRITE, 0);
  int *ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, PERMUTATION_WRITE);
  int e1 = ptr[0];
  int d1 = wordDetect(wAdd,wSize, PERMUTATION_WRITE);
  int del1 = ptr[2];
  int s1 = wordShift(wAdd,wSize, PERMUTATION_WRITE) + del1;
  ASSERT_EQ(e1, test.getIns_engy());
  ASSERT_EQ(e1, test.getIns_latcy());
  ASSERT_EQ(del1, test.getDel_engy());
  ASSERT_EQ(del1, test.getDel_latcy());
  ASSERT_EQ(4*8, test.getDet_engy());
  ASSERT_EQ(d1, test.getDet_latcy());
  ASSERT_EQ(s1, test.getSht_engy());
  ASSERT_EQ(s1, test.getSht_latcy());
  delete [] ptr;

  int pos = wAdd;
  int len = wSize;
  int testPos = 0;
  int rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j+testPos*8]); // here
  }

  // write 2
  wAdd = 0;
  wSize = 4;
  start = 4;
  test.writeData(wAdd, wSize, data+start, PERMUTATION_WRITE, 0);
  ptr = wordInsert(wAdd,wSize, data_bit, data_bit+start*8, PERMUTATION_WRITE);
  int e2 = ptr[0];
  int d2 = wordDetect(wAdd,wSize, PERMUTATION_WRITE);
  int del2 = ptr[2];
  int s2 = wordShift(wAdd,wSize, PERMUTATION_WRITE) + del2;
  ASSERT_EQ(e1+e2, test.getIns_engy());
  ASSERT_EQ(e1+e2, test.getIns_latcy());
  ASSERT_EQ(del1+del2, test.getDel_engy()); // 1 is because 14->13, 1 more redundant skyrmion
  ASSERT_EQ(del1+del2, test.getDel_latcy());
  ASSERT_EQ(4*8*2, test.getDet_engy());
  ASSERT_EQ(d1*2, test.getDet_latcy());
  ASSERT_EQ(s1+s2, test.getSht_engy()); // 1 is because 14->13, 1 more redundant skyrmion
  ASSERT_EQ(s1+s2, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 4;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }

  // write 3
  wAdd = 4;
  wSize = 4;
  start = 4;
  test.writeData(wAdd, wSize, data+start, PERMUTATION_WRITE, 0);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, PERMUTATION_WRITE);
  int e3 = ptr[0];
  int d3 = wordDetect(wAdd,wSize, PERMUTATION_WRITE);
  int del3 = ptr[2];
  int s3 = wordShift(wAdd,wSize, PERMUTATION_WRITE) + del3;
  ASSERT_EQ(e1+e2+e3, test.getIns_engy());
  ASSERT_EQ(e1+e2+e3, test.getIns_latcy());
  ASSERT_EQ(del1+del2+del3, test.getDel_engy());
  ASSERT_EQ(del1+del2+del3, test.getDel_latcy());
  ASSERT_EQ(4*8*3, test.getDet_engy());
  ASSERT_EQ(d1*2+d3, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 4;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }

  // write 4
  wAdd = 4;
  wSize = 4;
  start = 8;
  test.writeData(wAdd, wSize, data+start, PERMUTATION_WRITE, 0);
  ptr = wordInsert(wAdd,wSize, data_bit+32, data_bit+start*8, PERMUTATION_WRITE);
  int e4 = ptr[0];
  int d4 = wordDetect(wAdd,wSize, PERMUTATION_WRITE);
  int del4 = ptr[2];
  int s4 = wordShift(wAdd,wSize, PERMUTATION_WRITE) + del4;
  ASSERT_EQ(e1+e2+e3+e4, test.getIns_engy()); // 13->17, insert 4 skyrmions
  ASSERT_EQ(e1+e2+e3+e4, test.getIns_latcy());
  ASSERT_EQ(del1+del2+del3+del4, test.getDel_engy());
  ASSERT_EQ(del1+del2+del3+del4, test.getDel_latcy());
  ASSERT_EQ(4*8*4, test.getDet_engy());
  ASSERT_EQ(d1*2+d3*2, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 8;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }

  // write 5
  wAdd = 8;
  wSize = 4;
  start = 8;
  test.writeData(wAdd, wSize, data+start, PERMUTATION_WRITE, 0);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, PERMUTATION_WRITE);
  int e5 = ptr[0];
  int d5 = wordDetect(wAdd,wSize, PERMUTATION_WRITE);
  int del5 = ptr[2];
  int s5 = wordShift(wAdd,wSize, PERMUTATION_WRITE) + del5;
  ASSERT_EQ(e1+e2+e3+e4+e5, test.getIns_engy());
  ASSERT_EQ(e1+e2+e3+e4+e5, test.getIns_latcy());
  ASSERT_EQ(del1+del2+del3+del4+del5, test.getDel_engy());
  ASSERT_EQ(del1+del2+del3+del4+del5, test.getDel_latcy());
  ASSERT_EQ(4*8*5, test.getDet_engy());
  ASSERT_EQ(d1*2+d3*2+d5, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 8;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }

  // write 6
  wAdd = 8;
  wSize = 4;
  start = 12;
  test.writeData(wAdd, wSize, data+start, PERMUTATION_WRITE, 0);
  ptr = wordInsert(wAdd,wSize, data_bit+64, data_bit+start*8, PERMUTATION_WRITE);
  int e6 = ptr[0];
  int d6 = wordDetect(wAdd,wSize, PERMUTATION_WRITE);
  int del6 = ptr[2];
  int s6 = wordShift(wAdd,wSize, PERMUTATION_WRITE) + del6;
  ASSERT_EQ(e1+e2+e3+e4+e5+e6, test.getIns_engy());// 17->17, no need to insert any skyrmions
  ASSERT_EQ(e1+e2+e3+e4+e5+e6, test.getIns_latcy());
  ASSERT_EQ(del1+del2+del3+del4+del5+del6, test.getDel_engy());
  ASSERT_EQ(del1+del2+del3+del4+del5+del6, test.getDel_latcy());
  ASSERT_EQ(4*8*6, test.getDet_engy());
  ASSERT_EQ(d1*2+d3*2+d5*2, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 12;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }

  // write 7
  wAdd = 12;
  wSize = 4;
  start = 12;
  test.writeData(wAdd, wSize, data+start, PERMUTATION_WRITE, 0);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, PERMUTATION_WRITE);
  int e7 = ptr[0];
  int d7 = wordDetect(wAdd,wSize, PERMUTATION_WRITE);
  int del7 = ptr[2];
  int s7 = wordShift(wAdd,wSize, PERMUTATION_WRITE) + del7;
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7, test.getIns_engy());
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7, test.getIns_latcy());
  ASSERT_EQ(del1+del2+del3+del4+del5+del6+del7, test.getDel_engy());
  ASSERT_EQ(del1+del2+del3+del4+del5+del6+del7, test.getDel_latcy());
  ASSERT_EQ(4*8*7, test.getDet_engy());
  ASSERT_EQ(d1*2+d3*2+d5*2+d7, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 12;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }

  // write 8
  wAdd = 12;
  wSize = 4;
  start = 0;
  test.writeData(wAdd, wSize, data+start, PERMUTATION_WRITE, 0);
  ptr = wordInsert(wAdd,wSize, data_bit+96, data_bit+start*8, PERMUTATION_WRITE);
  int e8 = ptr[0];
  int d8 = wordDetect(wAdd,wSize, PERMUTATION_WRITE);
  int del8 = ptr[2];
  int s8 = wordShift(wAdd,wSize, PERMUTATION_WRITE) + del8;
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8, test.getIns_engy());// 17->14, no need to insert
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8, test.getIns_latcy());
  ASSERT_EQ(del1+del2+del3+del4+del5+del6+del7+del8, test.getDel_engy());
  ASSERT_EQ(del1+del2+del3+del4+del5+del6+del7+del8, test.getDel_latcy());
  ASSERT_EQ(4*8*8, test.getDet_engy());
  ASSERT_EQ(d1*2+d3*2+d5*2+d7*2, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8, test.getSht_engy()); // 17->14, need to remove 3 redundant skyrmions
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 0;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }

  // write 9
  wAdd = 251;
  wSize = 3;
  start = 0;
  test.writeData(wAdd, wSize, data+start, PERMUTATION_WRITE, 0);
  ptr = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, PERMUTATION_WRITE);
  int e9 = ptr[0];
  int d9 = wordDetect(wAdd,wSize, PERMUTATION_WRITE);
  int del9 = ptr[2];
  int s9 = wordShift(wAdd,wSize, PERMUTATION_WRITE) + del9;
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9, test.getIns_engy());
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9, test.getIns_latcy());
  ASSERT_EQ(del1+del2+del3+del4+del5+del6+del7+del8+del9, test.getDel_engy());
  ASSERT_EQ(del1+del2+del3+del4+del5+del6+del7+del8+del9, test.getDel_latcy());
  ASSERT_EQ(4*8*8+24, test.getDet_engy());
  ASSERT_EQ(d1*2+d3*2+d5*2+d7*2+d9, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 0;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }

  // write 10
  wAdd = 245;
  wSize = 8;
  start = 0;
  test.writeData(wAdd, wSize, data+start, PERMUTATION_WRITE, 0);
  sky_size_t data_bit2[64]={0,0,0,0,0,0,0,0,
                            0,0,0,0,0,0,0,0,
                            0,0,0,0,0,0,0,0,
                            0,0,0,0,0,0,0,0,
                            0,0,0,0,0,0,0,0,
                            0,0,0,0,0,0,0,0,
                            0,1,1,1,1,0,0,0,
                            1,0,1,1,0,1,1,0};
  ptr = wordInsert(wAdd,wSize, data_bit2, data_bit+start*8, PERMUTATION_WRITE);
  int e10 = ptr[0];
  int d10 = wordDetect(wAdd,wSize, PERMUTATION_WRITE);
  int del10 = ptr[2];
  int s10 = wordShift(wAdd,wSize, PERMUTATION_WRITE) + del10;
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10, test.getIns_engy());
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10, test.getIns_latcy());
  ASSERT_EQ(del1+del2+del3+del4+del5+del6+del7+del8+del9+del10, test.getDel_engy());
  ASSERT_EQ(del1+del2+del3+del4+del5+del6+del7+del8+del9+del10, test.getDel_latcy());
  ASSERT_EQ(4*8*8+24+64, test.getDet_engy());
  ASSERT_EQ(d1*2+d3*2+d5*2+d7*2+d9+d10, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10, test.getSht_engy()); // the most right 5->0, need to remove 5 redundant skyrmions
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 0;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }

  // write 11
  wAdd = 246;
  wSize = 6;
  start = 0;
  test.writeData(wAdd, wSize, data+start, PERMUTATION_WRITE, 0);
  sky_size_t data_bit3[48]={1,0,1,1,0,1,1,0,
                            0,0,1,1,0,1,0,0,
                            1,0,0,0,1,0,0,0,
                            0,1,1,1,0,0,1,1,
                            0,1,0,1,1,1,0,0,
                            1,0,0,1,0,1,1,0};
  ptr = wordInsert(wAdd,wSize, data_bit3, data_bit+start*8, PERMUTATION_WRITE);
  int e11 = ptr[0];
  int d11 = wordDetect(wAdd,wSize, PERMUTATION_WRITE);
  int del11 = ptr[2];
  int s11 = wordShift(wAdd,wSize, PERMUTATION_WRITE) + del11;
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11, test.getIns_engy());
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11, test.getIns_latcy());
  ASSERT_EQ(del1+del2+del3+del4+del5+del6+del7+del8+del9+del10+del11, test.getDel_engy());
  ASSERT_EQ(del1+del2+del3+del4+del5+del6+del7+del8+del9+del10+del11, test.getDel_latcy());
  ASSERT_EQ(4*8*8+24+64+48, test.getDet_engy());
  ASSERT_EQ(d1*2+d3*2+d5*2+d7*2+d9+d10+d11, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 0;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }

  // write 12
  wAdd = 249;
  wSize = 1;
  start = 0;
  sky_size_t new_data_bit2[16]={0,0,0,0,0,0,0,0,
                                0,1,1,1,1,0,0,0};
  test.writeData(wAdd, wSize, data+start, PERMUTATION_WRITE, 0);
  ptr = wordInsert(wAdd,wSize, data_bit+24, data_bit+start*8, PERMUTATION_WRITE);
  int e12 = ptr[0];
  int d12 = wordDetect(wAdd,wSize, PERMUTATION_WRITE);
  int del12 = ptr[2];
  int s12 = wordShift(wAdd,wSize, PERMUTATION_WRITE) + del12;
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12, test.getIns_engy());
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12, test.getIns_latcy());
  ASSERT_EQ(del1+del2+del3+del4+del5+del6+del7+del8+del9+del10+del11+del12, test.getDel_engy());
  ASSERT_EQ(del1+del2+del3+del4+del5+del6+del7+del8+del9+del10+del11+del12, test.getDel_latcy());
  ASSERT_EQ(4*8*8+24+64+48+8, test.getDet_engy());
  ASSERT_EQ(d1*2+d3*2+d5*2+d7*2+d9+d10+d11+d12, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 0;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }

  // write 13
  wAdd = 246;
  wSize = 4;
  start = 4;
  test.writeData(wAdd, wSize, data+start, PERMUTATION_WRITE, 0);
  sky_size_t data_bit4[32]={0,1,1,1,1,0,0,0,
                            1,0,1,1,0,1,1,0,
                            0,0,1,1,0,1,0,0,
                            0,1,1,1,1,0,0,0};
  ptr = wordInsert(wAdd,wSize, data_bit4, data_bit+start*8, PERMUTATION_WRITE);
  int e13 = ptr[0];
  int d13 = wordDetect(wAdd,wSize, PERMUTATION_WRITE);
  int del13 = ptr[2];
  int s13 = wordShift(wAdd,wSize, PERMUTATION_WRITE) + del13;
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13, test.getIns_engy());
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13, test.getIns_latcy());
  ASSERT_EQ(del1+del2+del3+del4+del5+del6+del7+del8+del9+del10+del11+del12+del13, test.getDel_engy());
  ASSERT_EQ(del1+del2+del3+del4+del5+del6+del7+del8+del9+del10+del11+del12+del13, test.getDel_latcy());
  ASSERT_EQ(4*8*8+24+64+48+8+32, test.getDet_engy());
  ASSERT_EQ(d1*2+d3*2+d5*2+d7*2+d9+d10+d11+d12+d13, test.getDet_latcy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13, test.getSht_engy());
  ASSERT_EQ(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13, test.getSht_latcy());
  delete [] ptr;

  pos = wAdd;
  len = wSize;
  testPos = 4;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
    testPos += BYTEBWPORTS - rem;
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + testPos*8]); // here
  }
}

TEST(checkWriteBit, case1)
{
  SkyrmionBit test;
  for (int k = 0; k < ROW; k++){
    for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
      for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
        test.setEntries(k, i, 0);
      }
    }
  }
  #ifdef DEBUG
    test.print();
  #endif

  sky_size_t data_bit[128]={0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,0}; // 61 bit 1
  sky_size_t data[16]={120, 182, 52, 136, 115, 92, 150, 0, 247, 33, 59, 152, 248, 213, 165, 98};
  sky_size_t data_bit_0[128]={0};

  test.setWriteType(BIT);
  // write 1
  int wBlk = 2;
  int wAdd = 0;
  int wSize = 4;
  int start = 12;
  test.write(wBlk, wAdd, wSize, data+start, BIT, 0);
  int *w = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, NAIVE);
  int e1 = w[0];
  int l1 = 0;
  if (e1 > 0) l1 = 1;
  int *ptr = bitShift(wBlk, 4);
  int se1 = ptr[0];
  int sl1 = ptr[1];
  ASSERT_EQ(4, test.getN_checkFull());
  ASSERT_EQ(1, test.getBlockUsedNumber(2));
  ASSERT_EQ(e1, test.getIns_engy());
  ASSERT_EQ(l1, test.getIns_latcy());
  ASSERT_EQ(4*8, test.getDel_engy());
  ASSERT_EQ(1, test.getDel_latcy());
  ASSERT_EQ(0, test.getDet_engy());
  ASSERT_EQ(0, test.getDet_latcy());
  ASSERT_EQ(se1, test.getSht_engy());
  ASSERT_EQ(sl1, test.getSht_latcy());
  delete [] ptr;
  delete [] w;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 2
  wBlk = 30;
  wAdd = 2;
  wSize = 5;
  start = 0;
  test.write(wBlk, wAdd, wSize, data+start, BIT, 0);
  w = wordInsert(wAdd,wSize, data_bit+96, data_bit+start*8, NAIVE);
  int e2 = w[0];
  int l2 = 0;
  if (e2 > 0) l2 = 1;
  ptr = bitShift(wBlk, wSize);
  int se2 = ptr[0];
  int sl2 = ptr[1];
  ASSERT_EQ(9, test.getN_checkFull());
  ASSERT_EQ(1, test.getBlockUsedNumber(30));
  ASSERT_EQ(e1+e2, test.getIns_engy());
  ASSERT_EQ(l1+l2, test.getIns_latcy());
  ASSERT_EQ(4*8+5*8, test.getDel_engy());
  ASSERT_EQ(2, test.getDel_latcy());
  ASSERT_EQ(0, test.getDet_engy());
  ASSERT_EQ(0, test.getDet_latcy());
  ASSERT_EQ(se1+se2, test.getSht_engy());
  ASSERT_EQ(sl1+sl2, test.getSht_latcy());
  delete [] ptr;
  delete [] w;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 3
  wBlk = 32;
  wAdd = 1;
  wSize = 1;
  start = 0;
  test.write(wBlk, wAdd, wSize, data+start, BIT, 0);
  w = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, NAIVE);
  int e3 = w[0];
  int l3 = 0;
  if (e3 > 0) l3 = 1;
  ptr = bitShift(wBlk, wSize);
  int se3 = ptr[0];
  int sl3 = ptr[1];
  ASSERT_EQ(10, test.getN_checkFull());
  ASSERT_EQ(1, test.getBlockUsedNumber(32));
  ASSERT_EQ(e1+e2+e3, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3, test.getIns_latcy());
  ASSERT_EQ(4*8+5*8+8, test.getDel_engy());
  ASSERT_EQ(3, test.getDel_latcy());
  ASSERT_EQ(0, test.getDet_engy());
  ASSERT_EQ(0, test.getDet_latcy());
  ASSERT_EQ(se1+se2+se3, test.getSht_engy());
  ASSERT_EQ(sl1+sl2+sl3, test.getSht_latcy());
  delete [] ptr;
  delete [] w;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 4
  wBlk = 34;
  wAdd = 1;
  wSize = 2;
  start = 1;
  test.write(wBlk, wAdd, wSize, data+start, BIT, 0);
  w = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, NAIVE);
  int e4 = w[0];
  int l4 = 0;
  if (e4 > 0) l4 = 1;
  ptr = bitShift(wBlk, wSize);
  int se4 = ptr[0];
  int sl4 = ptr[1];
  ASSERT_EQ(12, test.getN_checkFull());
  ASSERT_EQ(1, test.getBlockUsedNumber(34));
  ASSERT_EQ(e1+e2+e3+e4, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4, test.getIns_latcy());
  ASSERT_EQ(4*8+5*8+8+16, test.getDel_engy());
  ASSERT_EQ(4, test.getDel_latcy());
  ASSERT_EQ(0, test.getDet_engy());
  ASSERT_EQ(0, test.getDet_latcy());
  ASSERT_EQ(se1+se2+se3+se4, test.getSht_engy());
  ASSERT_EQ(sl1+sl2+sl3+sl4, test.getSht_latcy());
  delete [] ptr;
  delete [] w;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 5
  wBlk = 90;
  wAdd = 2;
  wSize = 3;
  start = 0;
  test.write(wBlk, wAdd, wSize, data+start, BIT, 0);
  w = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, NAIVE);
  int e5 = w[0];
  int l5 = 0;
  if (e5 > 0) l5 = 1;
  ptr = bitShift(wBlk, wSize);
  int se5 = ptr[0];
  int sl5 = ptr[1];
  ASSERT_EQ(15, test.getN_checkFull());
  ASSERT_EQ(1, test.getBlockUsedNumber(90));
  ASSERT_EQ(e1+e2+e3+e4+e5, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5, test.getIns_latcy());
  ASSERT_EQ(4*8+5*8+8+16+24, test.getDel_engy());
  ASSERT_EQ(5, test.getDel_latcy());
  ASSERT_EQ(0, test.getDet_engy());
  ASSERT_EQ(0, test.getDet_latcy());
  ASSERT_EQ(se1+se2+se3+se4+se5, test.getSht_engy());
  ASSERT_EQ(sl1+sl2+sl3+sl4+sl5, test.getSht_latcy());
  delete [] ptr;
  delete [] w;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 6
  wBlk = 2;
  wAdd = 0;
  wSize = 5;
  start = 0;
  test.write(wBlk, wAdd, wSize, data+start, BIT, 0);
  sky_size_t data_bit1[40] = {1,1,1,1,1,0,0,0,
                              1,1,0,1,0,1,0,1,
                              1,0,1,0,0,1,0,1,
                              0,1,1,0,0,0,1,0,
                              0,0,0,0,0,0,0,0};
  w = wordInsert(wAdd,wSize, data_bit1, data_bit+start*8, NAIVE);
  int e6 = w[0];
  int l6 = 0;
  if (e6 > 0) l6 = 1;
  ptr = bitShift(wBlk, wSize);
  int se6 = ptr[0];
  int sl6 = ptr[1];
  ASSERT_EQ(16, test.getN_checkFull());
  ASSERT_EQ(2, test.getBlockUsedNumber(2));
  ASSERT_EQ(e1+e2+e3+e4+e5+e6, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6, test.getIns_latcy());
  ASSERT_EQ(4*8+5*8+8+16+24+40, test.getDel_engy());
  ASSERT_EQ(6, test.getDel_latcy());
  ASSERT_EQ(0, test.getDet_engy());
  ASSERT_EQ(0, test.getDet_latcy());
  ASSERT_EQ(se1+se2+se3+se4+se5+se6, test.getSht_engy());
  ASSERT_EQ(sl1+sl2+sl3+sl4+sl5+sl6, test.getSht_latcy());
  delete [] ptr;
  delete [] w;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 7
  wBlk = 4;
  wAdd = 2;
  wSize = 1;
  start = 7;
  test.write(wBlk, wAdd, wSize, data+start, BIT, 0);
  w = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, NAIVE);
  int e7 = w[0];
  int l7 = 0;
  if (e7 > 0) l7 = 1;
  ptr = bitShift(wBlk, wSize);
  int se7 = ptr[0];
  int sl7 = ptr[1];
  ASSERT_EQ(17, test.getN_checkFull());
  ASSERT_EQ(1, test.getBlockUsedNumber(4));
  ASSERT_EQ(e1+e2+e3+e4+e5+e6+e7, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6+l7, test.getIns_latcy());
  ASSERT_EQ(4*8+5*8+8+16+24+40+8, test.getDel_engy());
  ASSERT_EQ(7, test.getDel_latcy());
  ASSERT_EQ(0, test.getDet_engy());
  ASSERT_EQ(0, test.getDet_latcy());
  ASSERT_EQ(se1+se2+se3+se4+se5+se6+se7, test.getSht_engy());
  ASSERT_EQ(sl1+sl2+sl3+sl4+sl5+sl6+sl7, test.getSht_latcy());
  delete [] ptr;
  delete [] w;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }
}

TEST(checkWriteBitDCW, case1)
{
  SkyrmionBit test;
  for (int k = 0; k < ROW; k++){
    for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
      for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
        test.setEntries(k, i, 0);
      }
    }
  }
  #ifdef DEBUG
    test.print();
  #endif

  sky_size_t data_bit[128]={0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,0}; // 61 bit 1
  sky_size_t data[16]={120, 182, 52, 136, 115, 92, 150, 0, 247, 33, 59, 152, 248, 213, 165, 98};
  sky_size_t data_bit_0[128]={0};

  // write 1
  int wBlk = 2;
  int wAdd = 0;
  int wSize = 4;
  int start = 12;
  test.write(wBlk, wAdd, wSize, data+start, BIT_DCW, 0);
  int *w = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, DCW);
  int e1 = w[0];
  int l1 = 0;
  if (e1 > 0) l1 = 1;
  int *ptr = bitShift(wBlk, wSize);
  int se1 = ptr[0];
  int sl1 = ptr[1];
  ASSERT_EQ(e1, test.getIns_engy());
  ASSERT_EQ(l1, test.getIns_latcy());
  ASSERT_EQ(0, test.getDel_engy());
  ASSERT_EQ(0, test.getDel_latcy());
  ASSERT_EQ(4*8, test.getDet_engy());
  ASSERT_EQ(1, test.getDet_latcy());
  ASSERT_EQ(se1, test.getSht_engy());
  ASSERT_EQ(sl1, test.getSht_latcy());
  delete [] ptr;
  delete [] w;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 2
  wBlk = 30;
  wAdd = 2;
  wSize = 5;
  start = 0;
  test.write(wBlk, wAdd, wSize, data+start, BIT_DCW, 0);
  w = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, DCW);
  int e2 = w[0];
  int l2 = 0;
  if (e2 > 0) l2 = 1;
  ptr = bitShift(wBlk, wSize);
  int se2 = ptr[0];
  int sl2 = ptr[1];
  ASSERT_EQ(e1+e2, test.getIns_engy());
  ASSERT_EQ(l1+l2, test.getIns_latcy());
  ASSERT_EQ(0, test.getDel_engy());
  ASSERT_EQ(0, test.getDel_latcy());
  ASSERT_EQ(4*8+5*8, test.getDet_engy());
  ASSERT_EQ(2, test.getDet_latcy());
  ASSERT_EQ(se1+se2, test.getSht_engy());
  ASSERT_EQ(sl1+sl2, test.getSht_latcy());
  delete [] ptr;
  delete [] w;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 3
  wBlk = 32;
  wAdd = 1;
  wSize = 1;
  start = 0;
  test.write(wBlk, wAdd, wSize, data+start, BIT_DCW, 0);
  w = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, DCW);
  int e3 = w[0];
  int l3 = 0;
  if (e3 > 0) l3 = 1;
  ptr = bitShift(wBlk, wSize);
  int se3 = ptr[0];
  int sl3 = ptr[1];
  ASSERT_EQ(e1+e2+e3, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3, test.getIns_latcy());
  ASSERT_EQ(0, test.getDel_engy());
  ASSERT_EQ(0, test.getDel_latcy());
  ASSERT_EQ(4*8+5*8+8, test.getDet_engy());
  ASSERT_EQ(3, test.getDet_latcy());
  ASSERT_EQ(se1+se2+se3, test.getSht_engy());
  ASSERT_EQ(sl1+sl2+sl3, test.getSht_latcy());
  delete [] ptr;
  delete [] w;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 4
  wBlk = 34;
  wAdd = 1;
  wSize = 2;
  start = 1;
  test.write(wBlk, wAdd, wSize, data+start, BIT_DCW, 0);
  w = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, DCW);
  int e4 = w[0];
  int l4 = 0;
  if (e4 > 0) l4 = 1;
  ptr = bitShift(wBlk, wSize);
  int se4 = ptr[0];
  int sl4 = ptr[1];
  ASSERT_EQ(e1+e2+e3+e4, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4, test.getIns_latcy());
  ASSERT_EQ(0, test.getDel_engy());
  ASSERT_EQ(0, test.getDel_latcy());
  ASSERT_EQ(4*8+5*8+8+2*8, test.getDet_engy());
  ASSERT_EQ(4, test.getDet_latcy());
  ASSERT_EQ(se1+se2+se3+se4, test.getSht_engy());
  ASSERT_EQ(sl1+sl2+sl3+sl4, test.getSht_latcy());
  delete [] ptr;
  delete [] w;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 5
  wBlk = 90;
  wAdd = 2;
  wSize = 3;
  start = 0;
  test.write(wBlk, wAdd, wSize, data+start, BIT_DCW, 0);
  w = wordInsert(wAdd,wSize, data_bit_0, data_bit+start*8, DCW);
  int e5 = w[0];
  int l5 = 0;
  if (e5 > 0) l5 = 1;
  ptr = bitShift(wBlk, wSize);
  int se5 = ptr[0];
  int sl5 = ptr[1];
  ASSERT_EQ(e1+e2+e3+e4+e5, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5, test.getIns_latcy());
  ASSERT_EQ(0, test.getDel_engy());
  ASSERT_EQ(0, test.getDel_latcy());
  ASSERT_EQ(4*8+5*8+8+2*8+3*8, test.getDet_engy());
  ASSERT_EQ(5, test.getDet_latcy());
  ASSERT_EQ(se1+se2+se3+se4+se5, test.getSht_engy());
  ASSERT_EQ(sl1+sl2+sl3+sl4+sl5, test.getSht_latcy());
  delete [] ptr;
  delete [] w;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 6
  wBlk = 2;
  wAdd = 0;
  wSize = 5;
  start = 0;
  test.write(wBlk, wAdd, wSize, data+start, BIT_DCW, 0);
  sky_size_t data_bit1[40] = {1,1,1,1,1,0,0,0,
                              1,1,0,1,0,1,0,1,
                              1,0,1,0,0,1,0,1,
                              0,1,1,0,0,0,1,0,
                              0,0,0,0,0,0,0,0};
  w = wordInsert(wAdd,wSize, data_bit1, data_bit+start*8, DCW);
  int e6 = w[0];
  int delE6 = w[2];
  int l6 = 0;
  if (e6 > 0) l6 = 1;
  int delL6 = 0;
  if (delE6 > 0) delL6 = 1;
  ptr = bitShift(wBlk, wSize);
  int se6 = ptr[0];
  int sl6 = ptr[1];
  ASSERT_EQ(e1+e2+e3+e4+e5+e6, test.getIns_engy());
  ASSERT_EQ(l1+l2+l3+l4+l5+l6, test.getIns_latcy());
  ASSERT_EQ(delE6, test.getDel_engy());
  ASSERT_EQ(delL6, test.getDel_latcy());
  ASSERT_EQ(4*8+5*8+8+2*8+3*8+5*8, test.getDet_engy());
  ASSERT_EQ(6, test.getDet_latcy());
  ASSERT_EQ(se1+se2+se3+se4+se5+se6, test.getSht_engy());
  ASSERT_EQ(sl1+sl2+sl3+sl4+sl5+sl6, test.getSht_latcy());
  delete [] ptr;
  delete [] w;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }
}

TEST(checkWriteBitPW, case1)
{
  SkyrmionBit test;
  for (int k = 0; k < ROW; k++){
    for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
      for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
        test.setEntries(k, i, 0);
      }
    }
  }
  #ifdef DEBUG
    test.print();
  #endif

  sky_size_t data_bit[128]={0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,0}; // 61 bit 1
  sky_size_t data[16]={120, 182, 52, 136, 115, 92, 150, 0, 247, 33, 59, 152, 248, 213, 165, 98};
  sky_size_t data_bit_0[128]={0};

  // write 1
  int wBlk = 2;
  int wAdd = 0;
  int wSize = 4;
  int start = 12;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PW, 0);
  int *ptr = bitShift(wBlk, wSize);
  int se1 = ptr[0];
  int sl1 = ptr[1];
  ASSERT_EQ(17, test.getIns_engy());
  ASSERT_EQ(1, test.getIns_latcy());
  ASSERT_EQ(0, test.getDel_engy());
  ASSERT_EQ(0, test.getDel_latcy());
  ASSERT_EQ(4*8, test.getDet_engy());
  ASSERT_EQ(1, test.getDet_latcy());
  ASSERT_EQ(se1, test.getSht_engy());
  ASSERT_EQ(sl1, test.getSht_latcy());
  ASSERT_EQ(0, test.getShtVrtcl_engy());
  ASSERT_EQ(0, test.getShtVrtcl_latcy());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 2
  wBlk = 2;
  wAdd = 1;
  wSize = 5;
  start = 0;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PW, 0);
  ptr = bitShift(wBlk, wSize);
  int se2 = ptr[0];
  int sl2 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5), test.getIns_engy());
  ASSERT_EQ(2, test.getIns_latcy());
  ASSERT_EQ(0, test.getDel_engy()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy());
  ASSERT_EQ(4*8+5*8, test.getDet_engy());
  ASSERT_EQ(2, test.getDet_latcy());
  ASSERT_EQ(se1+se2, test.getSht_engy());
  ASSERT_EQ(sl1+sl2, test.getSht_latcy());
  ASSERT_EQ(((8+8+7)+(5+6+6)), test.getShtVrtcl_engy());
  ASSERT_EQ((8+6), test.getShtVrtcl_latcy());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 3
  wBlk = 2;
  wAdd = 2;
  wSize = 6;
  start = 8;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PW, 0);
  ptr = bitShift(wBlk, wSize);
  int se3 = ptr[0];
  int sl3 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5), test.getIns_engy());
  ASSERT_EQ(3, test.getIns_latcy());
  ASSERT_EQ(0, test.getDel_engy()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy());
  ASSERT_EQ(4*8+5*8+6*8, test.getDet_engy());
  ASSERT_EQ(3, test.getDet_latcy());
  ASSERT_EQ(se1+se2+se3, test.getSht_engy());
  ASSERT_EQ(sl1+sl2+sl3, test.getSht_latcy());
  ASSERT_EQ(((8+8+7)+(5+6+6))+((7+6+5+8)+(6+8+4+5)), test.getShtVrtcl_engy());
  ASSERT_EQ((8+6)+(8+8), test.getShtVrtcl_latcy());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 4
  wBlk = 32;
  wAdd = 1;
  wSize = 1;
  start = 0;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PW, 0);
  ptr = bitShift(wBlk, wSize);
  int se4 = ptr[0];
  int sl4 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5)+4, test.getIns_engy());
  ASSERT_EQ(4, test.getIns_latcy());
  ASSERT_EQ(0, test.getDel_engy()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy());
  ASSERT_EQ(4*8+5*8+6*8+8, test.getDet_engy());
  ASSERT_EQ(4, test.getDet_latcy());
  ASSERT_EQ(se1+se2+se3+se4, test.getSht_engy());
  ASSERT_EQ(sl1+sl2+sl3+sl4, test.getSht_latcy());
  ASSERT_EQ(((8+8+7)+(5+6+6))+((7+6+5+8)+(6+8+4+5)), test.getShtVrtcl_engy());
  ASSERT_EQ((8+6)+(8+8), test.getShtVrtcl_latcy());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 5
  wBlk = 34;
  wAdd = 1;
  wSize = 2;
  start = 1;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PW, 0);
  ptr = bitShift(wBlk, wSize);
  int se5 = ptr[0];
  int sl5 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5)+4+8, test.getIns_engy());
  ASSERT_EQ(5, test.getIns_latcy());
  ASSERT_EQ(0, test.getDel_engy()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy());
  ASSERT_EQ(4*8+5*8+6*8+8+2*8, test.getDet_engy());
  ASSERT_EQ(5, test.getDet_latcy());
  ASSERT_EQ(se1+se2+se3+se4+se5, test.getSht_engy());
  ASSERT_EQ(sl1+sl2+sl3+sl4+sl5, test.getSht_latcy());
  ASSERT_EQ(((8+8+7)+(5+6+6))+((7+6+5+8)+(6+8+4+5)), test.getShtVrtcl_engy());
  ASSERT_EQ((8+6)+(8+8), test.getShtVrtcl_latcy());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 6
  wBlk = 90;
  wAdd = 2;
  wSize = 3;
  start = 0;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PW, 0);
  ptr = bitShift(wBlk, wSize);
  int se6 = ptr[0];
  int sl6 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5)+4+8+12, test.getIns_engy());
  ASSERT_EQ(6, test.getIns_latcy());
  ASSERT_EQ(0, test.getDel_engy()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy());
  ASSERT_EQ(4*8+5*8+6*8+8+2*8+3*8, test.getDet_engy());
  ASSERT_EQ(6, test.getDet_latcy());
  ASSERT_EQ(se1+se2+se3+se4+se5+se6, test.getSht_engy());
  ASSERT_EQ(sl1+sl2+sl3+sl4+sl5+sl6, test.getSht_latcy());
  ASSERT_EQ(((8+8+7)+(5+6+6))+((7+6+5+8)+(6+8+4+5)), test.getShtVrtcl_engy());
  ASSERT_EQ((8+6)+(8+8), test.getShtVrtcl_latcy());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 7
  wBlk = 90;
  wAdd = 2;
  wSize = 3;
  start = 1;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PW, 0);
  ptr = bitShift(wBlk, wSize);
  int se7 = ptr[0];
  int sl7 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5)+4+8+12+1, test.getIns_engy());
  ASSERT_EQ(7, test.getIns_latcy());
  ASSERT_EQ(0, test.getDel_engy()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy());
  ASSERT_EQ(4*8+5*8+6*8+8+2*8+3*8+3*8, test.getDet_engy());
  ASSERT_EQ(7, test.getDet_latcy());
  ASSERT_EQ(se1+se2+se3+se4+se5+se6+se7, test.getSht_engy());
  ASSERT_EQ(sl1+sl2+sl3+sl4+sl5+sl6+sl7, test.getSht_latcy());
  ASSERT_EQ(((8+8+7)+(5+6+6))+((7+6+5+8)+(6+8+4+5)+((5+7+6)+(6+6+5))), test.getShtVrtcl_engy());
  ASSERT_EQ((8+6)+(8+8)+(7+6), test.getShtVrtcl_latcy());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 8
  wBlk = 2;
  wAdd = 0;
  wSize = 9;
  start = 4;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PW, 0);
  ptr = bitShift(wBlk, wSize);
  int se8 = ptr[0];
  int sl8 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5)+4+8+12+1+7, test.getIns_engy());
  ASSERT_EQ(8, test.getIns_latcy());
  ASSERT_EQ(0, test.getDel_engy()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy());
  ASSERT_EQ(4*8+5*8+6*8+8+2*8+3*8+3*8+9*8, test.getDet_engy());
  ASSERT_EQ(8, test.getDet_latcy());
  ASSERT_EQ(se1+se2+se3+se4+se5+se6+se7+se8, test.getSht_engy());
  ASSERT_EQ(sl1+sl2+sl3+sl4+sl5+sl6+sl7+sl8, test.getSht_latcy());
  ASSERT_EQ(((8+8+7)+(5+6+6))+((7+6+5+8)+(6+8+4+5)+((5+7+6)+(6+6+5)))+((5+5+8+8+8+5+5+8)+(8+6+7+0+6+8+8+5)), test.getShtVrtcl_engy());
  ASSERT_EQ((8+6)+(8+8)+(7+6)+(8+8), test.getShtVrtcl_latcy());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }
}

TEST(checkShiftVertical, case1)
{
  SkyrmionBit test;
  for (int k = 0; k < ROW; k++){
    for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
      for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
        test.setEntries(k, i, k+j);
      }
    }
  }
  #ifdef DEBUG
    test.print();
  #endif
  test.shift(66, 0, 0, 2, 0);
  test.shift(66, 0, 0, 2, 0);
  test.shift(66, 0, 0, 2, 0);
  #ifdef DEBUG
    test.print();
  #endif
  test.shiftVertcl(0, 0, 1, 0);
  #ifdef DEBUG
    test.print();
  #endif
  test.shiftVertcl(0, 0, 0, 0);
  #ifdef DEBUG
    test.print();
  #endif
  test.shift(0, 66, 0, 2, 0);
  test.shift(0, 66, 0, 2, 0);
  test.shift(0, 66, 0, 2, 0);
  #ifdef DEBUG
    test.print();
  #endif

  for (int j = 0; j < 8; j++){
    if (j == 0)
      ASSERT_EQ(0, test.getEntries(0, OVER_HEAD + 1 + 2));
    else
      ASSERT_EQ(j, test.getEntries(0+j, OVER_HEAD + 1 + 2));
  }
}

TEST(checkCountNumShift, case1)
{
  int arr[3] = {5, 4, 5};
  sky_size_t data[24] = {0,1,1,0,0,1,0,1,
                         1,1,1,1,1,0,0,1,
                         1,1,0,1,0,0,0,0};
  ASSERT_EQ(8, SkyrmionBit::countNumShift(data, arr[0]));//32:9
  ASSERT_EQ(1, arr[0]);//32:0
  ASSERT_EQ(4, SkyrmionBit::countNumShift(data+8, arr[1]));//32:4
  ASSERT_EQ(0, arr[1]);//32:0
  ASSERT_EQ(4, SkyrmionBit::countNumShift(data+16, arr[2]));
  ASSERT_EQ(2, arr[2]);
}

TEST(checkReadDataNonModified, case1)
{
  SkyrmionWord test;
  #ifdef DEBUG
    test.print();
  #endif
  int save = 1;
  sky_size_t data_bit[128]={0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,0}; // 61 bit 1
  sky_size_t data[16]={120, 182, 52, 136, 115, 92, 150, 0, 247, 33, 59, 152, 248, 213, 165, 98};

  for(int i = 0; i < 16; i++){
    test.writeData(i * 16, 16, data, NAIVE_TRADITIONAL, save);
  }
  // read 1
  int wAdd = 1;
  int wSize = 2;
  sky_size_t *result = test.readData(wAdd, wSize, 0, save);
  int s1 = wordShift(wAdd, wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(61*16+0, test.getIns_engy_DMW());
  ASSERT_EQ(61*16+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_engy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_latcy_DMW());
  ASSERT_EQ(0+2*8, test.getDet_engy_DMW());
  ASSERT_EQ(0+2*8, test.getDet_latcy_DMW());
  ASSERT_EQ(64*16*4+s1, test.getSht_engy_DMW());
  ASSERT_EQ(64*16*4+s1, test.getSht_latcy_DMW());
  #ifdef DEBUG
    test.print();
    printf("data:\n");
    for (int i = 0; i < 2*8; i++){
      printf("data[%d] %d\n", i, data_bit[i]);
    }
    printf("\n");
  #endif
  int pos = wAdd;
  int len = wSize;
  int rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + (pos%16)*8]); // here
  }
  for (int j = 0; j < 2; j++){
    ASSERT_EQ(result[j], data[j + 1 % 16]);
  }
  #ifdef DEBUG
    printf("after read1\n");
    test.print();

  #endif

  // read 2
  wAdd = 0;
  wSize = 64;
  result = test.readData(wAdd, wSize, 0, save);
  int s2 = wordShift(wAdd, wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(61*16+0, test.getIns_engy_DMW());
  ASSERT_EQ(61*16+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_engy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_latcy_DMW());
  ASSERT_EQ(16+64*8, test.getDet_engy_DMW());
  ASSERT_EQ(16+64*8, test.getDet_latcy_DMW());
  ASSERT_EQ(64*16*4+s1+s2, test.getSht_engy_DMW());
  ASSERT_EQ(64*16*4+s1+s2, test.getSht_latcy_DMW());
  #ifdef DEBUG
    test.print();
    printf("data2:\n");
    for (int i = 0; i < 5*8; i++){
      printf("data[%d] %d\n", i, data_bit[i]);
    }
    printf("\n");
  #endif
  pos = wAdd;
  len = wSize;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + (pos%16)*8]); // here
  }
  for (int j = 0; j < 64; j++){
    ASSERT_EQ(result[j], data[j % 16]);
  }

  #ifdef DEBUG
    printf("after read2\n");
    test.print();
  #endif

  // read 3
  wAdd = 69;
  wSize = 9;
  result = test.readData(wAdd, wSize, 0, save);
  int s3 = wordShift(wAdd, wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(61*16+0, test.getIns_engy_DMW());
  ASSERT_EQ(61*16+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_engy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_latcy_DMW());
  ASSERT_EQ(16+64*8+9*8, test.getDet_engy_DMW());
  ASSERT_EQ(16+64*8+9*8, test.getDet_latcy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3, test.getSht_engy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3, test.getSht_latcy_DMW());

  #ifdef DEBUG
    test.print();
    printf("data3:\n");
    for (int i = 0; i < 5*8; i++){
      printf("data[%d] %d\n", i, data_bit[i]);
    }
  #endif

  pos = wAdd;
  len = wSize;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + (pos%16)*8]); // here
  }
  for (int j = 0; j < 9; j++){
    ASSERT_EQ(result[j], data[j + 69 % 16]);
  }

  #ifdef DEBUG
    printf("after read3\n");
    test.print();
  #endif

  // read 4
  wAdd = 72;
  wSize = 1;
  result = test.readData(wAdd, wSize, 0, save);
  int s4 = wordShift(wAdd, wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(61*16+0, test.getIns_engy_DMW());
  ASSERT_EQ(61*16+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_engy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_latcy_DMW());
  ASSERT_EQ(16+64*8+9*8+8, test.getDet_engy_DMW());
  ASSERT_EQ(16+64*8+9*8+8, test.getDet_latcy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4, test.getSht_engy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4, test.getSht_latcy_DMW());
  #ifdef DEBUG
    test.print();
    printf("data4:\n");
    for (int i = 0; i < 8; i++){
      printf("data[%d] %d\n", i, data[i]);
    }
  #endif

  pos = wAdd;
  len = wSize;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + (pos%16)*8]); // here
  }
  for (int j = 0; j < 1; j++){
    ASSERT_EQ(result[j], data[j + 72 % 16]);
  }

  #ifdef DEBUG
    printf("after read4\n");
    test.print();
  #endif

  // read 5
  wAdd = 76;
  wSize = 4;
  result = test.readData(wAdd, wSize, 0, save);
  int s5 = wordShift(wAdd, wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(61*16+0, test.getIns_engy_DMW());
  ASSERT_EQ(61*16+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_engy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_latcy_DMW());
  ASSERT_EQ(16+64*8+9*8+8+32, test.getDet_engy_DMW());
  ASSERT_EQ(16+64*8+9*8+8+32, test.getDet_latcy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4+s5, test.getSht_engy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4+s5, test.getSht_latcy_DMW());
  #ifdef DEBUG
    test.print();
    printf("data5:\n");
    for (int i = 0; i < 32; i++){
      printf("data[%d] %d\n", i, data[i]);
    }
  #endif

  pos = wAdd;
  len = wSize;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + (pos%16)*8]); // here
  }
  for (int j = 0; j < 4; j++){
    ASSERT_EQ(result[j], data[j + 76 % 16]);
  }

  #ifdef DEBUG
    printf("after read5\n");
    test.print();
  #endif

  // read 6
  wAdd = 247;
  wSize = 8;
  result = test.readData(wAdd, wSize, 0, save);
  int s6 = wordShift(wAdd, wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(61*16+0, test.getIns_engy_DMW());
  ASSERT_EQ(61*16+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_engy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_latcy_DMW());
  ASSERT_EQ(16+64*8+9*8+8+32+8*8, test.getDet_engy_DMW());
  ASSERT_EQ(16+64*8+9*8+8+32+8*8, test.getDet_latcy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4+s5+s6, test.getSht_engy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4+s5+s6, test.getSht_latcy_DMW());

  pos = wAdd;
  len = wSize;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + (pos%16)*8]); // here
  }
  for (int j = 0; j < 8; j++){
    ASSERT_EQ(result[j], data[j + 247 % 16]);
  }

  // read 7
  wAdd = 245;
  wSize = 8;
  result = test.readData(wAdd, wSize, 0, save);
  int s7 = wordShift(wAdd, wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(61*16+0, test.getIns_engy_DMW());
  ASSERT_EQ(61*16+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_engy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_latcy_DMW());
  ASSERT_EQ(16+64*8+9*8+8+32+8*8+8*8, test.getDet_engy_DMW());
  ASSERT_EQ(16+64*8+9*8+8+32+8*8+8*8, test.getDet_latcy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4+s5+s6+s7, test.getSht_engy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4+s5+s6+s7, test.getSht_latcy_DMW());

  pos = wAdd;
  len = wSize;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + (pos%16)*8]); // here
  }
  for (int j = 0; j < 8; j++){
    ASSERT_EQ(result[j], data[j + 245 % 16]);
  }

  // read 8
  wAdd = 249;
  wSize = 3;
  result = test.readData(wAdd, wSize, 0, save);
  int s8 = wordShift(wAdd, wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(61*16+0, test.getIns_engy_DMW());
  ASSERT_EQ(61*16+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_engy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_latcy_DMW());
  ASSERT_EQ(16+64*8+9*8+8+32+8*8+8*8+3*8, test.getDet_engy_DMW());
  ASSERT_EQ(16+64*8+9*8+8+32+8*8+8*8+3*8, test.getDet_latcy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4+s5+s6+s7+s8, test.getSht_engy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4+s5+s6+s7+s8, test.getSht_latcy_DMW());

  pos = wAdd;
  len = wSize;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + (pos%16)*8]); // here
  }
  for (int j = 0; j < 3; j++){
    ASSERT_EQ(result[j], data[j + 249 % 16]);
  }

  // read 9
  wAdd = 249;
  wSize = 4;
  result = test.readData(wAdd, wSize, 0, save);
  int s9 = wordShift(wAdd, wSize, NAIVE_TRADITIONAL);
  ASSERT_EQ(61*16+0, test.getIns_engy_DMW());
  ASSERT_EQ(61*16+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_engy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_latcy_DMW());
  ASSERT_EQ(16+64*8+9*8+8+32+8*8+8*8+3*8+32, test.getDet_engy_DMW());
  ASSERT_EQ(16+64*8+9*8+8+32+8*8+8*8+3*8+32, test.getDet_latcy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4+s5+s6+s7+s8+s9, test.getSht_engy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4+s5+s6+s7+s8+s9, test.getSht_latcy_DMW());

  pos = wAdd;
  len = wSize;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + (pos%16)*8]); // here
  }
  for (int j = 0; j < 4; j++){
    ASSERT_EQ(result[j], data[j + 249 % 16]);
  }
}

TEST(checkReadDataModified, case1)
{
  SkyrmionWord test;
  #ifdef DEBUG
    test.print();
  #endif
  int save = 1;
  sky_size_t data_bit[128]={0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,0}; // 61 bit 1
  sky_size_t data[16]={120, 182, 52, 136, 115, 92, 150, 0, 247, 33, 59, 152, 248, 213, 165, 98};

  for(int i = 0; i < 16; i++){
    test.writeData(i * 16, 16, data, NAIVE_TRADITIONAL, save);
  }

  // read 1
  int wAdd = 1;
  int wSize = 2;
  sky_size_t *result = test.readData(wAdd, wSize, 1, save);
  int s1 = wordShift(wAdd, wSize, NAIVE);
  int d1 = wordDetect(wAdd, wSize, DCW);
  ASSERT_EQ(61*16+0, test.getIns_engy_DMW());
  ASSERT_EQ(61*16+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_engy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_latcy_DMW());
  ASSERT_EQ(2*8, test.getDet_engy_DMW());
  ASSERT_EQ(d1, test.getDet_latcy_DMW());
  ASSERT_EQ(64*16*4+s1, test.getSht_engy_DMW());
  ASSERT_EQ(64*16*4+s1, test.getSht_latcy_DMW());
  #ifdef DEBUG
    test.print();
    printf("data:\n");
    for (int i = 0; i < 2*8; i++){
      printf("data[%d] %d\n", i, data[i]);
    }
    printf("\n");
  #endif

  int pos = wAdd;
  int len = wSize;
  int rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + (pos%16)*8]); // here
  }
  ASSERT_EQ(result[0], data[1]);
  ASSERT_EQ(result[1], data[2]);
  #ifdef DEBUG
    printf("after read1\n");
    test.print();
  #endif
  delete [] result;

  // read 2
  wAdd = 0;
  wSize = 64;
  result = test.readData(wAdd, wSize, 1, save);
  int s2 = wordShift(wAdd, wSize, NAIVE);
  int d2 = wordDetect(wAdd, wSize, DCW);
  ASSERT_EQ(61*16+0, test.getIns_engy_DMW());
  ASSERT_EQ(61*16+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_engy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_latcy_DMW());
  ASSERT_EQ(16+64*8, test.getDet_engy_DMW());
  ASSERT_EQ(d1+d2, test.getDet_latcy_DMW());
  ASSERT_EQ(64*16*4+s1+s2, test.getSht_engy_DMW());
  ASSERT_EQ(64*16*4+s1+s2, test.getSht_latcy_DMW());
  #ifdef DEBUG
    test.print();
    printf("data2:\n");
    for (int i = 0; i < 5*8; i++){
      printf("data[%d] %d\n", i, data[i]);
    }
    printf("\n");
  #endif

  pos = wAdd;
  len = wSize;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + (pos%16)*8]); // here
  }
  for (int i = 0; i < 64/16; i++){
    for (int j = 0; j < 16; j++){
      ASSERT_EQ(result[i*16+j], data[j]);
    }
  }

  #ifdef DEBUG
    printf("after read2\n");
    test.print();
  #endif
  delete [] result;

  // read 3
  wAdd = 69;
  wSize = 9;
  result = test.readData(wAdd, wSize, 1, save);
  int s3 = wordShift(wAdd, wSize, NAIVE);
  int d3 = wordDetect(wAdd, wSize, DCW);
  ASSERT_EQ(61*16+0, test.getIns_engy_DMW());
  ASSERT_EQ(61*16+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_engy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_latcy_DMW());
  ASSERT_EQ(16+64*8+9*8, test.getDet_engy_DMW());
  ASSERT_EQ(d1+d2+d3, test.getDet_latcy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3, test.getSht_engy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3, test.getSht_latcy_DMW());

  #ifdef DEBUG
    test.print();
    printf("data3:\n");
    for (int i = 0; i < 5*8; i++){
      printf("data[%d] %d\n", i, data[i]);
    }
  #endif

  pos = wAdd;
  len = wSize;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + (pos%16)*8]); // here
  }
  for (int j = 0; j < 9; j++){
    ASSERT_EQ(result[j], data[5+j]);
  }
  #ifdef DEBUG
    printf("after read3\n");
    test.print();
  #endif
  delete [] result;

  // read 4
  wAdd = 72;
  wSize = 1;
  result = test.readData(wAdd, wSize, 1, save);
  int s4 = wordShift(wAdd, wSize, NAIVE);
  int d4 = wordDetect(wAdd, wSize, DCW);
  ASSERT_EQ(61*16+0, test.getIns_engy_DMW());
  ASSERT_EQ(61*16+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_engy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_latcy_DMW());
  ASSERT_EQ(16+64*8+9*8+8, test.getDet_engy_DMW());
  ASSERT_EQ(d1+d2+d3+d4, test.getDet_latcy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4, test.getSht_engy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4, test.getSht_latcy_DMW());
  #ifdef DEBUG
    test.print();
    printf("data4:\n");
    for (int i = 0; i < 8; i++){
      printf("data[%d] %d\n", i, data[i]);
    }
  #endif

  pos = wAdd;
  len = wSize;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + (pos%16)*8]); // here
  }
  ASSERT_EQ(result[0], data[8]);

  #ifdef DEBUG
    printf("after read4\n");
    test.print();
  #endif
  delete [] result;

  // read 5
  wAdd = 76;
  wSize = 4;
  result = test.readData(wAdd, wSize, 1, save);
  int s5 = wordShift(wAdd, wSize, NAIVE);
  int d5 = wordDetect(wAdd, wSize, DCW);
  ASSERT_EQ(61*16+0, test.getIns_engy_DMW());
  ASSERT_EQ(61*16+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_engy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_latcy_DMW());
  ASSERT_EQ(16+64*8+9*8+8+32, test.getDet_engy_DMW());
  ASSERT_EQ(d1+d2+d3+d4+d5, test.getDet_latcy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4+s5, test.getSht_engy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4+s5, test.getSht_latcy_DMW());
  #ifdef DEBUG
    test.print();
    printf("data5:\n");
    for (int i = 0; i < 32; i++){
      printf("data[%d] %d\n", i, data[i]);
    }
  #endif

  pos = wAdd;
  len = wSize;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + (pos%16)*8]); // here
  }
  for (int j = 0; j < 4; j++){
    ASSERT_EQ(result[j], data[12+j]);
  }
  #ifdef DEBUG
    printf("after read5\n");
    test.print();
  #endif
  delete [] result;

  // read 6
  wAdd = 247;
  wSize = 8;
  result = test.readData(wAdd, wSize, 1, save);
  int s6 = wordShift(wAdd, wSize, NAIVE);
  int d6 = wordDetect(wAdd, wSize, DCW);
  ASSERT_EQ(61*16+0, test.getIns_engy_DMW());
  ASSERT_EQ(61*16+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_engy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_latcy_DMW());
  ASSERT_EQ(16+64*8+9*8+8+32+8*8, test.getDet_engy_DMW());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6, test.getDet_latcy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4+s5+s6, test.getSht_engy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4+s5+s6, test.getSht_latcy_DMW());

  pos = wAdd;
  len = wSize;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + (pos%16)*8]); // here
  }
  for (int j = 0; j < 8; j++){
    ASSERT_EQ(result[j], data[7+j]);
  }
  delete [] result;

  // read 7
  wAdd = 245;
  wSize = 8;
  result = test.readData(wAdd, wSize, 1, save);
  int s7 = wordShift(wAdd, wSize, NAIVE);
  int d7 = wordDetect(wAdd, wSize, DCW);
  ASSERT_EQ(61*16+0, test.getIns_engy_DMW());
  ASSERT_EQ(61*16+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_engy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_latcy_DMW());
  ASSERT_EQ(16+64*8+9*8+8+32+8*8+8*8, test.getDet_engy_DMW());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7, test.getDet_latcy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4+s5+s6+s7, test.getSht_engy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4+s5+s6+s7, test.getSht_latcy_DMW());

  pos = wAdd;
  len = wSize;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + (pos%16)*8]); // here
  }
  for (int j = 0; j < 8; j++){
    ASSERT_EQ(result[j], data[5+j]);
  }
  delete [] result;

  // read 8
  wAdd = 249;
  wSize = 3;
  result = test.readData(wAdd, wSize, 1, save);
  int s8 = wordShift(wAdd, wSize, NAIVE);
  int d8 = wordDetect(wAdd, wSize, DCW);
  ASSERT_EQ(61*16+0, test.getIns_engy_DMW());
  ASSERT_EQ(61*16+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_engy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_latcy_DMW());
  ASSERT_EQ(16+64*8+9*8+8+32+8*8+8*8+3*8, test.getDet_engy_DMW());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7+d8, test.getDet_latcy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4+s5+s6+s7+s8, test.getSht_engy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4+s5+s6+s7+s8, test.getSht_latcy_DMW());

  pos = wAdd;
  len = wSize;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + (pos%16)*8]); // here
  }
  for (int j = 0; j < 3; j++){
    ASSERT_EQ(result[j], data[9+j]);
  }
  delete [] result;

  // read 9
  wAdd = 249;
  wSize = 4;
  result = test.readData(wAdd, wSize, 1, save);
  int s9 = wordShift(wAdd, wSize, NAIVE);
  int d9 = wordDetect(wAdd, wSize, DCW);
  ASSERT_EQ(61*16+0, test.getIns_engy_DMW());
  ASSERT_EQ(61*16+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_engy_DMW());
  ASSERT_EQ(128*16+0, test.getDel_latcy_DMW());
  ASSERT_EQ(16+64*8+9*8+8+32+8*8+8*8+3*8+32, test.getDet_engy_DMW());
  ASSERT_EQ(d1+d2+d3+d4+d5+d6+d7+d8+d9, test.getDet_latcy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4+s5+s6+s7+s8+s9, test.getSht_engy_DMW());
  ASSERT_EQ(64*16*4+s1+s2+s3+s4+s5+s6+s7+s8+s9, test.getSht_latcy_DMW());

  pos = wAdd;
  len = wSize;
  rem = pos % BYTEBWPORTS;
  if (rem != 0) {
    pos += (BYTEBWPORTS - rem);
  }
  len -= (BYTEBWPORTS - rem);
  len = min(len, BYTEBWPORTS) *8;
  for (int j = 0; j < len; j++){
    ASSERT_EQ(test.getEntry(OVER_HEAD + (pos*8 + pos/BYTEBWPORTS) + j + 1), data_bit[j + (pos%16)*8]); // here
  }
  for (int j = 0; j < 4; j++){
    ASSERT_EQ(result[j], data[9+j]);
  }
  delete [] result;
}

TEST(checkReadBit, case1)
{
  SkyrmionBit test;
  #ifdef DEBUG
    test.print();
  #endif
  int save = 1;
  sky_size_t data_bit[128]={0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,0}; // 61 bit 1
  sky_size_t data[16]={120, 182, 52, 136, 115, 92, 150, 0, 247, 33, 59, 152, 248, 213, 165, 98};

  for (int i = 0; i < 4; i++){
    test.write(1, 16*i, 16, data, BIT, save);
    test.write(2, 16*i, 16, data, BIT, save);
    test.write(15, 16*i, 16, data, BIT, save);
    test.write(16, 16*i, 16, data, BIT, save);
    test.write(32, 16*i, 16, data, BIT, save);
    test.write(65, 16*i, 16, data, BIT, save);
    test.write(125, 16*i, 16, data, BIT, save);
  }


  // read 1
  int wBlk = 1;
  int wAdd = 0;
  int wSize = 64;
  sky_size_t *result = test.read(wBlk, wAdd, wSize, save);
  int w0 = (bitShift(1, 16)[0]+bitShift(2,16)[0]+bitShift(15,16)[0]+bitShift(16,16)[0]+bitShift(32,16)[0]+bitShift(65,16)[0]+bitShift(125,16)[0])*4;
  int w1 = (bitShift(1, 16)[1]+bitShift(2,16)[1]+bitShift(15,16)[1]+bitShift(16,16)[1]+bitShift(32,16)[1]+bitShift(65,16)[1]+bitShift(125,16)[1])*4;
  int *ptr = bitShift(wBlk, wSize);
  int s1 = ptr[0];
  int l1 = ptr[1];
  ASSERT_EQ(61*7*4+0, test.getIns_engy_DMW());
  ASSERT_EQ(7*4+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*7*4+0, test.getDel_engy_DMW());
  ASSERT_EQ(7*4+0, test.getDel_latcy_DMW());
  ASSERT_EQ(64*8, test.getDet_engy_DMW());
  ASSERT_EQ(1, test.getDet_latcy_DMW());
  ASSERT_EQ(w0+s1, test.getSht_engy_DMW());
  ASSERT_EQ(w1+l1, test.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
  printf("data:\n");
  for (int i = 0; i < DISTANCE; i++){
    printf("%3d %3d\n", i, data[i]);
  }
  #endif

  for (int j = 0; j < 64; j++){
    ASSERT_EQ(result[j], data[j % 16]);
  }
  #ifdef DEBUG
    printf("after read1\n");
    test.print();
  #endif
  delete [] result;

  // read 2
  wBlk = 2;
  wAdd = 62;
  wSize = 2;
  result = test.read(wBlk, wAdd, wSize, save);
  ptr = bitShift(wBlk, wSize);
  int s2 = ptr[0];
  int l2 = ptr[1];
  ASSERT_EQ(61*7*4+0, test.getIns_engy_DMW());
  ASSERT_EQ(7*4+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*7*4+0, test.getDel_engy_DMW());
  ASSERT_EQ(7*4+0, test.getDel_latcy_DMW());
  ASSERT_EQ(64*8+2*8, test.getDet_engy_DMW());
  ASSERT_EQ(1+1, test.getDet_latcy_DMW());
  ASSERT_EQ(w0+s1+s2, test.getSht_engy_DMW());
  ASSERT_EQ(w1+l1+l2, test.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
  printf("data:\n");
  for (int i = 0; i < 2*8; i++){
    printf("%3d %3d\n", i, data[i]);
  }
  #endif
  for (int j = 0; j < 2; j++){
    ASSERT_EQ(result[j], data[(j + 62) % 16]);
  }
  #ifdef DEBUG
    printf("after read2\n");
    test.print();
  #endif
  delete [] result;

  // read 3
  wBlk = 15;
  wAdd = 61;
  wSize = 3;
  result = test.read(wBlk, wAdd, wSize, save);
  ptr = bitShift(wBlk, wSize);
  int s3 = ptr[0];
  int l3 = ptr[1];
  ASSERT_EQ(61*7*4+0, test.getIns_engy_DMW());
  ASSERT_EQ(7*4+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*7*4+0, test.getDel_engy_DMW());
  ASSERT_EQ(7*4+0, test.getDel_latcy_DMW());
  ASSERT_EQ(64*8+2*8+3*8, test.getDet_engy_DMW());
  ASSERT_EQ(3, test.getDet_latcy_DMW());
  ASSERT_EQ(w0+s1+s2+s3, test.getSht_engy_DMW());
  ASSERT_EQ(w1+l1+l2+l3, test.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
  printf("data:\n");
  for (int i = 0; i < 3*8; i++){
    printf("%3d %3d\n", i, data[i]);
  }
  #endif
  for (int j = 0; j < 3; j++){
    ASSERT_EQ(result[j], data[(j + 61) % 16]);
  }
  #ifdef DEBUG
    printf("after read3\n");
    test.print();
  #endif
  delete [] result;

  // read 4
  wBlk = 16;
  wAdd = 1;
  wSize = 4;
  result = test.read(wBlk, wAdd, wSize, save);
  ptr = bitShift(wBlk, wSize);
  int s4 = ptr[0];
  int l4 = ptr[1];
  ASSERT_EQ(61*7*4+0, test.getIns_engy_DMW());
  ASSERT_EQ(7*4+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*7*4+0, test.getDel_engy_DMW());
  ASSERT_EQ(7*4+0, test.getDel_latcy_DMW());
  ASSERT_EQ(64*8+2*8+3*8+4*8, test.getDet_engy_DMW());
  ASSERT_EQ(4, test.getDet_latcy_DMW());
  ASSERT_EQ(w0+s1+s2+s3+s4, test.getSht_engy_DMW());
  ASSERT_EQ(w1+l1+l2+l3+l4, test.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
  printf("data:\n");
  for (int i = 0; i < DISTANCE; i++){
    printf("%3d %3d\n", i, data[i]);
  }
  #endif
  for (int j = 0; j < 4; j++){
    ASSERT_EQ(result[j], data[(j + 1) % 16]);
  }
  #ifdef DEBUG
    printf("after read4\n");
    test.print();
  #endif
  delete [] result;

  // read 5
  wBlk = 32;
  wAdd = 4;
  wSize = 10;
  result = test.read(wBlk, wAdd, wSize, save);
  ptr = bitShift(wBlk, wSize);
  int s5 = ptr[0];
  int l5 = ptr[1];
  ASSERT_EQ(61*7*4+0, test.getIns_engy_DMW());
  ASSERT_EQ(7*4+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*7*4+0, test.getDel_engy_DMW());
  ASSERT_EQ(7*4+0, test.getDel_latcy_DMW());
  ASSERT_EQ(64*8+2*8+3*8+4*8+10*8, test.getDet_engy_DMW());
  ASSERT_EQ(5, test.getDet_latcy_DMW());
  ASSERT_EQ(w0+s1+s2+s3+s4+s5, test.getSht_engy_DMW());
  ASSERT_EQ(w1+l1+l2+l3+l4+l5, test.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
  printf("data:\n");
  for (int i = 0; i < DISTANCE; i++){
    printf("%3d %3d\n", i, data[i]);
  }
  #endif
  for (int j = 0; j < 10; j++){
    ASSERT_EQ(result[j], data[(j + 4) % 16]);
  }
  #ifdef DEBUG
    printf("after read5\n");
    test.print();
  #endif
  delete [] result;

  // read 6
  wBlk = 65;
  wAdd = 5;
  wSize = 12;
  result = test.read(wBlk, wAdd, wSize, save);
  ptr = bitShift(wBlk, wSize);
  int s6 = ptr[0];
  int l6 = ptr[1];
  ASSERT_EQ(61*7*4+0, test.getIns_engy_DMW());
  ASSERT_EQ(7*4+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*7*4+0, test.getDel_engy_DMW());
  ASSERT_EQ(7*4+0, test.getDel_latcy_DMW());
  ASSERT_EQ(64*8+2*8+3*8+4*8+10*8+12*8, test.getDet_engy_DMW());
  ASSERT_EQ(6, test.getDet_latcy_DMW());
  ASSERT_EQ(w0+s1+s2+s3+s4+s5+s6, test.getSht_engy_DMW());
  ASSERT_EQ(w1+l1+l2+l3+l4+l5+l6, test.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
  printf("data:\n");
  for (int i = 0; i < DISTANCE; i++){
    printf("%3d %3d\n", i, data[i]);
  }
  #endif
  for (int j = 0; j < 12; j++){
    ASSERT_EQ(result[j], data[(j + 5) % 16]);
  }
  #ifdef DEBUG
    printf("after read6\n");
    test.print();
  #endif
  delete [] result;

  // read 7
  wBlk = 125;
  wAdd = 0;
  wSize = 64;
  result = test.read(wBlk, wAdd, wSize, save);
  ptr = bitShift(wBlk, wSize);
  int s7 = ptr[0];
  int l7 = ptr[1];
  ASSERT_EQ(61*7*4+0, test.getIns_engy_DMW());
  ASSERT_EQ(7*4+0, test.getIns_latcy_DMW());
  ASSERT_EQ(128*7*4+0, test.getDel_engy_DMW());
  ASSERT_EQ(7*4+0, test.getDel_latcy_DMW());
  ASSERT_EQ(64*8+2*8+3*8+4*8+10*8+12*8+64*8, test.getDet_engy_DMW());
  ASSERT_EQ(7, test.getDet_latcy_DMW());
  ASSERT_EQ(w0+s1+s2+s3+s4+s5+s6+s7, test.getSht_engy_DMW());
  ASSERT_EQ(w1+l1+l2+l3+l4+l5+l6+l7, test.getSht_latcy_DMW());
  delete [] ptr;
  #ifdef DEBUG
  printf("data:\n");
  for (int i = 0; i < DISTANCE; i++){
    printf("%3d %3d\n", i, data[i]);
  }
  #endif
  for (int j = 0; j < 64; j++){ //12 is over the length of data
    ASSERT_EQ(result[j], data[(j + 0) % 16]);
  }
  #ifdef DEBUG
    printf("after read7\n");
    test.print();
  #endif
  delete [] result;
}

TEST(checkVector, case1)
{
  vector<SkyrmionWord> test(3);
  //Packet pkt(64);
  for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
    for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
      test[0].setEntry(i, 0);
      test[1].setEntry(i, 0);
      test[2].setEntry(i, 0);
    }
  }

  sky_size_t data_bit[128]={0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,0}; // 61 bit 1
  sky_size_t data[16]={120, 182, 52, 136, 115, 92, 150, 0, 247, 33, 59, 152, 248, 213, 165, 98};

  // write 1
  for(int i = 0; i < 4; i++){
    test[1].writeData(i * 16, 16, data, NAIVE_TRADITIONAL, 0);
  }
  int s = wordShift(0,16,NAIVE_TRADITIONAL)+wordShift(16,16,NAIVE_TRADITIONAL)+wordShift(32,16,NAIVE_TRADITIONAL)+wordShift(48,16,NAIVE_TRADITIONAL);
  ASSERT_EQ(61*4, test[1].getIns_engy()); //
  ASSERT_EQ(61*4, test[1].getIns_latcy());
  ASSERT_EQ(128*4, test[1].getDel_engy()); //
  ASSERT_EQ(128*4, test[1].getDel_latcy());
  ASSERT_EQ(0, test[1].getDet_engy()); //
  ASSERT_EQ(0, test[1].getDet_latcy());
  ASSERT_EQ(s, test[1].getSht_engy()); //
  ASSERT_EQ(s, test[1].getSht_latcy());

  for (int i = 0; i < 64; i++){
    sky_size_t *out = test[1].readData(i, 64-i, 0, 0);
    for (int j = 0; j < 64-i; j++){
      ASSERT_EQ(out[j], data[(j + i) % 16]);
    }
  }
}

TEST(checkBitToByte, case1)
{
  sky_size_t array[16] = {1, 0, 0, 1, 1, 1, 1, 1,
                          0, 1, 1, 0, 1, 0, 0, 1};
  sky_size_t *result = Skyrmion::bitToByte(2, array);
  ASSERT_EQ(159, result[0]);
  ASSERT_EQ(105, result[1]);
}

TEST(checkByteToBit, case1)
{
  sky_size_t array[2] = {159, 105};
  sky_size_t *result = Skyrmion::byteToBit(2, array);
  ASSERT_EQ(1, result[0]);
  ASSERT_EQ(0, result[1]);
  ASSERT_EQ(0, result[2]);
  ASSERT_EQ(1, result[3]);
  ASSERT_EQ(1, result[4]);
  ASSERT_EQ(1, result[5]);
  ASSERT_EQ(1, result[6]);
  ASSERT_EQ(1, result[7]);
  ASSERT_EQ(0, result[8]);
  ASSERT_EQ(1, result[9]);
  ASSERT_EQ(1, result[10]);
  ASSERT_EQ(0, result[11]);
  ASSERT_EQ(1, result[12]);
  ASSERT_EQ(0, result[13]);
  ASSERT_EQ(0, result[14]);
  ASSERT_EQ(1, result[15]);

}
/*
TEST(checkReadBit, case2) //all address
{
  SkyrmionBit test;
  #ifdef DEBUG
    test.print();
  #endif

  sky_size_t data_bit[128]={0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,0}; // 61 bit 1
  sky_size_t data[16]={120, 182, 52, 136, 115, 92, 150, 0, 247, 33, 59, 152, 248, 213, 165, 98};

  for (int i = 0; i < MAX_SIZE; i++){
    for (int j = 0; j < 4; j++){
      test.write(i, 16*j, 16, data, BIT_PW, 0);
    }
  }
  //assert(test.getEntries(0, 0) == data_bit[0]);

  for (int m = 1; m <= 64; m++){
    for (int i = 0; i < DISTANCE; i++){
      for (int j = 0; j < 4; j++){
        for (int k = 0; k < 128; k++){
          //cout << "i = " << i << " j = " << j << " k = " << k << endl;
          ASSERT_EQ(test.getEntries(128*j+k, i+(DISTANCE+1)*m), data_bit[k]);
        }
      }
    }
  }


  for (int i = 0; i < MAX_SIZE; i++){
    for (int j = 0; j < 64; j++){
      sky_size_t *result = test.read(i, j, 64-j, 0);
      for (int k = 0; k < 64-j; k++){
        ASSERT_EQ(result[k], data[(k + j) % 16]);
      }
    }
  }
}*/

TEST(checkReadBit, case3)
{
  SkyrmionBit test;
  #ifdef DEBUG
    test.print();
  #endif

  sky_size_t data_bit[128]={0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,0}; // 61 bit 1
  sky_size_t data[16]={120, 182, 52, 136, 115, 92, 150, 0, 247, 33, 59, 152, 248, 213, 165, 98};
  sky_size_t data2_bit[24] = {1,0,1,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,0,0,1,1,0,0};
  sky_size_t data2[3] = {160, 22, 76};
  test.write(304, 32, 3, data2, BIT_PW, 0);
  //test.print();
  for (int i = 0; i < min(DISTANCE,24); i++){
    ASSERT_EQ(test.getEntries(32*8+i,  OVER_HEAD + (304+304/DISTANCE)+1), data2_bit[i]);
  }
  for (int j = 0; j < 4; j++){
    test.write(192, 16*j, 16, data, BIT_PW, 0);
  }
  //test.print();
  for (int j = 0; j < 4; j++){
    for (int k = 0; k < 128; k++){
      //cout << "j = " << j << " k = " << k << endl;
      ASSERT_EQ(test.getEntries(128*j+k, OVER_HEAD + (192+192/DISTANCE)+1), data_bit[k]);
    }
  }
  for (int j = 0; j < 64; j++){
    sky_size_t *result = test.read(192, j, 64-j, 0);
    for (int k = 0; k < 64-j; k++){
      ASSERT_EQ(result[k], data[(k + j) % 16]);
    }
  }
  cout << "block = 192, add = 3, size = 4\n";
  test.write(192, 3, 4, data, BIT_PW, 0);
  for (int i = 0; i < 4*8; i++){
    ASSERT_EQ(test.getEntries(24+i, OVER_HEAD + (192+192/DISTANCE)+1), data_bit[i]);
  }
  sky_size_t *result = test.read(192, 3, 4, 0);
  for (int i = 0; i < 4; i++){
    ASSERT_EQ(result[i], data[i]);
  }
  cout << "block = 192, add = 4, size = 16\n";
  test.write(192, 4, 16, data, BIT_PW, 0);
  for (int i = 0; i < 16*8; i++){
    ASSERT_EQ(test.getEntries(32+i, OVER_HEAD + (192+192/DISTANCE)+1), data_bit[i]);
  }
  result = test.read(192, 0, 64, 0);
  ASSERT_EQ(result[0], data[0]);
  ASSERT_EQ(result[1], data[1]);
  ASSERT_EQ(result[2], data[2]);
  ASSERT_EQ(result[3], data[0]);
  for (int i = 0; i < 16; i++){
    ASSERT_EQ(result[4+i], data[i]);
  }
  for (int i = 20; i < 64; i++){
    ASSERT_EQ(result[i], data[i%16]);
  }
}

TEST(checkCounBit1, case1)
{
  sky_size_t data1[8] = {0,1,1,0,0,1,0,1};
  sky_size_t data2[8] = {1,1,1,1,1,0,0,0};
  sky_size_t data3[8] = {1,1,0,1,0,0,0,0};
  sky_size_t data4[8] = {0,0,0,0,0,0,0,0};
  int start1 = 0;
  int start2 = 0;
  int start3 = 0;
  int start4 = 0;
  ASSERT_EQ(4, SkyrmionBit::countBit1(data1, start1, 0));
  ASSERT_EQ(1, start1);
  ASSERT_EQ(4, SkyrmionBit::countBit1(data1, start1, 1));
  ASSERT_EQ(7, start1);
  ASSERT_EQ(5, SkyrmionBit::countBit1(data2, start2, 0));
  ASSERT_EQ(0, start2);
  ASSERT_EQ(5, SkyrmionBit::countBit1(data2, start2, 1));
  ASSERT_EQ(4, start2);
  ASSERT_EQ(3, SkyrmionBit::countBit1(data3, start3, 0));
  ASSERT_EQ(0, start3);
  ASSERT_EQ(3, SkyrmionBit::countBit1(data3, start3, 1));
  ASSERT_EQ(3, start3);
  ASSERT_EQ(0, SkyrmionBit::countBit1(data4, start4, 0));
  ASSERT_EQ(0, start4);
  ASSERT_EQ(0, SkyrmionBit::countBit1(data4, start4, 1));
  ASSERT_EQ(0, start4);
}

TEST(checkCmpPattern, case1)
{
  sky_size_t data0[8] = {0,0,0,0,0,0,0,0};
  sky_size_t data1[8] = {1,0,0,1,1,0,0,0};
  sky_size_t data2[8] = {0,0,0,1,0,0,1,1};
  sky_size_t data3[8] = {0,0,1,0,0,1,0,0};
  sky_size_t data4[8] = {0,0,1,0,0,1,1,1};
  sky_size_t data5[8] = {1,0,0,1,0,0,0,1};
  sky_size_t data6[8] = {1,1,0,0,1,0,0,0};
  sky_size_t data7[8] = {0,1,0,0,0,0,0,0};
  sky_size_t data8[8] = {0,0,0,0,0,1,1,0};
  sky_size_t data9[8] = {1,0,1,0,0,1,1,0};
  vector<int> cmpPatternResult(4);

  cmpPatternResult = SkyrmionBit::cmpPattern(data1, data2, 0);
  ASSERT_EQ(3, cmpPatternResult[0]);
  ASSERT_EQ(3, cmpPatternResult[1]);
  ASSERT_EQ(5, cmpPatternResult[2]);
  ASSERT_EQ(8, cmpPatternResult[3]);
  cmpPatternResult = SkyrmionBit::cmpPattern(data1, data2, 1);
  ASSERT_EQ(3, cmpPatternResult[0]);
  ASSERT_EQ(3, cmpPatternResult[1]);
  ASSERT_EQ(-1, cmpPatternResult[2]);
  ASSERT_EQ(2, cmpPatternResult[3]);
  cmpPatternResult = SkyrmionBit::cmpPattern(data1, data3, 0);
  ASSERT_EQ(4, cmpPatternResult[0]);
  ASSERT_EQ(2, cmpPatternResult[1]);
  ASSERT_EQ(4, cmpPatternResult[2]);
  ASSERT_EQ(6, cmpPatternResult[3]);
  cmpPatternResult = SkyrmionBit::cmpPattern(data1, data8, 1);
  ASSERT_EQ(4, cmpPatternResult[0]);
  ASSERT_EQ(2, cmpPatternResult[1]);
  ASSERT_EQ(0, cmpPatternResult[2]);
  ASSERT_EQ(2, cmpPatternResult[3]);
  cmpPatternResult = SkyrmionBit::cmpPattern(data1, data4, 0);
  ASSERT_EQ(5, cmpPatternResult[0]);
  ASSERT_EQ(2, cmpPatternResult[1]);
  ASSERT_EQ(5, cmpPatternResult[2]);
  ASSERT_EQ(7, cmpPatternResult[3]);
  cmpPatternResult = SkyrmionBit::cmpPattern(data4, data8, 1);
  ASSERT_EQ(4, cmpPatternResult[0]);
  ASSERT_EQ(-1, cmpPatternResult[1]);
  ASSERT_EQ(5, cmpPatternResult[2]);
  ASSERT_EQ(4, cmpPatternResult[3]);
  cmpPatternResult = SkyrmionBit::cmpPattern(data2, data3, 0);
  ASSERT_EQ(4, cmpPatternResult[0]);
  ASSERT_EQ(-1, cmpPatternResult[1]);
  ASSERT_EQ(7, cmpPatternResult[2]);
  ASSERT_EQ(6, cmpPatternResult[3]);
  cmpPatternResult = SkyrmionBit::cmpPattern(data2, data9, 1);
  ASSERT_EQ(5, cmpPatternResult[0]);
  ASSERT_EQ(-1, cmpPatternResult[1]);
  ASSERT_EQ(1, cmpPatternResult[2]);
  ASSERT_EQ(0, cmpPatternResult[3]);
  cmpPatternResult = SkyrmionBit::cmpPattern(data2, data4, 0);
  ASSERT_EQ(5, cmpPatternResult[0]);
  ASSERT_EQ(-1, cmpPatternResult[1]);
  ASSERT_EQ(8, cmpPatternResult[2]);
  ASSERT_EQ(7, cmpPatternResult[3]);
  cmpPatternResult = SkyrmionBit::cmpPattern(data4, data9, 1);
  ASSERT_EQ(3, cmpPatternResult[0]);
  ASSERT_EQ(-1, cmpPatternResult[1]);
  ASSERT_EQ(5, cmpPatternResult[2]);
  ASSERT_EQ(4, cmpPatternResult[3]);
  cmpPatternResult = SkyrmionBit::cmpPattern(data3, data4, 0);
  ASSERT_EQ(5, cmpPatternResult[0]);
  ASSERT_EQ(0, cmpPatternResult[1]);
  ASSERT_EQ(6, cmpPatternResult[2]);
  ASSERT_EQ(6, cmpPatternResult[3]);
  cmpPatternResult = SkyrmionBit::cmpPattern(data3, data7, 1);
  ASSERT_EQ(4, cmpPatternResult[0]);
  ASSERT_EQ(-4, cmpPatternResult[1]);
  ASSERT_EQ(3, cmpPatternResult[2]);
  ASSERT_EQ(-1, cmpPatternResult[3]);
  cmpPatternResult = SkyrmionBit::cmpPattern(data1, data7, 0);
  ASSERT_EQ(4, cmpPatternResult[0]);
  ASSERT_EQ(1, cmpPatternResult[1]);
  ASSERT_EQ(3, cmpPatternResult[2]);
  ASSERT_EQ(4, cmpPatternResult[3]);
  cmpPatternResult = SkyrmionBit::cmpPattern(data1, data7, 1);
  ASSERT_EQ(4, cmpPatternResult[0]);
  ASSERT_EQ(-3, cmpPatternResult[1]);
  ASSERT_EQ(3, cmpPatternResult[2]);
  ASSERT_EQ(0, cmpPatternResult[3]);
  cmpPatternResult = SkyrmionBit::cmpPattern(data5, data6, 0);
  ASSERT_EQ(3, cmpPatternResult[0]);
  ASSERT_EQ(0, cmpPatternResult[1]);
  ASSERT_EQ(1, cmpPatternResult[2]);
  ASSERT_EQ(1, cmpPatternResult[3]);
  cmpPatternResult = SkyrmionBit::cmpPattern(data5, data6, 1);
  ASSERT_EQ(3, cmpPatternResult[0]);
  ASSERT_EQ(-3, cmpPatternResult[1]);
  ASSERT_EQ(4, cmpPatternResult[2]);
  ASSERT_EQ(1, cmpPatternResult[3]);
}

TEST(checkAssessPureShift, case1)
{
  sky_size_t data0[8] = {0,0,0,0,0,0,0,0};
  sky_size_t data1[8] = {1,1,0,0,0,0,0,1};
  sky_size_t data2[8] = {0,0,0,1,0,1,1,0};
  sky_size_t data3[8] = {0,0,0,0,0,0,1,1};
  sky_size_t data4[8] = {0,1,1,1,0,1,1,0};
  sky_size_t data5[8] = {1,0,0,0,0,0,1,1};
  sky_size_t data6[8] = {0,0,0,0,1,1,0,1};
  sky_size_t data7[8] = {1,1,0,0,0,0,0,0};
  sky_size_t data8[8] = {0,1,1,0,1,0,1,1};
  sky_size_t data9[8] = {1,1,0,0,0,0,1,1};
  sky_size_t data10[8] = {1,1,1,0,0,0,1,1};
  sky_size_t data11[8] = {1,0,1,0,1,1,1,1};
  sky_size_t data12[8] = {1,1,0,0,1,1,0,0};
  sky_size_t data13[8] = {0,1,1,0,0,1,1,0};
  sky_size_t data14[8] = {1,0,0,0,1,0,1,1};
  sky_size_t data15[8] = {0,1,1,1,1,1,1,0};
  sky_size_t data16[8] = {0,0,0,0,1,1,0,1};
  vector<int> cmpPatternResult(4);
  int shift = 0;
  int nextStart1 = 0;
  int nextStart2 = 0;

  cmpPatternResult = SkyrmionBit::cmpPattern(data1, data2, 0);
  shift = cmpPatternResult[1];
  nextStart1 = cmpPatternResult[2];
  nextStart2 = cmpPatternResult[3];
  ASSERT_EQ(45, SkyrmionBit::assessPureShift(data1, data2, cmpPatternResult, 0, 0)); //D
  ASSERT_EQ(25, SkyrmionBit::assessPureShift(data1, data2, cmpPatternResult, 0, 8)); //D
  ASSERT_EQ(3, shift);
  ASSERT_EQ(1, nextStart1);
  ASSERT_EQ(4, nextStart2);
  cmpPatternResult = SkyrmionBit::cmpPattern(data7, data2, 1);
  shift = cmpPatternResult[1];
  nextStart1 = cmpPatternResult[2];
  nextStart2 = cmpPatternResult[3];
  ASSERT_EQ(15, SkyrmionBit::assessPureShift(data7, data2, cmpPatternResult, 1, 0)); //E
  ASSERT_EQ(5, SkyrmionBit::assessPureShift(data7, data2, cmpPatternResult, 1, 3)); //E
  ASSERT_EQ(5, shift);
  ASSERT_EQ(-1, nextStart1);
  ASSERT_EQ(4, nextStart2);
  cmpPatternResult = SkyrmionBit::cmpPattern(data3, data4, 0);
  shift = cmpPatternResult[1];
  nextStart1 = cmpPatternResult[2];
  nextStart2 = cmpPatternResult[3];
  ASSERT_EQ(35, SkyrmionBit::assessPureShift(data3, data4, cmpPatternResult, 0, 0)); //A
  ASSERT_EQ(-5, shift);
  ASSERT_EQ(8, nextStart1);
  ASSERT_EQ(3, nextStart2);
  cmpPatternResult = SkyrmionBit::cmpPattern(data3, data4, 1);
  shift = cmpPatternResult[1];
  nextStart1 = cmpPatternResult[2];
  nextStart2 = cmpPatternResult[3];
  ASSERT_EQ(17, SkyrmionBit::assessPureShift(data3, data4, cmpPatternResult, 1, 2)); //H1
  ASSERT_EQ(-1, shift);
  ASSERT_EQ(4, nextStart1);
  ASSERT_EQ(3, nextStart2);
  cmpPatternResult = SkyrmionBit::cmpPattern(data5, data4, 1);
  shift = cmpPatternResult[1];
  nextStart1 = cmpPatternResult[2];
  nextStart2 = cmpPatternResult[3];
  ASSERT_EQ(9, SkyrmionBit::assessPureShift(data5, data4, cmpPatternResult, 1, 2)); //H0~H1
  ASSERT_EQ(10, SkyrmionBit::assessPureShift(data5, data4, cmpPatternResult, 1, 8)); //H0~H1
  ASSERT_EQ(25, SkyrmionBit::assessPureShift(data5, data4, cmpPatternResult, 1, 0)); //H0~H1
  ASSERT_EQ(-1, shift);
  ASSERT_EQ(4, nextStart1);
  ASSERT_EQ(3, nextStart2);
  cmpPatternResult = SkyrmionBit::cmpPattern(data9, data8, 1);
  shift = cmpPatternResult[1];
  nextStart1 = cmpPatternResult[2];
  nextStart2 = cmpPatternResult[3];
  ASSERT_EQ(10, SkyrmionBit::assessPureShift(data9, data8, cmpPatternResult, 1, 2)); //G0
  ASSERT_EQ(12, SkyrmionBit::assessPureShift(data9, data8, cmpPatternResult, 1, 8)); //G0
  ASSERT_EQ(16, SkyrmionBit::assessPureShift(data9, data8, cmpPatternResult, 1, 0)); //G0
  ASSERT_EQ(0, shift);
  ASSERT_EQ(4, nextStart1);
  ASSERT_EQ(4, nextStart2);
  cmpPatternResult = SkyrmionBit::cmpPattern(data11, data10, 1);
  shift = cmpPatternResult[1];
  nextStart1 = cmpPatternResult[2];
  nextStart2 = cmpPatternResult[3];
  ASSERT_EQ(12, SkyrmionBit::assessPureShift(data11, data10, cmpPatternResult, 1, 1)); //G0
  ASSERT_EQ(10, SkyrmionBit::assessPureShift(data11, data10, cmpPatternResult, 1, 8)); //G0
  ASSERT_EQ(12, SkyrmionBit::assessPureShift(data11, data10, cmpPatternResult, 1, 0)); //G0
  ASSERT_EQ(0, shift);
  ASSERT_EQ(5, nextStart1);
  ASSERT_EQ(5, nextStart2);
  cmpPatternResult = SkyrmionBit::cmpPattern(data12, data11, 1);
  shift = cmpPatternResult[1];
  nextStart1 = cmpPatternResult[2];
  nextStart2 = cmpPatternResult[3];
  ASSERT_EQ(26, SkyrmionBit::assessPureShift(data12, data11, cmpPatternResult, 1, 0)); //G0~G1
  ASSERT_EQ(12, SkyrmionBit::assessPureShift(data12, data11, cmpPatternResult, 1, 8)); //G0~G1
  ASSERT_EQ(18, SkyrmionBit::assessPureShift(data12, data11, cmpPatternResult, 1, 1)); //G0~G1
  ASSERT_EQ(2, shift);
  ASSERT_EQ(3, nextStart1);
  ASSERT_EQ(5, nextStart2);
  cmpPatternResult = SkyrmionBit::cmpPattern(data14, data13, 1);
  shift = cmpPatternResult[1];
  nextStart1 = cmpPatternResult[2];
  nextStart2 = cmpPatternResult[3];
  ASSERT_EQ(9, SkyrmionBit::assessPureShift(data14, data13, cmpPatternResult, 1, 0)); //H0~H1
  ASSERT_EQ(9, SkyrmionBit::assessPureShift(data14, data13, cmpPatternResult, 1, 8)); //H0~H1
  ASSERT_EQ(9, SkyrmionBit::assessPureShift(data14, data13, cmpPatternResult, 1, 1)); //H0~H1
  ASSERT_EQ(-1, shift);
  ASSERT_EQ(4, nextStart1);
  ASSERT_EQ(3, nextStart2);
  cmpPatternResult = SkyrmionBit::cmpPattern(data4, data3, 0);
  shift = cmpPatternResult[1];
  nextStart1 = cmpPatternResult[2];
  nextStart2 = cmpPatternResult[3];
  ASSERT_EQ(38, SkyrmionBit::assessPureShift(data4, data3, cmpPatternResult, 0, 0)); //B
  ASSERT_EQ(8, SkyrmionBit::assessPureShift(data4, data3, cmpPatternResult, 0, 8)); //B
  ASSERT_EQ(5, shift);
  ASSERT_EQ(3, nextStart1);
  ASSERT_EQ(8, nextStart2);
  cmpPatternResult = SkyrmionBit::cmpPattern(data6, data4, 0);
  shift = cmpPatternResult[1];
  nextStart1 = cmpPatternResult[2];
  nextStart2 = cmpPatternResult[3];
  ASSERT_EQ(44, SkyrmionBit::assessPureShift(data6, data4, cmpPatternResult, 0, 0)); //C
  ASSERT_EQ(34, SkyrmionBit::assessPureShift(data6, data4, cmpPatternResult, 0, 8)); //C
  ASSERT_EQ(-3, shift);
  ASSERT_EQ(6, nextStart1);
  ASSERT_EQ(3, nextStart2);
  cmpPatternResult = SkyrmionBit::cmpPattern(data16, data15, 0);
  shift = cmpPatternResult[1];
  nextStart1 = cmpPatternResult[2];
  nextStart2 = cmpPatternResult[3];
  ASSERT_EQ(33, SkyrmionBit::assessPureShift(data16, data15, cmpPatternResult, 0, 0)); //C
  ASSERT_EQ(-3, shift);
  ASSERT_EQ(6, nextStart1);
  ASSERT_EQ(3, nextStart2);
  cmpPatternResult = SkyrmionBit::cmpPattern(data8, data7, 1);
  shift = cmpPatternResult[1];
  nextStart1 = cmpPatternResult[2];
  nextStart2 = cmpPatternResult[3];
  ASSERT_EQ(6, SkyrmionBit::assessPureShift(data8, data7, cmpPatternResult, 1, 0)); //F
  ASSERT_EQ(8, SkyrmionBit::assessPureShift(data8, data7, cmpPatternResult, 1, 7)); //F
  ASSERT_EQ(-6, shift);
  ASSERT_EQ(5, nextStart1);
  ASSERT_EQ(-1, nextStart2);
}

TEST(checkAssessDCW, case1)
{
  sky_size_t data0[8] = {0,0,0,0,0,0,0,0};
  sky_size_t data1[8] = {1,1,0,0,0,0,0,1};
  sky_size_t data2[8] = {0,0,0,1,0,1,1,0};
  sky_size_t data3[8] = {0,0,0,0,0,0,1,1};
  sky_size_t data4[8] = {0,1,1,1,0,1,1,0};
  sky_size_t data5[8] = {1,0,0,0,0,0,1,1};
  sky_size_t data6[8] = {0,0,0,0,1,1,0,1};
  sky_size_t data7[8] = {1,1,0,0,0,0,0,0};
  sky_size_t data8[8] = {0,1,1,0,1,0,1,1};
  sky_size_t data9[8] = {1,1,0,0,0,0,1,1};
  sky_size_t data10[8] = {1,1,1,0,0,0,1,1};
  sky_size_t data11[8] = {1,0,1,0,1,1,1,1};
  sky_size_t data12[8] = {1,1,0,0,1,1,0,0};
  sky_size_t data13[8] = {0,1,1,0,0,1,1,0};
  sky_size_t data14[8] = {1,0,0,0,1,0,1,1};
  sky_size_t data15[8] = {0,1,1,1,1,1,1,0};
  sky_size_t data16[8] = {0,0,0,0,1,1,0,1};

  ASSERT_EQ(63, SkyrmionBit::assessDCW(data1, data2, 0));
  ASSERT_EQ(33, SkyrmionBit::assessDCW(data1, data2, 8));
  ASSERT_EQ(52, SkyrmionBit::assessDCW(data7, data2, 0));
  ASSERT_EQ(51, SkyrmionBit::assessDCW(data3, data4, 0));
  ASSERT_EQ(62, SkyrmionBit::assessDCW(data5, data4, 2));
  ASSERT_EQ(31, SkyrmionBit::assessDCW(data9, data8, 2));
  ASSERT_EQ(32, SkyrmionBit::assessDCW(data11, data10, 1));
  ASSERT_EQ(41, SkyrmionBit::assessDCW(data12, data11, 0));
  ASSERT_EQ(63, SkyrmionBit::assessDCW(data14, data13, 0));
  ASSERT_EQ(54, SkyrmionBit::assessDCW(data4, data3, 0));
  ASSERT_EQ(62, SkyrmionBit::assessDCW(data6, data4, 0));
  ASSERT_EQ(51, SkyrmionBit::assessDCW(data16, data15, 0));
  ASSERT_EQ(54, SkyrmionBit::assessDCW(data8, data7, 0));
}

TEST(checkAssessPW, case1)
{
  sky_size_t data0[8] = {0,0,0,0,0,0,0,0};
  sky_size_t data1[8] = {1,1,0,0,0,0,0,1};
  sky_size_t data2[8] = {0,0,0,1,0,1,1,0};
  sky_size_t data3[8] = {0,0,0,0,0,0,1,1};
  sky_size_t data4[8] = {0,1,1,1,0,1,1,0};
  sky_size_t data5[8] = {1,0,0,0,0,0,1,1};
  sky_size_t data6[8] = {0,0,0,0,1,1,0,1};
  sky_size_t data7[8] = {1,1,0,0,0,0,0,0};
  sky_size_t data8[8] = {0,1,1,0,1,0,1,1};
  sky_size_t data9[8] = {1,1,0,0,0,0,1,1};
  sky_size_t data10[8] = {1,1,1,0,0,0,1,1};
  sky_size_t data11[8] = {1,0,1,0,1,1,1,1};
  sky_size_t data12[8] = {1,1,0,0,1,1,0,0};
  sky_size_t data13[8] = {0,1,1,0,0,1,1,0};
  sky_size_t data14[8] = {1,0,0,0,1,0,1,1};
  sky_size_t data15[8] = {0,1,1,1,1,1,1,0};
  sky_size_t data16[8] = {0,0,0,0,1,1,0,1};

  ASSERT_EQ(15, SkyrmionBit::assessPW(data1, data2, 0));
  ASSERT_EQ(10, SkyrmionBit::assessPW(data1, data2, 6));
  ASSERT_EQ(10, SkyrmionBit::assessPW(data1, data2, 8));
  ASSERT_EQ(18, SkyrmionBit::assessPW(data7, data2, 0)); //pureShift:15
  ASSERT_EQ(9, SkyrmionBit::assessPW(data7, data2, 1));
  ASSERT_EQ(9, SkyrmionBit::assessPW(data7, data2, 3)); //pureShift:5
  ASSERT_EQ(9, SkyrmionBit::assessPW(data7, data2, 8));
  ASSERT_EQ(41, SkyrmionBit::assessPW(data3, data4, 0));
  ASSERT_EQ(24, SkyrmionBit::assessPW(data3, data4, 2)); //pureShift:17
  ASSERT_EQ(15, SkyrmionBit::assessPW(data3, data4, 7));
  ASSERT_EQ(9, SkyrmionBit::assessPW(data3, data4, 8));
  ASSERT_EQ(32, SkyrmionBit::assessPW(data5, data4, 0)); //pureShift:25
  ASSERT_EQ(15, SkyrmionBit::assessPW(data5, data4, 2)); //pureShift:9
  ASSERT_EQ(15, SkyrmionBit::assessPW(data5, data4, 6));
  ASSERT_EQ(10, SkyrmionBit::assessPW(data5, data4, 8)); //pureShift:10
  ASSERT_EQ(25, SkyrmionBit::assessPW(data9, data8, 0)); //pureShift:16
  ASSERT_EQ(16, SkyrmionBit::assessPW(data9, data8, 2)); //pureShift:10
  ASSERT_EQ(12, SkyrmionBit::assessPW(data9, data8, 7));
  ASSERT_EQ(12, SkyrmionBit::assessPW(data9, data8, 8)); //pureShift:12
  ASSERT_EQ(16, SkyrmionBit::assessPW(data11, data10, 0));
  ASSERT_EQ(16, SkyrmionBit::assessPW(data11, data10, 1));
  ASSERT_EQ(15, SkyrmionBit::assessPW(data11, data10, 6));
  ASSERT_EQ(14, SkyrmionBit::assessPW(data11, data10, 8));
  ASSERT_EQ(32, SkyrmionBit::assessPW(data12, data11, 0));
  ASSERT_EQ(14, SkyrmionBit::assessPW(data12, data11, 2));
  ASSERT_EQ(12, SkyrmionBit::assessPW(data12, data11, 7));
  ASSERT_EQ(12, SkyrmionBit::assessPW(data12, data11, 8));
  ASSERT_EQ(15, SkyrmionBit::assessPW(data14, data13, 0));
  ASSERT_EQ(14, SkyrmionBit::assessPW(data14, data13, 6));
  ASSERT_EQ(11, SkyrmionBit::assessPW(data14, data13, 8));
  ASSERT_EQ(15, SkyrmionBit::assessPW(data4, data3, 0));
  ASSERT_EQ(13, SkyrmionBit::assessPW(data4, data3, 8));
  ASSERT_EQ(32, SkyrmionBit::assessPW(data6, data4, 0));
  ASSERT_EQ(15, SkyrmionBit::assessPW(data6, data4, 3));
  ASSERT_EQ(10, SkyrmionBit::assessPW(data6, data4, 8));
  ASSERT_EQ(42, SkyrmionBit::assessPW(data16, data15, 0));
  ASSERT_EQ(24, SkyrmionBit::assessPW(data16, data15, 2));
  ASSERT_EQ(14, SkyrmionBit::assessPW(data16, data15, 7));
  ASSERT_EQ(10, SkyrmionBit::assessPW(data8, data7, 0));
  ASSERT_EQ(10, SkyrmionBit::assessPW(data8, data7, 2));
  ASSERT_EQ(7, SkyrmionBit::assessPW(data8, data7, 8));
}
/*
TEST(checkWriteBitPURESHIFTOrder0, case1)
{
  SkyrmionBit test;
  for (int k = 0; k < ROW; k++){
    for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
      for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
        test.setEntries(k, i, 0);
      }
    }
  }

  sky_size_t data_bit[128]={0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,0}; // 61 bit 1
  sky_size_t data[16]={120, 182, 52, 136, 115, 92, 150, 0, 247, 33, 59, 152, 248, 213, 165, 98};

  // write 1
  test.write(2, 0, 4, data+3*4, BIT_PURESHIFT, 0);

  ASSERT_EQ(17, test.getIns_engy());
  ASSERT_EQ(1, test.getIns_latcy());
  ASSERT_EQ(0, test.getDel_engy());
  ASSERT_EQ(0, test.getDel_latcy());
  ASSERT_EQ(4*8, test.getDet_engy());
  ASSERT_EQ(1, test.getDet_latcy());
  ASSERT_EQ(6*DISTANCE, test.getSht_engy());
  ASSERT_EQ(6, test.getSht_latcy());
  ASSERT_EQ(0, test.getShtVrtcl_engy());
  ASSERT_EQ(0, test.getShtVrtcl_latcy());

  for (int k = 0; k < DISTANCE; k++){
    ASSERT_EQ(test.getEntries(k, OVER_HEAD + 1 + 2), data_bit[k+3*DISTANCE]); // here
  }

  // write 2
  cout << "write 2\n";
  test.write(2, 1, 5, data, BIT_PURESHIFT, 0);

  ASSERT_EQ(17+(1+2+1+2+5), test.getIns_engy());
  ASSERT_EQ(2, test.getIns_latcy());
  ASSERT_EQ((2+1+1), test.getDel_engy()); // only when the buffer is full
  ASSERT_EQ(1, test.getDel_latcy());
  ASSERT_EQ(4*8+5*8, test.getDet_engy());
  ASSERT_EQ(2, test.getDet_latcy());
  ASSERT_EQ(6*DISTANCE+6*5*8, test.getSht_engy());
  ASSERT_EQ(6+6, test.getSht_latcy());
  ASSERT_EQ((1+0+1), test.getShtVrtcl_engy());
  ASSERT_EQ(1, test.getShtVrtcl_latcy());

  for (int k = 0; k < 40; k++){
    ASSERT_EQ(test.getEntries(1*8+k, OVER_HEAD + 1 + 2), data_bit[k]); // here
  }

  // write 3
  cout << "write3\n";
  test.write(2, 2, 6, data+2*4, BIT_PURESHIFT, 0);
  test.print();
  ASSERT_EQ(17+(1+2+1+2+5)+(2+1+3+2+5+5), test.getIns_engy());
  ASSERT_EQ(3, test.getIns_latcy());
  ASSERT_EQ((2+1+1)+(0+2+0+4+0+0), test.getDel_engy()); // only when the buffer is full
  ASSERT_EQ(2, test.getDel_latcy());
  ASSERT_EQ(4*8+5*8+6*8, test.getDet_engy());
  ASSERT_EQ(3, test.getDet_latcy());
  ASSERT_EQ(6*DISTANCE+6*5*8+6*6*8, test.getSht_engy());
  ASSERT_EQ(6+6+6, test.getSht_latcy());
  ASSERT_EQ((1+0+1)+(0+0+2+1+0+0), test.getShtVrtcl_engy());
  ASSERT_EQ(1+2, test.getShtVrtcl_latcy());

  for (int k = 0; k < 48; k++){
    ASSERT_EQ(test.getEntries(2*8+k, OVER_HEAD + 1 + 2), data_bit[k+2*DISTANCE]); // here
  }

  // write 4
  cout << "write4\n";
  test.write(32, 1, 1, data, BIT_PURESHIFT, 0);
  #ifdef DEBUG
    test.print();
  #endif
  ASSERT_EQ(17+(1+2+1+2+5)+(2+1+3+2+5+5)+4, test.getIns_engy());
  ASSERT_EQ(4, test.getIns_latcy());
  ASSERT_EQ((2+1+1)+(0+2+0+4+0+0), test.getDel_engy()); // only when the buffer is full
  ASSERT_EQ(2, test.getDel_latcy());
  ASSERT_EQ(4*8+5*8+6*8+8, test.getDet_engy());
  ASSERT_EQ(4, test.getDet_latcy());
  ASSERT_EQ(6*DISTANCE+6*5*8+6*6*8+2*8, test.getSht_engy());
  ASSERT_EQ(6+6+6+2, test.getSht_latcy());
  ASSERT_EQ((1+0+1)+(0+0+2+1+0+0), test.getShtVrtcl_engy());
  ASSERT_EQ(1+2, test.getShtVrtcl_latcy());

  for (int k = 0; k < 8; k++){
    ASSERT_EQ(test.getEntries(1*8+k, OVER_HEAD + 1*(DISTANCE+1) + 1), data_bit[k]); // here
  }

  // write 5
  cout << "write5\n";
  test.write(34, 1, 2, data+1, BIT_PURESHIFT, 0);
  #ifdef DEBUG
    test.print();
  #endif
  ASSERT_EQ(17+(1+2+1+2+5)+(2+1+3+2+5+5)+4+8, test.getIns_engy());
  ASSERT_EQ(5, test.getIns_latcy());
  ASSERT_EQ((2+1+1)+(0+2+0+4+0+0), test.getDel_engy()); // only when the buffer is full
  ASSERT_EQ(2, test.getDel_latcy());
  ASSERT_EQ(4*8+5*8+6*8+8+2*8, test.getDet_engy());
  ASSERT_EQ(5, test.getDet_latcy());
  ASSERT_EQ(6*DISTANCE+6*5*8+6*6*8+2*8+6*2*8, test.getSht_engy());
  ASSERT_EQ(6+6+6+2+6, test.getSht_latcy());
  ASSERT_EQ((1+0+1)+(0+0+2+1+0+0), test.getShtVrtcl_engy());
  ASSERT_EQ(1+2, test.getShtVrtcl_latcy());

  for (int k = 0; k < 16; k++){
    ASSERT_EQ(test.getEntries(8+k, OVER_HEAD + (DISTANCE+1) + 3), data_bit[k+8]); // here
  }

  // write 6
  test.write(90, 2, 3, data, BIT_PURESHIFT, 0);
  #ifdef DEBUG
    test.print();
  #endif
  ASSERT_EQ(17+(1+2+1+2+5)+(2+1+3+2+5+5)+4+8+12, test.getIns_engy());
  ASSERT_EQ(6, test.getIns_latcy());
  ASSERT_EQ((2+1+1)+(0+2+0+4+0+0), test.getDel_engy()); // only when the buffer is full
  ASSERT_EQ(2, test.getDel_latcy());
  ASSERT_EQ(4*8+5*8+6*8+8+2*8+3*8, test.getDet_engy());
  ASSERT_EQ(6, test.getDet_latcy());
  ASSERT_EQ(6*DISTANCE+6*5*8+6*6*8+2*8+6*2*8+12*3*8, test.getSht_engy());
  ASSERT_EQ(6+6+6+2+6+12, test.getSht_latcy());
  ASSERT_EQ((1+0+1)+(0+0+2+1+0+0), test.getShtVrtcl_engy());
  ASSERT_EQ(1+2, test.getShtVrtcl_latcy());

  for (int k = 0; k < 24; k++){
    ASSERT_EQ(test.getEntries(2*8+k, OVER_HEAD + 2*(DISTANCE+1) + 27), data_bit[k]); // here
  }

  // write 7
  test.write(90, 2, 3, data+1, BIT_PURESHIFT, 0);
  #ifdef DEBUG
    test.print();
  #endif
  ASSERT_EQ(17+(1+2+1+2+5)+(2+1+3+2+5+5)+4+8+12+(2+1+1), test.getIns_engy());
  ASSERT_EQ(7, test.getIns_latcy());
  ASSERT_EQ((2+1+1)+(0+2+0+4+0+0)+(1+3+2), test.getDel_engy()); // only when the buffer is full
  ASSERT_EQ(3, test.getDel_latcy());
  ASSERT_EQ(4*8+5*8+6*8+8+2*8+3*8+3*8, test.getDet_engy());
  ASSERT_EQ(7, test.getDet_latcy());
  ASSERT_EQ(6*DISTANCE+6*5*8+6*6*8+2*8+6*2*8+12*3*8+12*3*8, test.getSht_engy());
  ASSERT_EQ(6+6+6+2+6+12+12, test.getSht_latcy());
  ASSERT_EQ((1+0+1)+(0+0+2+1+0+0)+(1+2+2), test.getShtVrtcl_engy());
  ASSERT_EQ(1+2+2, test.getShtVrtcl_latcy());

  for (int k = 0; k < 24; k++){
    ASSERT_EQ(test.getEntries(2*8+k, OVER_HEAD + 2*(DISTANCE+1) + 27), data_bit[k+8]); // here
  }

  // write 8
  test.write(2, 0, 9, data+4, BIT_PURESHIFT, 0);
  #ifdef DEBUG
    test.print();
  #endif
  ASSERT_EQ(17+(1+2+1+2+5)+(2+1+3+2+5+5)+4+8+12+(2+1+1)+(2+1+0+0+3+1+1+1+5), test.getIns_engy());
  ASSERT_EQ(8, test.getIns_latcy());
  ASSERT_EQ((2+1+1)+(0+2+0+4+0+0)+(1+3+2)+(2+1+3+2+1+2+1+3+0), test.getDel_engy()); // only when the buffer is full
  ASSERT_EQ(4, test.getDel_latcy());
  ASSERT_EQ(4*8+5*8+6*8+8+2*8+3*8+3*8+9*8, test.getDet_engy());
  ASSERT_EQ(8, test.getDet_latcy());
  ASSERT_EQ(6*DISTANCE+6*5*8+6*6*8+2*8+6*2*8+12*3*8+12*3*8+6*9*8, test.getSht_engy());
  ASSERT_EQ(6+6+6+2+6+12+12+6, test.getSht_latcy());
  ASSERT_EQ((1+0+1)+(0+0+2+1+0+0)+(1+2+2)+(1+0+0+0+2+2+2+0+0), test.getShtVrtcl_engy());
  ASSERT_EQ(1+2+2+2, test.getShtVrtcl_latcy());

  for (int k = 0; k < 72; k++){
    ASSERT_EQ(test.getEntries(k, OVER_HEAD + 1 + 2), data_bit[k+DISTANCE]); // here
  }
}
*/
TEST(checkWriteBitPURESHIFTOrder1, case1)
{
  SkyrmionBit test;
  for (int k = 0; k < ROW; k++){
    for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
      for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
        test.setEntries(k, i, 0);
      }
    }
  }
  int save = 1;
  sky_size_t data_bit[128]={0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,0}; // 61 bit 1
  sky_size_t data[16]={120, 182, 52, 136, 115, 92, 150, 0, 247, 33, 59, 152, 248, 213, 165, 98};
  sky_size_t data_bit2[72]={0,0,1,0,0,0,0,0,
                            1,1,0,0,0,0,0,0, //data7
                            0,0,0,0,0,0,1,1, //data3
                            1,0,0,0,0,0,1,1, //data5
                            1,1,0,0,0,0,1,1, //data9
                            1,0,1,0,1,1,1,1, //data11
                            1,1,0,0,1,1,0,0, //data12
                            1,0,0,0,1,0,1,1, //data14
                            0,1,1,0,1,0,1,1}; //data8
  sky_size_t data2[9]={32,192,3,131,195,175,204,139,107};
  sky_size_t data_bit3[72]={0,0,0,0,0,1,0,0,
                            0,0,0,1,0,1,1,0, //data2
                            0,1,1,1,0,1,1,0, //data4
                            0,1,1,1,0,1,1,0, //data4
                            0,1,1,0,1,0,1,1, //data8
                            1,1,1,0,0,0,1,1, //data10
                            1,0,1,0,1,1,1,1, //data11
                            0,1,1,0,0,1,1,0, //data13
                            1,1,0,0,0,0,0,0}; //data7
  sky_size_t data3[9]={4,22,118,118,107,227,175,102,192};
  // write 1
  int wBlk = 2;
  int wAdd = 0;
  int wSize = 4;
  int start = 12;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PURESHIFT, save);
  int *ptr = bitShift(wBlk, wSize);
  int se1 = ptr[0];
  int sl1 = ptr[1];
  ASSERT_EQ(17, test.getIns_engy_DMW());
  ASSERT_EQ(1, test.getIns_latcy_DMW());
  ASSERT_EQ(0, test.getDel_engy_DMW());
  ASSERT_EQ(0, test.getDel_latcy_DMW());
  ASSERT_EQ(4*8, test.getDet_engy_DMW());
  ASSERT_EQ(1, test.getDet_latcy_DMW());
  ASSERT_EQ(se1, test.getSht_engy_DMW());
  ASSERT_EQ(sl1, test.getSht_latcy_DMW());
  ASSERT_EQ(0, test.getShtVrtcl_engy_DMW());
  ASSERT_EQ(0, test.getShtVrtcl_latcy_DMW());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 2
  wBlk = 2;
  wAdd = 1;
  wSize = 5;
  start = 0;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PURESHIFT, save);
  ptr = bitShift(wBlk, wSize);
  int se2 = ptr[0];
  int sl2 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5), test.getIns_engy_DMW());
  ASSERT_EQ(2, test.getIns_latcy_DMW());
  ASSERT_EQ((0+0+0+0+0), test.getDel_engy_DMW()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy_DMW());
  ASSERT_EQ(4*8+5*8, test.getDet_engy_DMW());
  ASSERT_EQ(2, test.getDet_latcy_DMW());
  ASSERT_EQ(se1+se2, test.getSht_engy_DMW());
  ASSERT_EQ(sl1+sl2, test.getSht_latcy_DMW());
  ASSERT_EQ((11+11+9+0+0), test.getShtVrtcl_engy_DMW());
  ASSERT_EQ(11, test.getShtVrtcl_latcy_DMW());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 3
  wBlk = 2;
  wAdd = 2;
  wSize = 6;
  start = 8;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PURESHIFT, save);
  ptr = bitShift(wBlk, wSize);
  int se3 = ptr[0];
  int sl3 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5), test.getIns_engy_DMW());
  ASSERT_EQ(3, test.getIns_latcy_DMW());
  ASSERT_EQ((0+0+0+0+0), test.getDel_engy_DMW()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy_DMW());
  ASSERT_EQ(4*8+5*8+6*8, test.getDet_engy_DMW());
  ASSERT_EQ(3, test.getDet_latcy_DMW());
  ASSERT_EQ(se1+se2+se3, test.getSht_engy_DMW());
  ASSERT_EQ(sl1+sl2+sl3, test.getSht_latcy_DMW());
  ASSERT_EQ((11+11+9+0+0)+(9+10+5+3+0+0), test.getShtVrtcl_engy_DMW());
  ASSERT_EQ(11+10, test.getShtVrtcl_latcy_DMW());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 4
  wBlk = 32;
  wAdd = 1;
  wSize = 1;
  start = 0;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PURESHIFT, save);
  ptr = bitShift(wBlk, wSize);
  int se4 = ptr[0];
  int sl4 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5)+4, test.getIns_engy_DMW());
  ASSERT_EQ(4, test.getIns_latcy_DMW());
  ASSERT_EQ((0+0+0+0+0), test.getDel_engy_DMW()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy_DMW());
  ASSERT_EQ(4*8+5*8+6*8+8, test.getDet_engy_DMW());
  ASSERT_EQ(4, test.getDet_latcy_DMW());
  ASSERT_EQ(se1+se2+se3+se4, test.getSht_engy_DMW());
  ASSERT_EQ(sl1+sl2+sl3+sl4, test.getSht_latcy_DMW());
  ASSERT_EQ((11+11+9+0+0)+(9+10+5+3+0+0), test.getShtVrtcl_engy_DMW());
  ASSERT_EQ(11+10, test.getShtVrtcl_latcy_DMW());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 5
  wBlk = 34;
  wAdd = 1;
  wSize = 2;
  start = 1;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PURESHIFT, save);
  ptr = bitShift(wBlk, wSize);
  int se5 = ptr[0];
  int sl5 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5)+4+8, test.getIns_engy_DMW());
  ASSERT_EQ(5, test.getIns_latcy_DMW());
  ASSERT_EQ((0+0+0+0+0), test.getDel_engy_DMW()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy_DMW());
  ASSERT_EQ(4*8+5*8+6*8+8+2*8, test.getDet_engy_DMW());
  ASSERT_EQ(5, test.getDet_latcy_DMW());
  ASSERT_EQ(se1+se2+se3+se4+se5, test.getSht_engy_DMW());
  ASSERT_EQ(sl1+sl2+sl3+sl4+sl5, test.getSht_latcy_DMW());
  ASSERT_EQ((11+11+9+0+0)+(9+10+5+3+0+0), test.getShtVrtcl_engy_DMW());
  ASSERT_EQ(11+10, test.getShtVrtcl_latcy_DMW());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 6
  wBlk = 90;
  wAdd = 2;
  wSize = 3;
  start = 0;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PURESHIFT, save);
  ptr = bitShift(wBlk, wSize);
  int se6 = ptr[0];
  int sl6 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5)+4+8+12, test.getIns_engy_DMW());
  ASSERT_EQ(6, test.getIns_latcy_DMW());
  ASSERT_EQ((0+0+0+0+0), test.getDel_engy_DMW()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy_DMW());
  ASSERT_EQ(4*8+5*8+6*8+8+2*8+3*8, test.getDet_engy_DMW());
  ASSERT_EQ(6, test.getDet_latcy_DMW());
  ASSERT_EQ(se1+se2+se3+se4+se5+se6, test.getSht_engy_DMW());
  ASSERT_EQ(sl1+sl2+sl3+sl4+sl5+sl6, test.getSht_latcy_DMW());
  ASSERT_EQ((11+11+9+0+0)+(9+10+5+3+0+0), test.getShtVrtcl_engy_DMW());
  ASSERT_EQ(11+10, test.getShtVrtcl_latcy_DMW());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 7
  wBlk = 90;
  wAdd = 2;
  wSize = 3;
  start = 1;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PURESHIFT, save);
  ptr = bitShift(wBlk, wSize);
  int se7 = ptr[0];
  int sl7 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5)+4+8+12+(1+0+0), test.getIns_engy_DMW());
  ASSERT_EQ(7, test.getIns_latcy_DMW());
  ASSERT_EQ((0+0+0+0+0), test.getDel_engy_DMW()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy_DMW());
  ASSERT_EQ(4*8+5*8+6*8+8+2*8+3*8+3*8, test.getDet_engy_DMW());
  ASSERT_EQ(7, test.getDet_latcy_DMW());
  ASSERT_EQ(se1+se2+se3+se4+se5+se6+se7, test.getSht_engy_DMW());
  ASSERT_EQ(sl1+sl2+sl3+sl4+sl5+sl6+sl7, test.getSht_latcy_DMW());
  ASSERT_EQ((11+11+9+0+0)+(9+10+5+3+0+0)+(8+11+7), test.getShtVrtcl_engy_DMW());
  ASSERT_EQ(11+10+11, test.getShtVrtcl_latcy_DMW());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 8
  wBlk = 2;
  wAdd = 0;
  wSize = 9;
  start = 4;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PURESHIFT, save);
  ptr = bitShift(wBlk, wSize);
  int se8 = ptr[0];
  int sl8 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5)+4+8+12+(1+0+0)+(0+0+0+0+2+0+0+0+5), test.getIns_engy_DMW());
  ASSERT_EQ(8, test.getIns_latcy_DMW());
  ASSERT_EQ((0+0+0+0+0), test.getDel_engy_DMW()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy_DMW());
  ASSERT_EQ(4*8+5*8+6*8+8+2*8+3*8+3*8+9*8, test.getDet_engy_DMW());
  ASSERT_EQ(8, test.getDet_latcy_DMW());
  ASSERT_EQ(se1+se2+se3+se4+se5+se6+se7+se8, test.getSht_engy_DMW());
  ASSERT_EQ(sl1+sl2+sl3+sl4+sl5+sl6+sl7+sl8, test.getSht_latcy_DMW());
  ASSERT_EQ((11+11+9+0+0)+(9+10+5+3+0+0)+(8+11+7)+(9+5+11+0+8+10+11+9+11), test.getShtVrtcl_engy_DMW());
  ASSERT_EQ(11+10+11+11, test.getShtVrtcl_latcy_DMW());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write9
  wBlk = 3;
  wAdd = 0;
  wSize = 9;
  start = 0;
  test.write(wBlk, wAdd, wSize, data2, BIT_PURESHIFT, save);
  ptr = bitShift(wBlk, wSize);
  int se9 = ptr[0];
  int sl9 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5)+4+8+12+(1+0+0)+(0+0+0+0+2+0+0+0+5)+31, test.getIns_engy_DMW());
  ASSERT_EQ(9, test.getIns_latcy_DMW());
  ASSERT_EQ((0+0+0+0+0), test.getDel_engy_DMW()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy_DMW());
  ASSERT_EQ(4*8+5*8+6*8+8+2*8+3*8+3*8+9*8+9*8, test.getDet_engy_DMW());
  ASSERT_EQ(9, test.getDet_latcy_DMW());
  ASSERT_EQ(se1+se2+se3+se4+se5+se6+se7+se8+se9, test.getSht_engy_DMW());
  ASSERT_EQ(sl1+sl2+sl3+sl4+sl5+sl6+sl7+sl8+sl9, test.getSht_latcy_DMW());
  ASSERT_EQ((11+11+9+0+0)+(9+10+5+3+0+0)+(8+11+7)+(9+5+11+0+8+10+11+9+11), test.getShtVrtcl_engy_DMW());
  ASSERT_EQ(11+10+11+11, test.getShtVrtcl_latcy_DMW());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit2[k]); // here
  }

  // write10
  wBlk = 3;
  wAdd = 0;
  wSize = 9;
  start = 0;
  test.write(wBlk, wAdd, wSize, data3, BIT_PURESHIFT, save);
  ptr = bitShift(wBlk, wSize);
  int se10 = ptr[0];
  int sl10 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5)+4+8+12+(1+0+0)+(0+0+0+0+2+0+0+0+5)+31+(0+1+3+2+1+0+2+0+0), test.getIns_engy_DMW());
  ASSERT_EQ(10, test.getIns_latcy_DMW());
  ASSERT_EQ((0+0+0+0+0), test.getDel_engy_DMW()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy_DMW());
  ASSERT_EQ(4*8+5*8+6*8+8+2*8+3*8+3*8+9*8+9*8+9*8, test.getDet_engy_DMW());
  ASSERT_EQ(10, test.getDet_latcy_DMW());
  ASSERT_EQ(se1+se2+se3+se4+se5+se6+se7+se8+se9+se10, test.getSht_engy_DMW());
  ASSERT_EQ(sl1+sl2+sl3+sl4+sl5+sl6+sl7+sl8+sl9+sl10, test.getSht_latcy_DMW());
  ASSERT_EQ((11+11+9+0+0)+(9+10+5+3+0+0)+(8+11+7)+(9+5+11+0+8+10+11+9+11)+(3+5+1+5+6+12+6+9+6), test.getShtVrtcl_engy_DMW());
  ASSERT_EQ(11+10+11+11+12, test.getShtVrtcl_latcy_DMW());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit3[k]); // here
  }
}

TEST(checkWriteBitPWIMPROVED, case1)
{
  SkyrmionBit test;
  for (int k = 0; k < ROW; k++){
    for (int j = 0; j < MAX_SIZE / DISTANCE; j++){
      for (int i = OVER_HEAD + 1 + j * (DISTANCE + 1); i <= OVER_HEAD + j * (DISTANCE + 1) + DISTANCE; i++){
        test.setEntries(k, i, 0);
      }
    }
  }
  int save = 1;
  sky_size_t data_bit[128]={0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,0}; // 61 bit 1
  sky_size_t data[16]={120, 182, 52, 136, 115, 92, 150, 0, 247, 33, 59, 152, 248, 213, 165, 98};
  sky_size_t data_bit2[88]={0,0,1,0,0,0,0,0,
                            1,1,0,0,0,0,0,0, //data7
                            0,0,0,0,0,0,1,1, //data3
                            1,0,0,0,0,0,1,1, //data5
                            1,1,0,0,0,0,1,1, //data9
                            1,0,1,0,1,1,1,1, //data11
                            1,1,0,0,1,1,0,0, //data12
                            1,0,0,0,1,0,1,1, //data14
                            0,1,1,0,1,0,1,1, //data8
                            0,0,1,1,0,0,1,0,
                            0,1,0,0,1,0,0,0};
  sky_size_t data2[11]={32,192,3,131,195,175,204,139,107,50,192};
  sky_size_t data_bit3[88]={0,0,0,0,0,1,0,0,
                            0,0,0,1,0,1,1,0, //data2
                            0,1,1,1,0,1,1,0, //data4
                            0,1,1,1,0,1,1,0, //data4
                            0,1,1,0,1,0,1,1, //data8
                            1,1,1,0,0,0,1,1, //data10
                            1,0,1,0,1,1,1,1, //data11
                            0,1,1,0,0,1,1,0, //data13
                            1,1,0,0,0,0,0,0, //data7
                            0,0,1,0,0,0,0,0,
                            1,0,1,0,0,0,0,0};
  sky_size_t data3[11]={4,22,118,118,107,227,175,102,192,32,160};
  // write 1
  int wBlk = 2;
  int wAdd = 0;
  int wSize = 4;
  int start = 12;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PW_IMPROVED, save);
  int *ptr = bitShift(wBlk, wSize);
  int se1 = ptr[0];
  int sl1 = ptr[1];
  ASSERT_EQ(17, test.getIns_engy_DMW());
  ASSERT_EQ(1, test.getIns_latcy_DMW());
  ASSERT_EQ(0, test.getDel_engy_DMW());
  ASSERT_EQ(0, test.getDel_latcy_DMW());
  ASSERT_EQ(4*8, test.getDet_engy_DMW());
  ASSERT_EQ(1, test.getDet_latcy_DMW());
  ASSERT_EQ(se1, test.getSht_engy_DMW());
  ASSERT_EQ(sl1, test.getSht_latcy_DMW());
  ASSERT_EQ(0, test.getShtVrtcl_engy_DMW());
  ASSERT_EQ(0, test.getShtVrtcl_latcy_DMW());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 2
  wBlk = 2;
  wAdd = 1;
  wSize = 5;
  start = 0;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PW_IMPROVED, save);
  ptr = bitShift(wBlk, wSize);
  int se2 = ptr[0];
  int sl2 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5), test.getIns_engy_DMW());
  ASSERT_EQ(2, test.getIns_latcy_DMW());
  ASSERT_EQ((0+0+0+0+0), test.getDel_engy_DMW()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy_DMW());
  ASSERT_EQ(4*8+5*8, test.getDet_engy_DMW());
  ASSERT_EQ(2, test.getDet_latcy_DMW());
  ASSERT_EQ(se1+se2, test.getSht_engy_DMW());
  ASSERT_EQ(sl1+sl2, test.getSht_latcy_DMW());
  ASSERT_EQ((11+11+9+0+0), test.getShtVrtcl_engy_DMW());
  ASSERT_EQ(11, test.getShtVrtcl_latcy_DMW());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 3
  wBlk = 2;
  wAdd = 2;
  wSize = 6;
  start = 8;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PW_IMPROVED, save);
  ptr = bitShift(wBlk, wSize);
  int se3 = ptr[0];
  int sl3 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5), test.getIns_engy_DMW());
  ASSERT_EQ(3, test.getIns_latcy_DMW());
  ASSERT_EQ((0+0+0+0+0), test.getDel_engy_DMW()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy_DMW());
  ASSERT_EQ(4*8+5*8+6*8, test.getDet_engy_DMW());
  ASSERT_EQ(3, test.getDet_latcy_DMW());
  ASSERT_EQ(se1+se2+se3, test.getSht_engy_DMW());
  ASSERT_EQ(sl1+sl2+sl3, test.getSht_latcy_DMW());
  ASSERT_EQ((11+11+9+0+0)+(0+10+2+3+0+0), test.getShtVrtcl_engy_DMW());
  ASSERT_EQ(11+10, test.getShtVrtcl_latcy_DMW());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 4
  wBlk = 32;
  wAdd = 1;
  wSize = 1;
  start = 0;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PW_IMPROVED, save);
  ptr = bitShift(wBlk, wSize);
  int se4 = ptr[0];
  int sl4 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5)+4, test.getIns_engy_DMW());
  ASSERT_EQ(4, test.getIns_latcy_DMW());
  ASSERT_EQ((0+0+0+0+0), test.getDel_engy_DMW()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy_DMW());
  ASSERT_EQ(4*8+5*8+6*8+8, test.getDet_engy_DMW());
  ASSERT_EQ(4, test.getDet_latcy_DMW());
  ASSERT_EQ(se1+se2+se3+se4, test.getSht_engy_DMW());
  ASSERT_EQ(sl1+sl2+sl3+sl4, test.getSht_latcy_DMW());
  ASSERT_EQ((11+11+9+0+0)+(0+10+2+3+0+0), test.getShtVrtcl_engy_DMW());
  ASSERT_EQ(11+10, test.getShtVrtcl_latcy_DMW());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 5
  wBlk = 34;
  wAdd = 1;
  wSize = 2;
  start = 1;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PW_IMPROVED, save);
  ptr = bitShift(wBlk, wSize);
  int se5 = ptr[0];
  int sl5 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5)+4+8, test.getIns_engy_DMW());
  ASSERT_EQ(5, test.getIns_latcy_DMW());
  ASSERT_EQ((0+0+0+0+0), test.getDel_engy_DMW()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy_DMW());
  ASSERT_EQ(4*8+5*8+6*8+8+2*8, test.getDet_engy_DMW());
  ASSERT_EQ(5, test.getDet_latcy_DMW());
  ASSERT_EQ(se1+se2+se3+se4+se5, test.getSht_engy_DMW());
  ASSERT_EQ(sl1+sl2+sl3+sl4+sl5, test.getSht_latcy_DMW());
  ASSERT_EQ((11+11+9+0+0)+(0+10+2+3+0+0), test.getShtVrtcl_engy_DMW());
  ASSERT_EQ(11+10, test.getShtVrtcl_latcy_DMW());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 6
  wBlk = 90;
  wAdd = 2;
  wSize = 3;
  start = 0;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PW_IMPROVED, save);
  ptr = bitShift(wBlk, wSize);
  int se6 = ptr[0];
  int sl6 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5)+4+8+12, test.getIns_engy_DMW());
  ASSERT_EQ(6, test.getIns_latcy_DMW());
  ASSERT_EQ((0+0+0+0+0), test.getDel_engy_DMW()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy_DMW());
  ASSERT_EQ(4*8+5*8+6*8+8+2*8+3*8, test.getDet_engy_DMW());
  ASSERT_EQ(6, test.getDet_latcy_DMW());
  ASSERT_EQ(se1+se2+se3+se4+se5+se6, test.getSht_engy_DMW());
  ASSERT_EQ(sl1+sl2+sl3+sl4+sl5+sl6, test.getSht_latcy_DMW());
  ASSERT_EQ((11+11+9+0+0)+(0+10+2+3+0+0), test.getShtVrtcl_engy_DMW());
  ASSERT_EQ(11+10, test.getShtVrtcl_latcy_DMW());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 7
  wBlk = 90;
  wAdd = 2;
  wSize = 3;
  start = 1;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PW_IMPROVED, save);
  ptr = bitShift(wBlk, wSize);
  int se7 = ptr[0];
  int sl7 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5)+4+8+12+(1+0+0), test.getIns_engy_DMW());
  ASSERT_EQ(7, test.getIns_latcy_DMW());
  ASSERT_EQ((0+0+0+0+0), test.getDel_engy_DMW()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy_DMW());
  ASSERT_EQ(4*8+5*8+6*8+8+2*8+3*8+3*8, test.getDet_engy_DMW());
  ASSERT_EQ(7, test.getDet_latcy_DMW());
  ASSERT_EQ(se1+se2+se3+se4+se5+se6+se7, test.getSht_engy_DMW());
  ASSERT_EQ(sl1+sl2+sl3+sl4+sl5+sl6+sl7, test.getSht_latcy_DMW());
  ASSERT_EQ((11+11+9+0+0)+(0+10+2+3+0+0)+(8+11+7), test.getShtVrtcl_engy_DMW());
  ASSERT_EQ(11+10+11, test.getShtVrtcl_latcy_DMW());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write 8
  wBlk = 2;
  wAdd = 0;
  wSize = 9;
  start = 4;
  test.write(wBlk, wAdd, wSize, data+start, BIT_PW_IMPROVED, save);
  ptr = bitShift(wBlk, wSize);
  int se8 = ptr[0];
  int sl8 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5)+4+8+12+(1+0+0)+(0+0+0+0+2+0+0+0+5), test.getIns_engy_DMW());
  ASSERT_EQ(8, test.getIns_latcy_DMW());
  ASSERT_EQ((0+0+0+0+0), test.getDel_engy_DMW()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy_DMW());
  ASSERT_EQ(4*8+5*8+6*8+8+2*8+3*8+3*8+9*8, test.getDet_engy_DMW());
  ASSERT_EQ(8, test.getDet_latcy_DMW());
  ASSERT_EQ(se1+se2+se3+se4+se5+se6+se7+se8, test.getSht_engy_DMW());
  ASSERT_EQ(sl1+sl2+sl3+sl4+sl5+sl6+sl7+sl8, test.getSht_latcy_DMW());
  ASSERT_EQ((11+11+9+0+0)+(0+10+2+3+0+0)+(8+11+7)+(9+5+11+0+8+10+11+9+11), test.getShtVrtcl_engy_DMW());
  ASSERT_EQ(11+10+11+11, test.getShtVrtcl_latcy_DMW());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit[k+start*8]); // here
  }

  // write9
  wBlk = 3;
  wAdd = 0;
  wSize = 11;
  start = 0;
  test.write(wBlk, wAdd, wSize, data2, BIT_PW_IMPROVED, save);
  ptr = bitShift(wBlk, wSize);
  int se9 = ptr[0];
  int sl9 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5)+4+8+12+(1+0+0)+(0+0+0+0+2+0+0+0+5)+36, test.getIns_engy_DMW());
  ASSERT_EQ(9, test.getIns_latcy_DMW());
  ASSERT_EQ((0+0+0+0+0), test.getDel_engy_DMW()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy_DMW());
  ASSERT_EQ(4*8+5*8+6*8+8+2*8+3*8+3*8+9*8+11*8, test.getDet_engy_DMW());
  ASSERT_EQ(9, test.getDet_latcy_DMW());
  ASSERT_EQ(se1+se2+se3+se4+se5+se6+se7+se8+se9, test.getSht_engy_DMW());
  ASSERT_EQ(sl1+sl2+sl3+sl4+sl5+sl6+sl7+sl8+sl9, test.getSht_latcy_DMW());
  ASSERT_EQ((11+11+9+0+0)+(0+10+2+3+0+0)+(8+11+7)+(9+5+11+0+8+10+11+9+11), test.getShtVrtcl_engy_DMW());
  ASSERT_EQ(11+10+11+11, test.getShtVrtcl_latcy_DMW());
  delete [] ptr;

  for (int k = 0; k < 80; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit2[k]); // here
  }

  // write10
  //test.buffer[9 * MAX_SIZE + 3] = 7;
  wBlk = 3;
  wAdd = 0;
  wSize = 11;
  start = 0;
  test.write(wBlk, wAdd, wSize, data3, BIT_PW_IMPROVED, save);
  ptr = bitShift(wBlk, wSize);
  int se10 = ptr[0];
  int sl10 = ptr[1];
  ASSERT_EQ(17+(0+1+0+2+5)+(2+0+3+0+5+5)+4+8+12+(1+0+0)+(0+0+0+0+2+0+0+0+5)+36+(0+1+3+2+1+0+2+0+0+0+0), test.getIns_engy_DMW());
  ASSERT_EQ(10, test.getIns_latcy_DMW());
  ASSERT_EQ((0+0+0+0+0), test.getDel_engy_DMW()); // only when the buffer is full
  ASSERT_EQ(0, test.getDel_latcy_DMW());
  ASSERT_EQ(4*8+5*8+6*8+8+2*8+3*8+3*8+9*8+11*8+11*8, test.getDet_engy_DMW());
  ASSERT_EQ(10, test.getDet_latcy_DMW());
  ASSERT_EQ(se1+se2+se3+se4+se5+se6+se7+se8+se9+se10, test.getSht_engy_DMW());
  ASSERT_EQ(sl1+sl2+sl3+sl4+sl5+sl6+sl7+sl8+sl9+sl10, test.getSht_latcy_DMW());
  ASSERT_EQ((11+11+9+0+0)+(0+10+2+3+0+0)+(8+11+7)+(9+5+11+0+8+10+11+9+11)+(3+5+1+5+6+12+6+9+6+4+3), test.getShtVrtcl_engy_DMW());
  ASSERT_EQ(11+10+11+11+12, test.getShtVrtcl_latcy_DMW());
  delete [] ptr;

  for (int k = 0; k < wSize*8; k++){
    ASSERT_EQ(test.getEntries(wAdd*8+k, OVER_HEAD + (wBlk+wBlk/DISTANCE)+1), data_bit3[k]); // here
  }
}
