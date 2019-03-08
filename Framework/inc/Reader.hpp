#ifndef reader_h
#define reader_h

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;

class Reader {

private:
  int verbosity;

public:
  void setVerbosity(int v);
  pair<int, int> read_Mnist(string filename, vector<vector<float>> &vec);
  bool read_Mnist_Label(string filename, vector<float> &vec);
  int ReverseInt(int i);
};

#endif