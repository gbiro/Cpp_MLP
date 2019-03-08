#ifndef preprocessor_h
#define preprocessor_h

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

class Preprocessor {

private:
  int verbosity;

public:
  void setVerbosity(int v);

  void normalize(vector<vector<float>> &images, int dimx = 0, int dimy = 0);

  void average(vector<vector<float>> &images, int dimx, int dimy);
};

#endif